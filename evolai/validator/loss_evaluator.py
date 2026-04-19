
from __future__ import annotations

import contextlib
import gc
import json
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HF_LOSS_BATCH_SIZE, HF_LOSS_MAX_SEQ_LEN

logger = logging.getLogger(__name__)


@dataclass
class ChallengeSpec:
    uid: int


    datasets: Dict[str, List[int]]


@dataclass
class LossRecord:
    loss: float
    timestamp: str
    model_name: str
    dataset_name: str
    revision: str = ""


@dataclass
class MinerLossState:
    uid: int
    coldkey: str = ""
    hotkey: str = ""
    model_name: str = ""

    loss_history: List[LossRecord] = field(default_factory=list)

    best_loss: float = float("inf")

    eval_count: int = 0

    last_reward: float = 0.0

    cumulative_reward: float = 0.0


@dataclass
class ChatSample:
    instruction: str
    response: str


def compute_cross_entropy_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[Union[str, ChatSample]],
    max_length: int = HF_LOSS_MAX_SEQ_LEN,
    batch_size: int = HF_LOSS_BATCH_SIZE,
    device: str = "cuda",
    progress_callback: Optional["Callable[[int, int], None]"] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    if hasattr(model, "config"):
        model.config.use_cache = False

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    model_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    use_autocast = is_cuda and model_dtype in (torch.float16, torch.bfloat16)


    autocast_ctx: contextlib.AbstractContextManager
    if use_autocast:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()


    plain_texts  = [t for t in texts if isinstance(t, str)]
    chat_samples = [t for t in texts if isinstance(t, ChatSample)]
    total_items  = len(texts)


    has_template = bool(getattr(tokenizer, "chat_template", None))
    with torch.inference_mode():
        for i, sample in enumerate(chat_samples):
            if has_template:
                full_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user",      "content": sample.instruction},
                        {"role": "assistant", "content": sample.response},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample.instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:

                full_text   = f"### Human: {sample.instruction}\n\n### Assistant: {sample.response}"
                prompt_text = f"### Human: {sample.instruction}\n\n### Assistant:"

            full_enc   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=max_length)
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)

            full_len   = full_enc["input_ids"].shape[1]
            prompt_len = min(prompt_enc["input_ids"].shape[1], full_len)

            input_ids      = full_enc["input_ids"].to(device)
            attention_mask = full_enc["attention_mask"].to(device)
            labels = input_ids.clone()

            labels[0, :prompt_len] = -100
            labels[attention_mask == 0] = -100

            response_tokens = int((labels[0] != -100).sum().item())
            if progress_callback is not None:
                progress_callback(i + 1, total_items)
            if response_tokens == 0:
                del input_ids, attention_mask, labels
                continue

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )

            loss_val = float(outputs.loss.detach().float().item())
            total_loss   += loss_val * response_tokens
            total_tokens += response_tokens
            del input_ids, attention_mask, labels, outputs

    if is_cuda and chat_samples:
        torch.cuda.empty_cache()


    sorted_texts = sorted((t for t in plain_texts if t and t.strip()), key=len)

    max_batch_size = max(1, batch_size)
    current_batch_size = max_batch_size
    consecutive_successes = 0
    oom_retries = 0
    max_oom_retries = 5
    batch_count = 0
    index = 0
    chat_done = len(chat_samples)

    with torch.inference_mode():
        while index < len(sorted_texts):
            batch_texts = sorted_texts[index : index + current_batch_size]
            input_ids = attention_mask = labels = outputs = None
            try:
                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    pad_to_multiple_of=8 if is_cuda else None,
                )

                input_ids = encodings["input_ids"].to(device, non_blocking=True)
                attention_mask = encodings["attention_mask"].to(device, non_blocking=True)

                valid_tokens = (attention_mask.sum(dim=1) - 1).clamp(min=0)
                num_tokens = int(valid_tokens.sum().item())
                if num_tokens == 0:
                    index += len(batch_texts)
                    consecutive_successes += 1
                    continue

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                with autocast_ctx:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )

                loss_val = float(outputs.loss.detach().float().item())
                total_loss += loss_val * num_tokens
                total_tokens += num_tokens
                index += len(batch_texts)
                consecutive_successes += 1
                oom_retries = 0
                batch_count += 1

                if progress_callback is not None:
                    progress_callback(chat_done + index, total_items)


                del input_ids, attention_mask, labels, outputs


                if is_cuda and batch_count % 8 == 0:
                    torch.cuda.empty_cache()


                if consecutive_successes >= 5 and current_batch_size < max_batch_size:
                    current_batch_size = min(current_batch_size * 2, max_batch_size)
                    consecutive_successes = 0
                    logger.debug(f"Recovered batch_size to {current_batch_size}")

            except torch.cuda.OutOfMemoryError:
                oom_retries += 1
                if not is_cuda or current_batch_size == 1 or oom_retries > max_oom_retries:

                    for _t in (input_ids, attention_mask, labels, outputs):
                        if _t is not None:
                            del _t
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3) if is_cuda else 0
                    raise RuntimeError(
                        f"GPU out of memory after {oom_retries} OOM retries "
                        f"(batch_size reduced to {current_batch_size}, "
                        f"seq_len={max_length}, free VRAM ~{free_gb:.1f} GB). "
                        f"The model is too large for this GPU. "
                        f"Try lowering HF_EVAL_BATCH_<SIZE> / HF_EVAL_SEQLEN_<SIZE> "
                        f"or use a GPU with more VRAM."
                    ) from None

                for _t in (input_ids, attention_mask, labels, outputs):
                    if _t is not None:
                        del _t
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                consecutive_successes = 0
                logger.warning(
                    f"OOM during loss evaluation (retry {oom_retries}/{max_oom_retries}); "
                    f"reducing batch_size to {current_batch_size}"
                )
                continue


    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    if total_tokens == 0:
        tokenizer.padding_side = _saved_padding_side
        return float("inf")

    tokenizer.padding_side = _saved_padding_side
    return total_loss / total_tokens


def compute_thinking_eval_loss(
    model: AutoModelForCausalLM,
    ref_tokenizer: AutoTokenizer,
    samples: "List[ChatSample]",
    max_new_tokens: int = 512,
    max_length: int = HF_LOSS_MAX_SEQ_LEN,
    temperature: float = 1.0,
    device: str = "cuda",
    progress_callback: "Optional[Callable[[int, int], None]]" = None,
    penalty_loss: float = 10.0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_items = len(samples)


    if ref_tokenizer.pad_token is None and ref_tokenizer.eos_token is not None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    _saved_padding_side = ref_tokenizer.padding_side
    ref_tokenizer.padding_side = "left"

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    use_autocast = is_cuda and model_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx: contextlib.AbstractContextManager
    if use_autocast:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    think_end = "</think>"
    do_sample = temperature > 0

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "stop_strings": [think_end],
        "tokenizer": ref_tokenizer,
        "use_cache": True,
        "pad_token_id": ref_tokenizer.pad_token_id or ref_tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        for i, sample in enumerate(samples):
            prompt_ids = attn_mask = gen_out = None
            resp_ids = full_ids = full_attn = labels = outputs = None
            try:

                prompt_text = ref_tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample.instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_text += "<think>"

                prompt_enc = ref_tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                prompt_ids = prompt_enc["input_ids"].to(device)
                attn_mask = prompt_enc["attention_mask"].to(device)


                with autocast_ctx:
                    gen_out = model.generate(
                        prompt_ids, attention_mask=attn_mask, **gen_kwargs,
                    )

                think_total_len = gen_out.shape[1]


                remaining_len = max(1, max_length - think_total_len)
                resp_enc = ref_tokenizer(
                    sample.response,
                    return_tensors="pt",
                    truncation=True,
                    max_length=remaining_len,
                    add_special_tokens=False,
                )
                resp_ids = resp_enc["input_ids"].to(device)
                resp_len = resp_ids.shape[1]

                if resp_len == 0:
                    del prompt_ids, attn_mask, gen_out, resp_ids
                    if progress_callback:
                        progress_callback(i + 1, total_items)
                    continue


                full_ids = torch.cat([gen_out, resp_ids], dim=1)
                full_attn = torch.ones(
                    1, full_ids.shape[1], dtype=torch.long, device=device,
                )


                labels = full_ids.clone()
                labels[0, :think_total_len] = -100


                with autocast_ctx:
                    outputs = model(
                        input_ids=full_ids,
                        attention_mask=full_attn,
                        labels=labels,
                        use_cache=False,
                    )

                loss_val = float(outputs.loss.detach().float().item())
                total_loss += loss_val * resp_len
                total_tokens += resp_len

                del prompt_ids, attn_mask, gen_out, resp_ids
                del full_ids, full_attn, labels, outputs

            except torch.cuda.OutOfMemoryError:
                for _t in (prompt_ids, attn_mask, gen_out, resp_ids,
                           full_ids, full_attn, labels, outputs):
                    if _t is not None:
                        del _t
                if is_cuda:
                    torch.cuda.empty_cache()
                gc.collect()


                estimated_resp_len = max(1, len(sample.response) // 4)
                total_loss += penalty_loss * estimated_resp_len
                total_tokens += estimated_resp_len

                logger.warning(
                    f"OOM during thinking eval sample {i + 1}/{total_items} "
                    f"— recording penalty loss ({penalty_loss})"
                )
                if progress_callback:
                    progress_callback(i + 1, total_items)
                continue

            if is_cuda and (i + 1) % 4 == 0:
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(i + 1, total_items)

    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    if total_tokens == 0:
        ref_tokenizer.padding_side = _saved_padding_side
        return float("inf")

    ref_tokenizer.padding_side = _saved_padding_side
    return total_loss / total_tokens


def evaluate_with_side_quests(
    model: AutoModelForCausalLM,
    ref_tokenizer: AutoTokenizer,
    samples: "List[ChatSample]",
    block_hash: str,
    max_new_tokens: int = 32,
    think_max_new_tokens: int = 512,
    max_length: int = HF_LOSS_MAX_SEQ_LEN,
    device: str = "cuda",
    progress_callback: "Optional[Callable[[int, int], None]]" = None,
    penalty_loss: float = 10.0,
) -> "Tuple[float, float]":
    """
    Evaluate miner via a 3-turn shuffled conversation per sample:

    * Side quest turns (2 per sample): model generates freely (max_new_tokens),
      then binary loss — 0.0 if the correct integer answer appears in the
      output, 1.0 if not.  No CE computed for these turns.

    * Real sample turn (1 per sample): model generates <think>...</think>
      tokens freely (think_max_new_tokens), then CE loss is measured on the
      ground-truth response tokens only (teacher-forcing after the thinking
      prefix).

    Turn order is shuffled deterministically by block_hash + sample_index.
    Prior turns always use ground-truth context so evaluation is deterministic.

    Returns:
        (ce_loss, sq_accuracy)
        ce_loss      – token-weighted mean CE over real response turns.
        sq_accuracy  – fraction of side quests answered correctly (0.0–1.0).
    """
    from .side_quests import generate_side_quests, shuffle_turn_order, check_side_quest_answer

    model.eval()
    total_items = len(samples)
    total_ce_loss = 0.0
    total_ce_tokens = 0
    sq_correct = 0
    sq_total = 0

    if ref_tokenizer.pad_token is None and ref_tokenizer.eos_token is not None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    _saved_padding_side = ref_tokenizer.padding_side
    _saved_truncation_side = getattr(ref_tokenizer, "truncation_side", "right")
    # Left padding for generation; left truncation preserves response end on overflow.
    ref_tokenizer.padding_side = "left"
    ref_tokenizer.truncation_side = "left"

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    model_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    use_autocast = is_cuda and model_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx: contextlib.AbstractContextManager
    if use_autocast:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    has_template = bool(getattr(ref_tokenizer, "chat_template", None))
    gen_base: dict = {
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": ref_tokenizer.pad_token_id or ref_tokenizer.eos_token_id,
    }
    think_end = "</think>"

    with torch.inference_mode():
        for sample_idx, sample in enumerate(samples):
            # Snapshot sq counters so OOM can roll them back cleanly.
            sq_correct_before = sq_correct
            sq_total_before   = sq_total
            accumulated       = False

            # Per-sample CE result — set only when the real turn succeeds.
            sample_ce_loss:   "Optional[float]" = None
            sample_ce_tokens: int = 0

            # Named tensor refs for safe OOM cleanup.
            prompt_ids = attn_mask = gen_out = gen_ids = None
            resp_ids = full_ids = full_attn = labels = outputs = None

            try:
                quests     = generate_side_quests(block_hash, sample_idx, n=2)
                turn_order = shuffle_turn_order(block_hash, sample_idx, n_turns=3)

                # index 0 = real sample, 1 = quest[0], 2 = quest[1]
                turns = [
                    {"type": "real", "q": sample.instruction, "a": sample.response},
                    {"type": "sq",   "q": quests[0].question, "a": quests[0].answer, "quest": quests[0]},
                    {"type": "sq",   "q": quests[1].question, "a": quests[1].answer, "quest": quests[1]},
                ]
                ordered_turns = [turns[i] for i in turn_order]
                messages_so_far: list = []

                for turn in ordered_turns:
                    q_text    = turn["q"]
                    a_text    = turn["a"]
                    turn_type = turn["type"]

                    # ---------- build generation prompt ----------
                    if has_template:
                        prompt_text = ref_tokenizer.apply_chat_template(
                            messages_so_far + [{"role": "user", "content": q_text}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        if turn_type == "real":
                            prompt_text += "<think>"
                    else:
                        ctx = ""
                        for m in messages_so_far:
                            role_prefix = "Human" if m["role"] == "user" else "Assistant"
                            ctx += f"### {role_prefix}: {m['content']}\n\n"
                        if turn_type == "real":
                            prompt_text = ctx + f"### Human: {q_text}\n\n### Assistant: <think>"
                        else:
                            prompt_text = ctx + f"### Human: {q_text}\n\n### Assistant:"

                    prompt_enc = ref_tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )
                    prompt_ids = prompt_enc["input_ids"].to(device)
                    attn_mask  = prompt_enc["attention_mask"].to(device)

                    if turn_type == "sq":
                        # ---- free generation → binary correctness -----------
                        with autocast_ctx:
                            gen_out = model.generate(
                                prompt_ids,
                                attention_mask=attn_mask,
                                max_new_tokens=max_new_tokens,
                                **gen_base,
                            )
                        gen_ids  = gen_out[0, prompt_ids.shape[1]:]
                        gen_text = ref_tokenizer.decode(gen_ids, skip_special_tokens=True)
                        correct  = check_side_quest_answer(gen_text, turn["quest"])
                        sq_correct += int(correct)
                        sq_total   += 1
                        del prompt_ids, attn_mask, gen_out, gen_ids
                        prompt_ids = attn_mask = gen_out = gen_ids = None

                    else:
                        # ---- generate thinking, then CE on ground-truth response ----
                        with autocast_ctx:
                            gen_out = model.generate(
                                prompt_ids,
                                attention_mask=attn_mask,
                                max_new_tokens=think_max_new_tokens,
                                stop_strings=[think_end],
                                tokenizer=ref_tokenizer,
                                **gen_base,
                            )
                        think_prefix_len = gen_out.shape[1]
                        del prompt_ids, attn_mask
                        prompt_ids = attn_mask = None

                        # Append ground-truth response tokens after thinking.
                        remaining = max(1, max_length - think_prefix_len)
                        resp_enc  = ref_tokenizer(
                            a_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=remaining,
                            add_special_tokens=False,
                        )
                        resp_ids = resp_enc["input_ids"].to(device)
                        resp_len = resp_ids.shape[1]

                        if resp_len > 0:
                            full_ids  = torch.cat([gen_out, resp_ids], dim=1)
                            full_attn = torch.ones(
                                1, full_ids.shape[1], dtype=torch.long, device=device,
                            )
                            labels = full_ids.clone()
                            labels[0, :think_prefix_len] = -100  # mask prompt + thinking
                            with autocast_ctx:
                                outputs = model(
                                    input_ids=full_ids,
                                    attention_mask=full_attn,
                                    labels=labels,
                                    use_cache=False,
                                )
                            sample_ce_loss   = float(outputs.loss.detach().float().item())
                            sample_ce_tokens = resp_len
                            del full_ids, full_attn, labels, outputs
                            full_ids = full_attn = labels = outputs = None

                        del gen_out, resp_ids
                        gen_out = resp_ids = None

                    if is_cuda:
                        torch.cuda.empty_cache()

                    # Always advance context with ground-truth (deterministic).
                    messages_so_far += [
                        {"role": "user",      "content": q_text},
                        {"role": "assistant", "content": a_text},
                    ]

                # --- accumulate this sample's CE result ----------------------
                if sample_ce_loss is not None:
                    total_ce_loss   += sample_ce_loss * sample_ce_tokens
                    total_ce_tokens += sample_ce_tokens
                else:
                    # Real turn was skipped (resp_len == 0 or never reached).
                    total_ce_loss   += penalty_loss
                    total_ce_tokens += 1
                accumulated = True

            except torch.cuda.OutOfMemoryError:
                for _t in (prompt_ids, attn_mask, gen_out, gen_ids,
                           resp_ids, full_ids, full_attn, labels, outputs):
                    if _t is not None:
                        del _t
                if is_cuda:
                    torch.cuda.empty_cache()
                gc.collect()
                if not accumulated:
                    total_ce_loss   += penalty_loss
                    total_ce_tokens += 1
                # Roll back partial sq counts; treat all 2 quests as wrong.
                sq_correct = sq_correct_before
                sq_total   = sq_total_before + 2
                logger.warning(
                    f"OOM during side-quest eval sample {sample_idx + 1}/"
                    f"{total_items} — recording penalty"
                )

            if is_cuda:
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(sample_idx + 1, total_items)

    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    ref_tokenizer.padding_side = _saved_padding_side
    ref_tokenizer.truncation_side = _saved_truncation_side

    ce_final    = total_ce_loss / total_ce_tokens if total_ce_tokens > 0 else penalty_loss
    sq_accuracy = sq_correct / sq_total if sq_total > 0 else 0.0
    return ce_final, sq_accuracy


def compute_loss_vllm(
    texts: List[str],
    model_name: str,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    max_tokens_per_text: int = 1,
) -> float:
    import httpx

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        try:
            resp = httpx.post(
                f"{vllm_base_url}/completions",
                json={
                    "model": model_name,
                    "prompt": text,
                    "max_tokens": max_tokens_per_text,
                    "logprobs": 1,
                    "echo": True,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()


            choice = data["choices"][0]
            logprobs = choice.get("logprobs", {})
            token_logprobs = logprobs.get("token_logprobs", [])


            valid_lps = [lp for lp in token_logprobs if lp is not None]
            if any(lp > 0 for lp in valid_lps):
                logger.warning(
                    f"vLLM returned positive logprob for text[:50]={text[:50]!r} — "
                    "discarding (corrupt response)"
                )
                continue

            valid_lps = [max(lp, -100.0) for lp in valid_lps]
            if valid_lps:
                total_nll += sum(-lp for lp in valid_lps)
                total_tokens += len(valid_lps)

        except Exception as exc:
            logger.warning(f"vLLM logprob request failed for text[:50]={text[:50]!r}: {exc}")
            continue

    if total_tokens == 0:
        return float("inf")

    return total_nll / total_tokens


def reward_shaping(loss: float, gamma: float) -> float:
    return math.exp(-gamma * loss)


def compute_reward(
    current_loss: float,
    previous_best_loss: float,
    gamma: float,
    reward_max: float,
) -> Tuple[float, float]:
    new_best = min(previous_best_loss, current_loss)
    f_new = reward_shaping(new_best, gamma)
    f_old = reward_shaping(previous_best_loss, gamma)
    raw_reward = f_new - f_old
    reward = max(0.0, min(raw_reward, reward_max))
    return reward, new_best


def dirichlet_weighted_loss(
    loss_window: List[float],
    beta: float = 1.0,
) -> float:
    n = len(loss_window)
    if n == 0:
        return float("inf")
    if n == 1:
        return loss_window[0]

    weights = np.random.dirichlet([beta] * n)
    return float(np.dot(weights, loss_window))


class RewardTracker:

    def __init__(
        self,
        window_size: int = 100,
        gamma: float = 1.0,
        beta: float = 1.0,
        reward_max: float = 1.0,
        reward_decay: float = 0.995,
        storage_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.gamma = gamma
        self.beta = beta
        self.reward_max = reward_max


        self.reward_decay = reward_decay
        self.storage_path = storage_path or (
            Path.home() / ".evolai" / "validator" / "loss_rewards.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.miners: Dict[int, MinerLossState] = {}
        self._load()


    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for uid_str, mdata in data.items():
                uid = int(uid_str)
                state = MinerLossState(uid=uid)
                state.coldkey = mdata.get("coldkey", "")
                state.hotkey = mdata.get("hotkey", "")
                state.model_name = mdata.get("model_name", "")
                state.best_loss = mdata.get("best_loss", float("inf"))
                state.eval_count = mdata.get("eval_count", 0)
                state.last_reward = mdata.get("last_reward", 0.0)
                state.cumulative_reward = mdata.get("cumulative_reward", 0.0)
                state.loss_history = [
                    LossRecord(
                        loss=lr["loss"],
                        timestamp=lr["timestamp"],
                        model_name=lr["model_name"],
                        dataset_name=lr["dataset_name"],
                        revision=lr.get("revision", ""),
                    )
                    for lr in mdata.get("loss_history", [])
                ]
                self.miners[uid] = state
        except Exception as exc:
            logger.warning(f"Failed to load loss rewards: {exc}")

    def _save(self) -> None:
        data = {}
        for uid, state in self.miners.items():
            data[str(uid)] = {
                "coldkey": state.coldkey,
                "hotkey": state.hotkey,
                "model_name": state.model_name,
                "best_loss": state.best_loss,
                "eval_count": state.eval_count,
                "last_reward": state.last_reward,
                "cumulative_reward": state.cumulative_reward,
                "loss_history": [
                    {"loss": lr.loss, "timestamp": lr.timestamp,
                     "model_name": lr.model_name, "dataset_name": lr.dataset_name,
                     "revision": lr.revision}
                    for lr in state.loss_history
                ],
            }


        tmp_path = self.storage_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.storage_path)


    def sync_uid(self, uid: int, coldkey: str, hotkey: str) -> bool:
        state = self.miners.get(uid)
        if state is None:

            self.miners[uid] = MinerLossState(uid=uid, coldkey=coldkey, hotkey=hotkey)
            self._save()
            return False

        if state.hotkey != hotkey:

            logger.warning(
                f"UID {uid} replaced: {state.hotkey[:16]}... → {hotkey[:16]}..."
            )
            self.miners[uid] = MinerLossState(uid=uid, coldkey=coldkey, hotkey=hotkey)
            self._save()
            return True

        return False

    def sync_model(self, uid: int, model_name: str) -> bool:
        state = self.miners.get(uid)
        if state is None:
            return False

        if state.model_name and state.model_name != model_name:
            logger.info(
                f"UID {uid}: model changed {state.model_name!r} → {model_name!r} — "
                "resetting loss history and best_loss (cumulative_reward preserved)"
            )
            state.model_name = model_name
            state.loss_history = []
            state.best_loss = float("inf")
            state.eval_count = 0
            state.last_reward = 0.0


            self._save()
            return True

        state.model_name = model_name
        return False


    def record_loss(
        self,
        uid: int,
        loss: float,
        model_name: str,
        dataset_name: str,
        revision: str = "main",
        timestamp: Optional[str] = None,
    ) -> Tuple[float, float]:
        ts = timestamp or datetime.utcnow().isoformat()
        state = self.miners.get(uid)
        if state is None:
            state = MinerLossState(uid=uid, model_name=model_name)
            self.miners[uid] = state


        record = LossRecord(
            loss=loss, timestamp=ts,
            model_name=model_name, dataset_name=dataset_name,
            revision=revision,
        )
        state.loss_history.append(record)


        if len(state.loss_history) > self.window_size:
            state.loss_history = state.loss_history[-self.window_size:]

        state.eval_count += 1


        loss_values = [lr.loss for lr in state.loss_history]
        aggregated_loss = dirichlet_weighted_loss(loss_values, self.beta)


        reward, new_best = compute_reward(
            aggregated_loss, state.best_loss, self.gamma, self.reward_max,
        )
        state.best_loss = new_best

        state.last_reward = reward


        state.cumulative_reward = state.cumulative_reward * self.reward_decay + reward

        self._save()
        return reward, new_best


    def get_miner_state(self, uid: int) -> Optional[MinerLossState]:
        return self.miners.get(uid)

    def get_effective_scores(self, min_evaluations: int = 10) -> Dict[int, float]:
        scores = {}
        for uid, state in self.miners.items():
            if state.eval_count < min_evaluations:
                continue
            scores[uid] = state.cumulative_reward
        return scores

    def get_latest_loss(self, uid: int) -> Optional[float]:
        state = self.miners.get(uid)
        if state and state.loss_history:
            return state.loss_history[-1].loss
        return None
