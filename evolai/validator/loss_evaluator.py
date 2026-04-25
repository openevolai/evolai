
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
) -> "Tuple[float, float, float]":
    """
    Evaluate a miner via a single 3-turn shuffled conversation per sample,
    processed turn-by-turn with growing context.

    For each sample the 3 turns are:
      - 2 side-quest turns (deterministic math questions)
      - 1 real-sample turn (from the dataset)
    Turn order is shuffled deterministically by ``block_hash + sample_index``.

    Side-quest turns
        Model generates freely (``max_new_tokens``).  Score is binary:
        0 if the correct answer appears in the output, 1 otherwise.

    Real-sample turn — **two measurements**:
        1. **think_ce**: model generates ``<think>…</think>`` freely, then CE
           is computed on the ground-truth response tokens (teacher-forced
           after the thinking prefix).
        2. **base_ce**: pure teacher-forced CE on the response tokens with
           chat-template context only (no thinking generated).
        Both use the *same* accumulated conversation context from prior turns.

    Context always advances with ground-truth answers so the conversation is
    deterministic regardless of what the model generates.

    Returns
    -------
    (think_ce, base_ce, sq_accuracy)
        think_ce     – token-weighted mean CE (with thinking) on response.
        base_ce      – token-weighted mean CE (no thinking) on response.
        sq_accuracy  – fraction of side-quest answers correct (0.0–1.0).
    """
    from .side_quests import generate_side_quests, shuffle_turn_order, check_side_quest_answer

    model.eval()
    total_items = len(samples)

    # Accumulators for think-CE.
    think_loss_sum = 0.0
    think_loss_tokens = 0
    # Accumulators for base-CE (no thinking).
    base_loss_sum = 0.0
    base_loss_tokens = 0
    # Side-quest accuracy.
    sq_correct = 0
    sq_total = 0

    if ref_tokenizer.pad_token is None and ref_tokenizer.eos_token is not None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    _saved_padding_side = ref_tokenizer.padding_side
    _saved_truncation_side = getattr(ref_tokenizer, "truncation_side", "right")
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
    # Resolve </think> to token IDs once. Checking token IDs at the CUDA
    # kernel level is O(1) per step; stop_strings requires per-step Python
    # string decoding which is much slower over long budgets.
    _think_end_ids = ref_tokenizer.encode(think_end, add_special_tokens=False)
    _eos_id = ref_tokenizer.eos_token_id
    if len(_think_end_ids) == 1:
        # Single-token </think> (e.g. Qwen3 151668): use eos_token_id override.
        _think_stop_ids = list({_think_end_ids[0], _eos_id} - {None})
        _think_gen_kwargs: dict = {"eos_token_id": _think_stop_ids}
    else:
        # Multi-token fallback: keep stop_strings (rare for Qwen3).
        _think_gen_kwargs = {"stop_strings": [think_end], "tokenizer": ref_tokenizer}

    # ------------------------------------------------------------------
    # Helpers for batched generation / CE across samples.
    # ------------------------------------------------------------------
    def _build_sq_prompt(messages_so_far, q_text):
        if has_template:
            return ref_tokenizer.apply_chat_template(
                messages_so_far + [{"role": "user", "content": q_text}],
                tokenize=False, add_generation_prompt=True,
            )
        ctx = ""
        for m in messages_so_far:
            pfx = "Human" if m["role"] == "user" else "Assistant"
            ctx += f"### {pfx}: {m['content']}\n\n"
        return ctx + f"### Human: {q_text}\n\n### Assistant:"

    def _build_think_prompt(messages_so_far, q_text):
        if has_template:
            return ref_tokenizer.apply_chat_template(
                messages_so_far + [{"role": "user", "content": q_text}],
                tokenize=False, add_generation_prompt=True,
            ) + "<think>"
        ctx = ""
        for m in messages_so_far:
            pfx = "Human" if m["role"] == "user" else "Assistant"
            ctx += f"### {pfx}: {m['content']}\n\n"
        return ctx + f"### Human: {q_text}\n\n### Assistant: <think>"

    def _build_base_texts(messages_so_far, q_text, a_text):
        if has_template:
            base_full_text = ref_tokenizer.apply_chat_template(
                messages_so_far + [
                    {"role": "user", "content": q_text},
                    {"role": "assistant", "content": a_text},
                ],
                tokenize=False, add_generation_prompt=False,
            )
            base_prompt_text = ref_tokenizer.apply_chat_template(
                messages_so_far + [{"role": "user", "content": q_text}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            ctx = ""
            for m in messages_so_far:
                pfx = "Human" if m["role"] == "user" else "Assistant"
                ctx += f"### {pfx}: {m['content']}\n\n"
            base_prompt_text = ctx + f"### Human: {q_text}\n\n### Assistant:"
            base_full_text = base_prompt_text + f" {a_text}"
        return base_full_text, base_prompt_text

    def _batched_generate(prompts, max_new, extra_kwargs=None):
        """Left-pad prompts and run one generate() call. Returns
        list of (gen_text, gen_ids_tensor) per prompt."""
        ref_tokenizer.padding_side = "left"
        enc = ref_tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        kwargs = dict(gen_base)
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        with autocast_ctx:
            out = model.generate(
                ids, attention_mask=attn,
                max_new_tokens=max_new, **kwargs,
            )
        prompt_pad_len = ids.shape[1]
        results = []
        eos = ref_tokenizer.eos_token_id
        pad = ref_tokenizer.pad_token_id
        for i in range(len(prompts)):
            tail = out[i, prompt_pad_len:]
            # Trim trailing pad tokens (post-eos padding from batched gen).
            if pad is not None:
                nonpad = (tail != pad).nonzero(as_tuple=False)
                if len(nonpad) > 0:
                    tail = tail[: int(nonpad[-1].item()) + 1]
                else:
                    tail = tail[:0]
            text = ref_tokenizer.decode(tail, skip_special_tokens=True)
            results.append((text, tail))
        del ids, attn, out
        if is_cuda:
            torch.cuda.empty_cache()
        return results

    def _batched_ce(input_texts, prompt_lens=None, prompt_texts=None):
        """Right-pad full sequences and forward once. Mask tokens before
        prompt_len (per sample) and pad positions; compute per-sample CE
        on the response tokens. Returns list of (ce_value, n_resp_tokens)."""
        ref_tokenizer.padding_side = "right"
        enc = ref_tokenizer(
            input_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        # Per-sample prompt lengths (in the padded tensor coordinates).
        if prompt_lens is None:
            prompt_lens = []
            for pt in (prompt_texts or []):
                penc = ref_tokenizer(
                    pt, return_tensors="pt",
                    truncation=True, max_length=max_length,
                )
                prompt_lens.append(int(penc["input_ids"].shape[1]))
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.shape[1])
            labels[i, :plen] = -100
        labels[attn == 0] = -100
        # Per-sample CE via reduction='none'.
        with autocast_ctx:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
        valid = (shift_labels != -100)
        per_sample_sum = (flat_loss * valid).sum(dim=1)
        per_sample_n = valid.sum(dim=1)
        results = []
        for i in range(input_ids.shape[0]):
            n = int(per_sample_n[i].item())
            if n > 0:
                results.append((float(per_sample_sum[i].item() / n), n))
            else:
                results.append((penalty_loss, 1))
        del input_ids, attn, labels, out, logits, shift_logits, shift_labels, flat_loss
        if is_cuda:
            torch.cuda.empty_cache()
        return results

    def _batched_think_ce(prompts_with_gen_ids, response_texts):
        """Forward (prompt + gen + response) with response masked except for
        teacher-forced loss on response tokens. Returns list of (ce, n)."""
        # Build per-sample full token tensors then right-pad manually.
        full_seqs = []
        prompt_gen_lens = []
        resp_lens = []
        for (prompt_gen_ids, resp_text) in zip(prompts_with_gen_ids, response_texts):
            pg = prompt_gen_ids
            remaining = max(1, max_length - pg.shape[0])
            resp_enc = ref_tokenizer(
                resp_text, return_tensors="pt",
                truncation=True, max_length=remaining,
                add_special_tokens=False,
            )
            resp = resp_enc["input_ids"][0].to(device)
            full = torch.cat([pg, resp], dim=0)
            full_seqs.append(full)
            prompt_gen_lens.append(pg.shape[0])
            resp_lens.append(resp.shape[0])
        max_len = max((s.shape[0] for s in full_seqs), default=0)
        if max_len == 0:
            return [(penalty_loss, 1) for _ in full_seqs]
        pad_id = ref_tokenizer.pad_token_id or ref_tokenizer.eos_token_id or 0
        B = len(full_seqs)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for i, full in enumerate(full_seqs):
            L = full.shape[0]
            input_ids[i, :L] = full
            attn[i, :L] = 1
            pl = prompt_gen_lens[i]
            if resp_lens[i] > 0:
                labels[i, pl:L] = full[pl:L]
        with autocast_ctx:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
        valid = (shift_labels != -100)
        per_sample_sum = (flat_loss * valid).sum(dim=1)
        per_sample_n = valid.sum(dim=1)
        results = []
        for i in range(B):
            n = int(per_sample_n[i].item())
            if n > 0:
                results.append((float(per_sample_sum[i].item() / n), n))
            else:
                results.append((penalty_loss, 1))
        del input_ids, attn, labels, out, logits, shift_logits, shift_labels, flat_loss
        if is_cuda:
            torch.cuda.empty_cache()
        return results

    # ------------------------------------------------------------------
    # Phased evaluation: process all samples turn-by-turn, batching
    # same-type operations across samples at each turn position.
    # ------------------------------------------------------------------
    plans = []
    for sample_idx, sample in enumerate(samples):
        quests = generate_side_quests(block_hash, sample_idx, n=2)
        turn_order = shuffle_turn_order(block_hash, sample_idx, n_turns=3)
        turns = [
            {"type": "real", "q": sample.instruction, "a": sample.response},
            {"type": "sq", "q": quests[0].question, "a": quests[0].answer, "quest": quests[0]},
            {"type": "sq", "q": quests[1].question, "a": quests[1].answer, "quest": quests[1]},
        ]
        plans.append({
            "sample_idx": sample_idx,
            "ordered": [turns[i] for i in turn_order],
            "messages_so_far": [],
            "ok": True,
        })

    with torch.inference_mode():
        for turn_pos in range(3):
            # Collect prompts at this position, grouped by type.
            sq_idx, sq_prompts, sq_quests, sq_qtexts = [], [], [], []
            real_idx, real_think_prompts = [], []
            real_base_full, real_base_prompt = [], []
            real_qtexts, real_atexts = [], []

            for k, p in enumerate(plans):
                if not p["ok"]:
                    continue
                turn = p["ordered"][turn_pos]
                if turn["type"] == "sq":
                    sq_idx.append(k)
                    sq_prompts.append(_build_sq_prompt(p["messages_so_far"], turn["q"]))
                    sq_quests.append(turn["quest"])
                    sq_qtexts.append(turn["q"])
                else:
                    real_idx.append(k)
                    real_think_prompts.append(_build_think_prompt(p["messages_so_far"], turn["q"]))
                    bf, bp = _build_base_texts(p["messages_so_far"], turn["q"], turn["a"])
                    real_base_full.append(bf)
                    real_base_prompt.append(bp)
                    real_qtexts.append(turn["q"])
                    real_atexts.append(turn["a"])

            # --- Batched SQ generation across samples at this turn pos ---
            if sq_prompts:
                try:
                    sq_results = _batched_generate(sq_prompts, max_new_tokens)
                    for k, qtext, quest, (gen_text, _gen_ids) in zip(
                        sq_idx, sq_qtexts, sq_quests, sq_results,
                    ):
                        correct = check_side_quest_answer(gen_text, quest)
                        sq_correct += int(correct)
                        sq_total += 1
                        plans[k]["messages_so_far"] += [
                            {"role": "user", "content": qtext},
                            {"role": "assistant", "content": gen_text},
                        ]
                except torch.cuda.OutOfMemoryError:
                    if is_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.warning(
                        f"OOM during batched SQ generation at turn {turn_pos} "
                        f"({len(sq_prompts)} samples) — recording penalty"
                    )
                    for k, qtext, quest in zip(sq_idx, sq_qtexts, sq_quests):
                        sq_total += 1
                        plans[k]["messages_so_far"] += [
                            {"role": "user", "content": qtext},
                            {"role": "assistant", "content": ""},
                        ]

            # --- Batched real-turn (think generate + think CE + base CE) ---
            if real_think_prompts:
                try:
                    think_results = _batched_generate(
                        real_think_prompts, think_max_new_tokens, _think_gen_kwargs,
                    )
                    # Reconstruct per-sample (prompt_ids + gen_ids) for think CE.
                    prompts_with_gen = []
                    for prompt_text, (_gen_text, gen_tail) in zip(
                        real_think_prompts, think_results,
                    ):
                        penc = ref_tokenizer(
                            prompt_text, return_tensors="pt",
                            truncation=True, max_length=max_length,
                        )
                        pids = penc["input_ids"][0].to(device)
                        prompts_with_gen.append(torch.cat([pids, gen_tail], dim=0))
                    think_ces = _batched_think_ce(prompts_with_gen, real_atexts)
                    base_ces = _batched_ce(real_base_full, prompt_texts=real_base_prompt)
                    for k, qtext, atext, (t_ce, t_n), (b_ce, b_n) in zip(
                        real_idx, real_qtexts, real_atexts, think_ces, base_ces,
                    ):
                        if math.isfinite(t_ce):
                            think_loss_sum += t_ce * t_n
                            think_loss_tokens += t_n
                        else:
                            think_loss_sum += penalty_loss
                            think_loss_tokens += 1
                        if math.isfinite(b_ce):
                            base_loss_sum += b_ce * b_n
                            base_loss_tokens += b_n
                        else:
                            base_loss_sum += penalty_loss
                            base_loss_tokens += 1
                        plans[k]["messages_so_far"] += [
                            {"role": "user", "content": qtext},
                            {"role": "assistant", "content": atext},
                        ]
                except torch.cuda.OutOfMemoryError:
                    if is_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.warning(
                        f"OOM during batched real turn at pos {turn_pos} "
                        f"({len(real_think_prompts)} samples) — recording penalty"
                    )
                    for k, qtext, atext in zip(real_idx, real_qtexts, real_atexts):
                        think_loss_sum += penalty_loss
                        think_loss_tokens += 1
                        base_loss_sum += penalty_loss
                        base_loss_tokens += 1
                        plans[k]["messages_so_far"] += [
                            {"role": "user", "content": qtext},
                            {"role": "assistant", "content": atext},
                        ]

            if progress_callback:
                progress_callback(
                    int((turn_pos + 1) * total_items / 3), total_items,
                )

    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    ref_tokenizer.padding_side = _saved_padding_side
    ref_tokenizer.truncation_side = _saved_truncation_side

    think_ce = think_loss_sum / think_loss_tokens if think_loss_tokens > 0 else penalty_loss
    base_ce = base_loss_sum / base_loss_tokens if base_loss_tokens > 0 else penalty_loss
    sq_accuracy = sq_correct / sq_total if sq_total > 0 else 0.0
    return think_ce, base_ce, sq_accuracy


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
