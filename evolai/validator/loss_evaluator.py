"""
Loss-Based Evaluation and Reward System for EvolAI Validators

Implements the reward mechanism described in the system spec:
  - Temporal evaluation window (sliding window of past challenges)
  - Dirichlet-weighted randomised evaluation
  - Best-so-far loss tracking
  - Exponential reward shaping: R_t = clip(F(L*_t) - F(L*_{t-1}), 0, R_MAX)
    where F(L) = exp(-gamma * L)

Each miner is evaluated by computing cross-entropy loss on its challenge
(dataset name + text indices fetched from the Owner API).  The validator
tracks the last CHALLENGE_WINDOW_SIZE loss values and rewards improvement.
"""

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


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChallengeSpec:
    """Challenge fetched from the Owner API for a specific UID."""
    uid: int
    # Maps each dataset name to the row indices the miner must train on.
    # e.g. {"evolai/dataset_a": [5, 42, 99, ...], "evolai/dataset_b": [3, 17, ...]}
    datasets: Dict[str, List[int]]


@dataclass
class LossRecord:
    """A single loss measurement for one challenge evaluation."""
    loss: float
    timestamp: str
    model_name: str
    dataset_name: str
    revision: str = ""


@dataclass
class MinerLossState:
    """Per-miner state for the loss-based reward system."""
    uid: int
    coldkey: str = ""
    hotkey: str = ""
    model_name: str = ""
    # Sliding window of loss records (most recent last)
    loss_history: List[LossRecord] = field(default_factory=list)
    # Best aggregated loss so far (L_tilde_star)
    best_loss: float = float("inf")
    # Total evaluation count
    eval_count: int = 0
    # Last reward value
    last_reward: float = 0.0
    # Cumulative reward (sum of all step rewards)
    cumulative_reward: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Chat sample type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatSample:
    """A single instruction/response pair from a chat-format dataset.

    When challenge rows contain both an instruction column and a response
    column the data is stored as a ChatSample so that:
      - the model's chat template is applied (matching its fine-tune format)
      - loss is computed only on response tokens (instruction tokens masked)

    This gives a per-token NLL that reflects the model's ability to generate
    the expected response, not to "predict" the question.
    """
    instruction: str
    response: str


# ──────────────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_cross_entropy_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[Union[str, ChatSample]],
    max_length: int = HF_LOSS_MAX_SEQ_LEN,
    batch_size: int = HF_LOSS_BATCH_SIZE,
    device: str = "cuda",
    progress_callback: Optional["Callable[[int, int], None]"] = None,
) -> float:
    """Compute average per-token cross-entropy loss on *texts*.

    Args:
        model: HuggingFace causal LM (already on device).
        tokenizer: Matching tokenizer.
        texts: Plain text strings **or** ChatSample pairs (instruction+response).
            For ChatSamples the model's chat template is applied and loss is
            computed only on response tokens (instruction tokens are masked with
            -100), matching the fine-tuning objective.
        max_length: Max token length per text.
        batch_size: Target batch size for plain-text evaluation.
        device: Torch device string.
        progress_callback: Optional callable(completed, total) called after each
            batch/sample. Used by the CLI to render a live progress bar.

    Returns:
        Average per-token cross-entropy loss across all texts.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if hasattr(model, "config"):
        model.config.use_cache = False

    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Detect model dtype for autocast: keeps activations in half-precision even
    # when HF outputs intermediate float32 buffers.
    model_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    use_autocast = is_cuda and model_dtype in (torch.float16, torch.bfloat16)

    # autocast context: enables half-precision intermediate ops when applicable.
    autocast_ctx: contextlib.AbstractContextManager
    if use_autocast:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    # ── Separate plain texts from chat instruction/response pairs ────────────
    plain_texts  = [t for t in texts if isinstance(t, str)]
    chat_samples = [t for t in texts if isinstance(t, ChatSample)]
    total_items  = len(texts)

    # ── Chat samples: apply chat template, mask instruction tokens ────────────
    # Processed one-by-one (no padding needed).  Loss is computed only on the
    # response portion so the metric matches the model's fine-tuning objective.
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
                # Generic fallback: works for Alpaca-style models without a template
                full_text   = f"### Human: {sample.instruction}\n\n### Assistant: {sample.response}"
                prompt_text = f"### Human: {sample.instruction}\n\n### Assistant:"

            full_enc   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=max_length)
            prompt_len = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1]

            input_ids      = full_enc["input_ids"].to(device)
            attention_mask = full_enc["attention_mask"].to(device)
            labels = input_ids.clone()
            # Mask instruction + template tokens; response tokens keep their ids.
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
            # outputs.loss is the MEAN over non-masked tokens — recover total NLL.
            loss_val = float(outputs.loss.detach().float().item())
            total_loss   += loss_val * response_tokens
            total_tokens += response_tokens
            del input_ids, attention_mask, labels, outputs

    if is_cuda and chat_samples:
        torch.cuda.empty_cache()

    # ── Plain texts: batched with adaptive OOM back-off ───────────────────────
    # Sort texts by character length as proxy for token length.
    # Grouping similar-length texts in a batch minimises padding within each
    # batch, directly reducing the VRAM footprint and wasted GPU ops.
    sorted_texts = sorted((t for t in plain_texts if t and t.strip()), key=len)

    max_batch_size = max(1, batch_size)
    current_batch_size = max_batch_size
    consecutive_successes = 0
    batch_count = 0
    index = 0
    chat_done = len(chat_samples)  # progress offset for plain-text batches

    with torch.inference_mode():
        while index < len(sorted_texts):
            batch_texts = sorted_texts[index : index + current_batch_size]
            input_ids = attention_mask = labels = outputs = None  # for safe del
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
                batch_count += 1

                if progress_callback is not None:
                    progress_callback(chat_done + index, total_items)

                # Explicit tensor cleanup to release VRAM as early as possible.
                del input_ids, attention_mask, labels, outputs

                # Flush GPU allocator cache every 8 batches (sync cost is modest
                # compared to the VRAM headroom it recovers between batches).
                if is_cuda and batch_count % 8 == 0:
                    torch.cuda.empty_cache()

                # Adaptive batch-size recovery: after 5 consecutive successful
                # batches following an OOM reduction, try doubling back toward
                # the original size so throughput is recovered automatically.
                if consecutive_successes >= 5 and current_batch_size < max_batch_size:
                    current_batch_size = min(current_batch_size * 2, max_batch_size)
                    consecutive_successes = 0
                    logger.debug(f"Recovered batch_size to {current_batch_size}")

            except torch.cuda.OutOfMemoryError:
                if not is_cuda or current_batch_size == 1:
                    raise
                # Free any partially-allocated tensors before retrying.
                for _t in (input_ids, attention_mask, labels, outputs):
                    if _t is not None:
                        del _t
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                consecutive_successes = 0
                logger.warning(
                    f"OOM during loss evaluation; retrying with batch_size={current_batch_size}"
                )
                continue

    # Final cleanup: release GPU allocator pages and Python objects.
    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    if total_tokens == 0:
        return float("inf")

    return total_loss / total_tokens


def compute_loss_vllm(
    texts: List[str],
    model_name: str,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    max_tokens_per_text: int = 1,
) -> float:
    """Compute approximate loss via vLLM logprobs endpoint.

    Uses the OpenAI-compatible /completions endpoint with ``logprobs``
    to get per-token log-probabilities, then averages the negative
    log-prob (= cross-entropy) across all tokens.

    Args:
        texts: Texts to evaluate.
        model_name: Model served by vLLM.
        vllm_base_url: Base URL of the vLLM server.
        max_tokens_per_text: Max new tokens to generate (1 = just score prompt).

    Returns:
        Average per-token negative log-probability (≈ cross-entropy loss).
    """
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

            # Extract token logprobs from the echo'd prompt tokens
            choice = data["choices"][0]
            logprobs = choice.get("logprobs", {})
            token_logprobs = logprobs.get("token_logprobs", [])

            # First token has None logprob (no conditioning).
            # Log-probs must be ≤ 0; positive values indicate a corrupt/
            # tampered response — discard the entire text in that case.
            valid_lps = [lp for lp in token_logprobs if lp is not None]
            if any(lp > 0 for lp in valid_lps):
                logger.warning(
                    f"vLLM returned positive logprob for text[:50]={text[:50]!r} — "
                    "discarding (corrupt response)"
                )
                continue
            # Clamp to [-100, 0] so extreme values don't dominate the average.
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


# ──────────────────────────────────────────────────────────────────────────────
# Reward computation
# ──────────────────────────────────────────────────────────────────────────────

def reward_shaping(loss: float, gamma: float) -> float:
    """F(L) = exp(-gamma * L)"""
    return math.exp(-gamma * loss)


def compute_reward(
    current_loss: float,
    previous_best_loss: float,
    gamma: float,
    reward_max: float,
) -> Tuple[float, float]:
    """Compute reward from loss improvement.

    R_t = clip(F(L*_t) - F(L*_{t-1}), 0, R_MAX)
    where L*_t = min(L*_{t-1}, current_loss)

    Args:
        current_loss: Loss from current evaluation.
        previous_best_loss: Best loss up to previous step.
        gamma: Exponential shaping coefficient.
        reward_max: Clipping ceiling.

    Returns:
        (reward, new_best_loss)
    """
    new_best = min(previous_best_loss, current_loss)
    f_new = reward_shaping(new_best, gamma)
    f_old = reward_shaping(previous_best_loss, gamma)
    raw_reward = f_new - f_old
    reward = max(0.0, min(raw_reward, reward_max))
    return reward, new_best


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated loss with Dirichlet weighting
# ──────────────────────────────────────────────────────────────────────────────

def dirichlet_weighted_loss(
    loss_window: List[float],
    beta: float = 1.0,
) -> float:
    """Compute aggregated loss with Dirichlet-sampled weights.

    L_tilde_t = sum_{i in W_t} w_i * L_t^{(i)}
    where w ~ Dirichlet(beta, ..., beta)

    Args:
        loss_window: List of loss values in the temporal window.
        beta: Dirichlet concentration parameter.

    Returns:
        Aggregated weighted loss.
    """
    n = len(loss_window)
    if n == 0:
        return float("inf")
    if n == 1:
        return loss_window[0]

    weights = np.random.dirichlet([beta] * n)
    return float(np.dot(weights, loss_window))


# ──────────────────────────────────────────────────────────────────────────────
# RewardTracker  — persistent per-miner state
# ──────────────────────────────────────────────────────────────────────────────

class RewardTracker:
    """Tracks loss history and rewards for all miners.

    Persists to ``~/.evolai/validator/loss_rewards.json``.
    When a UID's hotkey changes (detected from metagraph), all state for that
    UID is wiped.
    """

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
        # Per-step exponential decay applied to cumulative_reward.
        # At decay=0.995 and ~100 evals/day, a reward halves in ~140 evals.
        # Prevents genesis miners from permanently dominating purely via
        # banked early rewards.  Set to 1.0 to disable.
        self.reward_decay = reward_decay
        self.storage_path = storage_path or (
            Path.home() / ".evolai" / "validator" / "loss_rewards.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.miners: Dict[int, MinerLossState] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

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
        # Atomic write: write to .tmp then rename so a mid-write crash never
        # leaves a truncated/corrupt file (POSIX rename is atomic on Linux).
        tmp_path = self.storage_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.storage_path)

    # ── Metagraph sync ───────────────────────────────────────────────────

    def sync_uid(self, uid: int, coldkey: str, hotkey: str) -> bool:
        """Update UID ownership.  Returns True if the UID was replaced (hotkey changed)."""
        state = self.miners.get(uid)
        if state is None:
            # New UID
            self.miners[uid] = MinerLossState(uid=uid, coldkey=coldkey, hotkey=hotkey)
            self._save()
            return False

        if state.hotkey != hotkey:
            # UID was replaced — wipe all state
            logger.warning(
                f"UID {uid} replaced: {state.hotkey[:16]}... → {hotkey[:16]}..."
            )
            self.miners[uid] = MinerLossState(uid=uid, coldkey=coldkey, hotkey=hotkey)
            self._save()
            return True

        return False

    def sync_model(self, uid: int, model_name: str) -> bool:
        """Detect model change for a UID.  Returns True if model changed (resets loss history)."""
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
            # NOTE: cumulative_reward is intentionally NOT reset here.
            # Resetting it would let miners erase bad evaluation periods by
            # toggling their model name.  Past rewards remain banked.
            self._save()
            return True

        state.model_name = model_name
        return False

    # ── Evaluation ───────────────────────────────────────────────────────

    def record_loss(
        self,
        uid: int,
        loss: float,
        model_name: str,
        dataset_name: str,
        revision: str = "main",
        timestamp: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Record a loss measurement and compute reward.

        Reward rules:
        - First submission (best_loss == inf): establishes baseline, reward = 0.
        - New model submission (model_name or revision changed): compute reward
          from Dirichlet-weighted aggregated loss improvement.
        - Same model (no change): record loss for history but return 0 reward
          ("no new model = no progress").

        Returns:
            (reward, new_best_loss)
        """
        ts = timestamp or datetime.utcnow().isoformat()
        state = self.miners.get(uid)
        if state is None:
            state = MinerLossState(uid=uid, model_name=model_name)
            self.miners[uid] = state

        # Detect new submission: model name or revision differs from last record
        is_new_submission = True
        if state.loss_history:
            last_record = state.loss_history[-1]
            if last_record.model_name == model_name and last_record.revision == revision:
                is_new_submission = False

        # Append to history
        record = LossRecord(
            loss=loss, timestamp=ts,
            model_name=model_name, dataset_name=dataset_name,
            revision=revision,
        )
        state.loss_history.append(record)

        # Cap sliding window
        if len(state.loss_history) > self.window_size:
            state.loss_history = state.loss_history[-self.window_size:]

        state.eval_count += 1

        # Compute aggregated loss using Dirichlet weighting over the window
        loss_values = [lr.loss for lr in state.loss_history]
        aggregated_loss = dirichlet_weighted_loss(loss_values, self.beta)

        if state.best_loss == float("inf"):
            # First submission — establish baseline, zero reward
            state.best_loss = aggregated_loss
            reward = 0.0
            new_best = aggregated_loss
        elif is_new_submission:
            # New model submitted — compute reward from improvement
            reward, new_best = compute_reward(
                aggregated_loss, state.best_loss, self.gamma, self.reward_max,
            )
            state.best_loss = new_best
        else:
            # Same model, no new submission — no progress
            reward = 0.0
            new_best = state.best_loss

        state.last_reward = reward
        # Apply per-step decay before adding new reward so that old rewards
        # fade toward zero.  This prevents early miners from indefinitely
        # dominating weight-setting through banked cumulative rewards.
        state.cumulative_reward = state.cumulative_reward * self.reward_decay + reward

        self._save()
        return reward, new_best

    # ── Queries ──────────────────────────────────────────────────────────

    def get_miner_state(self, uid: int) -> Optional[MinerLossState]:
        return self.miners.get(uid)

    def get_effective_scores(self, min_evaluations: int = 10) -> Dict[int, float]:
        """Get scores for weight setting.

        Only includes miners with >= min_evaluations.
        Score = cumulative_reward (running sum of Dirichlet-weighted
        improvement rewards with per-step decay, tracked by record_loss()).

        First submission gets 0 cumulative_reward (zero score).
        Miners who don't push new models see their score decay toward 0.
        """
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
