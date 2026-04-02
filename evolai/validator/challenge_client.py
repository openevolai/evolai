"""
Challenge Client — fetches challenges from the Owner API.

Validators use this to get the challenge spec for a given UID.
Authentication is performed via Bittensor hotkey signature so that miners
cannot discover their own test sets through the public proxy.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import httpx

from .loss_evaluator import ChallengeSpec, ChatSample

logger = logging.getLogger(__name__)

# Maximum number of indices we'll accept per dataset from the Owner API response.
# Guards against a compromised/misconfigured Owner sending oversized challenges
# that would cause the validator to load gigabytes of dataset rows.
_MAX_INDICES_PER_DATASET = 2000

# Allowlist pattern for HuggingFace dataset names.
# Valid form: "owner/dataset-name" or plain "dataset-name".
# Characters allowed: alphanumeric, hyphen, underscore, dot, slash (one max).
_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,96}(/[a-zA-Z0-9][a-zA-Z0-9._-]{0,96})?$")


def _validate_dataset_name(name: str) -> bool:
    """Return True if *name* is a safe HuggingFace dataset identifier."""
    return bool(_DATASET_NAME_RE.match(name))


@dataclass
class ValidatorAuth:
    """Signing credentials for an authenticated validator request.

    Attributes:
        hotkey: SS58-encoded hotkey address of the validator.
        sign_fn: Callable that accepts a plaintext message string and returns
            a hex-encoded SR25519 signature string.  Typically wraps
            ``wallet.hotkey.sign(msg.encode()).hex()``.
    """

    hotkey: str
    sign_fn: Callable[[str], str]

    def make_headers(self) -> dict[str, str]:
        """Return the three auth headers required by the proxy."""
        message = f"evolai_validator:{self.hotkey}:{int(time.time())}"
        signature = self.sign_fn(message)
        return {
            "X-Validator-Hotkey": self.hotkey,
            "X-Validator-Message": message,
            "X-Validator-Signature": signature,
        }


def fetch_challenge(
    uid: int,
    owner_api_url: str,
    auth: Optional[ValidatorAuth] = None,
) -> Optional[ChallengeSpec]:
    """Fetch the current challenge for *uid* from the Owner API.

    Args:
        uid: Miner UID.
        owner_api_url: Base URL of the Owner API or proxy
            (e.g. ``https://your-org-evolai.hf.space``).
        auth: Optional validator authentication credentials.  When provided
            the request is signed with the validator's hotkey so the proxy
            can verify it.  Pass ``None`` only when calling the manager
            directly on localhost (owner-co-located setup).

    Returns:
        ChallengeSpec or None if the UID has no challenge.
    """
    url = f"{owner_api_url}/challenge/{uid}"
    headers = auth.make_headers() if auth is not None else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=30.0)
        if resp.status_code == 404:
            logger.warning(f"No challenge for UID {uid} (404)")
            return None
        resp.raise_for_status()
        data = resp.json()

        raw_datasets: Dict[str, List[int]] = data.get("datasets") or {}

        # Back-compat: old API returned dataset_name + text_indices
        if not raw_datasets and "dataset_name" in data and "text_indices" in data:
            raw_datasets = {data["dataset_name"]: data["text_indices"]}

        if not raw_datasets:
            logger.error(f"Challenge for UID {uid} has no datasets — rejecting")
            return None

        clean_datasets: Dict[str, List[int]] = {}
        for dataset_name, raw_indices in raw_datasets.items():
            if not _validate_dataset_name(dataset_name):
                logger.warning(
                    f"Challenge for UID {uid}: skipping invalid dataset name {dataset_name!r}"
                )
                continue

            # Cap indices per dataset
            if len(raw_indices) > _MAX_INDICES_PER_DATASET:
                logger.warning(
                    f"Challenge for UID {uid}: {dataset_name} has {len(raw_indices)} indices "
                    f"(cap={_MAX_INDICES_PER_DATASET}) — truncating"
                )
                raw_indices = raw_indices[:_MAX_INDICES_PER_DATASET]

            indices = [int(i) for i in raw_indices if isinstance(i, int) and i >= 0]
            if len(indices) != len(raw_indices):
                logger.warning(
                    f"Challenge for UID {uid}, {dataset_name}: dropped "
                    f"{len(raw_indices) - len(indices)} invalid indices"
                )
            if indices:
                clean_datasets[dataset_name] = indices

        if not clean_datasets:
            logger.error(f"Challenge for UID {uid}: no valid datasets after validation — rejecting")
            return None

        return ChallengeSpec(uid=data["uid"], datasets=clean_datasets)
    except Exception as exc:
        logger.error(f"Failed to fetch challenge for UID {uid}: {exc}")
        return None


def submit_evaluations(
    evaluation_round: int,
    judge_model: str,
    results: List[Dict],
    owner_api_url: str,
    auth: Optional[ValidatorAuth] = None,
) -> bool:
    """Submit validator evaluation results to the owner proxy.

    Args:
        evaluation_round: Monotonic evaluation round number.
        judge_model: Label stored by the owner for monitoring/aggregation.
        results: List of per-miner result payloads expected by the proxy.
        owner_api_url: Base URL of the owner proxy.
        auth: Validator signature auth. Required for proxy submission.

    Returns:
        True if the submission succeeded, else False.
    """
    if auth is None:
        logger.warning(
            "Skipping evaluation submission: validator auth unavailable "
            "(fake wallet or unsigned setup)"
        )
        return False

    url = f"{owner_api_url}/evaluations/submit"
    headers = {
        "Content-Type": "application/json",
        **auth.make_headers(),
    }
    payload = {
        "evaluation_round": evaluation_round,
        "judge_model": judge_model,
        "results": results,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        logger.info(
            f"Submitted {len(results)} evaluations to owner proxy "
            f"for round {evaluation_round}"
        )
        return True
    except Exception as exc:
        logger.error(f"Failed to submit evaluations for round {evaluation_round}: {exc}")
        return False


# Column candidates for the instruction/input side of a chat pair
_INSTRUCTION_COLUMNS = ("instruction", "input", "question", "prompt", "human")
# Column candidates for the response/output side of a chat pair
_RESPONSE_COLUMNS    = ("response", "output", "answer", "completion", "assistant", "gpt")
# Plain-text fallback columns (used when no chat pair is detected)
_PLAIN_TEXT_COLUMNS  = ("text", "content")

# HF Datasets Server rows endpoint.
_HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
# HF rows API accepts at most 100 rows per request.
_HF_ROWS_MAX_BATCH = 100


def _get_hf_token() -> Optional[str]:
    """Return an HF auth token from env vars or the huggingface_hub credential store.

    Authenticated requests to the HF Datasets Server have much higher rate
    limits (~5 000 req/min) compared to anonymous requests (~100 req/min).
    """
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(env_var, "").strip()
        if tok:
            return tok
    try:
        from huggingface_hub.utils import get_token  # type: ignore[import]
        return get_token()
    except Exception:
        return None


def _extract_sample_from_row(row: dict) -> Optional[Union[str, ChatSample]]:
    """Extract a ChatSample from instruction+response columns, or a plain str.

    Detection order:
    1. Both an instruction column AND a response column present → ChatSample.
       Enables chat-template application and response-only loss masking.
    2. Any plain-text or instruction column → str.
    3. Any non-empty string column as last resort.
    """
    instr_col = next(
        (c for c in _INSTRUCTION_COLUMNS if c in row and isinstance(row[c], str) and row[c].strip()),
        None,
    )
    resp_col = next(
        (c for c in _RESPONSE_COLUMNS if c in row and isinstance(row[c], str) and row[c].strip()),
        None,
    )
    if instr_col and resp_col:
        return ChatSample(instruction=row[instr_col].strip(), response=row[resp_col].strip())

    # Fall back to plain text
    for col in _PLAIN_TEXT_COLUMNS + _INSTRUCTION_COLUMNS:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Last resort: any string column
    for val in row.values():
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _fetch_via_rest_api(
    dataset_name: str,
    sorted_indices: List[int],
    token: Optional[str] = None,
) -> Optional[List[Union[str, ChatSample]]]:
    """Fetch specific rows via the HF Datasets Server REST API.

    Each call requests a contiguous window of up to ``_HF_ROWS_MAX_BATCH``
    rows, so only the needed rows are transferred.  With an auth token the
    rate limit is high enough for thousands of requests; without one the API
    is capped at ~100 req/min and a 429 response is returned.

    Returns:
        List of samples (str or ChatSample) if all requests succeeded, or
        ``None`` if the API returned HTTP 429 (caller should fall back to
        streaming).
    """
    needed: set = set(sorted_indices)
    texts: List[Union[str, ChatSample]] = []
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    requests_made = 0

    i = 0
    while i < len(sorted_indices):
        window_start = sorted_indices[i]
        # Extend window greedily while it fits within _HF_ROWS_MAX_BATCH rows.
        j = i
        while (
            j < len(sorted_indices)
            and (sorted_indices[j] - window_start + 1) <= _HF_ROWS_MAX_BATCH
        ):
            j += 1
        length = sorted_indices[j - 1] - window_start + 1

        try:
            resp = httpx.get(
                _HF_ROWS_URL,
                params={
                    "dataset": dataset_name,
                    "config": "default",
                    "split": "train",
                    "offset": window_start,
                    "length": length,
                },
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code == 429:
                logger.warning(
                    f"{dataset_name}: HF REST API rate-limited — "
                    "will fall back to streaming"
                )
                return None  # Signal caller to switch strategy
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"HF rows API HTTP error for {dataset_name} "
                f"offset={window_start} length={length}: {exc}"
            )
            i = j
            continue
        except Exception as exc:
            logger.error(
                f"HF rows API request failed for {dataset_name} "
                f"offset={window_start} length={length}: {exc}"
            )
            i = j
            continue

        requests_made += 1
        for row_item in payload.get("rows", []):
            row_idx = row_item.get("row_idx")
            if row_idx not in needed:
                continue
            row = row_item.get("row", {})
            sample = _extract_sample_from_row(row)
            if sample is not None:
                texts.append(sample)
        i = j

    logger.debug(f"{dataset_name}: REST API used {requests_made} request(s)")
    return texts


def _fetch_via_streaming(
    dataset_name: str,
    sorted_indices: List[int],
) -> List[Union[str, ChatSample]]:
    """Collect needed rows by streaming through the dataset.

    Uses ``datasets`` streaming mode: rows are fetched on-demand from remote
    Parquet files without writing the full dataset to disk.  Time complexity
    is O(max_index), which is acceptable for datasets up to a few hundred
    thousand rows.

    This is used as a fallback when the REST API is rate-limited (i.e. no
    HF token is configured and request volume exceeds the anonymous limit).
    """
    needed = set(sorted_indices)
    max_idx = max(sorted_indices)
    row_samples: Dict[int, Union[str, ChatSample]] = {}

    try:
        from datasets import load_dataset as _hf_load_dataset  # type: ignore[import]
        ds = _hf_load_dataset(dataset_name, split="train", streaming=True)
        for row_idx, row in enumerate(ds):
            if row_idx in needed:
                sample = _extract_sample_from_row(row)
                if sample is not None:
                    row_samples[row_idx] = sample
            if row_idx >= max_idx:
                break
    except Exception as exc:
        logger.error(f"{dataset_name}: streaming fallback failed: {exc}")

    return [row_samples[i] for i in sorted_indices if i in row_samples]


def fetch_challenge_texts(
    datasets: Dict[str, List[int]],
) -> List[Union[str, ChatSample]]:
    """Fetch text for specific row indices from HuggingFace datasets.

    Strategy (no full dataset download in either path):

    1. **REST API** — requests only the needed rows (up to 100 per HTTP call).
       Authenticated requests (via ``HF_TOKEN`` env var or ``huggingface-cli
       login``) support thousands of calls per minute.  Returns ``None`` on
       HTTP 429 to trigger the fallback.

    2. **Streaming fallback** — streams through the dataset in row order using
       ``datasets`` streaming mode, collecting only the rows at the requested
       indices.  No dataset is cached to disk; only enough Parquet chunks are
       downloaded to reach the highest requested index.

    Args:
        datasets: Mapping of HuggingFace dataset name to row indices
            (as returned by :func:`fetch_challenge`).

    Returns:
        Flat list of samples across all datasets.  Each item is either a plain
        ``str`` (plain-text datasets) or a ``ChatSample`` (instruction+response
        pair) for chat/instruction-following datasets.
    """
    all_texts: List[Union[str, ChatSample]] = []
    token = _get_hf_token()
    if not token:
        logger.debug(
            "No HF token found — REST API requests will be anonymous "
            "(rate-limited to ~100 req/min; streaming fallback will be used if exceeded)"
        )

    for dataset_name, indices in datasets.items():
        if not indices:
            continue

        sorted_indices = sorted(set(indices))

        # Primary path: REST API — only transfers the requested rows.
        texts = _fetch_via_rest_api(dataset_name, sorted_indices, token=token)

        if texts is None:
            # REST API rate-limited (no token). Stream instead.
            logger.info(f"{dataset_name}: streaming {len(sorted_indices)} rows (no HF token)")
            texts = _fetch_via_streaming(dataset_name, sorted_indices)

        logger.info(
            f"Fetched {len(texts)}/{len(indices)} texts from {dataset_name}"
        )
        all_texts.extend(texts)

    return all_texts
