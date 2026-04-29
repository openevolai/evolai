
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import httpx

from .loss_evaluator import ChallengeSpec, ChatSample

logger = logging.getLogger(__name__)


_MAX_INDICES_PER_DATASET = 2000


_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,96}(/[a-zA-Z0-9][a-zA-Z0-9._-]{0,96})?$")


def _validate_dataset_name(name: str) -> bool:
    return bool(_DATASET_NAME_RE.match(name))


@dataclass
class ValidatorAuth:

    hotkey: str
    sign_fn: Callable[[str], str]

    def make_headers(self) -> dict[str, str]:
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


def submit_weights(
    validator_hotkey: str,
    netuid: int,
    weights: Dict[int, float],
    block: Optional[int] = None,
    owner_api_url: str = "",
    auth: Optional[ValidatorAuth] = None,
) -> bool:
    if auth is None:
        logger.warning("Skipping weight submission: validator auth unavailable")
        return False
    if not owner_api_url:
        return False

    url = f"{owner_api_url}/weights/submit"
    headers = {
        "Content-Type": "application/json",
        **auth.make_headers(),
    }
    payload = {
        "validator_hotkey": validator_hotkey,
        "netuid": netuid,
        "weights": {str(uid): round(w, 8) for uid, w in weights.items()},
        "submitted_at": _utcnow_iso(),
    }
    if block is not None:
        payload["block"] = block

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        logger.info(
            f"Submitted weight distribution for {len(weights)} UIDs to owner proxy"
        )
        return True
    except Exception as exc:
        logger.warning(f"Failed to submit weights to owner proxy: {exc}")
        return False


def announce_miners(
    miners: List[Dict],
    track: str,
    netuid: int,
    owner_api_url: str = "",
    auth: Optional[ValidatorAuth] = None,
) -> bool:
    if auth is None:
        logger.warning("Skipping miner announcement: validator auth unavailable")
        return False
    if not owner_api_url or not miners:
        return False

    url = f"{owner_api_url}/miners/announce"
    headers = {
        "Content-Type": "application/json",
        **auth.make_headers(),
    }
    payload = {
        "track": track,
        "netuid": netuid,
        "announced_at": _utcnow_iso(),
        "miners": [
            {
                "uid": m["uid"],
                "hotkey": m["hotkey"],
                "model_name": m["model_name"],
                "revision": m.get("revision") or "main",
            }
            for m in miners
        ],
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        logger.info(
            f"Announced {len(miners)} {track} miners to owner proxy"
        )
        return True
    except Exception as exc:
        logger.warning(f"Failed to announce miners to owner proxy: {exc}")
        return False


def _utcnow_iso() -> str:
    from datetime import datetime as _dt
    return _dt.utcnow().isoformat() + "Z"


_INSTRUCTION_COLUMNS = ("instruction", "input", "question", "prompt", "human")

_RESPONSE_COLUMNS    = ("response", "output", "answer", "completion", "assistant", "gpt")

_PLAIN_TEXT_COLUMNS  = ("text", "content")


def _extract_sample_from_row(row: dict) -> Optional[Union[str, ChatSample]]:
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


    for col in _PLAIN_TEXT_COLUMNS + _INSTRUCTION_COLUMNS:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for val in row.values():
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ── Dataset loader ────────────────────────────────────────────────────────────

def _get_dataset(dataset_name: str):
    """Load a HuggingFace dataset (train split), checking for updates each call.

    The HF datasets library stores data under ~/.cache/huggingface/datasets/.
    On each call it performs a lightweight commit-hash check against the HF Hub:
      - No remote change → returns from local disk cache immediately (ms).
      - Remote dataset updated → re-downloads and reloads automatically.

    No in-process cache is kept so every evaluation round sees the latest
    version of the dataset without requiring a validator restart.
    """
    from datasets import load_dataset as _hf_load_dataset
    ds = _hf_load_dataset(dataset_name, split="train")
    logger.debug(f"Dataset {dataset_name!r}: {len(ds)} rows")
    return ds


def get_dataset_size(dataset_name: str) -> int:
    """Return the number of rows in the train split of dataset_name."""
    return len(_get_dataset(dataset_name))


def fetch_challenge_texts(
    datasets: Dict[str, List[int]],
) -> List[Union[str, ChatSample]]:
    all_texts: List[Union[str, ChatSample]] = []

    for dataset_name, indices in datasets.items():
        if not indices:
            continue

        ds = _get_dataset(dataset_name)
        ds_len = len(ds)
        fetched = 0
        for idx in indices:
            if idx < 0 or idx >= ds_len:
                logger.warning(
                    f"{dataset_name}: index {idx} out of range [0, {ds_len}) — skipping"
                )
                continue
            sample = _extract_sample_from_row(ds[idx])
            if sample is not None:
                all_texts.append(sample)
                fetched += 1
            else:
                logger.warning(
                    f"{dataset_name}[{idx}]: could not extract sample — skipping"
                )

        logger.info(f"Fetched {fetched}/{len(indices)} texts from {dataset_name}")

    return all_texts
