"""
Validator Evaluator - Core evaluation logic for EvolAI validators

This module implements the evaluation process:
- Dataset sampling based on vote weights
- Model loading and validation
- Interview-based evaluation
- Score calculation and aggregation
- EMA score tracking
"""

import random
import json
import logging
import signal
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi

from .resource_manager import ResourceManager, ResourceLimits
from .error_handling import (
    with_retry,
    RetryConfig,
    ErrorCategory,
    classify_error,
    ModelLoadError,
    NetworkError,
    GPUOutOfMemoryError
)
from .health_checks import HealthChecker, WatchdogTimer
from .metrics import get_metrics, Timer

logger = logging.getLogger(__name__)


def purge_hf_model_cache(model_name: str) -> None:
    """Delete the HuggingFace Hub cache directory for *model_name*.

    vLLM (and ``AutoConfig.from_pretrained``) download model artefacts into
    ``$HF_HOME/hub/models--<org>--<repo>`` (or ``~/.cache/huggingface/hub/…``
    when ``HF_HOME`` is unset).  Calling this after each evaluation round
    ensures the next round performs a fresh download, avoiding stale-cache
    bugs and preventing unbounded disk growth.

    The function is intentionally best-effort: a failure to delete is logged
    but never propagated.
    """
    try:
        # Determine the HF hub cache root
        hf_home = os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"))
        hub_cache = os.path.join(hf_home, "hub")

        # HF stores repos as  models--<org>--<repo>
        safe_name = model_name.replace("/", "--")
        model_cache_dir = os.path.join(hub_cache, f"models--{safe_name}")

        if os.path.isdir(model_cache_dir):
            shutil.rmtree(model_cache_dir)
            logger.info(f"Purged HF model cache: {model_cache_dir}")
        else:
            logger.debug(f"No HF cache found for {model_name} at {model_cache_dir}")
    except Exception as exc:
        logger.warning(f"Failed to purge HF cache for {model_name}: {exc}")


class TimeoutError(Exception):
    """Raised when operation times out"""
    pass


def timeout(seconds: int):
    """
    Decorator to add timeout to function execution.

    Uses signal.SIGALRM on Unix and threading.Timer on Windows.

    Args:
        seconds: Timeout in seconds

    Raises:
        TimeoutError: If function doesn't complete in time
    """
    def decorator(func):
        if os.name != 'nt' and hasattr(signal, 'SIGALRM'):
            # Unix: use signal-based timeout (works only in main thread)
            def _handle_timeout(signum, frame):
                raise TimeoutError(f"Operation timed out after {seconds}s")

            def wrapper(*args, **kwargs):
                old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
        else:
            # Windows / non-main thread: use threading.Timer fallback
            import threading
            import ctypes

            def wrapper(*args, **kwargs):
                result = [None]
                exception = [None]

                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=seconds)

                if thread.is_alive():
                    raise TimeoutError(f"Operation timed out after {seconds}s")

                if exception[0] is not None:
                    raise exception[0]

                return result[0]

        return wrapper
    return decorator


@dataclass
class EvaluationConfig:
    """Configuration for validator evaluation"""
    questions_per_test: int = 20
    interview_turns: int = 10
    judge_models: List[str] = None
    device: str = "cuda"
    batch_size: int = 1
    
    def __post_init__(self):
        if self.judge_models is None:
            from .config import JUDGE_MODELS
            self.judge_models = list(JUDGE_MODELS)


@dataclass
class EvaluationResult:
    """Result from evaluating a single miner"""
    miner_uid: int
    miner_hotkey: str
    track: str  # "transformer" or "mamba2"
    model_name: str
    revision: str
    raw_score: float  # 0-10
    questions_evaluated: int
    judge_model: str
    timestamp: str
    

class ModelValidator:
    """Validates and loads miner models"""
    
    def __init__(self, device: str = "cuda", health_checker: Optional[HealthChecker] = None):
        self.device = device
        self.hf_api = HfApi()
        self.resource_mgr = ResourceManager()
        self.health_checker = health_checker
        self.metrics = get_metrics()
        
        # GPU allocation: GPU 0 for judge, GPUs 1+ for miners (round-robin)
        self.num_gpus = torch.cuda.device_count()
        self.judge_gpu = 0  # Judge always on GPU 0
        self.miner_gpus = list(range(1, self.num_gpus)) if self.num_gpus > 1 else [0]  # GPUs 1, 2, 3... for miners
        self.current_miner_gpu_index = 0  # Round-robin counter
        
        logger.info(f"GPU allocation: {self.num_gpus} GPUs detected")
        logger.info(f"Judge model will use GPU {self.judge_gpu}")
        logger.info(f"Miner models will rotate across GPUs: {self.miner_gpus}")
    
    def validate_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        debug_mode: bool = False,
        uid: Optional[int] = None,
        model_registry: Optional["ModelRegistry"] = None,
        track: str = "transformer",
    ) -> Tuple[bool, List[str], Dict]:
        """
        Validate a model meets requirements.

        Runs up to 6 checks (steps 1-6 logged by number):
          1. GPU memory availability
          2. HuggingFace model config load
          3. CPU model load (weight download)
          4. Parameter count
          5. Parameter range validation
          6. Copy-gaming / plagiarism check  ← NEW

        Args:
            model_name:       HuggingFace model name (``org/repo``).
            revision:         Git revision (commit hash or branch).
            debug_mode:       Reserved parameter — kept for API compatibility.
                              All UIDs go through the full validation pipeline.
            uid:              Miner UID (used for copy-gaming check).
            model_registry:   Optional ModelRegistry used for fingerprint
                              copy-detection.  When provided a fingerprint is
                              computed, checked against known models, and
                              stored in ``info['fingerprint']`` so callers can
                              register it after evaluation.
            track:            ``"transformer"`` or ``"mamba2"``.

        Returns:
            ``(is_valid, issues, info_dict)``

            ``info_dict`` keys include (when model_registry is provided):
              - ``fingerprint``:     ModelFingerprint.to_dict()
              - ``copy_gaming``:     bool — True if a copy collision was found
              - ``copy_owner_uid``:  int  — UID of the original model owner
              - ``copy_reason``:     str  — description of the collision type
        """
        issues = []
        info = {}

        # Check model name format: must be org/name with 'evolai' in the name part
        if '/' not in model_name:
            issues.append("Model name must be in format: username/evolai-model-name")
        else:
            model_part = model_name.split('/')[-1]
            if 'evolai' not in model_part.lower():
                issues.append("Model name must contain 'evolai' in the model part (after /)")

        try:
            logger.info(f"[1/6] Checking GPU memory for {model_name}")
            # Check GPU memory before loading (estimate 8GB for typical models)
            from .config import VALIDATION_GPU_REQUIRED_GB
            if not self.resource_mgr.gpu_manager.check_available_memory(required_gb=VALIDATION_GPU_REQUIRED_GB):
                issues.append("Insufficient GPU memory available (need ~8GB free)")
                return False, issues, info
            logger.info(f"[1/6] GPU memory check passed")

            logger.info(f"[2/6] Loading config for {model_name}@{revision}")
            # SECURITY: trust_remote_code executes Python code bundled inside the
            # miner's HuggingFace model repo. Validator servers should be isolated
            # from wallets and private keys (see SECURITY.md).
            # Load model config
            config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
            logger.info(f"[2/6] Config loaded successfully")

            # Load on CPU only — no VRAM consumed, so no conflict with the vLLM
            # subprocess that runs immediately after this validation step.
            # (Loading to GPU here would keep VRAM reserved in the parent process's
            # CUDA context, causing the subsequent vLLM server to see insufficient
            # free VRAM and exit rc=1.)
            logger.info(f"[3/6] Loading model on CPU for parameter counting")
            _cpu_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                config=config,
                trust_remote_code=False,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            logger.info(f"[3/6] Model loaded on CPU")

            logger.info(f"[4/6] Counting parameters")
            total_params = sum(p.numel() for p in _cpu_model.parameters())
            total_params_b = total_params / 1e9
            logger.info(f"[4/6] Parameter count complete: {total_params_b:.2f}B params")

            # ── Step 6 (fingerprint) — MUST happen before del _cpu_model ────
            # Compute on CPU right after param count so the weights are still
            # in memory.  We use a try/except so fingerprint errors are never
            # fatal for the overall validation.
            _fingerprint_dict: Optional[Dict] = None
            if model_registry is not None and uid is not None:
                logger.info(f"[5/6] Computing model fingerprint for copy-gaming detection")
                try:
                    from .model_fingerprint import compute_model_fingerprint
                    from .config import (
                        FINGERPRINT_SAMPLE_N,
                        FINGERPRINT_MAX_BYTES_PER_TENSOR,
                        FINGERPRINT_BUCKET_COUNT,
                        FINGERPRINT_SEED,
                        FINGERPRINT_NORM_SLICE_ELEMENTS,
                    )
                    _fp = compute_model_fingerprint(
                        _cpu_model,
                        config=config,
                        seed=FINGERPRINT_SEED,
                        sample_n=FINGERPRINT_SAMPLE_N,
                        max_bytes_per_tensor=FINGERPRINT_MAX_BYTES_PER_TENSOR,
                        bucket_count=FINGERPRINT_BUCKET_COUNT,
                        norm_slice_elements=FINGERPRINT_NORM_SLICE_ELEMENTS,
                    )
                    _fingerprint_dict = _fp.to_dict()
                    info['fingerprint'] = _fingerprint_dict

                    # Cross-check against every other miner's registered fingerprints
                    is_copy, owner_uid, reason = model_registry.check_copy_gaming(
                        uid=uid,
                        track=track,
                        model_name=model_name,
                        fingerprint_dict=_fingerprint_dict,
                    )
                    info['copy_gaming'] = is_copy
                    info['copy_owner_uid'] = owner_uid
                    info['copy_reason'] = reason

                    if is_copy:
                        issues.append(
                            f"Copy-gaming detected: this model is a plagiarism of "
                            f"uid={owner_uid}'s model ({reason}). Score set to 0."
                        )
                        logger.warning(
                            f"[5/6] COPY GAMING: uid={uid} model={model_name!r} "
                            f"matches uid={owner_uid} — {reason}"
                        )
                    else:
                        logger.info(f"[5/6] Fingerprint check passed — no copy collision")
                except Exception as _fp_exc:
                    logger.warning(
                        f"[5/6] Fingerprint computation failed (non-fatal): {_fp_exc}"
                    )
            else:
                logger.info(f"[5/6] Skipping fingerprint check (no registry provided)")

            # Now safe to release the CPU model
            del _cpu_model
            import gc as _gc; _gc.collect()

            info['total_params'] = total_params
            info['total_params_b'] = round(total_params_b, 2)
            info['architecture'] = config.model_type if hasattr(config, 'model_type') else 'unknown'

            logger.info(f"[6/6] Validating parameter count")

            # Check parameter count against allowed ranges from config
            from .config import VALID_PARAM_RANGES_B
            if not any(lo <= total_params_b <= hi for lo, hi in VALID_PARAM_RANGES_B):
                ranges_str = ", ".join(f"{lo:.2f}-{hi:.2f}B" for lo, hi in VALID_PARAM_RANGES_B)
                issues.append(
                    f"Model has {total_params_b:.2f}B parameters. "
                    f"Must be one of: {ranges_str}"
                )

            logger.info(f"[6/6] Validation complete, is_valid={len(issues)==0}")
            logger.info(f"[Cleanup] No GPU model loaded during validation; cleanup complete")

        except torch.cuda.OutOfMemoryError as e:
            self.resource_mgr.emergency_cleanup()
            issues.append(f"GPU out of memory during validation: {str(e)}")
            return False, issues, info
        except Exception as e:
            issues.append(f"Failed to load model: {str(e)}")
            return False, issues, info

        is_valid = len(issues) == 0
        return is_valid, issues, info

    def load_model(self, model_name: str, revision: Optional[str] = None, use_vllm: bool = True, timeout_seconds: int = 600):
        """
        Load a model for evaluation with timeout
        
        Downloads to temp folder and auto-cleans up after evaluation.
        
        Args:
            model_name: HuggingFace model name
            revision: Model revision (commit hash or branch)
            use_vllm: Try to use vLLM for faster inference
            timeout_seconds: Maximum time to wait for model loading (default: 10 minutes)
            
        Returns:
            (model, tokenizer, is_vllm, cleanup_fn)
            
        Raises:
            TimeoutError: If model loading exceeds timeout
            ModelLoadError: If model loading fails
        """
        import tempfile
        import shutil
        
        self.metrics.get_counter("model_loads_total").inc()
        
        # Create temp directory for this model
        temp_dir = tempfile.mkdtemp(prefix=f"evolai_model_{model_name.replace('/', '_')}_")
        loaded_model = None  # Track loaded model for cleanup
        
        def cleanup():
            """Delete temp model files and stop vLLM process"""
            nonlocal loaded_model
            try:
                # Clean up model from memory first
                if loaded_model is not None:
                    logger.info(f"Cleaning up model {model_name} from GPU memory")
                    try:
                        # For vLLM models, we need to properly shut down the engine
                        if hasattr(loaded_model, 'llm_engine'):
                            # vLLM LLM class
                            del loaded_model.llm_engine
                        del loaded_model
                        loaded_model = None
                        
                        # Force GPU cache cleanup
                        import torch
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        logger.info(f"Model {model_name} cleaned from GPU")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup model from GPU: {e}")
                
                # Delete temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temp directory {temp_dir} deleted")
                # Purge HF hub cache so next round re-downloads from scratch
                purge_hf_model_cache(model_name)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        
        # Use watchdog timer for timeout
        def on_timeout():
            logger.error(f"Model loading timed out after {timeout_seconds}s: {model_name}")
            cleanup()
            self.metrics.get_counter("model_loads_failed").inc()
        
        try:
            with WatchdogTimer(timeout_seconds, on_timeout, name=f"load_model_{model_name}"):
                with Timer(self.metrics.get_histogram("model_load_duration_seconds")):

                    if use_vllm:
                        # ── vLLM path: start a subprocess server ──────────────────────────────
                        # vllm binary lives in a separate virtualenv (see scripts/setup-validator.sh)
                        # and is called via subprocess — not imported directly — to avoid the
                        # bittensor/vllm fastapi+setuptools version conflict.
                        vllm_bin = os.environ.get("VLLM_EXECUTABLE", "vllm")
                        import shutil as _shutil_check
                        if not _shutil_check.which(vllm_bin) and not os.path.isfile(vllm_bin):
                            raise RuntimeError(
                                f"vLLM binary not found: '{vllm_bin}'.\n"
                                f"Run: bash scripts/setup-validator.sh\n"
                                f"Or set VLLM_EXECUTABLE in .env to the full path of 'vllm'."
                            )

                        from .vllm_client import VLLMClient
                        from .config import (
                            VLLM_MINER_PORT,
                            VLLM_MINER_GPU_INDEX,
                            VLLM_MINER_GPU_MEMORY_UTILIZATION,
                            VLLM_MINER_MAX_MODEL_LEN,
                        )

                        miner_server = VLLMClient(
                            port=VLLM_MINER_PORT,
                            max_model_len=VLLM_MINER_MAX_MODEL_LEN,
                            gpu_memory_utilization=VLLM_MINER_GPU_MEMORY_UTILIZATION,
                        )
                        miner_server.start_server(
                            model_name=model_name,
                            revision=revision or "main",
                            gpu_index=VLLM_MINER_GPU_INDEX,
                        )
                        loaded_model = miner_server  # so cleanup can stop it

                        def cleanup_vllm():
                            try:
                                miner_server.stop_server()
                            except Exception as _e:
                                logger.warning(f"Failed to stop miner vLLM server: {_e}")
                            try:
                                if os.path.exists(temp_dir):
                                    import shutil as _shutil
                                    _shutil.rmtree(temp_dir)
                            except Exception:
                                pass
                            # Purge HF hub cache so next round re-downloads from scratch
                            purge_hf_model_cache(model_name)

                        # model/tokenizer are None — orchestrator uses HTTP to the server
                        return None, None, True, cleanup_vllm

                    else:
                        # ── HuggingFace transformers path ─────────────────────────────────────
                        # Check GPU memory before loading
                        from .config import (
                            EVALUATION_GPU_REQUIRED_GB,
                            HF_EVAL_ENABLE_4BIT,
                            HF_EVAL_PREFER_FLASH_ATTN,
                            HF_EVAL_TORCH_COMPILE,
                        )
                        if not self.resource_mgr.gpu_manager.check_available_memory(required_gb=EVALUATION_GPU_REQUIRED_GB):
                            raise RuntimeError("Insufficient GPU memory to load model (need ~10GB free)")

                        if torch.cuda.is_available():
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True

                        compute_dtype = (
                            torch.bfloat16
                            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                            else torch.float16
                        )

                        base_kwargs = {
                            "revision": revision,
                            "trust_remote_code": False,
                            "device_map": "auto",
                            "cache_dir": temp_dir,
                            "low_cpu_mem_usage": True,
                            "torch_dtype": compute_dtype,
                        }

                        load_attempts = []

                        quantization_config = None
                        if HF_EVAL_ENABLE_4BIT and torch.cuda.is_available():
                            try:
                                from transformers import BitsAndBytesConfig

                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=compute_dtype,
                                )
                            except Exception as exc:
                                logger.warning(f"4-bit quantization unavailable: {exc}")

                        if quantization_config is not None:
                            quant_kwargs = dict(base_kwargs)
                            quant_kwargs["quantization_config"] = quantization_config
                            load_attempts.append(("4bit+flash_attention_2", {**quant_kwargs, "attn_implementation": "flash_attention_2"}))
                            load_attempts.append(("4bit+sdpa", {**quant_kwargs, "attn_implementation": "sdpa"}))
                            load_attempts.append(("4bit", quant_kwargs))

                        if HF_EVAL_PREFER_FLASH_ATTN and torch.cuda.is_available():
                            load_attempts.append(("fp16+flash_attention_2", {**base_kwargs, "attn_implementation": "flash_attention_2"}))
                            load_attempts.append(("fp16+sdpa", {**base_kwargs, "attn_implementation": "sdpa"}))

                        load_attempts.append(("fp16", base_kwargs))

                        # Suppress HuggingFace / huggingface_hub tqdm progress
                        # bars so they don't flood the terminal during model
                        # download and shard-loading.  The caller already shows
                        # a rich spinner around load_model().
                        import os as _os
                        _prev_hf_bars = _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
                        _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                        # Also silence transformers' own tqdm wrapper.
                        try:
                            from transformers.utils import logging as _tf_logging
                            _tf_logging.disable_progress_bar()
                            _tf_progress_disabled = True
                        except Exception:
                            _tf_progress_disabled = False

                        last_error = None
                        model = None
                        try:
                            for attempt_name, attempt_kwargs in load_attempts:
                                try:
                                    logger.info(f"Loading {model_name} with HF path ({attempt_name})")
                                    model = AutoModelForCausalLM.from_pretrained(
                                        model_name,
                                        **attempt_kwargs,
                                    )
                                    logger.info(f"Loaded {model_name} with HF path ({attempt_name})")
                                    break
                                except Exception as exc:
                                    last_error = exc
                                    logger.warning(
                                        f"HF load attempt failed for {model_name} ({attempt_name}): {exc}"
                                    )
                        finally:
                            # Restore progress-bar settings regardless of outcome.
                            if _prev_hf_bars is None:
                                _os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                            else:
                                _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_hf_bars
                            if _tf_progress_disabled:
                                try:
                                    _tf_logging.enable_progress_bar()
                                except Exception:
                                    pass

                        if model is None:
                            raise RuntimeError(f"Failed to load model via HF path: {last_error}")

                        loaded_model = model  # Track for cleanup

                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            revision=revision,
                            trust_remote_code=False,
                            cache_dir=temp_dir
                        )
                        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                            tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.padding_side = "right"

                        if hasattr(model, "config"):
                            model.config.use_cache = False
                        model.eval()

                        # Optional torch.compile: reduces kernel-launch overhead
                        # and speeds up repeated forward passes ~15-30%.
                        # dynamic=True handles the variable sequence lengths we
                        # produce during batched loss evaluation without recompiling
                        # for every new shape.
                        if HF_EVAL_TORCH_COMPILE and hasattr(torch, "compile"):
                            try:
                                model = torch.compile(
                                    model,
                                    mode="reduce-overhead",
                                    dynamic=True,
                                )
                                logger.info(f"torch.compile applied to {model_name}")
                            except Exception as exc:
                                logger.warning(f"torch.compile skipped for {model_name}: {exc}")

                        return model, tokenizer, False, cleanup
                    
        except torch.cuda.OutOfMemoryError as e:
            cleanup()
            # Emergency GPU cleanup
            self.resource_mgr.emergency_cleanup()
            self.metrics.get_counter("oom_errors_total").inc()
            self.metrics.get_counter("model_loads_failed").inc()
            raise GPUOutOfMemoryError(f"OOM loading model {model_name}", original_error=e)
        except (RuntimeError, ModelLoadError, GPUOutOfMemoryError):
            cleanup()
            self.metrics.get_counter("model_loads_failed").inc()
            raise
        except Exception as e:
            cleanup()  # Clean up on error
            error = classify_error(e)
            self.metrics.get_counter("model_loads_failed").inc()
            self.metrics.get_counter("errors_total").inc()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(error)}", original_error=e)


class DatasetSampler:
    
    def __init__(self, datasets: Dict[str, Dict]):
        """
        Initialize sampler with datasets
        
        Args:
            datasets: Dict[dataset_id, {weight, name, description, questions}]
        """
        self.datasets = datasets
        self._prepare_sampling()
    
    def _prepare_sampling(self):
        """Prepare weighted sampling"""
        self.dataset_ids = list(self.datasets.keys())
        self.weights = [self.datasets[ds_id].get('weight', 1.0) for ds_id in self.dataset_ids]
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            # Equal weights if no votes
            self.weights = [1.0 / len(self.weights)] * len(self.weights)
    
    def sample_questions(self, num_questions: int = 20) -> List[Dict]:
        """
        Sample questions from datasets based on weights
        
        Args:
            num_questions: Number of questions to sample
            
        Returns:
            List of question dictionaries with dataset context
        """
        questions = []
        
        for _ in range(num_questions):
            # Sample dataset based on weight
            dataset_id = random.choices(self.dataset_ids, weights=self.weights, k=1)[0]
            dataset = self.datasets[dataset_id]
            
            # Sample a random question from this dataset
            if 'questions' in dataset and dataset['questions']:
                question = random.choice(dataset['questions'])
                questions.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset.get('name', dataset_id),
                    'question': question
                })
        
        return questions


class InterviewEvaluator:
    """
    Bridge to the new modular evaluation system.

    The old monolithic interview logic has been replaced by:
      - orchestrator.EvaluationOrchestrator  (async pipeline)
      - judge_client.call_judge_with_fallback (HTTP call layer)
      - streaming.stream_miner_response       (Ollama / vLLM streaming)
      - scoring / compaction / loop_detector / sanitizer / …

    This class preserves the old public API so that callers in
    ``evaluator.py`` (and external scripts) continue to work.
    Internally it delegates to ``EvaluationOrchestrator``.
    """

    def __init__(self, config: EvaluationConfig, health_checker: Optional[HealthChecker] = None):
        self.config = config
        self.health_checker = health_checker
        self.metrics = get_metrics()

        # Create validator which handles GPU allocation
        self.validator = ModelValidator(config.device, health_checker)

        # Expose the primary judge model name for CLI backward compat
        self.judge_model_name = config.judge_models[0] if config.judge_models else "unknown"

        # New orchestrator replaces all judge/interview logic
        from .orchestrator import EvaluationOrchestrator
        self._orchestrator = EvaluationOrchestrator(
            judge_model_pool=config.judge_models,
            miner_backend="hf",  # overridden per-call by evaluate_miner() based on is_vllm
        )
        logger.info(f"Judge model pool: {len(config.judge_models)} models (new orchestrator)")
        logger.info(f"Primary judge: {config.judge_models[0] if config.judge_models else 'none'}")

    def evaluate_miner(
        self,
        miner_model,
        miner_tokenizer,
        questions: list[dict],
        is_vllm: bool = False,
        skip_sanity_check: bool = False,
        sampled_instructions: Optional[list[str]] = None,
        miner_uid: int = 0,
        model_name: str = "",
    ) -> tuple[float, Optional[str]]:
        """
        Evaluate a miner — delegates to the new async orchestrator.

        Preserves the old (score, failure_reason) return signature.
        """
        import asyncio
        import random as _random

        self.metrics.get_counter("evaluations_total").inc()

        try:
            # asyncio.get_running_loop() raises RuntimeError when no loop is running
            # (safe on Python 3.7+; unlike get_event_loop() which errors on 3.10+)
            try:
                asyncio.get_running_loop()
                _in_running_loop = True
            except RuntimeError:
                _in_running_loop = False

            # Choose miner backend based on how the model was loaded
            _backend = "vllm" if is_vllm else "hf"
            _hf_model = None if is_vllm else miner_model
            _hf_tokenizer = None if is_vllm else miner_tokenizer

            # Pick a distinct sanity-check instruction from the pre-sampled pool so
            # it's different from the scored questions and consistent across miners.
            _validity_instruction = ""
            if not skip_sanity_check and sampled_instructions:
                _validity_instruction = _random.choice(sampled_instructions)

            if _in_running_loop:
                # Already inside an event loop (e.g. Jupyter / bittensor axon thread)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    output = pool.submit(
                        asyncio.run,
                        self._orchestrator.evaluate_miner(
                            miner_uid=miner_uid,
                            model_name=model_name,
                            questions=questions,
                            instruction_for_validity=_validity_instruction,
                            skip_sanity_check=skip_sanity_check,
                            miner_backend=_backend,
                            miner_hf_model=_hf_model,
                            miner_hf_tokenizer=_hf_tokenizer,
                        ),
                    ).result()
            else:
                output = asyncio.run(
                    self._orchestrator.evaluate_miner(
                        miner_uid=miner_uid,
                        model_name=model_name,
                        questions=questions,
                        instruction_for_validity=_validity_instruction,
                        skip_sanity_check=skip_sanity_check,
                        miner_backend=_backend,
                        miner_hf_model=_hf_model,
                        miner_hf_tokenizer=_hf_tokenizer,
                    )
                )
        except Exception as e:
            self.metrics.get_counter("evaluations_failed").inc()
            logger.error(f"Orchestrator evaluation failed: {e}", exc_info=True)
            return 0.0, str(e)

        if output.error:
            return 0.0, output.error

        self.metrics.get_counter("evaluations_success").inc()
        return output.normalized_score * 10.0, None  # scale to 0-10 for backward compat

    def cleanup(self):
        """Clean up — no in-process judge model to free any more."""
        torch.cuda.empty_cache()

    def sample_sanity_check_instructions(
        self,
        num_samples: int = 10,
    ) -> Optional[List[str]]:
        """
        Sample sanity-check instructions for validity screening.

        Loads the ``instruction`` column directly from the
        ``evolai/natural_questions`` HuggingFace dataset using streaming so
        no full download is required.  A pool of candidates is collected and
        then randomly sub-sampled to ``num_samples`` items.

        Falls back to a built-in list when the ``datasets`` library is not
        installed or the HF load fails.

        Args:
            num_samples: How many instructions to return.

        Returns:
            List[str] of instruction strings, or None on failure.
        """
        # ── Try loading directly from HuggingFace ─────────────────────────
        try:
            from datasets import load_dataset  # type: ignore

            # Collect a pool larger than num_samples so the final random
            # sample is drawn from a diverse set of questions.
            pool_size = max(num_samples * 20, 200)
            pool: List[str] = []

            ds = load_dataset(
                'evolai/natural_questions',
                split='train',
                streaming=True,
            )
            for item in ds:
                instr = item.get('instruction', '')
                if isinstance(instr, str) and instr.strip():
                    pool.append(instr.strip())
                if len(pool) >= pool_size:
                    break

            if pool:
                sampled = random.sample(pool, min(num_samples, len(pool)))
                logger.debug(
                    f"Sanity-check instructions loaded from evolai/natural_questions "
                    f"(pool={len(pool)}, sampled={len(sampled)})"
                )
                return sampled

            logger.warning(
                "evolai/natural_questions loaded but no 'instruction' values found; "
                "falling back to built-in instructions"
            )
        except ImportError:
            logger.warning(
                "'datasets' library not installed — "
                "falling back to built-in sanity check instructions"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load evolai/natural_questions from HuggingFace: {e} — "
                "falling back to built-in instructions"
            )

        # ── Fallback: built-in list ────────────────────────────────────────
        try:
            default_instructions = [
                "What is 2 + 2?",
                "Say hello in English.",
                "What color is the sky on a clear day?",
                "Name three fruits.",
                "What is the capital of France?",
                "Count from 1 to 5.",
                "What day comes after Monday?",
                "Is water wet? Answer yes or no.",
                "Complete this sentence: The sun rises in the ___.",
                "What language is this sentence written in?",
            ]
            return random.sample(default_instructions, min(num_samples, len(default_instructions)))
        except Exception as e:
            logger.warning(f"Failed to sample fallback sanity check instructions: {e}")
            return None


# Legacy compatibility: keep the old method names importable but unused
# (code scanning / grep may reference them)
InterviewEvaluator.load_judge_model = lambda self, rotate=False: None  # noqa: E731


class _RemovedInterviewMethods:
    """Placeholder — old InterviewEvaluator methods removed.

    All interview, sanity check, and direct judge calling logic has been
    replaced by the modular orchestrator pipeline:
      orchestrator.py  → EvaluationOrchestrator
      judge_client.py  → call_judge_with_fallback
      streaming.py     → stream_miner_response / stream_miner_response_vllm
      scoring.py       → calculate_final_score / normalize_score
      compaction.py    → compact_conversation_history
      loop_detector.py → AnswerLoopDetector
      sanitizer.py     → sanitize_miner_answer_for_judge
      lifecycle.py     → emit_* event helpers
    """
    pass


class ScoreCalculator:
    """Calculates final scores with variance and improvement penalties"""
    
    @staticmethod
    def calculate_effective_score(
        raw_score: float,
        significant_improve: float = 0.0,
        variance: float = 0.0,
        alpha: float = 1.0
    ) -> float:
        """
        Calculate effective score with penalties
        
        Args:
            raw_score: Raw evaluation score (0-10)
            significant_improve: Penalty for models that need significant improvement
            variance: Score variance across evaluations
            alpha: Variance penalty coefficient
            
        Returns:
            Effective score
        """
        return raw_score - significant_improve - alpha * variance
    
    @staticmethod
    def softmax_with_temperature(scores: List[float], temperature: float = 1.0) -> List[float]:
        """
        Apply softmax with temperature for emission distribution
        
        Args:
            scores: List of effective scores
            temperature: Temperature parameter (higher = more uniform)
            
        Returns:
            List of probabilities (weights)
        """
        import math
        
        # Scale by temperature
        scaled_scores = [s / temperature for s in scores]
        
        # Compute softmax
        max_score = max(scaled_scores) if scaled_scores else 0
        exp_scores = [math.exp(s - max_score) for s in scaled_scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [e / sum_exp for e in exp_scores]
    
    @staticmethod
    def update_ema_score(
        current_ema: float,
        new_score: float,
        alpha: float = 0.3
    ) -> float:
        """
        Update EMA score with new evaluation
        
        Args:
            current_ema: Current EMA score
            new_score: New evaluation score
            alpha: EMA coefficient (higher = more weight on new score)
            
        Returns:
            Updated EMA score
        """
        return alpha * new_score + (1 - alpha) * current_ema


class EMAScoreTracker:
    """Tracks EMA scores for miners across evaluation rounds"""
    
    def __init__(self, alpha: float = 0.3, storage_path: Optional[Path] = None):
        """
        Initialize EMA score tracker
        
        Args:
            alpha: EMA coefficient (0.3 = 30% new score, 70% historical)
            storage_path: Path to store scores (default: ~/.evolai/validator/scores.json)
        """
        self.alpha = alpha
        self.storage_path = storage_path or Path.home() / ".evolai" / "validator" / "scores.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing scores
        self.scores = self._load_scores()
    
    def _load_scores(self) -> Dict:
        """Load scores from disk"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {
            'transformer': {},  # {miner_uid: {ema_score, count, last_updated, score_history, variance, rank_history}}
            'mamba2': {}
        }
    
    def _save_scores(self):
        """Save scores to disk (atomic write to prevent corruption on crash)"""
        tmp_path = self.storage_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.scores, f, indent=2)
        tmp_path.replace(self.storage_path)
    
    def update_score(self, miner_uid: int, track: str, new_score: float, timestamp: str, model_name: str = "") -> float:
        """
        Update EMA score for a miner.

        If *model_name* is provided and differs from the last-known model for
        this UID the entire EMA history is wiped so a freshly-registered model
        cannot inherit the score of its predecessor.

        Args:
            miner_uid: Miner UID
            track: "transformer" or "mamba2"
            new_score: New raw score from evaluation
            timestamp: ISO timestamp
            model_name: HuggingFace repo name for the model being evaluated

        Returns:
            Updated EMA score
        """
        uid_str = str(miner_uid)

        # ── Reset EMA if the miner registered a different model ────────────
        existing = self.scores[track].get(uid_str)
        if existing and model_name:
            stored_model = existing.get('model_name', '')
            if stored_model and stored_model != model_name:
                logger.info(
                    f"UID {miner_uid} track={track}: model changed "
                    f"{stored_model!r} → {model_name!r} — resetting EMA"
                )
                del self.scores[track][uid_str]

        if uid_str not in self.scores[track]:
            # First evaluation (fresh or post-reset) — use raw score as initial EMA
            ema_score = new_score
            self.scores[track][uid_str] = {
                'ema_score': ema_score,
                'count': 1,
                'last_updated': timestamp,
                'last_raw_score': new_score,
                'score_history': [new_score],
                'variance': 0.0,
                'model_name': model_name,
            }
        else:
            # Normal EMA update
            current_ema = self.scores[track][uid_str]['ema_score']
            ema_score = ScoreCalculator.update_ema_score(current_ema, new_score, self.alpha)

            # Update score history (keep last 10 for variance calculation)
            score_history = self.scores[track][uid_str].get('score_history', [])
            score_history.append(new_score)
            if len(score_history) > 10:
                score_history = score_history[-10:]

            # Calculate variance
            import statistics
            variance = statistics.variance(score_history) if len(score_history) > 1 else 0.0

            self.scores[track][uid_str]['ema_score'] = ema_score
            self.scores[track][uid_str]['count'] += 1
            self.scores[track][uid_str]['last_updated'] = timestamp
            self.scores[track][uid_str]['last_raw_score'] = new_score
            self.scores[track][uid_str]['score_history'] = score_history
            self.scores[track][uid_str]['variance'] = variance
            if model_name:
                self.scores[track][uid_str]['model_name'] = model_name

        self._save_scores()
        return ema_score
    
    def get_scores(self, track: str) -> Dict[int, float]:
        """
        Get all EMA scores for a track
        
        Args:
            track: "transformer" or "mamba2"
            
        Returns:
            Dict mapping miner_uid to ema_score
        """
        return {
            int(uid): data['ema_score']
            for uid, data in self.scores[track].items()
        }
    
    def get_miner_stats(self, miner_uid: int, track: str) -> Optional[Dict]:
        """Get detailed stats for a miner"""
        uid_str = str(miner_uid)
        return self.scores[track].get(uid_str)
    
    def get_effective_scores(self, track: str, significant_improve: float = 0.5, variance_alpha: float = 0.1, min_evaluations: int = 10) -> Dict[int, float]:
        """
        Calculate effective scores with variance penalties.

        Miners with fewer than *min_evaluations* completed rounds are excluded
        entirely so a single lucky evaluation cannot earn on-chain weights.

        Args:
            track: "transformer" or "mamba2"
            significant_improve: Penalty for models needing improvement
            variance_alpha: Variance penalty coefficient
            min_evaluations: Minimum number of completed evaluations required
                before a miner is eligible to receive weights (default 10)

        Returns:
            Dict mapping miner_uid to effective_score
        """
        effective_scores = {}
        for uid_str, data in self.scores[track].items():
            # Exclude miners that haven't been evaluated enough times
            eval_count = data.get('count', 0)
            if eval_count < min_evaluations:
                logger.debug(
                    f"UID {uid_str} track={track}: only {eval_count}/{min_evaluations} "
                    "evaluations — excluded from weight calculation"
                )
                continue

            uid = int(uid_str)
            raw_score = data['ema_score']
            variance = data.get('variance', 0.0)

            # Apply penalty if score below threshold
            improve_penalty = significant_improve if raw_score < 7.0 else 0.0

            # Calculate effective score
            effective_score = ScoreCalculator.calculate_effective_score(
                raw_score, improve_penalty, variance, variance_alpha
            )
            effective_scores[uid] = max(0.0, effective_score)  # Floor at 0
        
        return effective_scores
    
    def update_rank_history(self, track: str, top_uid: int, timestamp: str):
        """
        Track rank history for stagnation detection.
        
        Appends the current top UID and timestamp each weight update.
        Keeps up to RANK_HISTORY_MAX_ENTRIES (~30 days at 30-min intervals).
        
        Args:
            track: "transformer" or "mamba2"
            top_uid: UID of current top miner
            timestamp: ISO timestamp
        """
        from .config import RANK_HISTORY_MAX_ENTRIES
        
        # Initialize rank history if not exists
        if 'rank_history' not in self.scores:
            self.scores['rank_history'] = {'transformer': [], 'mamba2': []}
        
        if track not in self.scores['rank_history']:
            self.scores['rank_history'][track] = []
        
        # Add entry
        self.scores['rank_history'][track].append({
            'top_uid': top_uid,
            'timestamp': timestamp
        })
        
        # Cap history length
        if len(self.scores['rank_history'][track]) > RANK_HISTORY_MAX_ENTRIES:
            self.scores['rank_history'][track] = self.scores['rank_history'][track][-RANK_HISTORY_MAX_ENTRIES:]
        
        self._save_scores()
    
    def check_stagnation(self, track: str) -> tuple[bool, int]:
        """
        Check if the current top miner has held #1 continuously for 3+ days.
        
        Walks backwards through the FULL rank history to find how long the
        current winner has been on top without interruption. When a new top
        miner appears, stagnation resets immediately.
        
        Args:
            track: "transformer" or "mamba2"
            
        Returns:
            (is_stagnant, days_unchanged)
        """
        from .config import STAGNATION_THRESHOLD_DAYS
        
        if 'rank_history' not in self.scores or track not in self.scores['rank_history']:
            return False, 0
        
        history = self.scores['rank_history'][track]
        if len(history) < 2:
            return False, 0
        
        # Current top UID is the most recent entry
        current_top = history[-1]['top_uid']
        
        # Walk backwards to find the earliest CONSECUTIVE entry with same UID
        earliest_idx = len(history) - 1
        for i in range(len(history) - 2, -1, -1):
            if history[i]['top_uid'] == current_top:
                earliest_idx = i
            else:
                break  # Different UID — this is where the streak started
        
        # Calculate how many days this UID has been continuously on top
        from datetime import datetime
        earliest_time = datetime.fromisoformat(history[earliest_idx]['timestamp'])
        latest_time = datetime.fromisoformat(history[-1]['timestamp'])
        days_unchanged = (latest_time - earliest_time).days
        
        if days_unchanged >= STAGNATION_THRESHOLD_DAYS:
            return True, days_unchanged
        
        return False, 0
    
    def get_decay_factor(self, days_stagnant: int) -> float:
        """
        Calculate linear decay factor for stagnant emissions.
        
        Timeline (with default constants):
          Day 0-2:  decay_factor = 1.0  (no decay, winner gets full weight)
          Day 3:    decay starts
          Day 3-9:  decay_factor linearly decreases from 1.0 → 0.0
          Day 10+:  decay_factor = 0.0  (fully burned to UID 0)
        
        When a new top miner appears, days_stagnant resets to 0 and the
        new winner immediately gets full weight (decay_factor = 1.0).
        
        Args:
            days_stagnant: Total days the same UID has been #1 continuously.
            
        Returns:
            Decay factor ∈ [0.0, 1.0]. 1.0 = full weight, 0.0 = fully burned.
        """
        from .config import STAGNATION_THRESHOLD_DAYS, STAGNATION_DECAY_PERIOD_DAYS
        
        if days_stagnant <= STAGNATION_THRESHOLD_DAYS:
            return 1.0  # No decay yet
        
        # Linear decay over STAGNATION_DECAY_PERIOD_DAYS after threshold
        days_in_decay = days_stagnant - STAGNATION_THRESHOLD_DAYS
        if days_in_decay >= STAGNATION_DECAY_PERIOD_DAYS:
            return 0.0  # Fully decayed → all weight burned to UID 0
        
        return 1.0 - (days_in_decay / STAGNATION_DECAY_PERIOD_DAYS)


class ModelRegistry:
    """
    Tracks model ownership to prevent theft
    
    Caches up to 10 latest models per UID per track.
    Prevents miners from stealing models uploaded by others.
    """
    
    MAX_MODELS_PER_MINER = 10
    
    def __init__(self, cache_dir: Path = Path.home() / ".evolai" / "validator"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "model_registry.json"
        self.registry = self._load_registry()
        self.hf_api = HfApi()
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry from {self.cache_file}: {e}")
        return {
            "transformer": {},  # {uid: [{model_name, registered_at, hf_upload_time}, ...]}
            "mamba2": {}
        }
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    @with_retry(
        config=RetryConfig(max_attempts=3),
        retryable_categories={ErrorCategory.NETWORK}
    )
    def get_hf_upload_time(self, model_name: str, revision: Optional[str] = None) -> Optional[datetime]:
        """Get HuggingFace upload timestamp for a model revision"""
        try:
            from datetime import datetime
            repo_info = self.hf_api.repo_info(model_name, revision=revision, repo_type="model")
            
            # Get commit timestamp
            commits = list(self.hf_api.list_repo_commits(model_name, repo_type="model"))
            revision_to_check = revision or repo_info.sha
            
            for commit in commits:
                if commit.commit_id == revision_to_check or commit.commit_id.startswith(revision_to_check or ""):
                    return commit.created_at
            
            # Fallback to repo last modified
            return repo_info.last_modified
        except Exception as e:
            logger.warning(f"Failed to get HF upload time for {model_name}@{revision}: {e}")
            return None
    
    def update_from_metagraph(self, metagraph, subtensor, netuid: int):
        """
        Update registry from current metagraph state
        
        Scans all miners and builds/updates the model ownership cache.
        Keeps latest 10 models per UID per track.
        """
        from datetime import datetime
        from evolai.utils.metadata import decompress_metadata
        
        for uid, hotkey in enumerate(metagraph.hotkeys):
            try:
                commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
                if not commit_data:
                    continue
                
                # Extract bytes from bittensor's nested structure
                try:
                    if isinstance(commit_data, dict) and 'info' in commit_data:
                        fields = commit_data['info']['fields']
                        if fields and len(fields) > 0:
                            field_data = fields[0]
                            if field_data and len(field_data) > 0:
                                raw_data = field_data[0]
                                raw_key = next((k for k in raw_data if k.startswith('Raw') and k[3:].isdigit()), None)
                                if raw_key is not None:
                                    byte_tuple = raw_data[raw_key][0]
                                    # Convert tuple of ints to bytes
                                    compressed_bytes = bytes(byte_tuple)
                                    metadata = decompress_metadata(compressed_bytes)
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                except Exception:
                    continue
                
                if not metadata:
                    continue
                
                # Check both tracks
                for track in ["transformer", "mamba2"]:
                    track_info = metadata.get(track, {})
                    if not track_info:
                        continue
                    
                    model_name = track_info.get('model_name')
                    revision = track_info.get('revision', 'main')
                    
                    if not model_name:
                        continue
                    
                    # Query HuggingFace for actual upload time (don't trust metadata)
                    hf_upload_time = self.get_hf_upload_time(model_name, revision)
                    
                    # Initialize UID's model list if needed
                    uid_str = str(uid)
                    if uid_str not in self.registry[track]:
                        self.registry[track][uid_str] = []
                    
                    models_list = self.registry[track][uid_str]
                    
                    # Check if model already in this UID's list
                    existing_idx = None
                    for idx, entry in enumerate(models_list):
                        if entry['model_name'] == model_name:
                            existing_idx = idx
                            break
                    
                    model_entry = {
                        'model_name': model_name,
                        'revision': revision,
                        'registered_at': datetime.utcnow().isoformat(),  # When we discovered it
                        'hf_upload_time': hf_upload_time.isoformat() if hf_upload_time else None
                    }
                    
                    if existing_idx is not None:
                        # Update existing entry
                        models_list[existing_idx] = model_entry
                    else:
                        # Add new entry
                        models_list.append(model_entry)
                    
                    # Keep only latest 10 models (sorted by registration time)
                    models_list.sort(key=lambda x: x['registered_at'], reverse=True)
                    self.registry[track][uid_str] = models_list[:self.MAX_MODELS_PER_MINER]
            except Exception as e:
                logger.warning(f"Failed to update registry for UID {uid}: {e}")
                continue
        
        self._save_registry()
    
    def check_ownership(self, model_name: str, track: str, uid: int, hf_upload_time: datetime) -> Tuple[bool, Optional[int]]:
        """
        Check if this UID can claim this model
        
        Returns:
            (is_owner, conflicting_uid)
        """
        from datetime import datetime
        
        # Check all UIDs to see if anyone else claimed this model first
        for other_uid_str, models_list in self.registry[track].items():
            other_uid = int(other_uid_str)
            
            if other_uid == uid:
                continue  # Skip self
            
            # Check if this model is in other UID's list
            for entry in models_list:
                if entry['model_name'] == model_name:
                    # Found conflict - check who uploaded to HF first
                    other_hf_time_str = entry.get('hf_upload_time')
                    if other_hf_time_str:
                        other_hf_time = datetime.fromisoformat(other_hf_time_str)
                        
                        # If other UID has earlier HF upload time, they own it
                        if other_hf_time < hf_upload_time:
                            return False, other_uid
        
        return True, None  # No conflicts or this UID uploaded first
    
    def get_model_info(self, model_name: str, track: str, uid: int) -> Optional[Dict]:
        """Get cached model info for specific UID"""
        uid_str = str(uid)
        if uid_str not in self.registry[track]:
            return None
        
        for entry in self.registry[track][uid_str]:
            if entry['model_name'] == model_name:
                return entry
        
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Anti-Gaming: Copy / Plagiarism Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _fingerprint_store(self, track: str) -> Dict:
        """Return (initialising if absent) the per-track fingerprint lookup.

        Shape::

            {track: {"<exact_hash>": {"uid": int, "model_name": str, "registered_at": str}}}
        """
        if "fingerprints" not in self.registry:
            self.registry["fingerprints"] = {"transformer": {}, "mamba2": {}}
        if track not in self.registry["fingerprints"]:
            self.registry["fingerprints"][track] = {}
        return self.registry["fingerprints"][track]

    def register_fingerprint(
        self,
        uid: int,
        track: str,
        model_name: str,
        fingerprint_dict: Dict,
        timestamp: str,
    ) -> None:
        """
        Persist a model's fingerprint so future miners can be checked against it.

        Called *after* the model has passed all other validation checks.

        Args:
            uid:              Miner UID that owns this model.
            model_name:       HuggingFace repo slug (``org/name``).
            track:            ``"transformer"`` or ``"mamba2"``.
            fingerprint_dict: Result of ``ModelFingerprint.to_dict()``.
            timestamp:        ISO-8601 registration time.
        """
        store = self._fingerprint_store(track)
        exact_hash = fingerprint_dict.get("exact_hash", "")
        if not exact_hash:
            logger.warning(f"register_fingerprint: empty exact_hash for {model_name} uid={uid}")
            return

        entry = {
            "uid": uid,
            "model_name": model_name,
            "registered_at": timestamp,
            "fingerprint": fingerprint_dict,
        }
        store[exact_hash] = entry
        self._save_registry()
        logger.info(
            f"[FP] Registered fingerprint for uid={uid} track={track} "
            f"model={model_name} exact_hash={exact_hash[:12]}…"
        )

    def check_copy_gaming(
        self,
        uid: int,
        track: str,
        model_name: str,
        fingerprint_dict: Dict,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Detect whether this model is a copy of a model already owned by a
        *different* UID.

        Checks are applied in three escalating layers (cheapest first):
          1. Exact ``exact_hash`` match in the fingerprint store.
          2. ``arch_hash`` + ``layer_names_hash`` + ``param_count`` match
             (structural clone — same architecture, possibly different exact bytes).
          3. Cosine similarity of ``fuzzy_vector`` ≥ FINGERPRINT_FUZZY_THRESHOLD
             (near-copy / trivial fine-tune evasion attempt).

        Args:
            uid:              UID that is being validated.
            track:            ``"transformer"`` or ``"mamba2"``.
            model_name:       Model being validated.
            fingerprint_dict: ``ModelFingerprint.to_dict()`` of the candidate model.

        Returns:
            ``(is_copy, owner_uid, reason)`` — ``is_copy=False`` means no
            collision was found.
        """
        from .model_fingerprint import ModelFingerprint, fingerprints_collide
        from .config import FINGERPRINT_FUZZY_THRESHOLD

        store = self._fingerprint_store(track)
        new_fp = ModelFingerprint.from_dict(fingerprint_dict)

        for exact_hash, entry in store.items():
            owner_uid: int = entry["uid"]
            if owner_uid == uid:
                continue  # Same miner — not plagiarism

            existing_fp_dict: Dict = entry.get("fingerprint", {})
            if not existing_fp_dict:
                # Legacy entry without fingerprint data — fall back to exact hash only
                if exact_hash == fingerprint_dict.get("exact_hash"):
                    return (
                        True,
                        owner_uid,
                        f"exact_weight_copy of uid={owner_uid} model={entry['model_name']}",
                    )
                continue

            existing_fp = ModelFingerprint.from_dict(existing_fp_dict)
            collision, reason = fingerprints_collide(
                new_fp, existing_fp, fuzzy_threshold=FINGERPRINT_FUZZY_THRESHOLD
            )
            if collision:
                logger.warning(
                    f"[FP] Copy-gaming detected: uid={uid} track={track} "
                    f"model={model_name!r} is a {reason} of "
                    f"uid={owner_uid} model={entry['model_name']!r}"
                )
                return (
                    True,
                    owner_uid,
                    f"{reason} (owner uid={owner_uid}, model={entry['model_name']})",
                )

        return False, None, ""
