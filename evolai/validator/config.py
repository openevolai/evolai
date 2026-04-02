"""
Configuration Constants for EvolAI Validator Evaluation System

All tunable parameters in one place.  Every constant can be overridden via an
environment variable of the same name (e.g. VLLM_JUDGE_PORT=9001).

Design basis: OpenClaw flat-constant style (no nested config objects, no YAML).
"""

from __future__ import annotations

import os as _os
import secrets as _secrets


def _env(name: str, default, *, cast=None):
    """Read *name* from the environment; return *default* if unset.

    When *cast* is given it is applied to the raw string.  Convenient for
    ints/floats/bools so callers don't repeat ``int(os.environ.get(...))``.
    """
    raw = _os.environ.get(name)
    if raw is None:
        return default
    if cast is not None:
        return cast(raw)
    return raw

# ──────────────────────────────────────────────────────────────────────────────
# Local Model Deployment
# ──────────────────────────────────────────────────────────────────────────────
VLLM_BASE_URL:   str   = _env("VLLM_BASE_URL",   "http://127.0.0.1:8001/v1")
VLLM_MINER_BASE_URL: str = _env("VLLM_MINER_BASE_URL", "http://127.0.0.1:8000/v1")

# Ollama (alternative local backend — optional)
OLLAMA_BASE_URL:  str = "http://127.0.0.1:11434"
OLLAMA_V1_URL:    str = "http://127.0.0.1:11434/v1"
OLLAMA_CHAT_URL:  str = "http://127.0.0.1:11434/api/chat"
OLLAMA_NUM_CTX:   int = 65536
OLLAMA_NUM_PREDICT: int = 4096

# GPU memory fractions
VLLM_GPU_MEMORY_UTILIZATION:       float = _env("VLLM_GPU_MEMORY_UTILIZATION",       0.90, cast=float)
VLLM_JUDGE_GPU_MEMORY_UTILIZATION: float = _env("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", 0.90, cast=float)
VLLM_MINER_GPU_MEMORY_UTILIZATION: float = _env("VLLM_MINER_GPU_MEMORY_UTILIZATION", 0.85, cast=float)

# Context lengths
VLLM_JUDGE_MAX_MODEL_LEN: int = _env("VLLM_JUDGE_MAX_MODEL_LEN", 65536, cast=int)
VLLM_MINER_MAX_MODEL_LEN: int = _env("VLLM_MINER_MAX_MODEL_LEN", 32768, cast=int)

# Ports
VLLM_JUDGE_PORT: int = _env("VLLM_JUDGE_PORT", 8001, cast=int)
VLLM_MINER_PORT: int = _env("VLLM_MINER_PORT", 8000, cast=int)

# GPU assignment
VLLM_JUDGE_GPU_INDEX: int = _env("VLLM_JUDGE_GPU_INDEX", 0, cast=int)
VLLM_MINER_GPU_INDEX: int = _env("VLLM_MINER_GPU_INDEX", 1, cast=int)
VLLM_JUDGE_TENSOR_PARALLEL_SIZE: int = _env("VLLM_JUDGE_TENSOR_PARALLEL_SIZE", 1, cast=int)

# Auth
# SECURITY: If VLLM_API_KEY is not set in the environment, a random key is
# generated at startup so vLLM servers are never protected by the well-known
# static token "local". Set VLLM_API_KEY in your .env to use a fixed key.
_raw_api_key: str = _env("VLLM_API_KEY", "")
LOCAL_API_KEY: str = _raw_api_key if _raw_api_key else _secrets.token_urlsafe(32)

# Path to the `vllm` binary.  Set this when vllm lives in a separate
# virtualenv (the recommended setup, see scripts/setup-validator.sh):
#   export VLLM_EXECUTABLE=/path/to/vllm_env/bin/vllm
VLLM_EXECUTABLE: str = _env("VLLM_EXECUTABLE", "vllm")

# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU Parallel Miner Slots  (4+ GPU mode)
# ──────────────────────────────────────────────────────────────────────────────
# With 4× A100 80 GB: GPU 0 = judge (warm), GPU 1-3 = parallel miner slots.
# Each slot owns one GPU and one vLLM server process (port 8010 + slot_index).
# All slots share the SAME judge server at VLLM_BASE_URL.
#
# To enable:  evolcli validator run --parallel-miners 3
# Port layout: slot 0 → 8010, slot 1 → 8011, slot 2 → 8012
VLLM_PARALLEL_MINER_BASE_PORT: int = 8010     # port = BASE + slot_index
VLLM_PARALLEL_MINER_GPU_INDICES: list[int] = [
    int(x) for x in _env("VLLM_PARALLEL_MINER_GPU_INDICES", "1,2,3").split(",")
]  # GPUs used for parallel slots  (override via VLLM_PARALLEL_MINER_GPU_INDICES=0,1,2)
NUM_PARALLEL_MINER_SLOTS: int = 1             # Default = 1 (single-miner mode); set to 3 for 4-GPU

# ──────────────────────────────────────────────────────────────────────────────
# Agent Run Loop (from OpenClaw run.ts)
# ──────────────────────────────────────────────────────────────────────────────
MAX_OVERFLOW_COMPACTION_ATTEMPTS: int = 3    # Max staged compactions on context overflow per Q
BASE_RUN_RETRY_ITERATIONS: int = 24          # Base judge call attempts per evaluation
RUN_RETRY_PER_PROFILE: int = 8              # Extra attempts per fallback judge in pool
# Effective max: max(32, min(160, BASE + max(1,pool_size) * PER_PROFILE))

# ──────────────────────────────────────────────────────────────────────────────
# Judge Call Layer (from OpenClaw retry.ts + model-fallback.ts)
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_RETRY_ATTEMPTS: int = 3           # Retries per judge call
JUDGE_MIN_DELAY_MS: int = 300           # First retry wait
JUDGE_MAX_DELAY_MS: int = 30_000        # Backoff cap (30 s)
JUDGE_JITTER: float = 0.2              # ±20% delay randomisation
JUDGE_COOLDOWN_OOM_MS: int = 300_000    # 5 min cooldown after OOM
JUDGE_COOLDOWN_ERROR_MS: int = 60_000   # 60 s cooldown after general error
PROBE_INTERVAL_MS: int = 30_000         # Min gap between primary-model probes
PROBE_MARGIN_MS: int = 120_000          # Probe primary within 2 min of cooldown expiry

# ──────────────────────────────────────────────────────────────────────────────
# Context Window Guard (from OpenClaw context-window-guard.ts)
# ──────────────────────────────────────────────────────────────────────────────
# Largest possible judge call (Phase 3 follow-up with full 16-turn history):
#   history_text  16 × 500 chars ≈ 2,700 tokens
#   previous_summary              ≈   240 tokens
#   system prompt                 ≈    50 tokens
#   JUDGE_PROMPT_OVERHEAD         = 4,096 tokens
#   max_tokens (output)           =   300 tokens
#   TOTAL worst-case              ≈ 7,400 tokens
# With VLLM_JUDGE_MAX_MODEL_LEN=32768 there is ~25K tokens of headroom.
# CONTEXT_WINDOW_HARD_MIN_TOKENS must be > 7,400 to catch misconfiguration.
CONTEXT_WINDOW_HARD_MIN_TOKENS: int = 8_000    # Block judge if fewer available tokens
CONTEXT_WINDOW_WARN_BELOW_TOKENS: int = 16_000 # Warn if close to limit
JUDGE_PROMPT_OVERHEAD_TOKENS: int = 4_096      # Reserved for prompt templates

# ──────────────────────────────────────────────────────────────────────────────
# Anti-Injection / Answer Sanitization
# ──────────────────────────────────────────────────────────────────────────────
MINER_ANSWER_MAX_CHARS: int = 2_000   # Truncate before any judge call

# ──────────────────────────────────────────────────────────────────────────────
# Answer Loop Detection (from OpenClaw tool-loop-detection.ts)
# ──────────────────────────────────────────────────────────────────────────────
ANSWER_LOOP_HISTORY_SIZE: int = 30     # Rolling SHA-256 hash window
LOOP_WARNING_THRESHOLD: int = 3        # Warn at 3 identical answer hashes
LOOP_CIRCUIT_BREAKER: int = 5          # score=0.0 at 5 identical hashes

# ──────────────────────────────────────────────────────────────────────────────
# Context Compaction (from OpenClaw compaction.ts)
# ──────────────────────────────────────────────────────────────────────────────
SAFETY_MARGIN: float = 1.2             # Token overestimate factor
COMPACTION_TRIGGER_K: int = 16         # Compact when history > 16 turns
SUMMARY_KEEP_RECENT: int = 3           # Keep N most recent turns verbatim
COMPACTION_MAX_TOKENS: int = 800       # Max tokens for summary block

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Validity
# ──────────────────────────────────────────────────────────────────────────────
VALIDITY_SCORE: float = 0.2
VALIDITY_TIMEOUT_S: int = 60
VALIDITY_CORRECTNESS_THRESH: float = 0.5   # Float threshold for validity gate

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Questions
# ──────────────────────────────────────────────────────────────────────────────
NUM_QUESTIONS: int = 1   # 1 question per evaluation round
DATASET_SELECTION: str = "voting_based"
DATASET_VOTING_WEIGHTS: list[float] = [0.6, 0.4]
QUESTION_TIMEOUT_S: int = 120
CONTESTED_SCORE_LOW: float = 0.3          # Two-judge consensus below this
CONTESTED_SCORE_HIGH: float = 0.7         # Two-judge consensus above this
LOW_CONFIDENCE_DISCOUNT: float = 0.8      # Multiply score when judge conf < 0.3
PHASE2_SKIP_INTERVIEW_THRESHOLD: float = CONTESTED_SCORE_LOW  # Skip Phase 3 interview when Phase 2 score < this (wrong answer)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Interview
# ──────────────────────────────────────────────────────────────────────────────
INTERVIEW_BASE_TURNS: int = 7
INTERVIEW_MAX_TURNS: int = 10
INTERVIEW_MIN_TURNS: int = 4
TOTAL_INTERVIEW_TOKEN_LIMIT: int = 20480  # Max cumulative miner output tokens across all interview turns per question
PER_RESPONSE_TOKEN_LIMIT: int = TOTAL_INTERVIEW_TOKEN_LIMIT  # No per-response cap — miner may use any length
                                                               # up to the total budget; the cumulative limit enforces it
INTERVIEW_TIMEOUT_S: int = 180

# ──────────────────────────────────────────────────────────────────────────────
# Judge Model Pool & Sampling
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_MODELS: list[str] = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",   # Primary (local vLLM, FP8 quantized)
]
JUDGE_TEMPERATURE: float = 0.0            # Deterministic scoring
JUDGE_TIMEOUT_S: int = 30
# Round-based sampling: each round picks a different judge from the pool.
# Round-robin with shuffle: no consecutive repeats, all judges get equal use.
JUDGE_SAMPLING_STRATEGY: str = "round_robin"  # "round_robin" | "random"
JUDGE_SERVER_SHUTDOWN_TIMEOUT_S: int = 30      # Max wait for graceful judge shutdown
MINER_SERVER_SHUTDOWN_TIMEOUT_S: int = 15       # Max wait for graceful miner shutdown

# Map model name → base_url for local servers
LOCAL_JUDGE_ENDPOINTS: dict[str, str] = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": VLLM_BASE_URL,
}

# ──────────────────────────────────────────────────────────────────────────────
# Weight Distribution — Proportional-to-Score + Stagnation Decay
# ──────────────────────────────────────────────────────────────────────────────
# Each active track independently targets ~1 TAO/day of miner emissions.
# If a track has no miners it is simply skipped — its allocation is not redistributed.
DAILY_ALPHA_EMISSION: int = 7200          # alpha tokens emitted per day by subnet
STAGNATION_THRESHOLD_DAYS: int = 3         # Days before stagnation decay starts
STAGNATION_DECAY_PERIOD_DAYS: int = 7      # Days over which weight decays to 0
STAGNATION_BURN_UID: int = 0               # UID that receives burned weight
RANK_HISTORY_MAX_ENTRIES: int = 1500       # ~30 days at 30-min weight intervals

# ──────────────────────────────────────────────────────────────────────────────
# Scoring  (legacy interview-based — temporarily disabled)
# ──────────────────────────────────────────────────────────────────────────────
INTERVIEW_DISABLED: bool = True       # Temporarily disable Phase 3 interview evaluation
PARTIAL_CREDIT_ENABLED: bool = True
# Weights for blending Phase 2 (initial answer) and Phase 3 (interview verdict)
PHASE2_WEIGHT: float = 0.4          # 40% weight on initial answer
INTERVIEW_WEIGHT: float = 0.6       # 60% weight on interview verdict
LOOP_WARNING_PENALTY: float = 0.5
MAX_RAW_SCORE: float = VALIDITY_SCORE + float(NUM_QUESTIONS)  # validity + 1.0 per question

# EMA score tracker
EMA_ALPHA: float = 0.3               # EMA smoothing factor
EMA_SIGNIFICANT_IMPROVE: float = 0.5 # Threshold for score change to count as improvement
EMA_VARIANCE_ALPHA: float = 0.1      # Variance weighting in effective score calculation
EMA_MIN_EVALUATIONS: int = 10        # Minimum evaluations before a miner's score counts for weights

# Improve penalty applied when a miner's current round score is low
IMPROVE_PENALTY_THRESHOLD: float = 7.0  # Raw score below this triggers an improve penalty
IMPROVE_PENALTY_VALUE: float = 0.5      # Penalty magnitude subtracted when below threshold

# ──────────────────────────────────────────────────────────────────────────────
# Loss-Based Reward System
# ──────────────────────────────────────────────────────────────────────────────
# Temporal evaluation window: validator evaluates miner on a sliding window of
# past challenges (datasets × text indices).  At each step, a Dirichlet-
# distributed random weighting mixes the losses across the window.
#
#   W_t = {t-k, ..., t}                    sliding window of evaluation steps
#   w ~ Dirichlet(beta)                    randomised per-step weights
#   L_tilde_t = sum_{i in W_t} w_i * L_t^{(i)}  aggregated evaluation loss
#   L_tilde_star_t = min(L_tilde_star_{t-1}, L_tilde_t)  best-so-far
#   R_t = clip(F(L_tilde_star_t) - F(L_tilde_star_{t-1}), 0, R_MAX)
#   where F(L) = exp(-gamma * L)           exponential reward shaping
#
CHALLENGE_WINDOW_SIZE: int = _env("CHALLENGE_WINDOW_SIZE", 100, cast=int)
# Number of past challenges to keep in the sliding window per miner.

DIRICHLET_BETA: float = _env("DIRICHLET_BETA", 1.0, cast=float)
# Dirichlet concentration parameter.  beta=1 → uniform random weights;
# higher beta → weights concentrate towards equal.

REWARD_GAMMA: float = _env("REWARD_GAMMA", 1.0, cast=float)
# Exponential reward shaping coefficient: F(L) = exp(-gamma * L).
# Higher gamma → more reward for small loss reductions.

REWARD_MAX: float = _env("REWARD_MAX", 1.0, cast=float)
# Maximum per-step reward (clipping ceiling).

REWARD_DECAY: float = _env("REWARD_DECAY", 0.995, cast=float)
# Per-evaluation multiplicative decay applied to cumulative_reward before
# adding the new step reward.  Prevents genesis miners from permanently
# dominating via banked early rewards.
# At 0.995 and ~100 evals/day: a reward halves in ~140 evaluations (~1.4 days).
# Set to 1.0 to disable decay.

HF_LOSS_MAX_SEQ_LEN: int = _env("HF_LOSS_MAX_SEQ_LEN", 4096, cast=int)
# Maximum sequence length for HuggingFace loss evaluation.

HF_LOSS_BATCH_SIZE: int = _env("HF_LOSS_BATCH_SIZE", 8, cast=int)
# Target batch size for HuggingFace loss evaluation. Automatically backs off on OOM.

HF_EVAL_ENABLE_4BIT: bool = _env(
    "HF_EVAL_ENABLE_4BIT",
    True,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)
# Try 4-bit quantized model loading first to reduce VRAM during HF loss evaluation.

HF_EVAL_PREFER_FLASH_ATTN: bool = _env(
    "HF_EVAL_PREFER_FLASH_ATTN",
    True,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)
# Prefer Flash Attention / SDPA attention implementations when available.

HF_EVAL_TORCH_COMPILE: bool = _env(
    "HF_EVAL_TORCH_COMPILE",
    False,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)
# Apply torch.compile(model, mode='reduce-overhead', dynamic=True) after load.
# Adds ~30-60 s warm-up on first batch; speeds up all subsequent batches ~15-30%.
# Requires PyTorch >= 2.0.  Disabled by default; set HF_EVAL_TORCH_COMPILE=1 to enable.

LOSS_EMA_ALPHA: float = _env("LOSS_EMA_ALPHA", 0.3, cast=float)
# EMA smoothing for model parameters (conceptual; tracked per-miner).

CHALLENGE_NUM_INDICES: int = _env("CHALLENGE_NUM_INDICES", 50, cast=int)
# Number of text indices sampled per challenge.

# URL of the owner gateway that validators contact to fetch challenges and
# submit evaluation results.
#
# Default points to the deployed public proxy (HF Spaces).
# Override with OWNER_API_URL env var for local development:
#   OWNER_API_URL=http://127.0.0.1:8669
OWNER_API_URL: str = _env("OWNER_API_URL", "https://evolai-gate.hf.space")

# ──────────────────────────────────────────────────────────────────────────────
# Model Validation
# ──────────────────────────────────────────────────────────────────────────────
# Valid model parameter ranges (in billions) — min/max pairs
VALID_PARAM_RANGES_B: list[tuple[float, float]] = [
    (0.45, 0.48),   # ~450M
    (1.5,  1.8),    # ~1.6B
    (3.5,  3.8),    # ~3.7B
    (9.0,  9.5),    # ~9B
    (21.0, 21.5),   # ~21B
]
VALIDATION_GPU_REQUIRED_GB: float = 8.0    # Min free VRAM before validate_model (GPU path)
EVALUATION_GPU_REQUIRED_GB: float = 10.0   # Min free VRAM before HF transformers eval load

# ──────────────────────────────────────────────────────────────────────────────
# Anti-Gaming — Model Copy / Plagiarism Detection
# ──────────────────────────────────────────────────────────────────────────────
# Miners who re-upload an existing model (even under a different repo name) or
# apply a trivial fine-tune to harvest emissions are penalised with score = 0.
#
# Detection uses three complementary fingerprints computed in model_fingerprint.py:
#   • exact_hash     — SHA-256 of deterministically sampled weight bytes
#   • arch_hash      — SHA-256 of numeric architecture config values
#   • fuzzy_vector   — quantised L2-norm vector for near-copy detection
FINGERPRINT_SAMPLE_N: int = _env("FINGERPRINT_SAMPLE_N", 30, cast=int)
# Number of parameter tensors sampled for the exact SHA-256 hash
FINGERPRINT_MAX_BYTES_PER_TENSOR: int = _env("FINGERPRINT_MAX_BYTES_PER_TENSOR", 4096, cast=int)
# Bytes read per sampled tensor — larger = more precise, slower
FINGERPRINT_BUCKET_COUNT: int = _env("FINGERPRINT_BUCKET_COUNT", 64, cast=int)
# Size of fuzzy norm vector (also = number of tensors sampled for fuzzy check)
FINGERPRINT_FUZZY_THRESHOLD: float = _env("FINGERPRINT_FUZZY_THRESHOLD", 0.995, cast=float)
# Cosine similarity above this is classified as a near-copy
FINGERPRINT_SEED: int = 42
# RNG seed for deterministic layer sampling — hardcoded, not env-overridable.
# All validators must use the same seed; exposing it as a config variable would
# allow individual operators to diverge and break anti-gaming detection.
FINGERPRINT_NORM_SLICE_ELEMENTS: int = _env("FINGERPRINT_NORM_SLICE_ELEMENTS", 32_768, cast=int)
# Max elements read per tensor for L2 norm computation.
# 32 768 elements = 128 KB (float32) — caps cost regardless of whether the
# sampled tensor is a 1 KB bias or a 1 GB 30B attention projection matrix.

# ──────────────────────────────────────────────────────────────────────────────
# Rate Limiting (from OpenClaw fixed-window-rate-limit.ts)
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_RATE_LIMIT_MAX_REQUESTS: int = 30    # Max judge calls per window
JUDGE_RATE_LIMIT_WINDOW_MS: int = 60_000   # 60-second fixed window
MINER_RATE_LIMIT_MAX_REQUESTS: int = 60    # Max miner calls per window
MINER_RATE_LIMIT_WINDOW_MS: int = 60_000   # 60-second fixed window

# ──────────────────────────────────────────────────────────────────────────────
# Proportional Response Truncation (from OpenClaw tool-result-truncation.ts)
# ──────────────────────────────────────────────────────────────────────────────
MAX_RESPONSE_CONTEXT_SHARE: float = 0.3       # No single answer > 30% of context
HARD_MAX_RESPONSE_CHARS: int = 400_000        # Absolute ceiling (~133K tokens)
MIN_KEEP_CHARS: int = 2_000                   # Never truncate below this

# ──────────────────────────────────────────────────────────────────────────────
# Server Health / Startup
# ──────────────────────────────────────────────────────────────────────────────
SERVER_HEALTH_RETRIES: int = 20
SERVER_HEALTH_INTERVAL_S: float = 3.0
SERVER_START_TIMEOUT_S: int = 1800         # 30 min — allows first-time model download
VLLM_HTTP_CLIENT_TIMEOUT_S: float = 300.0  # httpx client default timeout for inference calls
VLLM_HEALTH_CHECK_TIMEOUT_S: float = 5.0   # Timeout for /health probe
VLLM_POLL_INTERVAL_S: int = 5              # Seconds between readiness polls during startup
VLLM_STOP_WAIT_S: int = 10                 # Seconds after kill to wait for GPU VRAM reclaim

# ──────────────────────────────────────────────────────────────────────────────
# GPU Management (from OpenClaw auth-profiles + channel-health-monitor patterns)
# ──────────────────────────────────────────────────────────────────────────────
GPU_TOTAL_VRAM_GB: float = 0.0              # Auto-detected; 0 = probe at startup
GPU_RESERVE_VRAM_GB: float = 2.0            # Reserved for CUDA kernels + OS
JUDGE_VRAM_FRACTION: float = 0.90           # Judge server VRAM share (dedicated GPU 0)
MINER_VRAM_FRACTION: float = 0.85           # Miner server VRAM share (dedicated GPU 1)

GPU_HEALTH_POLL_INTERVAL_S: float = 30.0    # Health check frequency
GPU_STARTUP_GRACE_S: float = 60.0           # Skip health checks during model load
GPU_MAX_RESTARTS_PER_HOUR: int = 3          # Rate-limit auto-restarts per role
GPU_RESTART_COOLDOWN_S: float = 120.0       # Min gap between restarts per role

GPU_OOM_COOLDOWN_BASE_S: float = 60.0       # OOM backoff base (60s)
GPU_OOM_COOLDOWN_MULTIPLIER: float = 5.0    # Exponential: base × 5^(n-1)
GPU_OOM_COOLDOWN_CAP_S: float = 3600.0      # 1 hour cap
GPU_FAILURE_WINDOW_DECAY_S: float = 86400.0 # Error counts reset after 24h

GPU_VRAM_WARNING_THRESHOLD: float = 0.90    # Warn if VRAM > 90% utilised
GPU_VRAM_CRITICAL_THRESHOLD: float = 0.95   # Critical if > 95%

