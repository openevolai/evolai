"""
Validator module for EvolAI subnet

This module contains the modular evaluation pipeline:
- Orchestrator:    async three-phase evaluation pipeline (orchestrator.py)
- Judge Client:    HTTP call layer with retry + cooldown (judge_client.py)
- Streaming:       Ollama / vLLM miner response drivers (streaming.py)
- Scoring:         final score calculation + normalisation (scoring.py)
- Compaction:      staged context compaction (compaction.py)
- Loop Detector:   SHA-256 answer gaming detection (loop_detector.py)
- Sanitizer:       prompt injection + Unicode defense (sanitizer.py)
- Lifecycle:       W&B audit trail events (lifecycle.py)
- Usage:           token usage accumulator + normalization (usage.py)
- Rate Limiter:    fixed-window rate limiting (rate_limiter.py)
- Payload Trace:   SHA-256 diagnostic tracing (payload_trace.py)
- Cost Tracker:    per-model cost & latency tracking (cost_tracker.py)
- Error Handling:  FailoverError taxonomy + error classification (error_handling.py)
- GPU Manager:     VRAM budget, slot cooldown, multi-GPU scheduling (gpu_manager.py)
- GPU Health:      async health monitor with auto-restart (gpu_health_monitor.py)
- Round Manager:   round-based judge sampling & model lifecycle (round_manager.py)
- Model Fingerprint: weight-hash copy-gaming detection (model_fingerprint.py)
- Config:          all tunable constants (config.py)
- Prompts:         judge prompt templates (prompts.py)
- Legacy:          ModelValidator, DatasetSampler, EMAScoreTracker (evaluator.py)
"""

# ── New modular pipeline ────────────────────────────────────────────────
from evolai.validator.orchestrator import EvaluationOrchestrator, EvaluationOutput
from evolai.validator.judge_client import call_judge_with_fallback, parse_judge_json
from evolai.validator.vllm_client import VLLMClient, VLLMServerManager
from evolai.validator.scoring import calculate_final_score, normalize_score, score_zero_copy_gaming
from evolai.validator.loop_detector import AnswerLoopDetector
from evolai.validator.usage import UsageAccumulator, normalize_usage
from evolai.validator.rate_limiter import FixedWindowRateLimiter, get_judge_rate_limiter
from evolai.validator.payload_trace import PayloadTracer, get_tracer
from evolai.validator.cost_tracker import CostTracker
from evolai.validator.error_handling import (
    FailoverError, FailoverReason, classify_judge_error,
)
from evolai.validator.sanitizer import (
    sanitize_miner_answer_for_judge,
    sanitize_unicode_for_prompt,
    calculate_max_response_chars,
)
from evolai.validator.gpu_manager import (
    GPUManager, get_gpu_manager, GPUSlot, VRAMBudget, GPUInfo,
    detect_gpus, get_live_vram_usage, select_gpu_for_role,
)
from evolai.validator.gpu_health_monitor import (
    GPUHealthMonitor, get_health_monitor,
    ServerHealthStatus, GPUHealthStatus, RestartTracker,
)
from evolai.validator.round_manager import (
    EvaluationRoundManager, JudgeSampler, MinerEntry, RoundResult,
)
from evolai.validator.model_fingerprint import (
    ModelFingerprint,
    compute_model_fingerprint,
    fingerprints_collide,
)

# ── Retained from original evaluator ────────────────────────────────────
from evolai.validator.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    ModelValidator,
    DatasetSampler,
    InterviewEvaluator,
    ScoreCalculator,
    EMAScoreTracker,
    ModelRegistry,
)

__all__ = [
    # New pipeline
    "EvaluationOrchestrator",
    "EvaluationOutput",
    "call_judge_with_fallback",
    "parse_judge_json",
    "VLLMClient",
    "VLLMServerManager",
    "calculate_final_score",
    "normalize_score",
    "score_zero_copy_gaming",
    "AnswerLoopDetector",
    "UsageAccumulator",
    "normalize_usage",
    # New Phase 2 (OpenClaw-derived improvements)
    "FixedWindowRateLimiter",
    "get_judge_rate_limiter",
    "PayloadTracer",
    "get_tracer",
    "CostTracker",
    "FailoverError",
    "FailoverReason",
    "classify_judge_error",
    "sanitize_miner_answer_for_judge",
    "sanitize_unicode_for_prompt",
    "calculate_max_response_chars",
    # GPU Management (§15)
    "GPUManager",
    "get_gpu_manager",
    "GPUSlot",
    "VRAMBudget",
    "GPUInfo",
    "detect_gpus",
    "get_live_vram_usage",
    "select_gpu_for_role",
    "GPUHealthMonitor",
    "get_health_monitor",
    "ServerHealthStatus",
    "GPUHealthStatus",
    "RestartTracker",
    # Round-based lifecycle (§15.9)
    "EvaluationRoundManager",
    "JudgeSampler",
    "MinerEntry",
    "RoundResult",
    # Anti-gaming / copy detection
    "ModelFingerprint",
    "compute_model_fingerprint",
    "fingerprints_collide",
    # Legacy / retained
    "EvaluationConfig",
    "EvaluationResult",
    "ModelValidator",
    "DatasetSampler",
    "InterviewEvaluator",
    "ScoreCalculator",
    "EMAScoreTracker",
    "ModelRegistry",
]
