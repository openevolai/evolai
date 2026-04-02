"""
Error Handling and Recovery for Validator

Industry-standard error classification, retry logic, and circuit breakers.
Based on Google SRE and AWS best practices.
"""

import time
import logging
from typing import Optional, Callable, TypeVar, Any
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    TRANSIENT = "transient"  # Retry automatically
    RECOVERABLE = "recoverable"  # Retry with backoff
    PERMANENT = "permanent"  # Don't retry
    CRITICAL = "critical"  # Emergency shutdown


class ErrorCategory(Enum):
    """Error categories for better handling"""
    NETWORK = "network"  # Network/API failures
    GPU_OOM = "gpu_oom"  # GPU out of memory
    DISK_FULL = "disk_full"  # Disk space exhausted
    MODEL_LOAD = "model_load"  # Model loading failures
    INFERENCE = "inference"  # Inference failures
    VALIDATION = "validation"  # Validation failures
    UNKNOWN = "unknown"  # Uncategorized


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


class ValidatorError(Exception):
    """Base exception for validator errors"""
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.RECOVERABLE,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.original_error = original_error
        self.timestamp = time.time()


class GPUOutOfMemoryError(ValidatorError):
    """GPU OOM error"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.GPU_OOM,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class ModelLoadError(ValidatorError):
    """Model loading error"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_LOAD,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class DiskSpaceError(ValidatorError):
    """Disk space error"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.DISK_FULL,
            severity=ErrorSeverity.CRITICAL,
            original_error=original_error
        )


class NetworkError(ValidatorError):
    """Network/API error"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )


class ContextOverflowError(ValidatorError):
    """Context window overflow — judge model cannot fit the conversation."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class JudgeCallError(ValidatorError):
    """Judge model returned an unparseable or invalid response."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )


class RateLimitError(ValidatorError):
    """Rate-limited by LLM provider — honour retry-after header."""
    def __init__(self, message: str, retry_after_ms: Optional[int] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )
        self.retry_after_ms = retry_after_ms


# ──────────────────────────────────────────────────────────────────────────────
# Structured Failover Error Taxonomy (from OpenClaw failover-error.ts)
# ──────────────────────────────────────────────────────────────────────────────

class FailoverReason(Enum):
    """
    Structured failover reason codes.
    Derived from OpenClaw src/agents/failover-error.ts.
    Maps reasons to HTTP status codes for deterministic retry/rotation decisions.
    """
    BILLING = "billing"                   # 402 — quota exceeded
    RATE_LIMIT = "rate_limit"             # 429 — too many requests
    AUTH = "auth"                         # 401/403 — credentials invalid
    TIMEOUT = "timeout"                   # 408/504 — server too slow
    FORMAT = "format"                     # 400 — bad request shape
    MODEL_NOT_FOUND = "model_not_found"   # 404 — model not deployed
    CONTEXT_OVERFLOW = "context_overflow" # context window exceeded
    OOM = "oom"                           # GPU out of memory
    UNKNOWN = "unknown"


FAILOVER_HTTP_MAP: dict = {
    FailoverReason.BILLING: 402,
    FailoverReason.RATE_LIMIT: 429,
    FailoverReason.AUTH: 401,
    FailoverReason.TIMEOUT: 408,
    FailoverReason.FORMAT: 400,
    FailoverReason.MODEL_NOT_FOUND: 404,
}

# Retry policy per reason (from OpenClaw failover-error.ts behavior)
FAILOVER_SHOULD_RETRY: dict = {
    FailoverReason.BILLING: False,        # No retry — quota problem
    FailoverReason.RATE_LIMIT: True,      # Retry after delay
    FailoverReason.AUTH: False,            # Immediate rotation, no retry
    FailoverReason.TIMEOUT: True,         # Retry with backoff
    FailoverReason.FORMAT: False,          # Bad request — won't fix itself
    FailoverReason.MODEL_NOT_FOUND: False, # Immediate rotation
    FailoverReason.CONTEXT_OVERFLOW: False, # Needs compaction, not retry
    FailoverReason.OOM: False,            # Needs cooldown, not retry
    FailoverReason.UNKNOWN: True,         # Retry — might be transient
}


class FailoverError(ValidatorError):
    """
    Structured failover error with typed reason code.
    Derived from OpenClaw src/agents/failover-error.ts:FailoverError.

    Enables smarter retry/fallback decisions:
      - OOM → 5 min cooldown, rotate to fallback
      - rate_limit → retry with backoff, respect retry-after
      - auth → immediate rotation, no retry
      - billing → fail fast, no retry
      - timeout → retry with extended timeout
    """
    def __init__(
        self,
        message: str,
        reason: FailoverReason,
        original_error: Optional[Exception] = None,
    ):
        severity = (
            ErrorSeverity.TRANSIENT
            if FAILOVER_SHOULD_RETRY.get(reason, True)
            else ErrorSeverity.RECOVERABLE
        )
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=severity,
            original_error=original_error,
        )
        self.reason = reason

    @property
    def http_status(self) -> int:
        """Map reason to HTTP status code."""
        return FAILOVER_HTTP_MAP.get(self.reason, 500)

    @property
    def should_retry(self) -> bool:
        """Whether this error type warrants retry (vs immediate rotation)."""
        return FAILOVER_SHOULD_RETRY.get(self.reason, True)

    def __repr__(self) -> str:
        return f"FailoverError(reason={self.reason.value}, status={self.http_status})"


def classify_judge_error(error: Exception) -> FailoverReason:
    """
    Map exception → FailoverReason using HTTP status codes first,
    then string-match fallback.
    Derived from OpenClaw failover-error.ts reason classification.
    """
    # Try HTTP status code first (most reliable)
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", 0) or getattr(response, "status", 0)

    if status:
        for reason, code in FAILOVER_HTTP_MAP.items():
            if status == code:
                return reason

    # Fallback: string matching
    msg = str(error).lower()
    if "context" in msg and ("length" in msg or "overflow" in msg):
        return FailoverReason.CONTEXT_OVERFLOW
    if "out of memory" in msg or "oom" in msg or "cuda" in msg:
        return FailoverReason.OOM
    if "rate" in msg and "limit" in msg:
        return FailoverReason.RATE_LIMIT
    if "unauthorized" in msg or "forbidden" in msg or "auth" in msg:
        return FailoverReason.AUTH
    if "timeout" in msg or "timed out" in msg:
        return FailoverReason.TIMEOUT
    if "not found" in msg and "model" in msg:
        return FailoverReason.MODEL_NOT_FOUND
    if "billing" in msg or "quota" in msg or "insufficient" in msg:
        return FailoverReason.BILLING

    return FailoverReason.UNKNOWN


def classify_error(error: Exception) -> ValidatorError:
    """
    Classify exceptions into structured ValidatorError
    
    Args:
        error: Original exception
        
    Returns:
        ValidatorError with proper classification
    """
    import torch
    
    # Already classified
    if isinstance(error, ValidatorError):
        return error
    
    # CUDA OOM
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return GPUOutOfMemoryError(
            f"GPU out of memory: {str(error)}",
            original_error=error
        )
    
    # Context overflow heuristic (from OpenClaw isLikelyContextOverflowError)
    if is_likely_context_overflow(error):
        return ContextOverflowError(
            f"Context window overflow: {str(error)}",
            original_error=error
        )
    
    # Disk errors
    if isinstance(error, (OSError, IOError)):
        error_msg = str(error).lower()
        if "no space left" in error_msg or "disk full" in error_msg:
            return DiskSpaceError(
                f"Disk space exhausted: {str(error)}",
                original_error=error
            )
    
    # Network errors
    if "connection" in str(error).lower() or "timeout" in str(error).lower():
        return NetworkError(
            f"Network error: {str(error)}",
            original_error=error
        )
    
    # Rate limit errors
    if "rate" in str(error).lower() and "limit" in str(error).lower():
        return RateLimitError(
            f"Rate limited: {str(error)}",
            original_error=error
        )
    
    # Model loading errors
    if "pretrained" in str(error).lower() or "checkpoint" in str(error).lower():
        return ModelLoadError(
            f"Model load failed: {str(error)}",
            original_error=error
        )
    
    # Default classification
    return ValidatorError(
        f"Unclassified error: {str(error)}",
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.RECOVERABLE,
        original_error=error
    )


def is_likely_context_overflow(error: Exception) -> bool:
    """
    Heuristic check for context window overflow errors.
    Derived from OpenClaw context-window-guard.ts isLikelyContextOverflowError().
    
    Matches patterns from vLLM, OpenAI, Anthropic, and Ollama error messages.
    """
    msg = str(error).lower()
    overflow_patterns = [
        "context length",
        "context_length_exceeded",
        "maximum context",
        "token limit",
        "too many tokens",
        "prompt is too long",
        "input too long",
        "max_tokens",
        "model_max_len",
        "exceeds the model",
    ]
    return any(p in msg for p in overflow_patterns)


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_categories: Optional[set] = None
):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        config: Retry configuration
        retryable_categories: Error categories to retry (default: TRANSIENT, RECOVERABLE)
        
    Example:
        @with_retry(RetryConfig(max_attempts=5))
        def load_model(model_name):
            return AutoModel.from_pretrained(model_name)
    """
    config = config or RetryConfig()
    retryable = retryable_categories or {
        ErrorCategory.NETWORK,
        ErrorCategory.GPU_OOM,
        ErrorCategory.MODEL_LOAD
    }
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    # Classify error
                    classified_error = classify_error(e)
                    last_error = classified_error
                    
                    # Check if retryable
                    if classified_error.category not in retryable:
                        logger.error(
                            f"{func.__name__} failed with non-retryable error: "
                            f"{classified_error.category.value}"
                        )
                        raise classified_error
                    
                    # Check if we have attempts left
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts"
                        )
                        raise classified_error
                    
                    # Calculate backoff delay
                    delay = min(
                        config.initial_delay_seconds * (config.exponential_base ** (attempt - 1)),
                        config.max_delay_seconds
                    )
                    
                    # Add jitter
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())  # 50-150% of calculated delay
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                        f"retrying in {delay:.2f}s. Error: {str(classified_error)}"
                    )
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_error or RuntimeError(f"{func.__name__} failed")
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for failing operations
    
    Prevents cascading failures by stopping calls to failing services.
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing) -> CLOSED
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in HALF_OPEN before closing
            timeout_seconds: Time to wait before trying HALF_OPEN
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.name = name
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to function
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If circuit is OPEN
        """
        # Check if circuit is OPEN
        if self.state == "OPEN":
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time < self.timeout_seconds:
                raise RuntimeError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Failures: {self.failure_count}/{self.failure_threshold}"
                )
            else:
                # Try transitioning to HALF_OPEN
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = "HALF_OPEN"
                self.success_count = 0
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker {self.name} CLOSING after {self.success_count} successes")
                self.state = "CLOSED"
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker {self.name} OPENING after {self.failure_count} failures"
            )
            self.state = "OPEN"
        
        # If in HALF_OPEN and failed, go back to OPEN
        if self.state == "HALF_OPEN":
            logger.warning(f"Circuit breaker {self.name} returning to OPEN from HALF_OPEN")
            self.state = "OPEN"
            self.success_count = 0
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info(f"Manually resetting circuit breaker {self.name}")
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
