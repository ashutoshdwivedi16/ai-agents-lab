"""Retry logic with exponential backoff using tenacity.

Teaches: Decorator-based retry patterns for unreliable network calls.

Retry parameters (max_attempts, base_delay, max_delay) are driven by
config/default.yaml. Providers use get_retry_config() at import time.

Usage:
    @llm_retry()
    def call_api():
        ...
"""

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Standard retryable exceptions across HTTP-based LLM SDKs
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

# Include SDK-specific errors if available (transitive via openai/anthropic)
try:
    import httpx

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, httpx.HTTPStatusError)
except ImportError:
    pass

# Config-driven retry defaults (loaded once at module level)
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 30.0


def _load_retry_config() -> tuple[int, float, float]:
    """Load retry settings from YAML config. Falls back to defaults."""
    try:
        from shared.config import load_app_config
        config = load_app_config()
        return (
            config.max_retries,
            config.retry_base_delay,
            _DEFAULT_MAX_DELAY,
        )
    except Exception:
        return _DEFAULT_MAX_ATTEMPTS, _DEFAULT_BASE_DELAY, _DEFAULT_MAX_DELAY


def llm_retry(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
):
    """Decorator for retrying LLM API calls with exponential backoff.

    Args:
        max_attempts: Maximum number of tries. Default: from config.
        base_delay: Initial delay in seconds. Default: from config.
        max_delay: Maximum delay cap in seconds. Default: 30.
    """
    cfg_attempts, cfg_delay, cfg_max = _load_retry_config()
    attempts = max_attempts if max_attempts is not None else cfg_attempts
    delay = base_delay if base_delay is not None else cfg_delay
    cap = max_delay if max_delay is not None else cfg_max

    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=delay, min=delay, max=cap),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
