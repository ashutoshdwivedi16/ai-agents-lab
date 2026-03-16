"""Retry logic with exponential backoff using tenacity.

Teaches: Decorator-based retry patterns for unreliable network calls.

Usage:
    @llm_retry(max_attempts=3)
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

# Include SDK-specific errors if available
try:
    import httpx

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, httpx.HTTPStatusError)
except ImportError:
    pass


def llm_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for retrying LLM API calls with exponential backoff.

    Args:
        max_attempts: Maximum number of tries before giving up.
        base_delay: Initial delay in seconds (doubles each retry).
    """
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
