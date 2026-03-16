"""Tests for retry logic."""

import pytest

from shared.retry import llm_retry


class TestLlmRetry:
    def test_succeeds_on_first_try(self):
        call_count = 0

        @llm_retry(max_attempts=3)
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    def test_retries_on_connection_error(self):
        call_count = 0

        @llm_retry(max_attempts=3, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient failure")
            return "recovered"

        assert fn() == "recovered"
        assert call_count == 3

    def test_gives_up_after_max_attempts(self):
        @llm_retry(max_attempts=2, base_delay=0.01)
        def fn():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError, match="always fails"):
            fn()

    def test_does_not_retry_value_error(self):
        """Non-retryable exceptions should not be retried."""
        call_count = 0

        @llm_retry(max_attempts=3, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            fn()
        assert call_count == 1  # no retry
