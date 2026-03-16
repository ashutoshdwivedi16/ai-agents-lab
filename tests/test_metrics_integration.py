"""Integration tests — FakeProvider + real metrics pipeline."""

import pytest
from unittest.mock import MagicMock

from shared.llm_pkg.base import LLMProvider, register_provider, chat
from shared.models import ChatResponse, Usage
from shared.metrics import (
    set_backend,
    get_backend,
    record_llm_call,
    init_metrics,
    shutdown,
    get_session_id,
    set_session_id,
)
from shared.metrics.models import MetricRecord
from shared.metrics.backends.noop_backend import NoopBackend
from shared.metrics.backends.sqlite_backend import SQLiteBackend


class MetricsCapture(SQLiteBackend):
    """SQLite in-memory backend that captures records for assertions."""

    def __init__(self):
        super().__init__(":memory:")


class FakeProvider(LLMProvider):
    """Test provider that returns canned responses without hitting any API."""

    name = "fake-metrics"
    default_model = "fake-v1"
    env_key = "FAKE_API_KEY"

    def _create_client(self):
        return MagicMock()

    def _do_call(self, client, messages, model):
        return ChatResponse(
            content="fake response",
            usage=Usage(input_tokens=42, output_tokens=18, cost=0.001),
            model=model,
            provider=self.name,
        )


class TestRecordLlmCall:
    def test_record_persists(self):
        """record_llm_call() should persist to the active backend."""
        backend = MetricsCapture()
        set_backend(backend)
        set_session_id("test-int")

        record_llm_call(
            provider="groq",
            model="llama-3",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            latency_ms=200.0,
        )

        summary = backend.summary()
        assert summary.total_calls == 1
        assert summary.total_input_tokens == 100

    def test_record_uses_session_id(self):
        backend = MetricsCapture()
        set_backend(backend)
        set_session_id("my-session")

        record_llm_call(
            provider="groq",
            model="m1",
            input_tokens=10,
            output_tokens=5,
            cost=0.0,
            latency_ms=50.0,
        )

        records = backend.records()
        assert len(records) == 1
        assert records[0].session_id == "my-session"

    def test_record_custom_session_id(self):
        backend = MetricsCapture()
        set_backend(backend)

        record_llm_call(
            provider="groq",
            model="m1",
            input_tokens=10,
            output_tokens=5,
            cost=0.0,
            latency_ms=50.0,
            session_id="custom-id",
        )

        records = backend.records()
        assert records[0].session_id == "custom-id"

    def test_record_with_agent_name(self):
        backend = MetricsCapture()
        set_backend(backend)
        set_session_id("s1")

        record_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            cost=0.01,
            latency_ms=800.0,
            agent_name="rag-agent",
        )

        records = backend.records()
        assert records[0].agent_name == "rag-agent"


class TestChatIntegration:
    def test_chat_records_metrics(self):
        """chat() should trigger metrics recording through call_with_response()."""
        backend = MetricsCapture()
        set_backend(backend)
        set_session_id("chat-test")

        provider = FakeProvider()
        register_provider(provider)

        result = chat(
            [{"role": "user", "content": "hello"}],
            provider="fake-metrics",
        )

        assert result == "fake response"

        # Verify metrics were recorded
        summary = backend.summary()
        assert summary.total_calls == 1
        assert summary.total_input_tokens == 42
        assert summary.total_output_tokens == 18
        assert abs(summary.total_cost - 0.001) < 1e-9
        assert summary.by_provider == {"fake-metrics": 1}

    def test_chat_records_latency(self):
        """Latency should be > 0 (timing actually works)."""
        backend = MetricsCapture()
        set_backend(backend)
        set_session_id("latency-test")

        provider = FakeProvider()
        register_provider(provider)

        chat(
            [{"role": "user", "content": "hi"}],
            provider="fake-metrics",
        )

        records = backend.records()
        assert len(records) == 1
        assert records[0].latency_ms > 0  # Timing is working


class TestMetricsFireAndForget:
    def test_broken_backend_does_not_crash_chat(self):
        """If metrics backend throws, chat() should still work."""

        class BrokenBackend(NoopBackend):
            def record(self, metric):
                raise RuntimeError("Metrics DB is down!")

        set_backend(BrokenBackend())
        set_session_id("broken-test")

        provider = FakeProvider()
        register_provider(provider)

        # This should NOT raise — metrics errors are swallowed
        result = chat(
            [{"role": "user", "content": "test"}],
            provider="fake-metrics",
        )
        assert result == "fake response"


class TestInitMetrics:
    def test_init_with_noop_config(self):
        from shared.models import MetricsConfig

        config = MetricsConfig(enabled=False)
        init_metrics(config)
        assert isinstance(get_backend(), NoopBackend)

    def test_init_with_sqlite_config(self):
        from shared.models import MetricsConfig

        config = MetricsConfig(enabled=True, backend="sqlite", sqlite_path=":memory:")
        init_metrics(config)
        assert isinstance(get_backend(), SQLiteBackend)

    def test_init_unknown_backend_falls_back_to_noop(self):
        from shared.models import MetricsConfig

        config = MetricsConfig(enabled=True, backend="nonexistent")
        init_metrics(config)
        assert isinstance(get_backend(), NoopBackend)

    def test_set_and_get_backend(self):
        noop = NoopBackend()
        set_backend(noop)
        assert get_backend() is noop
