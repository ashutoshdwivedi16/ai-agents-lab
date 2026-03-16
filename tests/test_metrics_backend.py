"""Tests for metrics backends (SQLite and Noop)."""

import pytest
from datetime import datetime, timezone

from shared.metrics.models import MetricRecord, MetricsSummary
from shared.metrics.backends.sqlite_backend import SQLiteBackend
from shared.metrics.backends.noop_backend import NoopBackend


def _make_record(**overrides) -> MetricRecord:
    defaults = dict(
        session_id="test-session",
        agent_name="test-agent",
        provider="groq",
        model="llama-3",
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
        latency_ms=200.0,
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MetricRecord(**defaults)


class TestSQLiteBackend:
    @pytest.fixture
    def backend(self):
        b = SQLiteBackend(":memory:")
        yield b
        b.close()

    def test_record_and_summary(self, backend):
        backend.record(_make_record())
        backend.record(_make_record(cost=0.002))

        summary = backend.summary()
        assert summary.total_calls == 2
        assert abs(summary.total_cost - 0.003) < 1e-9

    def test_record_and_query(self, backend):
        backend.record(_make_record(provider="groq"))
        backend.record(_make_record(provider="openai"))

        records = backend.records()
        assert len(records) == 2
        providers = {r.provider for r in records}
        assert providers == {"groq", "openai"}

    def test_filter_passthrough(self, backend):
        backend.record(_make_record(session_id="s1"))
        backend.record(_make_record(session_id="s2"))

        summary = backend.summary(session_id="s1")
        assert summary.total_calls == 1

    def test_close_is_safe(self, backend):
        """close() should not raise, even on empty backend."""
        backend.close()


class TestNoopBackend:
    def test_record_does_nothing(self):
        noop = NoopBackend()
        noop.record(_make_record())  # Should not raise

    def test_summary_returns_empty(self):
        noop = NoopBackend()
        summary = noop.summary()
        assert isinstance(summary, MetricsSummary)
        assert summary.total_calls == 0

    def test_records_returns_empty(self):
        noop = NoopBackend()
        records = noop.records()
        assert records == []

    def test_close_does_nothing(self):
        noop = NoopBackend()
        noop.close()  # Should not raise
