"""Tests for metrics Pydantic models."""

from datetime import datetime, timezone

from shared.metrics.models import MetricRecord, MetricsSummary


class TestMetricRecord:
    def test_defaults(self):
        rec = MetricRecord(session_id="s1", provider="groq", model="llama-3")
        assert rec.agent_name == "unknown"
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.cost == 0.0
        assert rec.latency_ms == 0.0
        assert isinstance(rec.timestamp, datetime)
        assert rec.timestamp.tzinfo is not None  # UTC-aware

    def test_full_record(self):
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        rec = MetricRecord(
            session_id="abc123",
            agent_name="chatbot",
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.003,
            latency_ms=450.5,
            timestamp=ts,
        )
        assert rec.session_id == "abc123"
        assert rec.agent_name == "chatbot"
        assert rec.provider == "openai"
        assert rec.model == "gpt-4o"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.cost == 0.003
        assert rec.latency_ms == 450.5
        assert rec.timestamp == ts

    def test_auto_timestamp_is_utc(self):
        rec = MetricRecord(session_id="s1", provider="groq", model="m1")
        assert rec.timestamp.tzinfo == timezone.utc


class TestMetricsSummary:
    def test_empty_defaults(self):
        summary = MetricsSummary()
        assert summary.total_calls == 0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_cost == 0.0
        assert summary.avg_latency_ms == 0.0
        assert summary.by_provider == {}
        assert summary.by_model == {}
        assert summary.by_agent == {}

    def test_populated_summary(self):
        summary = MetricsSummary(
            total_calls=10,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_cost=0.15,
            avg_latency_ms=200.0,
            by_provider={"groq": 7, "openai": 3},
            by_model={"llama-3": 7, "gpt-4o": 3},
            by_agent={"chatbot": 10},
        )
        assert summary.total_calls == 10
        assert summary.by_provider["groq"] == 7
        assert summary.by_model["gpt-4o"] == 3
        assert summary.by_agent["chatbot"] == 10
