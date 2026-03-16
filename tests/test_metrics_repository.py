"""Tests for metrics repositories (SQLite :memory: and InMemory)."""

import pytest
from datetime import datetime, timezone, timedelta

from shared.metrics.models import MetricRecord
from shared.metrics.repositories.sqlite_repository import SQLiteRepository
from shared.metrics.repositories.inmemory_repository import InMemoryRepository


def _make_record(**overrides) -> MetricRecord:
    """Factory helper for creating test MetricRecords."""
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


# Parametrize tests to run against both repository implementations
@pytest.fixture(params=["sqlite", "inmemory"])
def repo(request):
    """Provide both repository implementations for each test."""
    if request.param == "sqlite":
        r = SQLiteRepository(":memory:")
    else:
        r = InMemoryRepository()
    yield r
    r.close()


class TestSaveAndQuery:
    def test_save_and_retrieve(self, repo):
        record = _make_record()
        repo.save(record)
        records = repo.query_records()
        assert len(records) == 1
        assert records[0].session_id == "test-session"
        assert records[0].provider == "groq"

    def test_multiple_saves(self, repo):
        for i in range(5):
            repo.save(_make_record(input_tokens=i * 10))
        records = repo.query_records()
        assert len(records) == 5

    def test_empty_query(self, repo):
        records = repo.query_records()
        assert records == []


class TestQuerySummary:
    def test_summary_aggregation(self, repo):
        repo.save(_make_record(input_tokens=100, output_tokens=50, cost=0.001, latency_ms=200))
        repo.save(_make_record(input_tokens=200, output_tokens=100, cost=0.002, latency_ms=400))

        summary = repo.query_summary()
        assert summary.total_calls == 2
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150
        assert abs(summary.total_cost - 0.003) < 1e-9
        assert abs(summary.avg_latency_ms - 300.0) < 1e-3

    def test_summary_by_provider(self, repo):
        repo.save(_make_record(provider="groq"))
        repo.save(_make_record(provider="groq"))
        repo.save(_make_record(provider="openai"))

        summary = repo.query_summary()
        assert summary.by_provider == {"groq": 2, "openai": 1}

    def test_summary_by_model(self, repo):
        repo.save(_make_record(model="llama-3"))
        repo.save(_make_record(model="gpt-4o"))

        summary = repo.query_summary()
        assert summary.by_model == {"llama-3": 1, "gpt-4o": 1}

    def test_summary_by_agent(self, repo):
        repo.save(_make_record(agent_name="chatbot"))
        repo.save(_make_record(agent_name="rag-agent"))
        repo.save(_make_record(agent_name="chatbot"))

        summary = repo.query_summary()
        assert summary.by_agent == {"chatbot": 2, "rag-agent": 1}

    def test_empty_summary(self, repo):
        summary = repo.query_summary()
        assert summary.total_calls == 0
        assert summary.total_cost == 0.0


class TestFilters:
    def test_filter_by_session(self, repo):
        repo.save(_make_record(session_id="s1"))
        repo.save(_make_record(session_id="s2"))
        repo.save(_make_record(session_id="s1"))

        records = repo.query_records(session_id="s1")
        assert len(records) == 2
        assert all(r.session_id == "s1" for r in records)

    def test_filter_by_agent(self, repo):
        repo.save(_make_record(agent_name="bot-a"))
        repo.save(_make_record(agent_name="bot-b"))

        records = repo.query_records(agent_name="bot-a")
        assert len(records) == 1
        assert records[0].agent_name == "bot-a"

    def test_filter_by_since(self, repo):
        old = datetime(2025, 1, 1, tzinfo=timezone.utc)
        new = datetime(2025, 6, 1, tzinfo=timezone.utc)
        repo.save(_make_record(timestamp=old))
        repo.save(_make_record(timestamp=new))

        cutoff = datetime(2025, 3, 1, tzinfo=timezone.utc)
        records = repo.query_records(since=cutoff)
        assert len(records) == 1

    def test_summary_filter_by_session(self, repo):
        repo.save(_make_record(session_id="s1", cost=0.01))
        repo.save(_make_record(session_id="s2", cost=0.05))

        summary = repo.query_summary(session_id="s1")
        assert summary.total_calls == 1
        assert abs(summary.total_cost - 0.01) < 1e-9

    def test_limit_records(self, repo):
        for i in range(10):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            repo.save(_make_record(timestamp=ts))

        records = repo.query_records(limit=3)
        assert len(records) == 3

    def test_records_ordered_newest_first(self, repo):
        ts1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        repo.save(_make_record(timestamp=ts1))
        repo.save(_make_record(timestamp=ts2))

        records = repo.query_records()
        assert records[0].timestamp > records[1].timestamp
