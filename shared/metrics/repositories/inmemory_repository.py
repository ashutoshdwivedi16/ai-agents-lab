"""In-memory implementation for tests. No file I/O."""

from datetime import datetime

from shared.metrics.repository import MetricsRepository
from shared.metrics.models import MetricRecord, MetricsSummary


class InMemoryRepository(MetricsRepository):
    """List-backed repository for unit tests. Zero disk I/O."""

    def __init__(self):
        self._records: list[MetricRecord] = []

    def save(self, record: MetricRecord) -> None:
        self._records.append(record)

    def query_summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        filtered = self._filter(session_id, agent_name, since)
        if not filtered:
            return MetricsSummary()
        return MetricsSummary(
            total_calls=len(filtered),
            total_input_tokens=sum(r.input_tokens for r in filtered),
            total_output_tokens=sum(r.output_tokens for r in filtered),
            total_cost=sum(r.cost for r in filtered),
            avg_latency_ms=sum(r.latency_ms for r in filtered) / len(filtered),
            by_provider=self._group_count(filtered, "provider"),
            by_model=self._group_count(filtered, "model"),
            by_agent=self._group_count(filtered, "agent_name"),
        )

    def query_records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        filtered = self._filter(session_id, agent_name, since)
        return sorted(filtered, key=lambda r: r.timestamp, reverse=True)[:limit]

    def close(self) -> None:
        self._records.clear()

    def _filter(
        self,
        session_id: str | None,
        agent_name: str | None,
        since: datetime | None,
    ) -> list[MetricRecord]:
        result = self._records
        if session_id:
            result = [r for r in result if r.session_id == session_id]
        if agent_name:
            result = [r for r in result if r.agent_name == agent_name]
        if since:
            result = [r for r in result if r.timestamp >= since]
        return result

    @staticmethod
    def _group_count(records: list[MetricRecord], attr: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in records:
            key = getattr(r, attr)
            counts[key] = counts.get(key, 0) + 1
        return counts
