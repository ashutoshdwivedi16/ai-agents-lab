"""SQLite metrics backend — default for local development."""

from datetime import datetime

from shared.metrics.backend import MetricsBackend
from shared.metrics.models import MetricRecord, MetricsSummary
from shared.metrics.repositories.sqlite_repository import SQLiteRepository


class SQLiteBackend(MetricsBackend):
    """Delegates to SQLiteRepository.

    Separates "what to record" (backend) from "where to store" (repository).
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        self._repo = SQLiteRepository(db_path)

    def record(self, metric: MetricRecord) -> None:
        self._repo.save(metric)

    def summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        return self._repo.query_summary(session_id, agent_name, since)

    def records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        return self._repo.query_records(session_id, agent_name, since, limit)

    def close(self) -> None:
        self._repo.close()
