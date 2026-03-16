"""Repository Pattern ABC for metrics storage.

Swap SQLite for Postgres/DynamoDB by implementing this interface.
Consumer code never touches SQL or a specific DB client.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from shared.metrics.models import MetricRecord, MetricsSummary


class MetricsRepository(ABC):
    """Abstract base for metrics persistence."""

    @abstractmethod
    def save(self, record: MetricRecord) -> None:
        """Persist a single metric record."""

    @abstractmethod
    def query_summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        """Return aggregated metrics, optionally filtered."""

    @abstractmethod
    def query_records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        """Return raw metric records, optionally filtered."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources (DB connections, etc.)."""
