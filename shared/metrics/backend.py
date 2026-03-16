"""MetricsBackend ABC — Strategy Pattern for metrics collection.

Mirrors LLMProvider ABC from shared/llm_pkg/base.py.
Swap implementations via config (sqlite, noop, future: prometheus).
"""

from abc import ABC, abstractmethod
from datetime import datetime

from shared.metrics.models import MetricRecord, MetricsSummary


class MetricsBackend(ABC):
    """Abstract metrics backend. Swap implementations via config."""

    @abstractmethod
    def record(self, metric: MetricRecord) -> None:
        """Record a single metric data point."""

    @abstractmethod
    def summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        """Return aggregated metrics summary."""

    @abstractmethod
    def records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        """Return raw metric records."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the backend and release resources."""
