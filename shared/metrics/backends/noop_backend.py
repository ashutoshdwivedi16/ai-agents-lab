"""No-op backend — metrics disabled or test mode. Zero overhead."""

from datetime import datetime

from shared.metrics.backend import MetricsBackend
from shared.metrics.models import MetricRecord, MetricsSummary


class NoopBackend(MetricsBackend):
    """Silently discards all metrics. Used when metrics are disabled."""

    def record(self, metric: MetricRecord) -> None:
        pass

    def summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        return MetricsSummary()

    def records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        return []

    def close(self) -> None:
        pass
