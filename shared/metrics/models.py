"""Pydantic models for metrics data."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """A single metric data point recorded from an LLM call."""

    session_id: str
    agent_name: str = "unknown"
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsSummary(BaseModel):
    """Aggregated metrics summary for CLI/API output."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    by_provider: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_agent: dict[str, int] = {}
