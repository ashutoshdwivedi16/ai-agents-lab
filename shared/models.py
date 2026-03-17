"""Pydantic models for type-safe data across the lab."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=100_000)


class Usage(BaseModel):
    """Token usage from a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class ChatResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    usage: Usage = Usage()
    model: str = ""
    provider: str = ""


class SessionUsageReport(BaseModel):
    """Cumulative session usage stats."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    calls: int = 0


class ProviderConfig(BaseModel):
    """Config for a single LLM provider.

    These values are wired into providers at registration time via
    LLMProvider.apply_config(), so YAML changes take effect at runtime.
    """

    default_model: str
    env_key: str
    max_tokens: int = 4096  # max output tokens (configurable per provider)
    pricing: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # pricing: model_name -> (input_price_per_1M, output_price_per_1M)


class AgentConfig(BaseModel):
    """Config for an individual agent."""

    system_prompt: str = "You are a helpful assistant."
    max_history: int = 50
    max_input_length: int = 10_000
    provider: str = "groq"
    model: str | None = None


class MetricsConfig(BaseModel):
    """Config for the metrics subsystem."""

    enabled: bool = True
    backend: Literal["sqlite", "noop"] = "sqlite"
    sqlite_path: str = "data/metrics.db"
    session_id: str = "auto"  # "auto" = generate UUID per session


class AppConfig(BaseModel):
    """Top-level application config."""

    default_provider: str = "groq"
    max_retries: int = 3
    retry_base_delay: float = 1.0
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    metrics: MetricsConfig = MetricsConfig()
