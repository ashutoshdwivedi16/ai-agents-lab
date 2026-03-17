"""LLM provider base class and registry.

The Strategy Pattern with client reuse:
- _create_client() is called once and cached
- _do_call() is called on every request using the cached client
- Cost calculation and usage tracking happen once in the base class
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from shared.models import ChatResponse, Usage
from shared.llm_pkg.usage import SessionUsage
from shared.logging import get_logger

logger = get_logger(__name__)

_session = SessionUsage()
_REGISTRY: dict[str, "LLMProvider"] = {}


class LLMProvider(ABC):
    """Base class for all LLM providers.

    Subclasses must define:
        name: str           -- registry key (e.g., "groq")
        default_model: str  -- fallback model name
        env_key: str        -- environment variable for API key

    Subclasses must implement:
        _create_client() -> Any
        _do_call(client, messages, model) -> ChatResponse

    Config overrides from YAML are applied via apply_config() during
    registration — so changing default.yaml actually takes effect.
    """

    name: str
    default_model: str
    env_key: str

    def __init__(self) -> None:
        self._client: Any = None

    def _api_key(self) -> str:
        """Read the API key from the configured environment variable."""
        key = os.getenv(self.env_key)
        if not key:
            raise ValueError(f"Missing env var: {self.env_key}")
        return key

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the SDK client. Called once, then cached."""

    @abstractmethod
    def _do_call(self, client: Any, messages: list[dict], model: str) -> ChatResponse:
        """Execute the API call. Return a ChatResponse."""

    @property
    def client(self) -> Any:
        """Lazy-initialized, reused client instance."""
        if self._client is None:
            logger.info("Creating %s client", self.name)
            self._client = self._create_client()
        return self._client

    def apply_config(self, default_model: str, env_key: str) -> None:
        """Override class defaults with YAML config values.

        Called during provider registration to wire config into the provider.
        """
        self.default_model = default_model
        self.env_key = env_key

    def call(self, messages: list[dict], model: str) -> str:
        """Send messages, track usage, return response text.

        Backward-compatible method returning a plain string.
        """
        response = self.call_with_response(messages, model)
        return response.content

    def call_with_response(self, messages: list[dict], model: str) -> ChatResponse:
        """Send messages, calculate cost, track usage, return full ChatResponse."""
        start = time.perf_counter()
        response = self._do_call(self.client, messages, model)
        latency_ms = (time.perf_counter() - start) * 1000

        # Calculate cost from config-driven pricing table
        cost = _session.calc_cost(model, response.usage.input_tokens, response.usage.output_tokens)
        response.usage.cost = cost

        _session.track(model, response.usage.input_tokens, response.usage.output_tokens, cost)
        logger.debug(
            "LLM call: provider=%s model=%s tokens=%d+%d cost=$%.6f latency=%.1fms",
            self.name,
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost,
            latency_ms,
        )

        # Record to persistent metrics (fire-and-forget, never crashes)
        from shared.metrics import record_llm_call

        record_llm_call(
            provider=self.name,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=response.usage.cost,
            latency_ms=latency_ms,
        )

        return response


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_provider(provider: LLMProvider) -> None:
    """Register a provider instance and apply config overrides from YAML.

    Wires ProviderConfig.default_model and env_key into the provider
    so that YAML changes actually take effect (not dead config).
    """
    from shared.config import load_app_config

    config = load_app_config()
    prov_config = config.providers.get(provider.name)
    if prov_config:
        provider.apply_config(
            default_model=prov_config.default_model,
            env_key=prov_config.env_key,
        )

    _REGISTRY[provider.name] = provider


def get_provider(name: str) -> LLMProvider:
    """Get a registered provider by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown provider: '{name}'. Available: {available}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat(messages: list[dict], provider: str = "groq", model: str | None = None) -> str:
    """Send messages to an LLM and return the response text.

    This is the primary public API — signature unchanged from v0.
    """
    p = get_provider(provider)
    return p.call(messages, model or p.default_model)


def get_usage() -> dict:
    """Return session usage summary as a dict."""
    return _session.to_dict()


def reset_usage():
    """Reset session usage counters."""
    _session.reset()
