"""LLM provider base class and registry.

The Strategy Pattern with client reuse:
- _create_client() is called once and cached
- _do_call() is called on every request using the cached client
- Usage tracking happens once in the base class, not per provider
"""

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
    """

    name: str
    default_model: str
    env_key: str
    _client: Any = None

    def _api_key(self) -> str:
        import os

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

    def call(self, messages: list[dict], model: str) -> str:
        """Send messages, track usage, return response text.

        Backward-compatible method returning a plain string.
        """
        response = self.call_with_response(messages, model)
        return response.content

    def call_with_response(self, messages: list[dict], model: str) -> ChatResponse:
        """Send messages, track usage, return full ChatResponse."""
        response = self._do_call(self.client, messages, model)
        _session.track(model, response.usage.input_tokens, response.usage.output_tokens)
        logger.debug(
            "LLM call: provider=%s model=%s tokens=%d+%d cost=$%.6f",
            self.name,
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost,
        )
        return response


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_provider(provider: LLMProvider):
    """Register a provider instance."""
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


def chat(messages: list, provider: str = "groq", model: str | None = None) -> str:
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
