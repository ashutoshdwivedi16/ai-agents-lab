"""Anthropic (Claude) LLM provider."""

from typing import Any

from shared.llm_pkg.base import LLMProvider
from shared.models import ChatResponse, Usage
from shared.retry import llm_retry

# Default max_tokens — overridden by ProviderConfig.max_tokens from YAML
_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """Anthropic API — Claude models.

    Note: Anthropic requires `max_tokens` in every request (unlike OpenAI).
    The value comes from config/default.yaml → providers.anthropic.max_tokens.
    """

    name = "anthropic"
    default_model = "claude-sonnet-4-5-20250929"
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._max_tokens = _DEFAULT_MAX_TOKENS

    def apply_config(self, default_model: str, env_key: str) -> None:
        """Override defaults with YAML config, including max_tokens."""
        super().apply_config(default_model, env_key)
        # max_tokens is loaded separately via ProviderConfig
        from shared.config import load_app_config
        config = load_app_config()
        prov_config = config.providers.get(self.name)
        if prov_config:
            self._max_tokens = prov_config.max_tokens

    def _create_client(self) -> Any:
        from anthropic import Anthropic

        return Anthropic(api_key=self._api_key())

    @llm_retry()
    def _do_call(self, client: Any, messages: list[dict], model: str) -> ChatResponse:
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        content = ""
        if response.content:
            content = response.content[0].text
        return ChatResponse(
            content=content,
            usage=usage,
            model=model,
            provider=self.name,
        )
