"""Anthropic (Claude) LLM provider."""

from typing import Any

from shared.llm_pkg.base import LLMProvider
from shared.models import ChatResponse, Usage
from shared.retry import llm_retry


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-5-20250929"
    env_key = "ANTHROPIC_API_KEY"

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
            "max_tokens": 1024,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return ChatResponse(
            content=response.content[0].text,
            usage=usage,
            model=model,
            provider=self.name,
        )
