"""Groq LLM provider (free tier, fast inference)."""

from typing import Any

from shared.llm_pkg.base import LLMProvider
from shared.models import ChatResponse, Usage
from shared.retry import llm_retry


class GroqProvider(LLMProvider):
    """Groq cloud inference — OpenAI-compatible API."""

    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    env_key = "GROQ_API_KEY"

    def _create_client(self) -> Any:
        from groq import Groq

        return Groq(api_key=self._api_key())

    @llm_retry()
    def _do_call(self, client: Any, messages: list[dict], model: str) -> ChatResponse:
        response = client.chat.completions.create(model=model, messages=messages)
        usage = Usage()
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
        return ChatResponse(
            content=content,
            usage=usage,
            model=model,
            provider=self.name,
        )
