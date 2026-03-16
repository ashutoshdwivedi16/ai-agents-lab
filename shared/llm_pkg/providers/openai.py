"""OpenAI LLM provider."""

from typing import Any

from shared.llm_pkg.base import LLMProvider
from shared.models import ChatResponse, Usage
from shared.retry import llm_retry


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"
    env_key = "OPENAI_API_KEY"

    def _create_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=self._api_key())

    @llm_retry()
    def _do_call(self, client: Any, messages: list[dict], model: str) -> ChatResponse:
        response = client.chat.completions.create(model=model, messages=messages)
        usage = Usage()
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return ChatResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            model=model,
            provider=self.name,
        )
