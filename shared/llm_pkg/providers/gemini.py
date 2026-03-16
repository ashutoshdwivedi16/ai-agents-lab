"""Google Gemini LLM provider."""

from typing import Any

from shared.llm_pkg.base import LLMProvider
from shared.models import ChatResponse, Usage
from shared.retry import llm_retry


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"
    env_key = "GOOGLE_API_KEY"

    def _create_client(self) -> Any:
        from google import genai

        return genai.Client(api_key=self._api_key())

    @llm_retry()
    def _do_call(self, client: Any, messages: list[dict], model: str) -> ChatResponse:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            if msg["role"] == "system":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})
            else:
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(model=model, contents=contents)
        usage = Usage()
        if response.usage_metadata:
            usage = Usage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )
        return ChatResponse(
            content=response.text or "",
            usage=usage,
            model=model,
            provider=self.name,
        )
