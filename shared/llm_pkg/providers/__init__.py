"""Auto-register all built-in providers."""

from shared.llm_pkg.providers.groq import GroqProvider
from shared.llm_pkg.providers.openai import OpenAIProvider
from shared.llm_pkg.providers.anthropic import AnthropicProvider
from shared.llm_pkg.providers.gemini import GeminiProvider
from shared.llm_pkg.base import register_provider

for _cls in [GroqProvider, OpenAIProvider, AnthropicProvider, GeminiProvider]:
    register_provider(_cls())
