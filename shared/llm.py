"""
Thin wrapper for multiple LLM providers using the Strategy Pattern.
Each provider is a self-contained class — adding a new one requires
zero changes to existing code.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class SessionUsage:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    calls: int = 0

    def track(self, model: str, input_tokens: int, output_tokens: int):
        cost = _calc_cost(model, input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.calls += 1

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "calls": self.calls,
        }


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens) — updated March 2026
# ---------------------------------------------------------------------------

PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant":    (0.05, 0.08),
    "gpt-4o-mini":             (0.15, 0.60),
    "claude-sonnet-4-5-20250929":    (3.00, 15.00),
    "gemini-2.0-flash":        (0.10, 0.40),
}

_session = SessionUsage()


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    inp, out = PRICING[model]
    return (input_tokens * inp / 1_000_000) + (output_tokens * out / 1_000_000)


# ---------------------------------------------------------------------------
# Base class (Strategy interface)
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Every provider implements this — one method, one contract."""

    name: str
    default_model: str
    env_key: str

    @abstractmethod
    def call(self, messages: list, model: str) -> str:
        """Send messages and return the response text."""

    def _api_key(self) -> str:
        key = os.getenv(self.env_key)
        if not key:
            raise ValueError(f"Missing env var: {self.env_key}")
        return key


# ---------------------------------------------------------------------------
# Concrete strategies — each provider is self-contained
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    env_key = "GROQ_API_KEY"

    def call(self, messages: list, model: str) -> str:
        from groq import Groq

        client = Groq(api_key=self._api_key())
        response = client.chat.completions.create(model=model, messages=messages)
        if response.usage:
            _session.track(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"
    env_key = "OPENAI_API_KEY"

    def call(self, messages: list, model: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key())
        response = client.chat.completions.create(model=model, messages=messages)
        if response.usage:
            _session.track(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-5-20250929"
    env_key = "ANTHROPIC_API_KEY"

    def call(self, messages: list, model: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key())
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {"model": model, "max_tokens": 1024, "messages": chat_messages}
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        _session.track(model, response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"
    env_key = "GOOGLE_API_KEY"

    def call(self, messages: list, model: str) -> str:
        from google import genai

        client = genai.Client(api_key=self._api_key())
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            if msg["role"] == "system":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})
            else:
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(model=model, contents=contents)
        if response.usage_metadata:
            _session.track(model, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
        return response.text


# ---------------------------------------------------------------------------
# Provider registry — add new providers here, nothing else changes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, LLMProvider] = {
    "groq":      GroqProvider(),
    "openai":    OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "gemini":    GeminiProvider(),
}


def register_provider(provider: LLMProvider):
    """Register a custom provider at runtime."""
    _REGISTRY[provider.name] = provider


# ---------------------------------------------------------------------------
# Public API — unchanged interface, clean internals
# ---------------------------------------------------------------------------

def chat(messages: list, provider: str = "groq", model: str = None) -> str:
    """Send messages to an LLM and return the response text."""
    if provider not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown provider: '{provider}'. Available: {available}")

    p = _REGISTRY[provider]
    return p.call(messages, model or p.default_model)


def get_usage() -> dict:
    """Return session usage summary."""
    return _session.to_dict()


def reset_usage():
    """Reset session usage counters."""
    _session.reset()
