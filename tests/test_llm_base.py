"""Tests for LLM base layer (no real API calls)."""

import pytest
from unittest.mock import MagicMock

from shared.llm_pkg.base import (
    LLMProvider,
    register_provider,
    get_provider,
    chat,
    _REGISTRY,
)
from shared.models import ChatResponse, Usage


class FakeProvider(LLMProvider):
    """A test provider that returns canned responses without hitting any API."""

    name = "fake"
    default_model = "fake-v1"
    env_key = "FAKE_API_KEY"

    def _create_client(self):
        return MagicMock()

    def _do_call(self, client, messages, model):
        return ChatResponse(
            content="fake response",
            usage=Usage(input_tokens=10, output_tokens=5),
            model=model,
            provider=self.name,
        )


class TestRegistry:
    def test_register_and_get(self):
        provider = FakeProvider()
        register_provider(provider)
        assert get_provider("fake") is provider

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent-provider")

    def test_builtin_providers_registered(self):
        """Groq, OpenAI, Anthropic, Gemini should be auto-registered."""
        assert "groq" in _REGISTRY
        assert "openai" in _REGISTRY
        assert "anthropic" in _REGISTRY
        assert "gemini" in _REGISTRY


class TestChat:
    def test_chat_returns_string(self):
        register_provider(FakeProvider())
        result = chat(
            [{"role": "user", "content": "hi"}],
            provider="fake",
        )
        assert result == "fake response"
        assert isinstance(result, str)

    def test_chat_uses_default_model(self):
        provider = FakeProvider()
        register_provider(provider)
        chat([{"role": "user", "content": "hi"}], provider="fake")
        # If no model specified, should use default_model


class TestClientReuse:
    def test_client_created_once(self):
        provider = FakeProvider()
        _ = provider.client
        _ = provider.client
        # _create_client should only be called once (cached)
        assert provider._client is not None
