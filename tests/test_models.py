"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from shared.models import Message, Usage, ChatResponse, AgentConfig


class TestMessage:
    def test_valid_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Hello")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="")


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cost == 0.0

    def test_with_values(self):
        u = Usage(input_tokens=100, output_tokens=50, cost=0.01)
        assert u.input_tokens == 100


class TestChatResponse:
    def test_minimal(self):
        r = ChatResponse(content="Hello!")
        assert r.content == "Hello!"
        assert r.usage.input_tokens == 0

    def test_full(self):
        r = ChatResponse(
            content="Hi",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
            provider="test",
        )
        assert r.model == "test-model"
        assert r.provider == "test"


class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.max_history == 50
        assert c.max_input_length == 10_000
        assert c.provider == "groq"
        assert c.model is None

    def test_custom_values(self):
        c = AgentConfig(system_prompt="Be brief.", max_history=10)
        assert c.system_prompt == "Be brief."
        assert c.max_history == 10
