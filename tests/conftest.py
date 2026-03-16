"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock

from shared.llm_pkg.base import LLMProvider, _REGISTRY, reset_usage


@pytest.fixture(autouse=True)
def clean_state():
    """Reset registry and usage before each test."""
    original = dict(_REGISTRY)
    reset_usage()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = "mock"
    provider.default_model = "mock-model"
    provider.call.return_value = "Mock response"
    return provider
