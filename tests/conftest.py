"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock

from shared.llm_pkg.base import LLMProvider, _REGISTRY, reset_usage
from shared.metrics import set_backend, shutdown
from shared.metrics.backends.noop_backend import NoopBackend


@pytest.fixture(autouse=True)
def clean_state():
    """Reset registry and usage before each test."""
    original = dict(_REGISTRY)
    reset_usage()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original)


@pytest.fixture(autouse=True)
def clean_metrics():
    """Reset metrics to NoopBackend before each test.

    Prevents SQLite file creation during tests and ensures
    test isolation for the metrics subsystem.
    """
    set_backend(NoopBackend())
    yield
    shutdown()


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = "mock"
    provider.default_model = "mock-model"
    provider.call.return_value = "Mock response"
    return provider
