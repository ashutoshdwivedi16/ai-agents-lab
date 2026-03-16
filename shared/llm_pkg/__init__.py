"""
LLM provider abstraction layer.

Public API (unchanged from the original llm.py):
    from shared.llm import chat, get_usage, reset_usage, register_provider
"""

from shared.llm_pkg.base import (
    LLMProvider,
    chat,
    get_usage,
    reset_usage,
    register_provider,
    get_provider,
)
from shared.llm_pkg.usage import SessionUsage

# Register built-in providers on import
from shared.llm_pkg import providers as _providers  # noqa: F401

__all__ = [
    "LLMProvider",
    "chat",
    "get_usage",
    "reset_usage",
    "register_provider",
    "get_provider",
    "SessionUsage",
]
