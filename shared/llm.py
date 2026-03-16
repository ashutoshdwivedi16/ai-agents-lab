"""Backward-compatible re-export from the new llm subpackage.

Usage unchanged:
    from shared.llm import chat, get_usage, reset_usage
"""

from shared.llm_pkg import (  # noqa: F401
    LLMProvider,
    chat,
    get_usage,
    reset_usage,
    register_provider,
    get_provider,
    SessionUsage,
)
