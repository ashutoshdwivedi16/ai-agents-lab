"""Conversation utilities: validation, history management."""

from shared.logging import get_logger

logger = get_logger(__name__)


def validate_input(text: str, max_length: int) -> str | None:
    """Validate user input. Returns error message string, or None if valid.

    Empty strings are allowed (caller handles them separately).
    """
    if not text:
        return None
    if len(text) > max_length:
        return f"Input too long ({len(text)} chars). Max: {max_length}"
    return None


def trim_history(messages: list[dict], max_history: int) -> list[dict]:
    """Keep system prompt + last N messages.

    Prevents unbounded conversation history growth which would
    exceed context windows and rack up token costs.

    Assumes the first message is a system prompt (preserved always).
    If max_history <= 0, returns only the system prompt.
    """
    if not messages:
        return messages
    if max_history <= 0:
        return [messages[0]]
    if len(messages) <= max_history + 1:  # +1 for system message
        return messages
    logger.info(
        "Trimming history from %d to %d messages",
        len(messages) - 1,
        max_history,
    )
    return [messages[0]] + messages[-max_history:]
