"""Structured logging setup for the lab.

Uses Python's built-in logging module — no external dependencies.
Logs go to stderr so they don't interfere with chatbot stdout output.

Usage:
    from shared.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Starting chatbot", extra={"provider": "groq"})
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with structured formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
