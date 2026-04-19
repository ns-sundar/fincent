"""Centralised logger factory used across the codebase."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED: bool = False


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging exactly once.

    Args:
        level: Log level name ("DEBUG", "INFO", ...).
        log_file: Optional path to additionally log to a file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring root on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
