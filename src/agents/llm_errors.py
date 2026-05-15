"""User-safe LLM error messages."""

from __future__ import annotations

from typing import Any


def is_context_overflow_error(exc: Any) -> bool:
    """Return True when an LLM provider rejected an over-large prompt."""

    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "contextoverflow" in text
        or "context overflow" in text
        or "input tokens exceed" in text
        or "maximum context length" in text
        or "context_length_exceeded" in text
    )


def context_overflow_user_message(task_label: str) -> str:
    """Friendly message for a request that exceeded the model context window."""

    return (
        f"I ran out of model context while working through that {task_label}. "
        "Some retrieved market data was too large to fit in a single pass. "
        "Please try a narrower question, fewer tickers, or a shorter time window."
    )
