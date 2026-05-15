"""User-safe LLM error messages and compact diagnostics."""

from __future__ import annotations

import re
import traceback
from typing import Any

_MAX_ERROR_MESSAGE_CHARS = 1200
_MAX_TRACEBACK_CHARS = 6000
_SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password)(['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+"
)


def _compact(text: str, limit: int) -> str:
    """Return a one-line-safe, bounded version of diagnostic text."""

    if len(text) <= limit:
        return text
    return text[: limit - 24].rstrip() + "\n... [truncated]"


def _redact(text: str) -> str:
    """Remove obvious credential values from exception diagnostics."""

    return _SECRET_VALUE_RE.sub(r"\1\2[redacted]", text)


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


def exception_diagnostic_metadata(exc: BaseException, *, phase: str) -> dict[str, Any]:
    """Compact exception details for non-user-facing UI diagnostics."""

    message = _redact(str(exc) or repr(exc))
    tb = _redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {
        "error": True,
        "error_phase": phase,
        "error_type": type(exc).__name__,
        "error_message": _compact(message, _MAX_ERROR_MESSAGE_CHARS),
        "error_traceback": _compact(tb, _MAX_TRACEBACK_CHARS),
    }
