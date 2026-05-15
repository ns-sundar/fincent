"""Detect benign exceptions during asyncio / anyio teardown (e.g. uvicorn exit)."""

from __future__ import annotations

import asyncio


def is_lifespan_shutdown_noise(exc: BaseException) -> bool:
    """Return True if *exc* is expected when closing MCP sessions during app shutdown.

    ``AsyncExitStack.aclose()`` and anyio cancel scopes can surface
    ``asyncio.CancelledError``, :class:`BaseExceptionGroup` wrapping
    cancellation, or cancel-scope bookkeeping errors — none of which
    should abort process exit or spam ERROR logs.
    """
    if isinstance(exc, asyncio.CancelledError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        if not exc.exceptions:
            return True
        return all(is_lifespan_shutdown_noise(e) for e in exc.exceptions)
    if "cancel scope" in str(exc).lower():
        return True
    return False
