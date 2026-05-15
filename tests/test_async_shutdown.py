"""Tests for shutdown-noise classification during MCP / lifespan teardown."""

from __future__ import annotations

import asyncio

import pytest

from src.utils.async_shutdown import is_lifespan_shutdown_noise


def test_cancelled_error_is_noise() -> None:
    assert is_lifespan_shutdown_noise(asyncio.CancelledError())


def test_base_exception_group_all_cancelled_is_noise() -> None:
    exc = BaseExceptionGroup(
        "teardown",
        [asyncio.CancelledError(), asyncio.CancelledError()],
    )
    assert is_lifespan_shutdown_noise(exc)


def test_base_exception_group_mixed_not_noise() -> None:
    exc = BaseExceptionGroup(
        "teardown",
        [asyncio.CancelledError(), RuntimeError("real")],
    )
    assert not is_lifespan_shutdown_noise(exc)


def test_cancel_scope_message_is_noise() -> None:
    assert is_lifespan_shutdown_noise(
        RuntimeError("Attempted to exit a cancel scope that isn't the current")
    )
