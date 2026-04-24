"""MCP tool loader for the Portfolio agent.

The Portfolio agent is a ReAct-style tool user: it grounds its replies
in the user's personal portfolio snapshot but may additionally call
out to MCP tool servers for live market data (via OpenBB) and
retrieval-augmented context (via the Fincent RAG MCP server).

This module centralises:

* building a :class:`MultiServerMCPClient` from the relevant
  ``cfg.portfolio.tools.*`` specs,
* loading the advertised tools once per process and caching the list,
* and returning an empty list -- rather than crashing -- when the
  optional dependencies are missing, a server fails to start, or the
  operator disabled the servers via config.

The loader is intentionally import-light: ``langchain_mcp_adapters``
and ``mcp`` are only imported inside :func:`_abuild_tools`. Tests and
other call sites that never invoke the Portfolio agent never pay the
import cost.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from src.core.config import (
    AppConfig,
    PortfolioMcpServerSpec,
    get_config,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# Cached tool list. ``None`` -> never tried; ``[]`` -> tried and got
# nothing usable; populated list -> ready for the agent.
_TOOLS_CACHE: Optional[List[Any]] = None


# ---------------------------------------------------------------------
# Config -> MultiServerMCPClient shape
# ---------------------------------------------------------------------


def _server_config(spec: PortfolioMcpServerSpec) -> Optional[Dict[str, Any]]:
    """Convert a spec into the dict shape MultiServerMCPClient wants.

    Returns ``None`` when the spec is disabled or missing a command
    (so the caller can skip it without a branch).
    """
    if not spec.enabled:
        return None
    command = (spec.command or "").strip()
    if not command:
        return None

    # Merge the current process env with the per-server overrides so
    # child servers inherit PATH, HOME, OPENAI_API_KEY, etc. and still
    # see the extra overrides from config (e.g. forcing the RAG MCP
    # server's transport to stdio).
    child_env: Dict[str, str] = dict(os.environ)
    for key, value in (spec.env or {}).items():
        child_env[str(key)] = str(value)

    return {
        "transport": "stdio",
        "command": command,
        "args": list(spec.args or []),
        "env": child_env,
    }


def _build_client_specs(cfg: AppConfig) -> Dict[str, Dict[str, Any]]:
    """Collect all enabled server specs keyed by a stable server name."""
    specs: Dict[str, Dict[str, Any]] = {}
    tools_cfg = cfg.portfolio.tools

    for server_name, spec in (
        ("openbb", tools_cfg.openbb),
        ("fincent_rag", tools_cfg.rag),
    ):
        entry = _server_config(spec)
        if entry is not None:
            specs[server_name] = entry
    return specs


# ---------------------------------------------------------------------
# Async loader
# ---------------------------------------------------------------------


async def _abuild_tools(specs: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Open an MCP client, fetch tools from every server, return them."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        _logger.warning(
            "langchain-mcp-adapters not installed; Portfolio agent "
            "will run without MCP tools. Install with "
            "`pip install langchain-mcp-adapters`. (%s)",
            exc,
        )
        return []

    client = MultiServerMCPClient(specs)
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001 -- never break the agent
        _logger.warning(
            "Failed to load MCP tools for Portfolio agent (specs=%s): %s",
            sorted(specs.keys()),
            exc,
        )
        return []

    _logger.info(
        "Loaded %d MCP tool(s) for Portfolio agent from %d server(s): %s",
        len(tools),
        len(specs),
        sorted(specs.keys()),
    )
    return list(tools)


# ---------------------------------------------------------------------
# Sync-friendly public entry point
# ---------------------------------------------------------------------


def get_portfolio_tools(
    cfg: Optional[AppConfig] = None,
    *,
    force_refresh: bool = False,
) -> List[Any]:
    """Load (and cache) the MCP tools the Portfolio agent may call.

    The call is a no-op when every configured server is disabled, when
    ``langchain-mcp-adapters`` is not installed, or when server
    startup fails. In those cases the returned list is empty and the
    agent falls back to the legacy direct-LLM answer path.

    Args:
        cfg:           Optional pre-loaded config (tests).
        force_refresh: When True, re-load the tools even if the module
                       already cached a list from a previous call.

    Returns:
        A list of LangChain ``BaseTool`` objects (possibly empty).
    """
    global _TOOLS_CACHE

    if _TOOLS_CACHE is not None and not force_refresh:
        return _TOOLS_CACHE

    cfg = cfg or get_config()
    specs = _build_client_specs(cfg)
    if not specs:
        _logger.debug(
            "No enabled Portfolio MCP tool servers in config; "
            "Portfolio agent will answer without tools."
        )
        _TOOLS_CACHE = []
        return _TOOLS_CACHE

    # FastAPI dispatches sync routes to a threadpool, so there is no
    # running loop in this thread and ``asyncio.run`` works. Tests
    # running under pytest-asyncio would hit a running loop; to stay
    # safe we fall back to a fresh loop on a worker thread in that
    # case.
    try:
        _TOOLS_CACHE = asyncio.run(_abuild_tools(specs))
    except RuntimeError as exc:
        if "running event loop" not in str(exc):
            _logger.warning("Portfolio MCP tool loader failed: %s", exc)
            _TOOLS_CACHE = []
            return _TOOLS_CACHE
        _TOOLS_CACHE = _run_in_worker_thread(specs)
    except Exception as exc:  # noqa: BLE001 -- never break the agent
        _logger.warning("Portfolio MCP tool loader failed: %s", exc)
        _TOOLS_CACHE = []
    return _TOOLS_CACHE


def _run_in_worker_thread(specs: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Run :func:`_abuild_tools` on a worker thread with its own loop.

    Used only when ``asyncio.run`` would collide with a loop owned by
    the caller (e.g. pytest-asyncio). We spin up a throwaway thread
    with a fresh event loop so the one-shot MCP bootstrap never
    interferes with the caller's loop.
    """
    import concurrent.futures

    def _target() -> List[Any]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_abuild_tools(specs))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()


def reset_portfolio_tools_cache() -> None:
    """Drop the cached tool list so the next call re-loads (tests)."""
    global _TOOLS_CACHE
    _TOOLS_CACHE = None


__all__ = [
    "get_portfolio_tools",
    "reset_portfolio_tools_cache",
]
