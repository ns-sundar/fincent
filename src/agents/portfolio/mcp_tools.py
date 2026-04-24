"""MCP tool loader for the Portfolio agent.

The Portfolio agent is a ReAct-style tool user: it grounds its replies
in the user's personal portfolio snapshot but may additionally call
out to MCP tool servers for live market data (via OpenBB) and
retrieval-augmented context (via the Fincent RAG MCP server).

FastAPI lifespan calls :func:`start_portfolio_mcp_sessions` so stdio
MCP servers (OpenBB + Fincent RAG) stay running for the API process
lifetime; :func:`stop_portfolio_mcp_sessions` closes them on shutdown.
Without lifespan (e.g. ad-hoc scripts), :func:`get_portfolio_tools`
falls back to stateless discovery (new subprocess per tool call).

This module returns an empty tool list when dependencies are missing,
servers are disabled, or startup fails -- never crashing the agent.

``langchain_mcp_adapters`` / ``mcp`` are imported only inside async
startup helpers to keep import cost low for non-portfolio code paths.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import (
    AppConfig,
    PortfolioMcpServerSpec,
    get_config,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)

# Cached tool list. ``None`` -> lifespan not run / cleared; ``[]`` -> no
# tools; populated list -> ready for the agent.
_TOOLS_CACHE: Optional[List[Any]] = None
# Holds stdio MCP subprocesses open for the app lifetime (FastAPI lifespan).
_MCP_EXIT_STACK: Optional[AsyncExitStack] = None
# The uvicorn event loop — captured at lifespan startup so sync threadpool
# workers can submit coroutines to it via run_coroutine_threadsafe.
_UVICORN_LOOP: Optional[asyncio.AbstractEventLoop] = None

def _fincent_repo_root() -> Path:
    """Resolve the ``fincent/`` directory (contains ``config.yaml`` + ``src/``).

    ``mcp_tools.py`` lives at ``src/agents/portfolio/``; ``parents[2]`` is
    ``src/``, which is **not** the repo root and breaks ``data/`` paths.
    """
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "config.yaml").is_file() and (
            cand / "src" / "agents" / "portfolio"
        ).is_dir():
            return cand
    return here.parents[3]


# Project root (``fincent/``), never ``fincent/src/``.
_PROJECT_ROOT: Path = _fincent_repo_root()
# Isolated POSIX $HOME for the OpenBB MCP subprocess only. OpenBB reads
# ``$HOME/.openbb_platform/user_settings.json``; without this, the
# first alphabetically sorted provider (often FMP) is used and requires
# ``fmp_api_key``. The yfinance connector needs no API key.
_FINCENT_OPENBB_HOME: Path = _PROJECT_ROOT / ".fincent_openbb_home"
_DEFAULT_OPENBB_USER_SETTINGS: Path = (
    _PROJECT_ROOT / "data" / "openbb_default_user_settings.json"
)


def _ensure_fincent_openbb_mcp_home() -> str:
    """Create a dedicated HOME for ``openbb-mcp`` with yfinance defaults.

    Operators who want ``~/.openbb_platform`` credentials instead can
    set ``portfolio.tools.openbb.env.HOME`` in ``config.yaml``.
    """
    obb_dir = _FINCENT_OPENBB_HOME / ".openbb_platform"
    obb_dir.mkdir(parents=True, exist_ok=True)
    dest = obb_dir / "user_settings.json"
    if not dest.exists() and _DEFAULT_OPENBB_USER_SETTINGS.is_file():
        shutil.copyfile(_DEFAULT_OPENBB_USER_SETTINGS, dest)
        _logger.info(
            "Installed OpenBB user_settings for Fincent MCP home at %s",
            dest,
        )
    elif not dest.exists():
        dest.write_text(
            json.dumps(
                {
                    "credentials": {},
                    "preferences": {},
                    "defaults": {
                        "commands": {
                            "/equity/price/quote": {"provider": "yfinance"},
                            "/equity/price/historical": {"provider": "yfinance"},
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _logger.warning(
            "Wrote minimal OpenBB user_settings (template JSON missing): %s",
            dest,
        )
    return str(_FINCENT_OPENBB_HOME.resolve())


# ---------------------------------------------------------------------
# Config -> MultiServerMCPClient shape
# ---------------------------------------------------------------------


def _server_config(
    server_key: str,
    spec: PortfolioMcpServerSpec,
) -> Optional[Dict[str, Any]]:
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
    spec_env_keys_upper: set[str] = set()
    for key, value in (spec.env or {}).items():
        ks = str(key)
        spec_env_keys_upper.add(ks.upper())
        child_env[ks] = str(value)

    # OpenBB MCP: use an isolated $HOME with yfinance-first user_settings
    # (see ``data/openbb_default_user_settings.json``). Do not treat the
    # parent process HOME as an override — only ``spec.env`` may set
    # HOME to opt out (e.g. use ``~/.openbb_platform`` + API keys).
    if server_key == "openbb" and "HOME" not in spec_env_keys_upper:
        child_env["HOME"] = _ensure_fincent_openbb_mcp_home()

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
        entry = _server_config(server_name, spec)
        if entry is not None:
            specs[server_name] = entry
    return specs


# ---------------------------------------------------------------------
# Async loader
# ---------------------------------------------------------------------


async def _abuild_tools_stateless(specs: Dict[str, Dict[str, Any]]) -> List[Any]:
    """One-shot tool discovery (new stdio subprocess per tool call on invoke).

    Used only when :func:`start_portfolio_mcp_sessions` did not run
    (e.g. CLI scripts without FastAPI).
    """
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
        "Loaded %d MCP tool(s) (stateless) from %d server(s): %s",
        len(tools),
        len(specs),
        sorted(specs.keys()),
    )
    return list(tools)


async def start_portfolio_mcp_sessions(cfg: Optional[AppConfig] = None) -> None:
    """Open persistent MCP client sessions (stdio servers stay running).

    Called from FastAPI lifespan startup. Tool handles reuse these
    sessions so each model turn does not respawn ``openbb-mcp`` / RAG
    subprocesses. Shutdown via :func:`stop_portfolio_mcp_sessions`.
    """
    global _TOOLS_CACHE, _MCP_EXIT_STACK, _UVICORN_LOOP

    cfg = cfg or get_config()
    await stop_portfolio_mcp_sessions()

    specs = _build_client_specs(cfg)
    if not specs:
        _TOOLS_CACHE = []
        _logger.info("Portfolio MCP: no enabled servers.")
        return

    try:
        from langchain_mcp_adapters.sessions import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        _logger.warning(
            "langchain-mcp-adapters not installed; Portfolio MCP disabled. (%s)",
            exc,
        )
        _TOOLS_CACHE = []
        return

    stack = AsyncExitStack()
    all_tools: List[Any] = []
    try:
        for name, conn in specs.items():
            session = await stack.enter_async_context(create_session(conn))
            await session.initialize()
            part = await load_mcp_tools(
                session, connection=conn, server_name=name
            )
            all_tools.extend(part)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Portfolio MCP persistent session startup failed (specs=%s): %s",
            sorted(specs.keys()),
            exc,
        )
        await stack.aclose()
        _TOOLS_CACHE = []
        return

    _MCP_EXIT_STACK = stack
    _TOOLS_CACHE = all_tools
    _UVICORN_LOOP = asyncio.get_event_loop()
    _logger.info(
        "Portfolio MCP: persistent sessions active (%d tools; servers=%s).",
        len(all_tools),
        sorted(specs.keys()),
    )


def get_uvicorn_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the event loop captured during lifespan startup, or None."""
    return _UVICORN_LOOP


async def stop_portfolio_mcp_sessions() -> None:
    """Close MCP stdio sessions and subprocesses (FastAPI lifespan shutdown)."""
    global _MCP_EXIT_STACK, _TOOLS_CACHE, _UVICORN_LOOP

    if _MCP_EXIT_STACK is not None:
        try:
            await _MCP_EXIT_STACK.aclose()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Portfolio MCP shutdown error: %s", exc)
        finally:
            _MCP_EXIT_STACK = None
    _TOOLS_CACHE = None
    _UVICORN_LOOP = None


async def _restart_portfolio_mcp(cfg: AppConfig) -> None:
    await stop_portfolio_mcp_sessions()
    await start_portfolio_mcp_sessions(cfg)


# ---------------------------------------------------------------------
# Sync-friendly public entry point
# ---------------------------------------------------------------------


def get_portfolio_tools(
    cfg: Optional[AppConfig] = None,
    *,
    force_refresh: bool = False,
) -> List[Any]:
    """Return MCP tools (persistent sessions when FastAPI lifespan ran).

    The call is a no-op when every configured server is disabled, when
    ``langchain-mcp-adapters`` is not installed, or when server
    startup fails. In those cases the returned list is empty and the
    agent falls back to the legacy direct-LLM answer path.

    Args:
        cfg:           Optional pre-loaded config (tests).
        force_refresh: When True, close persistent sessions (if any) and
                       rebuild from config. Uses ``asyncio.run`` when no
                       loop is running.

    Returns:
        A list of LangChain ``BaseTool`` objects (possibly empty).
    """
    global _TOOLS_CACHE

    cfg = cfg or get_config()

    if force_refresh:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_restart_portfolio_mcp(cfg))
        else:
            _logger.warning(
                "get_portfolio_tools(force_refresh=True) ignored: event loop running"
            )

    if _TOOLS_CACHE is not None:
        return _TOOLS_CACHE

    specs = _build_client_specs(cfg)
    if not specs:
        _logger.debug(
            "No enabled Portfolio MCP tool servers in config; "
            "Portfolio agent will answer without tools."
        )
        _TOOLS_CACHE = []
        return _TOOLS_CACHE

    _logger.info(
        "Portfolio MCP: lazy-loading tools (no lifespan); stateless stdio per call."
    )
    try:
        _TOOLS_CACHE = asyncio.run(_abuild_tools_stateless(specs))
    except RuntimeError as exc:
        if "running event loop" not in str(exc):
            _logger.warning("Portfolio MCP tool loader failed: %s", exc)
            _TOOLS_CACHE = []
            return _TOOLS_CACHE
        _TOOLS_CACHE = _run_stateless_in_worker_thread(specs)
    except Exception as exc:  # noqa: BLE001 -- never break the agent
        _logger.warning("Portfolio MCP tool loader failed: %s", exc)
        _TOOLS_CACHE = []
    return _TOOLS_CACHE


def _run_stateless_in_worker_thread(
    specs: Dict[str, Dict[str, Any]],
) -> List[Any]:
    """Run :func:`_abuild_tools_stateless` on a worker thread with its own loop."""
    import concurrent.futures

    def _target() -> List[Any]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_abuild_tools_stateless(specs))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result()


def reset_portfolio_tools_cache() -> None:
    """Drop the cached tool list so the next call re-loads (tests).

    Does not close live MCP subprocesses; use :func:`stop_portfolio_mcp_sessions`
    in async code or restart the API process.
    """
    global _TOOLS_CACHE
    _TOOLS_CACHE = None


__all__ = [
    "get_portfolio_tools",
    "get_uvicorn_loop",
    "reset_portfolio_tools_cache",
    "start_portfolio_mcp_sessions",
    "stop_portfolio_mcp_sessions",
]
