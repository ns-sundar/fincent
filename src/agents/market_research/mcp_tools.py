"""MCP tool loader for the Market Research agent."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.openbb_mcp_invoke import filter_openbb_mcp_tools_for_available_credentials
from src.core.config import AppConfig, PortfolioMcpServerSpec, get_config
from src.utils.async_shutdown import is_lifespan_shutdown_noise
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def _filter_fmp_tools_by_name_substrings(
    tools: List[Any],
    substrings: List[str],
    *,
    server_key: str,
) -> List[Any]:
    """Remove FMP MCP tools whose names match configured substrings."""

    if not substrings or not tools:
        return list(tools)
    needles = [s.strip().lower() for s in substrings if str(s).strip()]
    if not needles:
        return list(tools)
    kept: List[Any] = []
    dropped = 0
    for t in tools:
        name_l = str(getattr(t, "name", "") or "").lower()
        if any(n in name_l for n in needles):
            dropped += 1
            continue
        kept.append(t)
    if dropped:
        _logger.info(
            "Market Research %s MCP: excluded %d tool(s) with name matching %s "
            "(set market_research.fmp_exclude_tool_name_substrings to [] for "
            "full FMP, e.g. paid plans).",
            server_key,
            dropped,
            needles,
        )
    return kept


_TOOLS_CACHE: Optional[List[Any]] = None
_MCP_EXIT_STACK: Optional[AsyncExitStack] = None
_UVICORN_LOOP: Optional[asyncio.AbstractEventLoop] = None

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# Canonical names used in config.yaml placeholders, plus common aliases
# (e.g. FMP docs often say "API key" while our MCP server expects
# ``FMP_ACCESS_TOKEN``).
_ENV_PLACEHOLDER_ALIASES: Dict[str, tuple[str, ...]] = {
    "FMP_ACCESS_TOKEN": (
        "FMP_ACCESS_TOKEN",
        "FMP_API_KEY",
        "FINANCIAL_MODELING_PREP_API_KEY",
    ),
    "ALPHA_VANTAGE_API_KEY": (
        "ALPHA_VANTAGE_API_KEY",
        "ALPHAVANTAGE_API_KEY",
    ),
    "TAVILY_API_KEY": ("TAVILY_API_KEY",),
}


def _fincent_repo_root() -> Path:
    """Resolve the ``fincent/`` directory."""

    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "config.yaml").is_file() and (
            cand / "src" / "agents" / "market_research"
        ).is_dir():
            return cand
    return here.parents[3]


_PROJECT_ROOT: Path = _fincent_repo_root()
_PORTFOLIO_OPENBB_HOME: Path = _PROJECT_ROOT / ".fincent_openbb_home"


def _ensure_openbb_home() -> Optional[str]:
    """Reuse Portfolio's isolated OpenBB HOME when available."""

    try:
        from src.agents.portfolio.mcp_tools import _ensure_fincent_openbb_mcp_home

        return _ensure_fincent_openbb_mcp_home()
    except Exception as exc:  # noqa: BLE001 -- fall back to a simple env
        _logger.warning("Could not prepare isolated OpenBB HOME: %s", exc)
        return str(_PORTFOLIO_OPENBB_HOME.resolve())


def _expand_env_placeholder(value: str, *, server_key: str) -> Optional[str]:
    """Resolve a literal ``${VAR}`` value from ``os.environ``."""

    match = _ENV_PLACEHOLDER_RE.match(value)
    if not match:
        return value
    env_key = match.group(1)
    keys = _ENV_PLACEHOLDER_ALIASES.get(env_key, (env_key,))
    for k in keys:
        raw = os.environ.get(k)
        if raw is None:
            continue
        stripped = str(raw).strip()
        if stripped:
            if k != env_key:
                _logger.debug(
                    "Market Research MCP %s: using %s for placeholder ${%s}",
                    server_key,
                    k,
                    env_key,
                )
            return stripped
    _logger.warning(
        "Market Research MCP server %s disabled: required env var %s is missing "
        "(tried: %s).",
        server_key,
        env_key,
        ", ".join(keys),
    )
    return None


def _expand_values(
    values: List[str],
    *,
    server_key: str,
) -> Optional[List[str]]:
    """Resolve env placeholders in command arguments."""

    expanded: List[str] = []
    for raw in values:
        resolved = _expand_env_placeholder(str(raw), server_key=server_key)
        if resolved is None:
            return None
        expanded.append(resolved)
    return expanded


def _server_config(
    server_key: str,
    spec: PortfolioMcpServerSpec,
) -> Optional[Dict[str, Any]]:
    """Convert a configured MCP server spec into MultiServerMCPClient shape."""

    if not spec.enabled:
        return None
    command = (spec.command or "").strip()
    if not command:
        return None

    args = _expand_values(list(spec.args or []), server_key=server_key)
    if args is None:
        return None

    child_env: Dict[str, str] = dict(os.environ)
    spec_env_keys_upper: set[str] = set()
    for key, value in (spec.env or {}).items():
        ks = str(key)
        resolved = _expand_env_placeholder(str(value), server_key=server_key)
        if resolved is None:
            return None
        spec_env_keys_upper.add(ks.upper())
        child_env[ks] = resolved

    if server_key == "openbb" and "HOME" not in spec_env_keys_upper:
        home = _ensure_openbb_home()
        if home:
            child_env["HOME"] = home

    return {
        "transport": "stdio",
        "command": command,
        "args": args,
        "env": child_env,
    }


def _build_client_specs(cfg: AppConfig) -> Dict[str, Dict[str, Any]]:
    """Collect enabled Market Research MCP server specs."""

    specs: Dict[str, Dict[str, Any]] = {}
    tools_cfg = cfg.market_research.tools
    for server_name, spec in (
        ("openbb", tools_cfg.openbb),
        ("alpha_vantage", tools_cfg.alpha_vantage),
        ("tavily", tools_cfg.tavily),
        ("fmp", tools_cfg.fmp),
    ):
        entry = _server_config(server_name, spec)
        if entry is not None:
            specs[server_name] = entry
    return specs


async def _abuild_tools_stateless(
    specs: Dict[str, Dict[str, Any]],
    *,
    fmp_exclude_tool_name_substrings: Optional[List[str]] = None,
) -> List[Any]:
    """One-shot tool discovery used when lifespan sessions are unavailable."""

    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        _logger.warning(
            "langchain-mcp-adapters not installed; Market Research agent "
            "will run without MCP tools. (%s)",
            exc,
        )
        return []

    exclude = fmp_exclude_tool_name_substrings
    all_tools: List[Any] = []
    for name, conn in specs.items():
        try:
            part = await load_mcp_tools(
                None,
                connection=conn,
                callbacks=None,
                server_name=name,
            )
        except Exception as exc:  # noqa: BLE001 -- never break the agent
            _logger.warning(
                "Failed to load Market Research MCP tools for server %s: %s",
                name,
                exc,
            )
            continue
        if name == "fmp" and exclude is not None:
            part = _filter_fmp_tools_by_name_substrings(
                part,
                exclude,
                server_key=name,
            )
        if name == "openbb":
            part = filter_openbb_mcp_tools_for_available_credentials(part)
        all_tools.extend(part)

    _logger.info(
        "Loaded %d Market Research MCP tool(s) from %d server(s): %s",
        len(all_tools),
        len(specs),
        sorted(specs.keys()),
    )
    return all_tools


async def start_market_research_mcp_sessions(cfg: Optional[AppConfig] = None) -> None:
    """Open persistent MCP client sessions for Market Research tools."""

    global _TOOLS_CACHE, _MCP_EXIT_STACK, _UVICORN_LOOP

    cfg = cfg or get_config()
    await stop_market_research_mcp_sessions()

    specs = _build_client_specs(cfg)
    if not specs:
        _TOOLS_CACHE = []
        _logger.info("Market Research MCP: no enabled servers.")
        return

    try:
        from langchain_mcp_adapters.sessions import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        _logger.warning("Market Research MCP disabled; adapters missing. (%s)", exc)
        _TOOLS_CACHE = []
        return

    stack = AsyncExitStack()
    all_tools: List[Any] = []
    try:
        for name, conn in specs.items():
            if shutil.which(str(conn.get("command") or "")) is None:
                _logger.warning(
                    "Market Research MCP server %s skipped: command %r not found.",
                    name,
                    conn.get("command"),
                )
                continue
            session = await stack.enter_async_context(create_session(conn))
            await session.initialize()
            part = await load_mcp_tools(session, connection=conn, server_name=name)
            if name == "fmp":
                part = _filter_fmp_tools_by_name_substrings(
                    part,
                    cfg.market_research.fmp_exclude_tool_name_substrings,
                    server_key=name,
                )
            if name == "openbb":
                part = filter_openbb_mcp_tools_for_available_credentials(part)
            all_tools.extend(part)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Market Research MCP persistent startup failed (specs=%s): %s",
            sorted(specs.keys()),
            exc,
        )
        await stack.aclose()
        _TOOLS_CACHE = []
        _MCP_EXIT_STACK = None
        return

    _MCP_EXIT_STACK = stack
    _TOOLS_CACHE = all_tools
    _UVICORN_LOOP = asyncio.get_running_loop()
    _logger.info("Market Research MCP: loaded %d tool(s).", len(all_tools))


async def stop_market_research_mcp_sessions() -> None:
    """Close persistent MCP sessions and clear caches.

    Same shutdown caveats as :func:`stop_portfolio_mcp_sessions`: lifespan
    cancellation during process exit must not surface as an unhandled
    ``CancelledError`` or anyio cancel-scope errors.
    """

    global _TOOLS_CACHE, _MCP_EXIT_STACK, _UVICORN_LOOP

    stack = _MCP_EXIT_STACK
    _MCP_EXIT_STACK = None
    _TOOLS_CACHE = None
    _UVICORN_LOOP = None
    if stack is None:
        return
    try:
        await stack.aclose()
    except BaseException as exc:  # noqa: BLE001
        if is_lifespan_shutdown_noise(exc):
            _logger.debug(
                "Market Research MCP shutdown: teardown noise during app exit (%s).",
                exc,
            )
        else:
            _logger.warning("Market Research MCP shutdown error: %s", exc)


def get_uvicorn_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the uvicorn event loop captured during MCP startup."""

    return _UVICORN_LOOP


def reset_market_research_tools_cache() -> None:
    """Clear cached tools; useful for tests."""

    global _TOOLS_CACHE
    _TOOLS_CACHE = None


def get_market_research_tools(cfg: Optional[AppConfig] = None) -> List[Any]:
    """Return cached or statelessly discovered Market Research MCP tools."""

    global _TOOLS_CACHE

    if _TOOLS_CACHE is not None:
        return list(_TOOLS_CACHE)

    cfg = cfg or get_config()
    specs = _build_client_specs(cfg)
    if not specs:
        _TOOLS_CACHE = []
        return []

    mr = cfg.market_research
    tools = asyncio.run(
        _abuild_tools_stateless(
            specs,
            fmp_exclude_tool_name_substrings=mr.fmp_exclude_tool_name_substrings,
        )
    )
    _TOOLS_CACHE = list(tools)
    return list(tools)
