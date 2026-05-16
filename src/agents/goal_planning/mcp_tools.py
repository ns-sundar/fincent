"""MCP tool loading for the Goal Planning agent."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from src.agents.goal_planning.context import load_portfolio_summary
from src.agents.goal_planning.financial_math import calculate_fv, run_monte_carlo
from src.agents.openbb_mcp_invoke import filter_openbb_mcp_tools_for_available_credentials
from src.core.config import AppConfig, PortfolioMcpServerSpec, get_config
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_TOOLS_CACHE: Optional[List[Any]] = None
_MCP_EXIT_STACK: Optional[AsyncExitStack] = None
_UVICORN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _json_tool_output(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _portfolio_defaults(cfg: Optional[AppConfig] = None) -> Dict[str, Any]:
    summary = load_portfolio_summary(cfg)
    return {
        "portfolio_mix": summary.get("allocation") or {"unknown": 1.0},
        "starting_balance": float(summary.get("total_balance") or 0.0),
    }


def _build_native_goal_tools(cfg: Optional[AppConfig] = None) -> List[Any]:
    """Build in-process fallback tools with the same names as the MCP server."""

    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        _logger.warning("Goal Planning native tools disabled; LangChain missing. (%s)", exc)
        return []

    cfg = cfg or get_config()

    @tool("calculate_fv")
    def calculate_fv_tool(pv: float, rate: float, nper: int, pmt: float = 0.0) -> str:
        """Calculate future value with end-of-period contributions."""

        return _json_tool_output(
            {
                "future_value": calculate_fv(pv=pv, rate=rate, nper=nper, pmt=pmt),
                "pv": pv,
                "rate": rate,
                "nper": nper,
                "pmt": pmt,
            }
        )

    @tool("run_monte_carlo")
    def run_monte_carlo_tool(
        years: float,
        monthly_contribution: float,
        target: float,
        portfolio_mix: Optional[Dict[str, Any]] = None,
        starting_balance: float = 0.0,
        simulations: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> str:
        """Run a Monte Carlo goal-success simulation."""

        sim_count = int(simulations or cfg.goal_planning.monte_carlo_simulations)
        sim_count = max(1, min(sim_count, cfg.goal_planning.max_monte_carlo_simulations))
        bounded_years = max(0.0, min(float(years), float(cfg.goal_planning.max_projection_years)))
        if not portfolio_mix:
            defaults = _portfolio_defaults(cfg)
            portfolio_mix = defaults["portfolio_mix"]
            if not starting_balance:
                starting_balance = defaults["starting_balance"]
        return _json_tool_output(
            run_monte_carlo(
                portfolio_mix=portfolio_mix,
                years=bounded_years,
                monthly_contribution=monthly_contribution,
                target=target,
                starting_balance=starting_balance,
                simulations=sim_count,
                seed=seed,
            )
        )

    @tool("portfolio_summary")
    def portfolio_summary_tool() -> str:
        """Return a compact JSON summary of the user's portfolio."""

        return _json_tool_output(load_portfolio_summary(cfg))

    return [calculate_fv_tool, run_monte_carlo_tool, portfolio_summary_tool]


def _append_native_goal_tools(tools: List[Any], cfg: Optional[AppConfig] = None) -> None:
    existing = {str(getattr(t, "name", "") or "") for t in tools}
    native = [
        t
        for t in _build_native_goal_tools(cfg)
        if str(getattr(t, "name", "") or "") not in existing
    ]
    if native:
        tools.extend(native)
        _logger.info("Goal Planning: added %d native planning tool(s).", len(native))


def _server_config(spec: PortfolioMcpServerSpec) -> Optional[Dict[str, Any]]:
    if not spec.enabled:
        return None
    command = (spec.command or "").strip()
    if not command:
        return None
    return {
        "transport": "stdio",
        "command": command,
        "args": list(spec.args or []),
        "env": {**dict(os.environ), **dict(spec.env or {})},
    }


def _build_client_specs(cfg: AppConfig) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    tools_cfg = cfg.goal_planning.tools
    for server_name, spec in (
        ("internal", tools_cfg.internal),
        ("openbb", tools_cfg.openbb),
    ):
        entry = _server_config(spec)
        if entry is not None:
            specs[server_name] = entry
    return specs


async def _abuild_tools_stateless(
    specs: Dict[str, Dict[str, Any]],
    cfg: AppConfig,
) -> List[Any]:
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        _logger.warning("Goal Planning MCP disabled; adapters missing. (%s)", exc)
        tools: List[Any] = []
        _append_native_goal_tools(tools, cfg)
        return tools

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
            _logger.warning("Failed to load Goal Planning MCP tools for %s: %s", name, exc)
            continue
        if name == "openbb":
            part = filter_openbb_mcp_tools_for_available_credentials(part)
        all_tools.extend(part)
    _append_native_goal_tools(all_tools, cfg)
    return all_tools


async def start_goal_planning_mcp_sessions(cfg: Optional[AppConfig] = None) -> None:
    """Open persistent MCP sessions for Goal Planning tools."""

    global _TOOLS_CACHE, _MCP_EXIT_STACK, _UVICORN_LOOP
    cfg = cfg or get_config()
    await stop_goal_planning_mcp_sessions()
    specs = _build_client_specs(cfg)
    if not specs:
        tools: List[Any] = []
        _append_native_goal_tools(tools, cfg)
        _TOOLS_CACHE = tools
        return

    try:
        from langchain_mcp_adapters.sessions import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        _logger.warning("Goal Planning MCP disabled; adapters missing. (%s)", exc)
        tools = []
        _append_native_goal_tools(tools, cfg)
        _TOOLS_CACHE = tools
        return

    stack = AsyncExitStack()
    all_tools: List[Any] = []
    loaded_servers: List[str] = []
    failed_servers: List[str] = []

    for name, conn in specs.items():
        if shutil.which(str(conn.get("command") or "")) is None:
            _logger.warning(
                "Goal Planning MCP server %s skipped: command %r not found.",
                name,
                conn.get("command"),
            )
            failed_servers.append(name)
            continue
        server_stack = AsyncExitStack()
        try:
            session = await server_stack.enter_async_context(create_session(conn))
            await session.initialize()
            part = await load_mcp_tools(session, connection=conn, server_name=name)
            if name == "openbb":
                part = filter_openbb_mcp_tools_for_available_credentials(part)
            stack.push_async_exit(server_stack.pop_all())
            all_tools.extend(part)
            loaded_servers.append(name)
        except Exception as exc:  # noqa: BLE001
            failed_servers.append(name)
            _logger.warning("Goal Planning MCP server %s skipped: %s", name, exc)
            await server_stack.aclose()

    _append_native_goal_tools(all_tools, cfg)
    _MCP_EXIT_STACK = stack
    _TOOLS_CACHE = all_tools
    _UVICORN_LOOP = asyncio.get_running_loop()
    _logger.info(
        "Goal Planning MCP: loaded %d tool(s) from %d/%d server(s): loaded=%s failed=%s",
        len(all_tools),
        len(loaded_servers),
        len(specs),
        loaded_servers,
        failed_servers,
    )


async def stop_goal_planning_mcp_sessions() -> None:
    """Close persistent Goal Planning MCP sessions."""

    global _TOOLS_CACHE, _MCP_EXIT_STACK, _UVICORN_LOOP
    stack = _MCP_EXIT_STACK
    _MCP_EXIT_STACK = None
    _TOOLS_CACHE = None
    _UVICORN_LOOP = None
    if stack is not None:
        await stack.aclose()


def reset_goal_planning_tools_cache() -> None:
    """Clear cached tools without closing any active sessions."""

    global _TOOLS_CACHE
    _TOOLS_CACHE = None


def get_uvicorn_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _UVICORN_LOOP


def _run_async(coro) -> Any:
    try:
        asyncio.get_running_loop()

        def _target() -> Any:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_target).result()
    except RuntimeError:
        uvicorn_loop = get_uvicorn_loop()
        if uvicorn_loop is not None and uvicorn_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, uvicorn_loop)
            return future.result()
        return asyncio.run(coro)


def get_goal_planning_tools(cfg: Optional[AppConfig] = None) -> List[Any]:
    """Return cached Goal Planning tools or discover them once."""

    global _TOOLS_CACHE
    if _TOOLS_CACHE is not None:
        return list(_TOOLS_CACHE)
    cfg = cfg or get_config()
    specs = _build_client_specs(cfg)
    tools = _run_async(_abuild_tools_stateless(specs, cfg))
    _TOOLS_CACHE = tools
    return list(tools)

