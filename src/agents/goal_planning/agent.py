"""Goal Planning specialist agent."""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.goal_planning.context import portfolio_summary_json
from src.agents.goal_planning.mcp_tools import get_goal_planning_tools, get_uvicorn_loop
from src.agents.goal_planning.prompts import (
    GOAL_PLANNING_CONTEXT_TEMPLATE,
    GOAL_PLANNING_SYSTEM_PROMPT,
)
from src.agents.llm_errors import (
    context_overflow_user_message,
    exception_diagnostic_metadata,
    is_context_overflow_error,
)
from src.agents.openbb_mcp_invoke import ainvoke_openbb_tool
from src.agents.portfolio.loader import PortfolioSnapshot, load_portfolio
from src.core.config import AppConfig, get_config
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_MAX_TOOL_ITERATIONS: int = 6


def _tools_by_name(tools: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            out[str(name)] = tool
    return out


def _infer_goal_type(query: str) -> str:
    text = query.lower()
    if any(term in text for term in ("retire", "retirement")):
        return "retirement"
    if any(term in text for term in ("college", "529", "tuition")):
        return "college"
    if any(term in text for term in ("house", "home", "downpayment", "down payment")):
        return "home_purchase"
    if "vacation" in text or "travel" in text:
        return "vacation"
    if any(term in text for term in ("recession", "drops", "drop", "stress")):
        return "stress_test"
    return "general_goal"


def _extract_first_money(query: str) -> Optional[float]:
    match = re.search(r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|m|/month|per month)?", query, re.I)
    if not match:
        return None
    amount = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return amount


def _extract_years(query: str) -> Optional[float]:
    text = query.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(years?|yrs?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*months?", text)
    if match:
        return float(match.group(1)) / 12.0
    if "next summer" in text:
        return 1.0
    return None


async def _ainvoke_tool(tool: Any, args: Dict[str, Any]) -> str:
    name = str(getattr(tool, "name", "") or "")
    return await ainvoke_openbb_tool(
        tool,
        dict(args or {}),
        tool_name=name,
        logger=_logger,
        log_context="Goal Planning",
    )


async def _arun_react_loop(
    llm: BaseChatModel,
    tools: List[Any],
    messages: List[Any],
    *,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> Tuple[AIMessage, List[str]]:
    """Run the Goal Planning ReAct loop."""

    from langchain_core.messages import ToolMessage

    bound = llm.bind_tools(tools) if tools else llm
    by_name = _tools_by_name(tools)
    working = list(messages)
    final: Optional[AIMessage] = None
    tools_invoked: List[str] = []

    for _ in range(max_iterations):
        response = await bound.ainvoke(working)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(getattr(response, "content", response)))
        working.append(response)
        final = response
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            name = str(call.get("name") or "")
            if name:
                tools_invoked.append(name)
            args = call.get("args") or {}
            call_id = str(call.get("id") or name or "tool")
            tool = by_name.get(name)
            content = (
                f"[tool error] unknown tool '{name}'"
                if tool is None
                else await _ainvoke_tool(tool, args)
            )
            working.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

    if final is None:
        final = AIMessage(
            content=(
                "I wasn't able to produce a goal-planning answer. Please try "
                "adding a goal amount, timeline, and monthly contribution."
            )
        )
    return final, tools_invoked


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


def answer(
    query: str,
    *,
    history: Optional[List[Any]] = None,
    llm: Optional[BaseChatModel] = None,
    cfg: Optional[AppConfig] = None,
    tools: Optional[List[Any]] = None,
    snapshot: Optional[PortfolioSnapshot] = None,
) -> AgentResponse:
    """Answer a portfolio-grounded financial goal-planning question."""

    _logger.debug("Goal Planning agent invoked with query: %s", query)
    effective_cfg = cfg or get_config()
    llm = llm or get_default_chat_model()
    snap = snapshot or load_portfolio(effective_cfg)
    portfolio_json = portfolio_summary_json(effective_cfg, snapshot=snap)
    if tools is None:
        tools = get_goal_planning_tools(effective_cfg)
    tool_names = [getattr(t, "name", "?") for t in tools]

    context = GOAL_PLANNING_CONTEXT_TEMPLATE.format(
        portfolio_summary=portfolio_json,
        inflation_rate=effective_cfg.goal_planning.default_inflation_rate,
        risk_free_rate=effective_cfg.goal_planning.default_risk_free_rate,
        simulations=effective_cfg.goal_planning.monte_carlo_simulations,
        max_years=effective_cfg.goal_planning.max_projection_years,
    )
    messages: List[Any] = [
        SystemMessage(content=GOAL_PLANNING_SYSTEM_PROMPT),
        SystemMessage(content=context),
        *list(history or []),
        HumanMessage(content=query),
    ]

    tools_invoked: List[str] = []
    error_meta: Dict[str, Any] = {}
    try:
        final_ai, tools_invoked = _run_async(_arun_react_loop(llm, tools, messages))
    except Exception as exc:  # noqa: BLE001 -- user-safe response
        _logger.exception("Goal Planning agent tool loop failed")
        error_meta = exception_diagnostic_metadata(exc, phase="goal_planning_tool_loop")
        if is_context_overflow_error(exc):
            final_ai = AIMessage(
                content=context_overflow_user_message("goal-planning question")
            )
        else:
            final_ai = AIMessage(
                content=(
                    "I hit an internal error while working through that goal-planning "
                    "question. Please try a narrower request or ask again."
                )
            )

    meta: Dict[str, Any] = {
        "goal_type": _infer_goal_type(query),
        "time_horizon_years": _extract_years(query),
        "target_amount": _extract_first_money(query),
        "portfolio_allocation": snap.allocation,
        "portfolio_total_balance": snap.total_balance,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tools_invoked": tools_invoked,
    }
    meta.update(error_meta)
    return AgentResponse(
        agent=AgentName.GOAL_PLANNING,
        content=str(final_ai.content).strip(),
        metadata=meta,
    )

