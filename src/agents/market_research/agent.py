"""Market Research specialist agent."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.llm_errors import (
    context_overflow_user_message,
    exception_diagnostic_metadata,
    is_context_overflow_error,
)
from src.agents.market_research.mcp_tools import (
    get_market_research_tools,
    get_uvicorn_loop,
)
from src.agents.openbb_mcp_invoke import (
    ainvoke_openbb_tool,
    fmp_sources_footprint_note,
    openbb_tool_text_suggests_fmp_paywall_fallback,
)
from src.agents.market_research.prompts import MARKET_RESEARCH_SYSTEM_PROMPT
from src.core.config import AppConfig
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_MAX_TOOL_ITERATIONS: int = 8


def _tools_by_name(tools: List[Any]) -> Dict[str, Any]:
    """Index a list of LangChain tools by name."""

    out: Dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            out[str(name)] = tool
    return out


_OPENBB_FREE_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "equity_price_quote": {"provider": "yfinance"},
    "equity_price_historical": {"provider": "yfinance"},
}


def _openbb_provider_token(raw: Any) -> str:
    """Normalise provider from schemas that use str or list."""

    if raw is None:
        return ""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str):
        return str(raw).strip().lower()
    return raw.strip().lower()


def _openbb_equity_fundamental_rejects_yfinance(tool_name: str) -> bool:
    """OpenBB fundamental routes reject yfinance provider values."""

    return str(tool_name).startswith("equity_fundamental_")


def _merge_openbb_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize provider for OpenBB MCP tools before invoke.

    * Fundamental tools (``equity_fundamental_*``) do not accept ``yfinance``.
      Models often pass it from general equity instructions, which triggers
      HTTP 422 — coerce to ``fmp``. Paid providers such as Intrinio are skipped
      later unless explicitly enabled.
    * Quote/history tools are pinned to ``yfinance`` when keys are disallowed
      (same behaviour as the Portfolio agent).
    """

    merged = dict(args or {})

    if _openbb_equity_fundamental_rejects_yfinance(tool_name):
        token = _openbb_provider_token(merged.get("provider"))
        if token in {"yfinance", "yf", "yahoo", "yahoofinance"}:
            merged["provider"] = "fmp"

    extra = _OPENBB_FREE_PROVIDER_DEFAULTS.get(tool_name)
    if not extra:
        return merged
    if os.environ.get("FINCENT_OPENBB_ALLOW_KEYED_PROVIDERS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return merged
    merged["provider"] = extra["provider"]
    return merged


async def _ainvoke_tool(tool: Any, args: Dict[str, Any]) -> str:
    """Invoke a LangChain tool and normalize the result to text."""

    name = str(getattr(tool, "name", "") or "")
    call_args = _merge_openbb_tool_args(name, dict(args or {}))
    return await ainvoke_openbb_tool(
        tool,
        call_args,
        tool_name=name,
        logger=_logger,
        log_context="Market Research",
    )


async def _arun_react_loop(
    llm: BaseChatModel,
    tools: List[Any],
    messages: List[Any],
    *,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> Tuple[AIMessage, List[str], bool]:
    """Run a minimal ReAct loop over the configured MCP tools."""

    from langchain_core.messages import ToolMessage

    bound = llm.bind_tools(tools) if tools else llm
    by_name = _tools_by_name(tools)

    working = list(messages)
    final: Optional[AIMessage] = None
    tools_invoked: List[str] = []
    saw_runtime_fmp_paywall: bool = False
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
            if tool is None:
                content = f"[tool error] unknown tool '{name}'"
            else:
                content = await _ainvoke_tool(tool, args)
            if openbb_tool_text_suggests_fmp_paywall_fallback(content):
                saw_runtime_fmp_paywall = True
            working.append(
                ToolMessage(content=content, tool_call_id=call_id, name=name)
            )

    if final is None:
        final = AIMessage(
            content=(
                "I'm sorry -- I wasn't able to produce a market research "
                "answer for that question. Please try rephrasing it."
            )
        )
    return final, tools_invoked, saw_runtime_fmp_paywall


def _run_async(coro) -> Any:
    """Execute an async coroutine from sync agent code."""

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
) -> AgentResponse:
    """Answer a non-personal market research question."""

    _logger.debug("Market Research agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()
    if tools is None:
        tools = get_market_research_tools(cfg)
    tool_names = [getattr(t, "name", "?") for t in tools]

    messages: List[Any] = [
        SystemMessage(content=MARKET_RESEARCH_SYSTEM_PROMPT),
        *list(history or []),
        HumanMessage(content=query),
    ]

    tools_invoked: List[str] = []
    saw_runtime_fmp_paywall = False
    error_meta: Dict[str, Any] = {}
    try:
        final_ai, tools_invoked, saw_runtime_fmp_paywall = _run_async(
            _arun_react_loop(llm, tools, messages)
        )
    except Exception as exc:  # noqa: BLE001 -- surface as agent response
        _logger.exception("Market Research agent tool loop failed")
        error_meta = exception_diagnostic_metadata(
            exc,
            phase="market_research_tool_loop",
        )
        if is_context_overflow_error(exc):
            final_ai = AIMessage(
                content=context_overflow_user_message("market research question")
            )
        else:
            final_ai = AIMessage(
                content=(
                    "I hit an internal error while working through that market "
                    "research question. Please try a narrower request or ask again."
                )
            )

    meta: Dict[str, Any] = {
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tools_invoked": tools_invoked,
    }
    meta.update(error_meta)
    note = fmp_sources_footprint_note(
        saw_runtime_fmp_paywall_signal=saw_runtime_fmp_paywall,
    )
    if note:
        meta["data_sources_note"] = note

    return AgentResponse(
        agent=AgentName.MARKET_RESEARCH,
        content=str(final_ai.content).strip(),
        metadata=meta,
    )
