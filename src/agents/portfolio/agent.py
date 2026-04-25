"""Portfolio specialist agent.

Given a natural-language question about the user's own portfolio, the
agent renders a compact JSON/markdown snapshot of the data (accounts
sorted by balance, asset-class allocation, ten most recent
transactions) into a system prompt and runs a ReAct tool-calling
loop. The agent has access to two families of MCP tools (loaded once
per process from ``cfg.portfolio.tools``):

* **OpenBB MCP tools** -- real-world financial data (live quotes,
  company news, ETF holdings, economic indicators, ...) via the
  OpenBB Platform's official MCP server (``openbb-mcp-server``).
* **Fincent RAG MCP tool** -- ``rag_search`` over the same FAISS
  vector_db the Q&A agent uses, exposed via
  :mod:`src.rag.mcp_server` in stdio mode.

If no tools load successfully (MCP adapters missing, every server
disabled, or subprocess launch failed), the agent degrades
gracefully to a single LLM call with the snapshot as grounding --
identical to the pre-tooling behaviour.

The same ``PortfolioSnapshot`` object feeds the Streamlit right-hand
graphics panel, so chat answers cannot drift from what the user sees.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.portfolio.loader import (
    AccountSummary,
    PortfolioSnapshot,
    load_portfolio,
)
from src.agents.portfolio.mcp_tools import get_portfolio_tools, get_uvicorn_loop
from src.agents.portfolio.prompts import (
    PORTFOLIO_CONTEXT_TEMPLATE,
    PORTFOLIO_SYSTEM_PROMPT,
)
from src.core.config import AppConfig
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.utils.logging import get_logger

_logger = get_logger(__name__)

# Hard cap on the number of tool-call iterations the ReAct loop may
# take for one user turn. Prevents a pathological model from spinning
# up dozens of OpenBB calls in a single answer.
_MAX_TOOL_ITERATIONS: int = 6


# ---------------------------------------------------------------------
# Snapshot -> LLM context
# ---------------------------------------------------------------------


def _account_to_dict(acc: AccountSummary) -> Dict[str, Any]:
    """Render a single account summary as JSON-friendly data."""
    return {
        "account_id": acc.account_id,
        "name": acc.name,
        "type": acc.type,
        "broker": acc.broker,
        "currency": acc.currency,
        "balance": acc.balance,
        "holdings": acc.holdings,
    }


def _build_portfolio_block(snapshot: PortfolioSnapshot) -> str:
    """Render the snapshot as a JSON block for the LLM to ground on.

    The agent receives **every** transaction (not just the 10 shown in
    the UI) plus the total count, so questions like "how many
    transactions do I have?" or "what was my first deposit?" can be
    answered correctly.
    """
    payload: Dict[str, Any] = {
        "as_of": snapshot.as_of,
        "total_balance": snapshot.total_balance,
        "allocation": snapshot.allocation,
        "accounts_sorted_by_balance_desc": [
            _account_to_dict(a) for a in snapshot.accounts
        ],
        "transaction_count": snapshot.transaction_count,
        "transactions_newest_first": snapshot.all_transactions,
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------
# Tool-calling loop (async)
# ---------------------------------------------------------------------


def _tools_by_name(tools: List[Any]) -> Dict[str, Any]:
    """Index a list of LangChain tools by their ``name`` attribute."""
    out: Dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            out[str(name)] = tool
    return out


# When ``provider`` is omitted, OpenBB often defaults to the first
# provider alphabetically (e.g. FMP) which requires paid API keys.
# yfinance works without keys for many US listings.
_OPENBB_FREE_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "equity_price_quote": {"provider": "yfinance"},
    "equity_price_historical": {"provider": "yfinance"},
}


def _merge_openbb_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Pin OpenBB to yfinance for key equity tools (no API key).

    The LLM or MCP tool schema may supply ``provider`` as ``\"fmp\"`` or
    ``[\"fmp\"]`` (OpenBB's alphabetical default). That bypasses
    ``user_settings`` and triggers ``fmp_api_key`` errors. Fincent forces
    yfinance for these tool names unless the operator opts out via
    ``FINCENT_OPENBB_ALLOW_KEYED_PROVIDERS=true``.
    """
    extra = _OPENBB_FREE_PROVIDER_DEFAULTS.get(tool_name)
    if not extra:
        return args
    if os.environ.get("FINCENT_OPENBB_ALLOW_KEYED_PROVIDERS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return args
    merged = dict(args)
    merged["provider"] = extra["provider"]
    return merged


async def _ainvoke_tool(tool: Any, args: Dict[str, Any]) -> str:
    """Invoke a LangChain tool and normalise the result to a string."""
    name = str(getattr(tool, "name", "") or "")
    call_args = _merge_openbb_tool_args(name, dict(args or {}))
    try:
        result = await tool.ainvoke(call_args)
    except Exception as exc:  # noqa: BLE001 -- feed errors back to the LLM
        _logger.warning("Tool %s failed: %s", name or "?", exc)
        return f"[tool error] {type(exc).__name__}: {exc}"
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


async def _arun_react_loop(
    llm: BaseChatModel,
    tools: List[Any],
    messages: List[Any],
    *,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> Tuple[AIMessage, List[str]]:
    """Run a minimal ReAct tool-calling loop.

    Binds the tools to the chat model, asks for a reply, executes any
    requested tool calls, and feeds the ``ToolMessage`` results back
    until the model stops asking for tools (or we hit
    ``max_iterations``).

    The loop is inlined here -- rather than delegated to
    ``langgraph.prebuilt.create_react_agent`` -- so that:
      * the whole agent stays sync-compatible (we own the event loop),
      * we can rely on the tools list possibly being empty without
        pulling in an extra LangGraph compile path per call,
      * and test doubles (``FakeListChatModel``) that never emit
        ``tool_calls`` short-circuit cleanly on the first iteration.
    """
    from langchain_core.messages import ToolMessage

    bound = llm.bind_tools(tools) if tools else llm
    by_name = _tools_by_name(tools)

    working = list(messages)
    final: Optional[AIMessage] = None
    tools_invoked: List[str] = []
    for iteration in range(max_iterations):
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
            working.append(
                ToolMessage(content=content, tool_call_id=call_id, name=name)
            )

    if final is None:
        # Pathological: max_iterations==0 or the model raised before
        # producing any AI message. Return a polite placeholder so
        # ``answer`` does not crash the workflow.
        final = AIMessage(
            content=(
                "I'm sorry -- I wasn't able to produce a portfolio answer "
                "for that question. Please try rephrasing it."
            )
        )
    return final, tools_invoked


def _run_async(coro) -> Any:
    """Execute an async coroutine from sync code.

    FastAPI dispatches sync routes to a threadpool (no running loop in
    the worker thread).  When persistent MCP sessions are live, their
    asyncio streams belong to the uvicorn event loop; submitting the
    coroutine to that same loop via ``run_coroutine_threadsafe`` prevents
    cross-loop I/O deadlocks.  Falls back to ``asyncio.run`` (fresh loop)
    when no uvicorn loop is known (e.g. CLI / no lifespan) and to a
    worker thread when a loop is already running (pytest-asyncio).
    """
    try:
        asyncio.get_running_loop()
        # A loop is already running (pytest-asyncio / notebook): spin a
        # worker thread with its own loop to avoid nested-run errors.
        def _target() -> Any:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_target).result()

    except RuntimeError:
        # No running loop in this thread (normal FastAPI threadpool path).
        uvicorn_loop = get_uvicorn_loop()
        if uvicorn_loop is not None and uvicorn_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, uvicorn_loop)
            return future.result()
        return asyncio.run(coro)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def answer(
    query: str,
    *,
    history: Optional[List[Any]] = None,
    llm: Optional[BaseChatModel] = None,
    cfg: Optional[AppConfig] = None,
    snapshot: Optional[PortfolioSnapshot] = None,
    tools: Optional[List[Any]] = None,
) -> AgentResponse:
    """Answer a portfolio question grounded in the snapshot.

    Args:
        query:    User question (routed here by the central planner
                  when it detects a portfolio intent).
        llm:      Optional chat model (tests).
        cfg:      Optional config override (tests).
        snapshot: Optional pre-loaded snapshot (tests). When ``None``
                  the loader's on-disk cache is used.
        tools:    Optional pre-loaded MCP tool list (tests). When
                  ``None`` the cached loader at
                  :func:`src.agents.portfolio.mcp_tools.get_portfolio_tools`
                  is used. Pass ``[]`` to force the tool-free path.

    Returns:
        An ``AgentResponse`` attributed to the portfolio agent with
        rollup metadata so the UI can surface "which data was shown
        to the agent" and which tools the agent reached for.
    """
    _logger.debug("Portfolio agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()
    snap = snapshot or load_portfolio(cfg)

    if tools is None:
        tools = get_portfolio_tools(cfg)
    tool_names = [getattr(t, "name", "?") for t in tools]

    context = PORTFOLIO_CONTEXT_TEMPLATE.format(
        portfolio_block=_build_portfolio_block(snap)
    )
    messages: List[Any] = [
        SystemMessage(content=PORTFOLIO_SYSTEM_PROMPT),
        SystemMessage(content=context),
        *list(history or []),
        HumanMessage(content=query),
    ]

    tools_invoked: List[str] = []
    try:
        final_ai, tools_invoked = _run_async(
            _arun_react_loop(llm, tools, messages)
        )
    except Exception as exc:  # noqa: BLE001 -- surface to aggregator
        _logger.exception("Portfolio agent tool loop failed")
        final_ai = AIMessage(
            content=(
                "I hit an internal error while working through that "
                f"portfolio question: {type(exc).__name__}: {exc}."
            )
        )

    body = str(final_ai.content).strip()

    metadata: Dict[str, Any] = {
        "account_count": len(snap.accounts),
        "total_balance": snap.total_balance,
        "allocation": snap.allocation,
        "as_of": snap.as_of,
        "transaction_count": snap.transaction_count,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "tools_invoked": tools_invoked,
    }
    return AgentResponse(
        agent=AgentName.PORTFOLIO,
        content=body,
        metadata=metadata,
    )
