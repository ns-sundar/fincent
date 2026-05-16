"""Unit tests for the Streamlit routing-details helpers.

These cover the pure functions that compute which agents contributed
to an answer and which MCP tools they reached for. The rendering
itself (``st.markdown`` / ``st.json``) is not exercised; the helpers
are pure dicts-in / lists-out so a plain pytest run is enough.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.web_app.streamlit_app import (
    _agents_involved,
    _agent_error_details,
    _data_sources_footprint_notes,
    _tool_limit_notes,
    _tools_called,
)


# ---------------------------------------------------------------------
# _agents_involved
# ---------------------------------------------------------------------


def test_agents_involved_single_specialist_returns_that_specialist_only():
    """Central -> Portfolio alone: the list must be exactly ['portfolio'].

    This is the behaviour the user asked for explicitly -- routing
    details should surface the specialist that actually produced the
    answer, not the central agent that merely dispatched.
    """
    plan: Dict[str, Any] = {"handled_by_central": False, "intents": ["portfolio"]}
    responses: List[Dict[str, Any]] = [
        {"agent": "portfolio", "content": "Your stock allocation is 70%.", "metadata": {}},
    ]
    assert _agents_involved(plan, responses) == ["portfolio"]


def test_agents_involved_market_research_specialist():
    """Market Research responses should display as the contributing agent."""
    plan: Dict[str, Any] = {
        "handled_by_central": False,
        "intents": ["market_research"],
    }
    responses: List[Dict[str, Any]] = [
        {
            "agent": "market_research",
            "content": "Nvidia has strong AI demand.",
            "metadata": {},
        },
    ]
    assert _agents_involved(plan, responses) == ["market_research"]


def test_agents_involved_goal_planning_specialist():
    """Goal Planning responses should display as the contributing agent."""
    plan: Dict[str, Any] = {
        "handled_by_central": False,
        "intents": ["goal_planning"],
    }
    responses: List[Dict[str, Any]] = [
        {
            "agent": "goal_planning",
            "content": "Your retirement plan needs a higher savings rate.",
            "metadata": {},
        },
    ]
    assert _agents_involved(plan, responses) == ["goal_planning"]


def test_agents_involved_central_direct_when_handled_by_central():
    """With a direct central answer, the list should show 'central'."""
    plan = {"handled_by_central": True, "intents": ["app_identity"]}
    responses = [{"agent": "central", "content": "Fincent is ...", "metadata": {}}]
    assert _agents_involved(plan, responses) == ["central"]


def test_agents_involved_empty_falls_back_to_central_when_flag_set():
    """If no responses are present but plan claims central-handling."""
    plan = {"handled_by_central": True, "intents": []}
    assert _agents_involved(plan, []) == ["central"]


def test_agents_involved_empty_all_around_returns_empty_list():
    """No responses AND plan did not claim central: empty list."""
    plan = {"handled_by_central": False, "intents": []}
    assert _agents_involved(plan, []) == []


def test_agents_involved_deduplicates_preserves_order():
    """Multiple responses from the same agent collapse to one entry."""
    plan = {"handled_by_central": False, "intents": ["portfolio", "qna"]}
    responses = [
        {"agent": "portfolio", "content": "a"},
        {"agent": "qna", "content": "b"},
        {"agent": "portfolio", "content": "c"},
    ]
    assert _agents_involved(plan, responses) == ["portfolio", "qna"]


# ---------------------------------------------------------------------
# _tools_called
# ---------------------------------------------------------------------


def test_tools_called_aggregates_tools_invoked_across_agents():
    """``tools_invoked`` from every agent response are concatenated in order."""
    responses = [
        {
            "agent": "portfolio",
            "content": "Your holdings include AAPL at $191.",
            "metadata": {"tools_invoked": ["rag_search", "equity_price_quote"]},
        },
    ]
    assert _tools_called(responses) == ["rag_search", "equity_price_quote"]


def test_tools_called_includes_market_research_tool_metadata():
    responses = [
        {
            "agent": "market_research",
            "content": "Sentiment is positive.",
            "metadata": {"tools_invoked": ["NEWS_SENTIMENT", "tavily_search"]},
        },
    ]
    assert _tools_called(responses) == ["NEWS_SENTIMENT", "tavily_search"]


def test_tools_called_empty_when_no_metadata():
    """Agents that omit ``tools_invoked`` contribute nothing."""
    responses = [
        {"agent": "qna", "content": "An ETF is ...", "metadata": {"sources": [{}]}},
    ]
    assert _tools_called(responses) == []


def test_tools_called_skips_non_list_tools_invoked():
    """A malformed ``tools_invoked`` value must never crash the UI."""
    responses = [
        {"agent": "portfolio", "content": "x", "metadata": {"tools_invoked": "rag_search"}},
        {"agent": "portfolio", "content": "y", "metadata": {"tools_invoked": None}},
        {"agent": "portfolio", "content": "z", "metadata": {"tools_invoked": ["ok_tool"]}},
    ]
    assert _tools_called(responses) == ["ok_tool"]


def test_tools_called_ignores_legacy_tool_names_only_payloads():
    """Bound-tool lists must not be mistaken for calls."""
    responses = [
        {
            "agent": "portfolio",
            "content": "x",
            "metadata": {"tool_names": ["equity_quote", "rag_search"]},
        },
    ]
    assert _tools_called(responses) == []


def test_data_sources_footprint_notes_dedupes_and_skips_invalid():
    from src.agents.openbb_mcp_invoke import FMP_FOOTPRINT_FREE_DATA_DISCLAIMER

    note = FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    responses: List[Dict[str, Any]] = [
        {
            "agent": "portfolio",
            "content": "x",
            "metadata": {"data_sources_note": note},
        },
        {
            "agent": "market_research",
            "content": "y",
            "metadata": {"data_sources_note": note},
        },
        {"agent": "qna", "content": "z", "metadata": {"data_sources_note": 123}},
    ]
    assert _data_sources_footprint_notes(responses) == [note]


def test_tool_limit_notes_describes_pruned_tools():
    responses = [
        {
            "agent": "market_research",
            "content": "ok",
            "metadata": {
                "tool_count": 128,
                "available_tool_count": 143,
                "dropped_tool_count": 15,
            },
        }
    ]
    assert _tool_limit_notes(responses) == [
        "market_research: bound 128 of 143 available tools; dropped 15 "
        "lower-priority tools to stay within provider limits."
    ]


def test_agent_error_details_collects_diagnostics():
    responses: List[Dict[str, Any]] = [
        {
            "agent": "market_research",
            "content": "I hit an internal error.",
            "metadata": {
                "error": True,
                "error_phase": "market_research_tool_loop",
                "error_type": "RuntimeError",
                "error_message": "upstream exploded",
                "error_traceback": "Traceback...",
            },
        },
        {"agent": "qna", "content": "ok", "metadata": {}},
    ]
    assert _agent_error_details(responses) == [
        {
            "agent": "market_research",
            "phase": "market_research_tool_loop",
            "type": "RuntimeError",
            "message": "upstream exploded",
            "traceback": "Traceback...",
        }
    ]


def test_tools_called_returns_empty_for_no_responses():
    """A central-handled turn has no specialist responses and no tools."""
    assert _tools_called([]) == []
