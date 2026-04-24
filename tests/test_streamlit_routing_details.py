"""Unit tests for the Streamlit routing-details helpers.

These cover the pure functions that compute which agents contributed
to an answer and which MCP tools they reached for. The rendering
itself (``st.markdown`` / ``st.json``) is not exercised; the helpers
are pure dicts-in / lists-out so a plain pytest run is enough.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.web_app.streamlit_app import _agents_involved, _tools_called


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


def test_agents_involved_central_direct_when_handled_by_central():
    """With a direct central answer, the list should show 'central'."""
    plan = {"handled_by_central": True, "intents": ["app_info"]}
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


def test_tools_called_aggregates_tool_names_across_agents():
    """Tool lists from every agent response are concatenated in order."""
    responses = [
        {
            "agent": "portfolio",
            "content": "Your holdings include AAPL at $191.",
            "metadata": {"tool_names": ["rag_search", "equity_price_quote"]},
        },
    ]
    assert _tools_called(responses) == ["rag_search", "equity_price_quote"]


def test_tools_called_empty_when_no_metadata():
    """Agents that omit ``tool_names`` contribute nothing."""
    responses = [
        {"agent": "qna", "content": "An ETF is ...", "metadata": {"sources": [{}]}},
    ]
    assert _tools_called(responses) == []


def test_tools_called_skips_non_list_tool_names():
    """A malformed ``tool_names`` value must never crash the UI."""
    responses = [
        {"agent": "portfolio", "content": "x", "metadata": {"tool_names": "rag_search"}},
        {"agent": "portfolio", "content": "y", "metadata": {"tool_names": None}},
        {"agent": "portfolio", "content": "z", "metadata": {"tool_names": ["ok_tool"]}},
    ]
    assert _tools_called(responses) == ["ok_tool"]


def test_tools_called_returns_empty_for_no_responses():
    """A central-handled turn has no specialist responses and no tools."""
    assert _tools_called([]) == []
