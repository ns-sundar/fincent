"""Unit tests for the central orchestrator agent."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.central import aggregate, answer_directly, plan_route
from src.core.schemas import AgentName, AgentResponse, Intent


def _fake_llm(*responses: str) -> FakeListChatModel:
    """Build a fake chat model that emits ``responses`` in order."""
    return FakeListChatModel(responses=list(responses))


# ---------------------------------------------------------------------
# plan_route
# ---------------------------------------------------------------------


def test_plan_route_central_handled_app_info():
    """An app-info classification should set handled_by_central=True."""
    payload = json.dumps(
        {
            "handled_by_central": True,
            "intents": ["app_info"],
            "rationale": "Asking what the app does.",
        }
    )
    plan = plan_route("What can this app do?", llm=_fake_llm(payload))

    assert plan.handled_by_central is True
    assert plan.intents == [Intent.APP_INFO]


def test_plan_route_specialist_qna():
    """A qna intent should be preserved and dispatched."""
    payload = json.dumps(
        {
            "handled_by_central": False,
            "intents": ["qna"],
            "rationale": "Generic financial question.",
        }
    )
    plan = plan_route("What is an ETF?", llm=_fake_llm(payload))

    assert plan.handled_by_central is False
    assert plan.intents == [Intent.QNA]


def test_plan_route_personal_finance_routes_to_portfolio_only():
    """Portfolio-tagged questions must route ONLY to the Portfolio agent.

    Under the new orchestration rules, any question that touches the
    user's own holdings must go to the Portfolio agent alone (even if
    it also asks a general concept) -- the Portfolio agent has its
    own RAG tool for generic context, so fanning out to Q&A would
    just duplicate work.
    """
    payload = json.dumps(
        {
            "handled_by_central": False,
            "intents": ["portfolio"],
            "rationale": "User references 'my holdings'.",
        }
    )
    plan = plan_route(
        "Explain dividend taxation for my current holdings",
        llm=_fake_llm(payload),
    )
    assert plan.handled_by_central is False
    assert plan.intents == [Intent.PORTFOLIO]


def test_plan_route_non_financial_small_talk_handled_centrally():
    """Non-financial chit-chat should be answered by the central agent."""
    payload = json.dumps(
        {
            "handled_by_central": True,
            "intents": ["user_generic"],
            "rationale": "Greeting.",
        }
    )
    plan = plan_route("Hi there!", llm=_fake_llm(payload))
    assert plan.handled_by_central is True
    assert plan.intents == [Intent.USER_GENERIC]


def test_plan_route_strips_code_fence():
    """The router must tolerate ```json ... ``` fences."""
    payload = (
        "```json\n"
        + json.dumps({"handled_by_central": False, "intents": ["qna"]})
        + "\n```"
    )
    plan = plan_route("Test", llm=_fake_llm(payload))
    assert plan.intents == [Intent.QNA]


def test_plan_route_drops_disabled_specialists():
    """Disabled specialists must be filtered out of the plan."""
    payload = json.dumps(
        {
            "handled_by_central": False,
            "intents": ["agent_three", "qna"],
            "rationale": "",
        }
    )
    plan = plan_route("Test", llm=_fake_llm(payload))
    assert Intent.AGENT_THREE not in plan.intents
    assert Intent.QNA in plan.intents


def test_plan_route_falls_back_on_bad_json():
    """Malformed JSON should fall back to qna instead of crashing."""
    plan = plan_route("Test", llm=_fake_llm("not really json {{"))
    assert plan.intents == [Intent.QNA]
    assert plan.handled_by_central is False


def test_plan_route_falls_back_when_no_enabled_specialist():
    """If the router picks only disabled specialists, fall back to qna."""
    payload = json.dumps(
        {
            "handled_by_central": False,
            "intents": ["agent_three", "agent_four"],
        }
    )
    plan = plan_route("Test", llm=_fake_llm(payload))
    assert plan.intents == [Intent.QNA]


def test_plan_route_honors_intent_hint():
    """An explicit caller hint bypasses the LLM classifier."""
    # No responses primed: if the LLM were called, FakeListChatModel
    # would raise. This proves the hint short-circuited the router.
    plan = plan_route(
        "show my portfolio",
        llm=_fake_llm(),
        intent_hint=Intent.PORTFOLIO,
    )
    assert plan.handled_by_central is False
    assert plan.intents == [Intent.PORTFOLIO]


def test_plan_route_ignores_hint_for_disabled_specialist():
    """A hint pointing at a disabled agent should be ignored."""
    payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"]}
    )
    plan = plan_route(
        "Test",
        llm=_fake_llm(payload),
        intent_hint=Intent.AGENT_THREE,
    )
    assert plan.intents == [Intent.QNA]


# ---------------------------------------------------------------------
# answer_directly
# ---------------------------------------------------------------------


def test_answer_directly_attribution():
    """Direct answers must be attributed to the central agent."""
    response = answer_directly("Hi!", llm=_fake_llm("Hello there."))
    assert response.agent == AgentName.CENTRAL
    assert response.content == "Hello there."
    assert response.metadata.get("mode") == "direct"


# ---------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------


def test_aggregate_single_response_returned_verbatim():
    """With one specialist response, no LLM call is made."""
    only = AgentResponse(agent=AgentName.QNA, content="An ETF is ...")
    out = aggregate("What is an ETF?", [only])
    assert out == "An ETF is ..."


def test_aggregate_empty_responses_returns_polite_message():
    """No specialist responses should produce a graceful fallback."""
    out = aggregate("Anything?", [])
    assert "wasn't able to produce" in out


def test_aggregate_multiple_responses_invokes_llm():
    """Multiple responses are merged via the LLM call."""
    responses = [
        AgentResponse(agent=AgentName.QNA, content="ETFs trade on exchanges."),
        AgentResponse(agent=AgentName.QNA, content="Mutual funds price daily."),
    ]
    merged = aggregate(
        "Compare ETFs and mutual funds",
        responses,
        llm=_fake_llm("ETFs trade intraday; mutual funds price once a day."),
    )
    assert "ETFs" in merged or "mutual funds" in merged
