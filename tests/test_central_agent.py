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
            "intents": ["agent_two", "qna"],
            "rationale": "",
        }
    )
    plan = plan_route("Test", llm=_fake_llm(payload))
    assert Intent.AGENT_TWO not in plan.intents
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
            "intents": ["agent_two", "agent_three"],
        }
    )
    plan = plan_route("Test", llm=_fake_llm(payload))
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
