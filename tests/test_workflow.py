"""Integration tests for the LangGraph workflow.

We patch the LLM factory so the graph runs end-to-end without any
network or API key.
"""

from __future__ import annotations

import json
from itertools import cycle
from typing import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.core.schemas import AgentName, Intent, QueryRequest


class _ScriptedFakeLLM(FakeListChatModel):
    """FakeListChatModel that cycles indefinitely over its responses."""

    def __init__(self, responses):
        super().__init__(responses=list(responses))
        # Replace the iterator with a cycling one so the graph can call
        # the model an arbitrary number of times during one test.
        self._cycle: Iterator[str] = cycle(self.responses)

    def _call(self, *args, **kwargs):  # type: ignore[override]
        return next(self._cycle)


@pytest.fixture
def patch_llm(monkeypatch):
    """Return a helper that monkey-patches ``get_default_chat_model``."""

    def _apply(*responses: str) -> _ScriptedFakeLLM:
        fake = _ScriptedFakeLLM(responses)
        # Patch every import site so all agents pick up the fake.
        from src.agents.central import agent as central_module
        from src.agents.goal_planning import agent as goal_planning_module
        from src.agents.market_research import agent as market_research_module
        from src.agents.portfolio import agent as portfolio_module
        from src.agents.qna import agent as qna_module

        monkeypatch.setattr(central_module, "get_default_chat_model", lambda: fake)
        monkeypatch.setattr(
            market_research_module, "get_default_chat_model", lambda: fake
        )
        monkeypatch.setattr(qna_module, "get_default_chat_model", lambda: fake)
        monkeypatch.setattr(portfolio_module, "get_default_chat_model", lambda: fake)
        monkeypatch.setattr(
            goal_planning_module, "get_default_chat_model", lambda: fake
        )
        monkeypatch.setattr(
            market_research_module, "get_market_research_tools", lambda _cfg=None: []
        )
        monkeypatch.setattr(
            goal_planning_module, "get_goal_planning_tools", lambda _cfg=None: []
        )
        return fake

    return _apply


def test_workflow_central_handled(patch_llm):
    """If the planner says handled_by_central, the central direct path runs."""
    from src.workflow.graph import run_query

    plan_payload = json.dumps(
        {
            "handled_by_central": True,
            "intents": ["app_identity"],
            "rationale": "App identity question.",
        }
    )
    direct_answer = "Fincent is a multi-agent financial assistant."
    patch_llm(plan_payload, direct_answer)

    out = run_query(QueryRequest(query="What can this app do?"))
    assert out.plan.handled_by_central is True
    assert out.plan.intents == [Intent.APP_IDENTITY]
    assert out.answer == direct_answer
    assert len(out.agent_responses) == 1
    assert out.agent_responses[0].agent == AgentName.CENTRAL
    assert out.agent_responses[0].metadata["intent"] == "app_identity"


def test_workflow_routes_to_qna(patch_llm):
    """A specialist intent should fan out to the qna node."""
    from src.workflow.graph import run_query

    plan_payload = json.dumps(
        {
            "handled_by_central": False,
            "intents": ["qna"],
            "rationale": "Generic financial question.",
        }
    )
    qna_text = "An ETF is an exchange-traded fund."
    patch_llm(plan_payload, qna_text)

    out = run_query(QueryRequest(query="What is an ETF?"))
    assert out.plan.handled_by_central is False
    assert Intent.QNA in out.plan.intents
    assert out.answer == qna_text  # single-response shortcut
    assert any(r.agent == AgentName.QNA for r in out.agent_responses)


def test_workflow_routes_portfolio_via_intent_hint(patch_llm):
    """An explicit intent_hint must bypass the router and run the portfolio agent."""
    from src.workflow.graph import run_query

    # Only the portfolio agent's LLM call is expected; the router is
    # short-circuited by the hint.
    portfolio_text = "You hold four accounts worth just over $1M."
    patch_llm(portfolio_text)

    out = run_query(
        QueryRequest(
            query="summarise my portfolio",
            intent_hint=Intent.PORTFOLIO,
        )
    )
    assert out.plan.handled_by_central is False
    assert Intent.PORTFOLIO in out.plan.intents
    assert out.answer == portfolio_text
    assert any(r.agent == AgentName.PORTFOLIO for r in out.agent_responses)


def test_workflow_routes_market_research_via_intent_hint(patch_llm):
    """The Market Research tab should bypass routing and run its specialist."""
    from src.workflow.graph import run_query

    market_text = "Nvidia has strong AI demand, but valuation risk matters."
    patch_llm(market_text)

    out = run_query(
        QueryRequest(
            query="Is Nvidia a good investment?",
            intent_hint=Intent.MARKET_RESEARCH,
        )
    )
    assert out.plan.handled_by_central is False
    assert Intent.MARKET_RESEARCH in out.plan.intents
    assert out.answer == market_text
    assert any(r.agent == AgentName.MARKET_RESEARCH for r in out.agent_responses)


def test_workflow_routes_goal_planning_via_intent_hint(patch_llm):
    """The Goal Planning tab should bypass routing and run its specialist."""
    from src.workflow.graph import run_query

    goal_text = "Your retirement plan needs a higher savings rate."
    patch_llm(goal_text)

    out = run_query(
        QueryRequest(
            query="Can I retire at 60?",
            intent_hint=Intent.GOAL_PLANNING,
        )
    )
    assert out.plan.handled_by_central is False
    assert Intent.GOAL_PLANNING in out.plan.intents
    assert out.answer == goal_text
    assert any(r.agent == AgentName.GOAL_PLANNING for r in out.agent_responses)
