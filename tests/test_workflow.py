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
        from src.agents.qna import agent as qna_module

        monkeypatch.setattr(central_module, "get_default_chat_model", lambda: fake)
        monkeypatch.setattr(qna_module, "get_default_chat_model", lambda: fake)
        return fake

    return _apply


def test_workflow_central_handled(patch_llm):
    """If the planner says handled_by_central, the central direct path runs."""
    from src.workflow.graph import run_query

    plan_payload = json.dumps(
        {
            "handled_by_central": True,
            "intents": ["app_info"],
            "rationale": "App-info question.",
        }
    )
    direct_answer = "Fincent is a multi-agent financial assistant."
    patch_llm(plan_payload, direct_answer)

    out = run_query(QueryRequest(query="What can this app do?"))
    assert out.plan.handled_by_central is True
    assert out.plan.intents == [Intent.APP_INFO]
    assert out.answer == direct_answer
    assert len(out.agent_responses) == 1
    assert out.agent_responses[0].agent == AgentName.CENTRAL


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
