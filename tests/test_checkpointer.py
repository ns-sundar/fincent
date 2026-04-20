"""Tests for the SQLite checkpointer integration.

These exercise the real ``SqliteSaver`` (in-memory) so we can verify
that ``/history`` and ``/reset`` round-trip correctly across multiple
turns on the same thread. The LLM is still mocked.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from itertools import cycle
from typing import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.schemas import AgentName, Intent, QueryRequest
from src.workflow.graph import (
    build_graph,
    get_history,
    reset_thread,
    run_query,
)


class _CyclingFakeLLM(FakeListChatModel):
    """FakeListChatModel that cycles its responses indefinitely."""

    def __init__(self, responses):
        super().__init__(responses=list(responses))
        self._cycle: Iterator[str] = cycle(self.responses)

    def _call(self, *args, **kwargs):  # type: ignore[override]
        return next(self._cycle)


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch both the central and qna modules' LLM factory."""

    def _apply(*responses: str) -> _CyclingFakeLLM:
        fake = _CyclingFakeLLM(responses)
        from src.agents.central import agent as central_module
        from src.agents.qna import agent as qna_module

        monkeypatch.setattr(central_module, "get_default_chat_model", lambda: fake)
        monkeypatch.setattr(qna_module, "get_default_chat_model", lambda: fake)
        return fake

    return _apply


@pytest.fixture
def checkpointed_graph():
    """Build a graph wired to an in-memory SQLite checkpointer."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=saver)
    yield graph
    conn.close()


def test_history_is_empty_for_new_thread(checkpointed_graph):
    """A brand-new thread must return an empty transcript, not an error."""
    tid = f"t-{uuid.uuid4()}"
    assert get_history(tid, graph=checkpointed_graph) == []


def test_multi_turn_history_accumulates(checkpointed_graph, fake_llm):
    """Two turns on the same thread must produce 4 messages (H, A, H, A)."""
    # Two plan responses + two qna responses (cycled).
    plan_payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"], "rationale": "ok"}
    )
    fake_llm(plan_payload, "An ETF is a fund.", plan_payload, "A bond is debt.")

    tid = f"t-{uuid.uuid4()}"
    out1 = run_query(
        QueryRequest(query="What is an ETF?"),
        graph=checkpointed_graph,
        thread_id=tid,
    )
    out2 = run_query(
        QueryRequest(query="What is a bond?"),
        graph=checkpointed_graph,
        thread_id=tid,
    )

    assert out1.answer == "An ETF is a fund."
    assert out2.answer == "A bond is debt."

    # agent_responses reducer must reset between turns.
    assert len(out2.agent_responses) == 1
    assert out2.agent_responses[0].agent == AgentName.QNA

    history = get_history(tid, graph=checkpointed_graph)
    roles = [m["role"] for m in history]
    assert roles == ["human", "ai", "human", "ai"]
    assert history[0]["content"] == "What is an ETF?"
    assert history[1]["content"] == "An ETF is a fund."
    assert history[2]["content"] == "What is a bond?"
    assert history[3]["content"] == "A bond is debt."


def test_reset_clears_current_state(checkpointed_graph, fake_llm):
    """reset_thread removes every message from the *current* state."""
    plan_payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"], "rationale": "ok"}
    )
    fake_llm(plan_payload, "An ETF is a fund.")

    tid = f"t-{uuid.uuid4()}"
    run_query(
        QueryRequest(query="What is an ETF?"),
        graph=checkpointed_graph,
        thread_id=tid,
    )
    assert len(get_history(tid, graph=checkpointed_graph)) == 2

    removed = reset_thread(tid, graph=checkpointed_graph)
    assert removed == 2
    assert get_history(tid, graph=checkpointed_graph) == []


def test_reset_on_empty_thread_is_noop(checkpointed_graph):
    """Resetting a thread with no messages should be a safe no-op."""
    tid = f"t-{uuid.uuid4()}"
    assert reset_thread(tid, graph=checkpointed_graph) == 0


def test_threads_are_isolated(checkpointed_graph, fake_llm):
    """Two different threads must maintain separate transcripts."""
    plan_payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"], "rationale": "ok"}
    )
    fake_llm(plan_payload, "Alice answer.", plan_payload, "Bob answer.")

    tid_a = f"alice-{uuid.uuid4()}"
    tid_b = f"bob-{uuid.uuid4()}"
    run_query(QueryRequest(query="Alice Q"), graph=checkpointed_graph, thread_id=tid_a)
    run_query(QueryRequest(query="Bob Q"), graph=checkpointed_graph, thread_id=tid_b)

    h_a = get_history(tid_a, graph=checkpointed_graph)
    h_b = get_history(tid_b, graph=checkpointed_graph)

    assert [m["content"] for m in h_a] == ["Alice Q", "Alice answer."]
    assert [m["content"] for m in h_b] == ["Bob Q", "Bob answer."]


def test_plan_mentions_qna_intent(checkpointed_graph, fake_llm):
    """Smoke-check that the plan still carries the right intent."""
    plan_payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"], "rationale": "ok"}
    )
    fake_llm(plan_payload, "hi")
    out = run_query(
        QueryRequest(query="What is a stock?"),
        graph=checkpointed_graph,
        thread_id=f"t-{uuid.uuid4()}",
    )
    assert Intent.QNA in out.plan.intents
