"""End-to-end HTTP tests for the /query, /history, /reset endpoints.

Uses FastAPI's TestClient plus an in-memory SQLite checkpointer.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from itertools import cycle
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite import SqliteSaver


class _CyclingFakeLLM(FakeListChatModel):
    def __init__(self, responses):
        super().__init__(responses=list(responses))
        self._cycle: Iterator[str] = cycle(self.responses)

    def _call(self, *args, **kwargs):  # type: ignore[override]
        return next(self._cycle)


def _patch_llm(monkeypatch, *responses: str) -> None:
    fake = _CyclingFakeLLM(responses)
    from src.agents.central import agent as central_module
    from src.agents.qna import agent as qna_module

    monkeypatch.setattr(central_module, "get_default_chat_model", lambda: fake)
    monkeypatch.setattr(qna_module, "get_default_chat_model", lambda: fake)


@pytest.fixture
def app_and_client(monkeypatch):
    """Build a FastAPI app using an in-memory checkpointed graph."""
    from src.workflow.graph import build_graph

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=saver)

    # Swap the cached default_graph() so server.create_app() picks
    # up our in-memory version.
    import src.workflow.graph as graph_module
    import src.workflow.server as server_module

    monkeypatch.setattr(graph_module, "default_graph", lambda: graph)
    monkeypatch.setattr(server_module, "default_graph", lambda: graph)

    app: FastAPI = server_module.create_app()
    client = TestClient(app)
    yield client
    conn.close()


def test_health_endpoint(app_and_client):
    """/health returns a status payload."""
    resp = app_and_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_then_history_then_reset_round_trip(app_and_client, monkeypatch):
    """Full round trip: POST /query -> GET /history -> POST /reset."""
    plan_payload = json.dumps(
        {"handled_by_central": False, "intents": ["qna"], "rationale": "ok"}
    )
    _patch_llm(monkeypatch, plan_payload, "An ETF is a fund.")

    tid = f"t-{uuid.uuid4()}"

    # 1) POST /query with a session_id
    q = app_and_client.post(
        "/query",
        json={"query": "What is an ETF?", "session_id": tid},
    )
    assert q.status_code == 200, q.text
    assert q.json()["answer"] == "An ETF is a fund."

    # 2) GET /history/{thread_id}
    h = app_and_client.get(f"/history/{tid}")
    assert h.status_code == 200
    body = h.json()
    assert body["thread_id"] == tid
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["human", "ai"]
    assert msgs[0]["content"] == "What is an ETF?"
    assert msgs[1]["content"] == "An ETF is a fund."

    # 3) POST /reset/{thread_id}
    r = app_and_client.post(f"/reset/{tid}")
    assert r.status_code == 200
    assert r.json() == {"thread_id": tid, "removed": 2}

    # 4) /history is now empty for that thread
    h2 = app_and_client.get(f"/history/{tid}")
    assert h2.status_code == 200
    assert h2.json()["messages"] == []


def test_history_is_empty_for_unknown_thread(app_and_client):
    """Unknown threads return an empty transcript, not an error."""
    resp = app_and_client.get(f"/history/unknown-{uuid.uuid4()}")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []
