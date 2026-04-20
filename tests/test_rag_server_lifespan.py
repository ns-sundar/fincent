"""Integration test: FastAPI lifespan runs RAG ingestion at startup.

We verify:
  * /rag/status reflects the outcome of ``ingest_if_needed``.
  * A failure during ingestion does NOT prevent FastAPI from serving
    (GET /health still returns 200).
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver


def _build_client_with_patched_ingest(monkeypatch, fake_ingest):
    """Build a TestClient whose server uses ``fake_ingest`` in place of
    the real ``ingest_if_needed``."""
    from src.workflow.graph import build_graph
    import src.workflow.graph as graph_module
    import src.workflow.server as server_module

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=saver)

    monkeypatch.setattr(graph_module, "default_graph", lambda: graph)
    monkeypatch.setattr(server_module, "default_graph", lambda: graph)
    monkeypatch.setattr(server_module, "ingest_if_needed", fake_ingest)

    app = server_module.create_app()
    return TestClient(app), conn


def test_lifespan_sets_rag_status_to_ready(monkeypatch):
    """Lifespan writes a "ready" snapshot when ingestion succeeds."""
    from src.rag import status as status_mod

    def _fake_ingest(cfg=None):
        return status_mod.set_status(
            state=status_mod.STATE_READY,
            detail="test ingestion complete",
            chunk_count=42,
            ingested_articles=5,
        )

    client, conn = _build_client_with_patched_ingest(monkeypatch, _fake_ingest)
    try:
        with client:
            # Entering the context manager triggers FastAPI lifespan.
            r = client.get("/rag/status")
            assert r.status_code == 200
            body = r.json()
            assert body["state"] == "ready"
            assert body["chunk_count"] == 42
            assert body["ingested_articles"] == 5
    finally:
        conn.close()


def test_lifespan_surfaces_failure_but_still_serves(monkeypatch):
    """Ingestion failure must not prevent /health from returning 200."""
    from src.rag import status as status_mod

    def _fake_ingest(cfg=None):
        return status_mod.set_status(
            state=status_mod.STATE_FAILED,
            detail="simulated failure",
            error="RuntimeError: boom",
        )

    client, conn = _build_client_with_patched_ingest(monkeypatch, _fake_ingest)
    try:
        with client:
            # /health should still work.
            h = client.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

            # /rag/status reports the failure.
            s = client.get("/rag/status")
            assert s.status_code == 200
            body = s.json()
            assert body["state"] == "failed"
            assert body["error"] == "RuntimeError: boom"
    finally:
        conn.close()
