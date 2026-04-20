"""Tests for the FAISS-backed retriever.

Depends on the same DeterministicEmbeddings shim used for ingestion
so we can round-trip a small corpus entirely in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

from src.core.config import get_config, reset_config_cache
from src.rag import ingest as ingest_mod
from src.rag import retriever as retriever_mod

from tests.test_rag_ingest import DeterministicEmbeddings


def _articles() -> List[Dict[str, Any]]:
    return [
        {
            "url": "https://example.com/etf",
            "title": "Intro to ETFs",
            "tags": {"source": "example", "category": ["etf"]},
        },
        {
            "url": "https://irs.example.com/ira",
            "title": "IRS on IRAs",
            "tags": {"source": "irs", "category": ["tax", "retirement"]},
        },
    ]


def _loader(article: Dict[str, Any]) -> List[Document]:
    if article["tags"].get("source") == "irs":
        body = (
            "An Individual Retirement Arrangement (IRA) is a tax-advantaged "
            "account used to save for retirement. "
        ) * 10
    else:
        body = (
            "An ETF is a pooled investment vehicle that trades on an "
            "exchange like a stock. ETFs bundle many underlying "
            "securities into a single ticker symbol. "
        ) * 10
    return [
        Document(
            page_content=body,
            metadata={
                "url": article["url"],
                "title": article["title"],
                "tags": article["tags"],
                "source": article["url"],
            },
        )
    ]


@pytest.fixture
def built_index(monkeypatch, tmp_path: Path):
    """Build a real FAISS index in tmp_path and reset retriever cache."""
    vector_dir = tmp_path / "vector_db"
    monkeypatch.setenv("FINCENT__RAG__ENABLED", "true")
    monkeypatch.setenv("FINCENT__RAG__VECTOR_DB_PATH", str(vector_dir))
    reset_config_cache()
    retriever_mod.reset_default_retriever()

    snapshot = ingest_mod.ingest_if_needed(
        embeddings=DeterministicEmbeddings(),
        loader=_loader,
        articles=_articles(),
    )
    assert snapshot.state == "ready", snapshot.error
    yield get_config(), vector_dir
    retriever_mod.reset_default_retriever()
    reset_config_cache()


def test_load_retriever_returns_none_when_no_index(monkeypatch, tmp_path: Path):
    """Missing index -> ``load_retriever`` returns None (graceful)."""
    monkeypatch.setenv("FINCENT__RAG__ENABLED", "true")
    monkeypatch.setenv("FINCENT__RAG__VECTOR_DB_PATH", str(tmp_path / "none"))
    reset_config_cache()
    assert retriever_mod.load_retriever() is None


def test_retriever_round_trip_returns_metadata(built_index):
    """A query against the freshly built index surfaces our metadata."""
    _, _vector_dir = built_index
    retriever = retriever_mod.load_retriever(embeddings=DeterministicEmbeddings())
    assert retriever is not None
    assert retriever.size >= 1

    results = retriever.retrieve("What is an ETF?", k=1, use_mmr=False)
    assert len(results) == 1
    hit = results[0]
    assert hit.url.startswith("https://")
    assert hit.title
    assert hit.tags.get("source") in {"example", "irs"}


def test_retriever_empty_query_returns_empty(built_index):
    """A blank query is handled without raising."""
    retriever = retriever_mod.load_retriever(embeddings=DeterministicEmbeddings())
    assert retriever is not None
    assert retriever.retrieve("   ", k=3) == []


def test_retriever_mmr_path_returns_results(built_index):
    """MMR re-ranking is reachable end-to-end through the real FAISS backend."""
    retriever = retriever_mod.load_retriever(embeddings=DeterministicEmbeddings())
    assert retriever is not None
    results = retriever.retrieve(
        "ETF or IRA?",
        k=2,
        use_mmr=True,
        fetch_k=5,
        lambda_mult=0.5,
    )
    assert 0 < len(results) <= 2
    for r in results:
        assert r.tags.get("source") in {"example", "irs"}


def test_retriever_source_filter_restricts_hits(built_index):
    """A ``source_filter`` of 'irs' excludes all non-IRS chunks."""
    retriever = retriever_mod.load_retriever(embeddings=DeterministicEmbeddings())
    assert retriever is not None

    # Similarity path
    results = retriever.retrieve(
        "retirement account",
        k=3,
        use_mmr=False,
        source_filter="irs",
    )
    assert len(results) >= 1
    for r in results:
        assert r.tags.get("source") == "irs"

    # MMR path
    results_mmr = retriever.retrieve(
        "retirement account",
        k=3,
        use_mmr=True,
        fetch_k=5,
        lambda_mult=0.5,
        source_filter="irs",
    )
    assert len(results_mmr) >= 1
    for r in results_mmr:
        assert r.tags.get("source") == "irs"
