"""Tests for the RAG ingestion pipeline.

These tests exercise ``ingest_if_needed`` end-to-end using:
  * a fake article catalog in-memory (no real HTTP),
  * a fake per-article loader that returns pre-built Documents,
  * an in-memory ``DeterministicEmbeddings`` backend so FAISS can
    actually build an index without calling OpenAI,
  * a tmp_path-rooted vector store path so nothing is written to the
    production /data/vector_db location.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import get_config, reset_config_cache
from src.rag import ingest as ingest_mod
from src.rag import status as status_mod


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


class DeterministicEmbeddings(Embeddings):
    """Tiny, seed-free hashing embedder so tests don't need OpenAI.

    Each input string becomes a 32-dim float vector derived from
    SHA-256 bytes; the output is deterministic and L2-normalised, so
    FAISS similarity search returns stable ordering.
    """

    _DIM: int = 32

    def _embed_one(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        # Expand the 32-byte digest into 32 floats in [-1, 1].
        vec = [(b / 127.5) - 1.0 for b in digest[: self._DIM]]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self._embed_one(text)


def _make_articles() -> List[Dict[str, Any]]:
    return [
        {
            "url": "https://example.com/etf",
            "title": "Intro to ETFs",
            "tags": {"source": "example", "category": ["etf", "basics"]},
        },
        {
            "url": "https://example.com/bonds",
            "title": "Bond Fundamentals",
            "tags": {"source": "example", "category": ["bonds", "basics"]},
        },
    ]


def _fake_loader(article: Dict[str, Any]) -> List[Document]:
    """Return a single Document per article, pre-stamped with metadata."""
    base_body = (
        "An ETF is a pooled investment vehicle that trades on an exchange. "
        "It bundles many underlying securities into a single ticker. "
    ) * 20  # long enough to produce multiple chunks after splitting
    content = f"{article['title']}\n\n{base_body}"
    return [
        Document(
            page_content=content,
            metadata={
                "url": article["url"],
                "title": article["title"],
                "tags": article["tags"],
                "source": article["url"],
            },
        )
    ]


@pytest.fixture
def rag_enabled_cfg(monkeypatch, tmp_path: Path):
    """Re-enable RAG (disabled by default in conftest) and redirect to tmp."""
    vector_dir = tmp_path / "vector_db"
    monkeypatch.setenv("FINCENT__RAG__ENABLED", "true")
    monkeypatch.setenv("FINCENT__RAG__VECTOR_DB_PATH", str(vector_dir))
    reset_config_cache()
    yield get_config(), vector_dir
    reset_config_cache()


# ---------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------


def test_load_articles_reads_real_catalog():
    """The shipped catalog at rag/fincent_rag_articles.json is valid."""
    project_root = Path(__file__).resolve().parents[1]
    catalog = ingest_mod.load_articles(project_root / "rag" / "fincent_rag_articles.json")
    assert isinstance(catalog, list) and len(catalog) > 0
    first = catalog[0]
    assert "url" in first and first["url"].startswith("http")
    assert "title" in first
    assert "tags" in first and "source" in first["tags"]


def test_load_articles_rejects_malformed(tmp_path: Path):
    """Malformed payloads raise ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_mod.load_articles(bad)


# ---------------------------------------------------------------------
# ingest_if_needed
# ---------------------------------------------------------------------


def test_ingest_disabled_returns_disabled_status(monkeypatch):
    """When rag.enabled is false the pipeline short-circuits."""
    monkeypatch.setenv("FINCENT__RAG__ENABLED", "false")
    reset_config_cache()
    snapshot = ingest_mod.ingest_if_needed()
    assert snapshot.state == status_mod.STATE_DISABLED


def test_ingest_builds_faiss_index_and_reports_ready(rag_enabled_cfg):
    """Happy path: build a FAISS index, metadata flows through."""
    _, vector_dir = rag_enabled_cfg
    articles = _make_articles()

    snapshot = ingest_mod.ingest_if_needed(
        embeddings=DeterministicEmbeddings(),
        loader=_fake_loader,
        articles=articles,
    )
    assert snapshot.state == status_mod.STATE_READY, snapshot.error
    assert snapshot.chunk_count > 0
    assert snapshot.ingested_articles == len(articles)
    assert ingest_mod.index_exists(vector_dir)


def test_ingest_skips_when_index_already_exists(rag_enabled_cfg):
    """A second call detects the existing index and returns STATE_SKIPPED."""
    _, vector_dir = rag_enabled_cfg
    # First run actually builds the index.
    ingest_mod.ingest_if_needed(
        embeddings=DeterministicEmbeddings(),
        loader=_fake_loader,
        articles=_make_articles(),
    )
    # Second run must NOT invoke the loader again: pass a loader that
    # explodes if called -- the code should never reach it.
    def _exploding_loader(article: Dict[str, Any]) -> List[Document]:  # pragma: no cover
        raise AssertionError("loader must not be called when index exists")

    snapshot = ingest_mod.ingest_if_needed(
        embeddings=DeterministicEmbeddings(),
        loader=_exploding_loader,
        articles=_make_articles(),
    )
    assert snapshot.state == status_mod.STATE_SKIPPED
    assert ingest_mod.index_exists(vector_dir)


def test_ingest_failure_yields_failed_status(rag_enabled_cfg):
    """Total-failure path: every loader call fails -> STATE_FAILED."""

    def _always_fail(article: Dict[str, Any]) -> List[Document]:
        raise RuntimeError(f"boom for {article['url']}")

    snapshot = ingest_mod.ingest_if_needed(
        embeddings=DeterministicEmbeddings(),
        loader=_always_fail,
        articles=_make_articles(),
    )
    assert snapshot.state == status_mod.STATE_FAILED
    assert snapshot.error  # non-empty error message
