"""FAISS-backed retriever for the Q&A agent.

The retriever lazily opens the persisted FAISS index written by
``src.rag.ingest`` and performs similarity search at query time.
Metadata (url, title, tags) stamped during ingestion flows through to
each returned result so the agent can cite sources.

If the index is not available (ingestion was disabled or failed) the
retriever degrades gracefully: ``retrieve`` returns an empty list and
the Q&A agent falls back to its non-RAG behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from src.core.config import AppConfig, get_config
from src.rag.ingest import index_exists
from src.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass
class RetrievedDoc:
    """A single retrieval result exposed to agents."""

    text: str
    url: str
    title: str
    tags: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        """Back-compat alias for callers that referenced ``.source``."""
        return self.url


class Retriever:
    """Thin wrapper around a LangChain FAISS vector store.

    The wrapper hides the embedding backend and the 2-tuple
    ``(Document, score)`` return shape from agents, exposing a simple
    ``retrieve(query, k=...)`` API that yields :class:`RetrievedDoc`.
    """

    def __init__(
        self,
        vector_store: Any,
        *,
        default_k: int,
    ) -> None:
        self._vs = vector_store
        self._default_k: int = default_k

    @property
    def size(self) -> int:
        """Number of vectors in the underlying index (0 if unknown)."""
        index = getattr(self._vs, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    def retrieve(self, query: str, *, k: Optional[int] = None) -> List[RetrievedDoc]:
        """Run similarity search and return up to ``k`` documents."""
        top_k = k or self._default_k
        if not query.strip() or top_k <= 0:
            return []
        try:
            pairs = self._vs.similarity_search_with_score(query, k=top_k)
        except Exception as exc:  # noqa: BLE001 -- keep agent responsive
            _logger.warning("FAISS similarity search failed: %s", exc)
            return []

        out: List[RetrievedDoc] = []
        for doc, score in pairs:
            meta = dict(doc.metadata or {})
            out.append(
                RetrievedDoc(
                    text=str(doc.page_content or ""),
                    url=str(meta.get("url") or meta.get("source") or ""),
                    title=str(meta.get("title") or ""),
                    tags=dict(meta.get("tags") or {}),
                    score=float(score),
                )
            )
        return out


# ---------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------


def _build_default_embeddings(cfg: AppConfig) -> Embeddings:
    """Construct the same embedding backend used during ingestion."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=cfg.rag.embedding_model)


def load_retriever(
    cfg: Optional[AppConfig] = None,
    *,
    embeddings: Optional[Embeddings] = None,
) -> Optional[Retriever]:
    """Open the persisted FAISS index and wrap it in a :class:`Retriever`.

    Returns ``None`` when the index is missing on disk (so callers can
    fall back to a non-RAG code path without an exception).
    """
    cfg = cfg or get_config()
    if not cfg.rag.enabled:
        return None

    vector_dir = Path(cfg.rag.vector_db_path).expanduser().resolve()
    if not index_exists(vector_dir):
        _logger.info("No FAISS index at %s; retriever unavailable.", vector_dir)
        return None

    from langchain_community.vectorstores import FAISS

    embed = embeddings or _build_default_embeddings(cfg)
    # ``allow_dangerous_deserialization`` is required for pickled
    # metadata in newer LangChain; our index is produced by our own
    # ingestion pipeline, so this is safe.
    vs = FAISS.load_local(
        str(vector_dir),
        embeddings=embed,
        allow_dangerous_deserialization=True,
    )
    _logger.info("Loaded FAISS retriever from %s (%d vectors)", vector_dir, vs.index.ntotal)
    return Retriever(vs, default_k=cfg.rag.top_k)


@lru_cache(maxsize=1)
def get_default_retriever() -> Optional[Retriever]:
    """Process-wide cached retriever (lazily opens the FAISS index)."""
    try:
        return load_retriever()
    except Exception as exc:  # noqa: BLE001 -- never break callers at startup
        _logger.warning("Could not load default retriever: %s", exc)
        return None


def reset_default_retriever() -> None:
    """Clear the cached retriever (tests and post-reingestion hooks)."""
    get_default_retriever.cache_clear()


# ---------------------------------------------------------------------
# Back-compat function the old skeleton exposed
# ---------------------------------------------------------------------


def retrieve(query: str, *, k: int = 4) -> List[RetrievedDoc]:
    """Convenience wrapper that uses the default retriever.

    Returns an empty list if the retriever is unavailable.
    """
    r = get_default_retriever()
    if r is None:
        return []
    return r.retrieve(query, k=k)


__all__ = [
    "RetrievedDoc",
    "Retriever",
    "get_default_retriever",
    "load_retriever",
    "reset_default_retriever",
    "retrieve",
]
