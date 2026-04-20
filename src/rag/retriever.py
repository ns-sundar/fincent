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


def _doc_source(doc: Any) -> str:
    """Extract the ``tags.source`` value from a LangChain Document (lowercased)."""
    meta = getattr(doc, "metadata", None) or {}
    tags = meta.get("tags") or {}
    return str(tags.get("source") or "").strip().lower()


def _make_source_filter(source: str):
    """Return a LangChain FAISS-compatible ``filter`` callable.

    FAISS accepts a callable ``filter(metadata) -> bool`` in recent
    LangChain versions. We normalise the needle once and compare
    case-insensitively so callers can pass ``"IRS"`` or ``"irs"``.
    """
    needle = source.strip().lower()

    def _predicate(metadata: Dict[str, Any]) -> bool:
        tags = (metadata or {}).get("tags") or {}
        return str(tags.get("source") or "").strip().lower() == needle

    return _predicate


class Retriever:
    """Thin wrapper around a LangChain FAISS vector store.

    The wrapper hides the embedding backend and the 2-tuple
    ``(Document, score)`` return shape from agents, exposing a simple
    ``retrieve(query, ...)`` API that yields :class:`RetrievedDoc`.

    Supports two search modes:

    * **similarity**  - raw cosine/L2 similarity.
    * **MMR**         - re-ranks a larger candidate pool with Maximal
                         Marginal Relevance to diversify results.

    Supports post-filtering by the ``tags.source`` metadata value
    stamped at ingestion time (e.g. ``"irs"``, ``"sec"``, ``"finra"``).
    """

    def __init__(
        self,
        vector_store: Any,
        *,
        default_k: int,
        use_mmr: bool = True,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> None:
        self._vs = vector_store
        self._default_k: int = default_k
        self._default_use_mmr: bool = use_mmr
        self._default_fetch_k: int = fetch_k
        self._default_lambda_mult: float = lambda_mult

    @property
    def size(self) -> int:
        """Number of vectors in the underlying index (0 if unknown)."""
        index = getattr(self._vs, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    def retrieve(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        """Run retrieval and return up to ``k`` documents.

        Args:
            query:          The search string.
            k:              Number of results to return (defaults to
                            ``cfg.rag.top_k``).
            use_mmr:        Toggle MMR re-ranking (defaults to
                            ``cfg.rag.use_mmr``).
            fetch_k:        Candidate pool size for MMR (defaults to
                            ``cfg.rag.mmr_fetch_k``).
            lambda_mult:    MMR diversity/relevance in [0, 1].
            source_filter:  Restrict hits to chunks whose
                            ``metadata.tags.source`` matches (case
                            insensitive). Example: ``"irs"``, ``"sec"``.
        """
        top_k = k or self._default_k
        if not query.strip() or top_k <= 0:
            return []

        effective_use_mmr = self._default_use_mmr if use_mmr is None else use_mmr
        effective_fetch_k = fetch_k or self._default_fetch_k
        effective_lambda = (
            self._default_lambda_mult if lambda_mult is None else lambda_mult
        )
        flt = _make_source_filter(source_filter) if source_filter else None

        try:
            if effective_use_mmr:
                docs = self._mmr_search(
                    query,
                    k=top_k,
                    fetch_k=max(effective_fetch_k, top_k),
                    lambda_mult=effective_lambda,
                    source_filter=source_filter,
                    filter_callable=flt,
                )
                pairs = [(d, 0.0) for d in docs]
            else:
                pairs = self._similarity_search(
                    query,
                    k=top_k,
                    source_filter=source_filter,
                    filter_callable=flt,
                )
        except Exception as exc:  # noqa: BLE001 -- keep agent responsive
            _logger.warning("FAISS retrieval failed: %s", exc)
            return []

        out: List[RetrievedDoc] = []
        for doc, score in pairs:
            meta = dict(getattr(doc, "metadata", None) or {})
            out.append(
                RetrievedDoc(
                    text=str(getattr(doc, "page_content", "") or ""),
                    url=str(meta.get("url") or meta.get("source") or ""),
                    title=str(meta.get("title") or ""),
                    tags=dict(meta.get("tags") or {}),
                    score=float(score),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Internal search helpers -- isolated so they can be monkey-patched
    # in tests and so the two code paths (similarity / MMR) don't clutter
    # the public ``retrieve`` method.
    # ------------------------------------------------------------------

    def _similarity_search(
        self,
        query: str,
        *,
        k: int,
        source_filter: Optional[str],
        filter_callable: Optional[Any],
    ) -> List[tuple[Any, float]]:
        """Similarity search with optional LangChain-native filter.

        Falls back to Python-side filtering when the underlying store
        does not accept a ``filter`` kwarg (older LangChain versions or
        custom test doubles).
        """
        try:
            if filter_callable is not None:
                return list(
                    self._vs.similarity_search_with_score(
                        query, k=k, filter=filter_callable
                    )
                )
        except TypeError:
            pass

        over_fetch = k * 5 if source_filter else k
        pairs = list(self._vs.similarity_search_with_score(query, k=over_fetch))
        if source_filter:
            pairs = [p for p in pairs if _doc_source(p[0]) == source_filter.lower()]
        return pairs[:k]

    def _mmr_search(
        self,
        query: str,
        *,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        source_filter: Optional[str],
        filter_callable: Optional[Any],
    ) -> List[Any]:
        """MMR search; returns Documents (FAISS MMR does not expose scores)."""
        kwargs: Dict[str, Any] = {
            "k": k,
            "fetch_k": max(fetch_k, k),
            "lambda_mult": lambda_mult,
        }
        try:
            if filter_callable is not None:
                return list(
                    self._vs.max_marginal_relevance_search(
                        query, filter=filter_callable, **kwargs
                    )
                )
            return list(self._vs.max_marginal_relevance_search(query, **kwargs))
        except TypeError:
            # Older backends / doubles may not support ``filter``; emulate it.
            docs = list(
                self._vs.max_marginal_relevance_search(
                    query,
                    k=fetch_k if source_filter else k,
                    fetch_k=max(fetch_k, k),
                    lambda_mult=lambda_mult,
                )
            )
            if source_filter:
                docs = [d for d in docs if _doc_source(d) == source_filter.lower()]
            return docs[:k]


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
    return Retriever(
        vs,
        default_k=cfg.rag.top_k,
        use_mmr=cfg.rag.use_mmr,
        fetch_k=cfg.rag.mmr_fetch_k,
        lambda_mult=cfg.rag.mmr_lambda,
    )


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
