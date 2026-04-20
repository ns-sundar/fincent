"""RAG ingestion pipeline for the Q&A agent.

Flow:

    1. Read the JSON catalog ``rag/fincent_rag_articles.json``.
    2. For each entry, download the URL and parse PDF/HTML into
       LangChain ``Document`` objects (see ``src.rag.loaders``).
    3. Chunk with the recursive character splitter calibrated on
       ``tiktoken`` (~1000 tokens, 200 overlap) so definitions stay
       next to their examples.
    4. Build a FAISS index and persist it at ``cfg.rag.vector_db_path``
       (absolute path, typically ``/data/vector_db``).

The pipeline is idempotent: if a FAISS index is already present at
the target path we skip ingestion entirely. This treats the corpus as
static.

The FastAPI lifespan hook calls :func:`ingest_if_needed` once at
startup and does not begin serving until it returns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import AppConfig, get_config
from src.rag import status as status_mod
from src.rag.loaders import ArticleFetcher, load_article
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# The two files that ``FAISS.save_local`` writes. If both exist we
# consider the index present.
_FAISS_SENTINELS: tuple[str, ...] = ("index.faiss", "index.pkl")


# ---------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------


def load_articles(path: Path | str) -> List[Dict[str, Any]]:
    """Read the JSON catalog of articles from disk.

    Raises ``FileNotFoundError`` if the file is missing and
    ``ValueError`` if the payload is not a list of objects.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Article catalog not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"Article catalog {p} must contain a JSON list, got {type(data).__name__}"
        )
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(
                f"Catalog entry #{i} is malformed (needs at least a 'url' field)"
            )
    return data


# ---------------------------------------------------------------------
# Vector-store helpers
# ---------------------------------------------------------------------


def _resolve_vector_dir(raw_path: str) -> Path:
    """Expand and normalise the configured vector-store directory."""
    return Path(raw_path).expanduser().resolve()


def index_exists(vector_dir: Path | str) -> bool:
    """Return True when a FAISS index is already persisted at ``vector_dir``.

    The check looks for both files ``FAISS.save_local`` writes; a
    half-written directory (e.g. from a crashed prior run) is treated
    as "needs re-ingestion".
    """
    p = Path(vector_dir)
    return all((p / name).is_file() for name in _FAISS_SENTINELS)


def _build_default_embeddings(cfg: AppConfig) -> Embeddings:
    """Construct the default ``OpenAIEmbeddings`` backend.

    Notes:
      * ``chunk_size`` here is OpenAIEmbeddings' *batch size*, not our
        document chunk size. Default is 1000 texts per request, which
        produces ~1-2 MB POST bodies for our catalog. That is large
        enough to trip MTU / TLS-record issues on some networks
        (classic WSL2 + path-MTU blackhole -> ``SSLV3_ALERT_BAD_RECORD_MAC``).
        Smaller batches => smaller requests => higher reliability and
        less blast radius if one batch fails. 100 is a good default.
      * ``max_retries`` defaults to 2 in langchain/openai; we bump it
        a little to better tolerate transient edges.

    Kept behind a helper (rather than inline) so tests can substitute a
    fake embedder without importing ``langchain_openai``.
    """
    # Imported lazily: the module and its transitive deps are heavy and
    # unnecessary when callers supply their own embeddings.
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg.rag.embedding_model,
        # Small batches minimise request-body size. Some networks
        # (corporate TLS-inspection proxies, WSL2 with TSO/GSO bugs)
        # corrupt or reset HTTPS uploads larger than ~64 KB; 20 texts
        # per batch keeps each POST comfortably under that threshold.
        chunk_size=20,
        max_retries=5,
    )


def _build_splitter(cfg: AppConfig):
    """Create a recursive character splitter sized in tokens."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        # Preferred: token-accurate chunks via tiktoken.
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg.rag.chunk_size,
            chunk_overlap=cfg.rag.chunk_overlap,
        )
    except Exception as exc:  # noqa: BLE001 -- tiktoken optional in tests
        _logger.warning(
            "tiktoken-based splitter unavailable (%s); falling back to char sizes",
            exc,
        )
        # Fallback treats ``chunk_size`` as characters (rough approximation).
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg.rag.chunk_size * 4,  # ~4 chars / token
            chunk_overlap=cfg.rag.chunk_overlap * 4,
        )


def chunk_documents(docs: Sequence[Document], cfg: AppConfig) -> List[Document]:
    """Split loaded documents into RAG chunks with preserved metadata."""
    splitter = _build_splitter(cfg)
    # ``split_documents`` carries metadata from parent into children.
    return list(splitter.split_documents(list(docs)))


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


LoaderFn = Callable[[Dict[str, Any]], List[Document]]


def _default_loader(cfg: AppConfig, fetcher: Optional[ArticleFetcher]) -> LoaderFn:
    """Return a loader bound to the configured timeout / UA / fetcher."""

    def _load(article: Dict[str, Any]) -> List[Document]:
        return load_article(
            article,
            timeout=cfg.rag.request_timeout,
            user_agent=cfg.rag.user_agent,
            fetcher=fetcher,
        )

    return _load


def _collect_documents(
    articles: Sequence[Dict[str, Any]],
    *,
    loader: LoaderFn,
) -> tuple[List[Document], List[Dict[str, str]]]:
    """Load all articles, logging and collecting per-URL failures.

    Returns:
        ``(documents, failures)`` where ``failures`` is a list of
        ``{"url", "title", "error"}`` dicts.
    """
    docs: List[Document] = []
    failures: List[Dict[str, str]] = []

    for i, article in enumerate(articles, start=1):
        url = article.get("url", "")
        title = article.get("title", "")
        try:
            loaded = loader(article)
            if not loaded:
                _logger.warning("No extractable text for %s -- skipping", url)
                failures.append(
                    {"url": url, "title": title, "error": "no extractable text"}
                )
                continue
            _logger.info(
                "[%d/%d] Loaded %d doc(s) from %s",
                i,
                len(articles),
                len(loaded),
                url,
            )
            docs.extend(loaded)
        except Exception as exc:  # noqa: BLE001 -- per-URL isolation
            _logger.warning("[%d/%d] Failed to load %s: %s", i, len(articles), url, exc)
            failures.append({"url": url, "title": title, "error": str(exc)})

    return docs, failures


def _build_and_persist_index(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    vector_dir: Path,
) -> int:
    """Compute embeddings, build a FAISS index, and save to ``vector_dir``.

    Returns the number of vectors written.
    """
    from langchain_community.vectorstores import FAISS

    vector_dir.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(list(chunks), embeddings)
    store.save_local(str(vector_dir))
    return store.index.ntotal  # type: ignore[attr-defined]


def ingest_if_needed(
    cfg: Optional[AppConfig] = None,
    *,
    embeddings: Optional[Embeddings] = None,
    fetcher: Optional[ArticleFetcher] = None,
    loader: Optional[LoaderFn] = None,
    articles: Optional[Sequence[Dict[str, Any]]] = None,
) -> status_mod.RagStatus:
    """Run the full ingestion pipeline (once) and update the status singleton.

    Args:
        cfg:        Optional pre-loaded ``AppConfig``.
        embeddings: Optional embedding backend (tests inject fakes).
        fetcher:    Optional URL fetcher override.
        loader:     Optional per-article loader override (takes
                    precedence over ``fetcher`` if supplied).
        articles:   Optional in-memory article catalog. When ``None``
                    the catalog is read from ``cfg.rag.articles_path``.

    Returns:
        The resulting ``RagStatus`` snapshot. Always returns; callers
        should inspect ``status.state``.
    """
    cfg = cfg or get_config()

    if not cfg.rag.enabled:
        _logger.info("RAG is disabled via config; skipping ingestion.")
        return status_mod.set_status(
            state=status_mod.STATE_DISABLED,
            detail="RAG ingestion disabled via config.",
            error=None,
            chunk_count=0,
            ingested_articles=0,
            meta={},
        )

    vector_dir = _resolve_vector_dir(cfg.rag.vector_db_path)
    if index_exists(vector_dir):
        _logger.info(
            "FAISS index already present at %s; skipping ingestion.",
            vector_dir,
        )
        return status_mod.set_status(
            state=status_mod.STATE_SKIPPED,
            detail=f"Existing index found at {vector_dir}; ingestion skipped.",
            error=None,
            # We don't open the index here; the retriever does that lazily.
            chunk_count=0,
            ingested_articles=0,
            meta={"vector_db_path": str(vector_dir)},
        )

    status_mod.set_status(
        state=status_mod.STATE_INGESTING,
        detail="Ingestion in progress...",
        error=None,
    )
    started = time.monotonic()

    try:
        # ---- 1. Resolve inputs ------------------------------------------------
        if articles is None:
            articles_path = Path(cfg.rag.articles_path)
            if not articles_path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                articles_path = (project_root / articles_path).resolve()
            articles = load_articles(articles_path)
        _logger.info("Ingesting %d article(s) into %s", len(articles), vector_dir)

        effective_loader: LoaderFn = loader or _default_loader(cfg, fetcher)

        # ---- 2. Load + parse URLs --------------------------------------------
        docs, failures = _collect_documents(articles, loader=effective_loader)
        if not docs:
            raise RuntimeError(
                "No documents could be loaded from the catalog "
                f"({len(failures)} failures)."
            )

        # ---- 3. Chunk ---------------------------------------------------------
        chunks = chunk_documents(docs, cfg)
        _logger.info(
            "Produced %d chunk(s) from %d loaded doc(s)", len(chunks), len(docs)
        )
        if not chunks:
            raise RuntimeError("Chunker produced zero chunks; refusing to build index.")

        # ---- 4. Embed + persist ----------------------------------------------
        embed = embeddings or _build_default_embeddings(cfg)
        vectors_written = _build_and_persist_index(chunks, embed, vector_dir)

        elapsed = time.monotonic() - started
        ingested_articles = len(articles) - len(failures)
        detail = (
            f"{vectors_written} chunk(s) from {ingested_articles}/"
            f"{len(articles)} article(s) in {elapsed:0.1f}s"
        )
        _logger.info("RAG ingestion complete: %s", detail)
        return status_mod.set_status(
            state=status_mod.STATE_READY,
            detail=detail,
            error=None,
            chunk_count=vectors_written,
            ingested_articles=ingested_articles,
            meta={
                "vector_db_path": str(vector_dir),
                "failures": failures,
                "elapsed_seconds": round(elapsed, 2),
            },
        )

    except Exception as exc:  # noqa: BLE001 -- surface to status, keep API up
        _logger.exception("RAG ingestion failed")
        return status_mod.set_status(
            state=status_mod.STATE_FAILED,
            detail="Ingestion failed; API will serve without RAG context.",
            error=f"{type(exc).__name__}: {exc}",
            chunk_count=0,
            ingested_articles=0,
            meta={"vector_db_path": str(vector_dir)},
        )


__all__ = [
    "chunk_documents",
    "index_exists",
    "ingest_if_needed",
    "load_articles",
]
