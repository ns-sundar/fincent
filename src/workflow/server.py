"""FastAPI ingress that exposes the compiled LangGraph via LangServe.

Endpoints:
    GET  /health                  -> liveness probe.
    GET  /rag/status              -> snapshot of the RAG ingestion
                                     pipeline state (see
                                     ``src.rag.status``).
    POST /query                   -> typed convenience endpoint
                                     returning ``QueryResponse`` (uses
                                     the checkpointed graph).
    GET  /history/{thread_id}     -> Chat transcript for a thread, in
                                     LangGraph role vocabulary
                                     ("human" / "ai"). The client
                                     translates to its own roles.
    POST /reset/{thread_id}       -> Clear the current message list
                                     for a thread via ``update_state``.
                                     SQLite versioning preserves the
                                     previous checkpoints.
    *    {graph_path}/...         -> LangServe routes (invoke, stream,
                                     etc.) for the raw graph runnable.

Startup behaviour:
    The application uses FastAPI's ``lifespan`` context manager to run
    the RAG ingestion pipeline *before* FastAPI begins serving. When
    ingestion fails the app still starts; ``/rag/status`` reports the
    failure and the Streamlit UI surfaces a banner while still
    accepting queries.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.config import AppConfig, get_config
from src.core.schemas import QueryRequest, QueryResponse
from src.rag import status as rag_status_mod
from src.rag.ingest import ingest_if_needed
from src.rag.retriever import reset_default_retriever
from src.utils.logging import configure_logging, get_logger
from src.workflow.graph import (
    default_graph,
    get_history,
    reset_thread,
    run_query,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """A single chat message in LangGraph role vocabulary."""

    role: str = Field(..., description="'human' or 'ai' (client maps to its own).")
    content: str


class HistoryResponse(BaseModel):
    """Response envelope for ``GET /history/{thread_id}``."""

    thread_id: str
    messages: List[HistoryMessage] = Field(default_factory=list)


class ResetResponse(BaseModel):
    """Response envelope for ``POST /reset/{thread_id}``."""

    thread_id: str
    removed: int


class RagStatusResponse(BaseModel):
    """Response envelope for ``GET /rag/status``."""

    state: str = Field(
        ...,
        description=(
            "One of: pending, ingesting, ready, skipped, disabled, failed."
        ),
    )
    detail: str = ""
    error: Optional[str] = None
    chunk_count: int = 0
    ingested_articles: int = 0
    meta: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------
# Lifespan: run RAG ingestion before serving
# ---------------------------------------------------------------------


def _build_lifespan(cfg: AppConfig):
    """Return a lifespan context manager bound to a specific config."""

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        _logger.info("RAG lifespan: starting ingestion (enabled=%s)", cfg.rag.enabled)
        try:
            # ingest_if_needed is synchronous + potentially long-running
            # (network + embeddings). That is intentional: FastAPI must
            # not start serving until it resolves one way or another.
            snapshot = ingest_if_needed(cfg)
        except Exception as exc:  # noqa: BLE001 -- never block startup
            _logger.exception("Unexpected ingestion failure at startup")
            snapshot = rag_status_mod.set_status(
                state=rag_status_mod.STATE_FAILED,
                detail="Unexpected ingestion failure.",
                error=f"{type(exc).__name__}: {exc}",
            )
        _logger.info(
            "RAG lifespan: ingestion finished (state=%s, detail=%s)",
            snapshot.state,
            snapshot.detail,
        )
        # Force the retriever cache to re-open the freshly built index
        # on its next use.
        reset_default_retriever()
        yield
        # No explicit shutdown work for RAG today.

    return _lifespan


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI app, including LangServe routes."""
    cfg = cfg or get_config()
    configure_logging(level=cfg.logging.level, log_file=cfg.logging.file or None)

    app = FastAPI(
        title=cfg.app.name,
        version=cfg.app.version,
        description=cfg.app.description,
        lifespan=_build_lifespan(cfg),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graph = default_graph()

    # ---- Plain endpoints ---------------------------------------------------

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Simple liveness probe."""
        return {
            "status": "ok",
            "app": cfg.app.name,
            "version": cfg.app.version,
        }

    @app.get("/rag/status", response_model=RagStatusResponse)
    def rag_status() -> RagStatusResponse:
        """Expose the RAG ingestion pipeline state."""
        snap = rag_status_mod.get_status()
        return RagStatusResponse(
            state=snap.state,
            detail=snap.detail,
            error=snap.error,
            chunk_count=snap.chunk_count,
            ingested_articles=snap.ingested_articles,
            meta=snap.meta,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Typed wrapper around the graph for the Streamlit UI.

        ``request.session_id`` is used as the LangGraph ``thread_id``
        so that state persists between calls.
        """
        try:
            return run_query(request, graph=graph)
        except Exception as exc:  # noqa: BLE001 -- surface as 500
            _logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/history/{thread_id}", response_model=HistoryResponse)
    def history(thread_id: str) -> HistoryResponse:
        """Return the chat transcript for a thread."""
        try:
            raw = get_history(thread_id, graph=graph)
        except Exception as exc:  # noqa: BLE001 -- surface as 500
            _logger.exception("History lookup failed for %s", thread_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return HistoryResponse(
            thread_id=thread_id,
            messages=[HistoryMessage(**m) for m in raw],
        )

    @app.post("/reset/{thread_id}", response_model=ResetResponse)
    def reset(thread_id: str) -> ResetResponse:
        """Clear the current message list for a thread."""
        try:
            removed = reset_thread(thread_id, graph=graph)
        except Exception as exc:  # noqa: BLE001 -- surface as 500
            _logger.exception("Reset failed for %s", thread_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ResetResponse(thread_id=thread_id, removed=removed)

    # ---- LangServe routes --------------------------------------------------
    try:
        from langserve import add_routes

        add_routes(
            app,
            graph,
            path=cfg.server.graph_path,
            input_type=Dict[str, Any],
            output_type=Dict[str, Any],
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort
        _logger.warning("LangServe routes not registered: %s", exc)

    return app


# Module-level singleton for ``uvicorn src.workflow.server:app``.
app: FastAPI = create_app()
