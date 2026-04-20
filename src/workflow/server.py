"""FastAPI ingress that exposes the compiled LangGraph via LangServe.

Endpoints:
    GET  /health                  -> liveness probe.
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
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.config import AppConfig, get_config
from src.core.schemas import QueryRequest, QueryResponse
from src.utils.logging import configure_logging, get_logger
from src.workflow.graph import (
    default_graph,
    get_history,
    reset_thread,
    run_query,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Response envelopes for the history / reset endpoints
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
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Compile-and-cache the checkpointed graph once (lru_cache inside).
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
        """Return the chat transcript for a thread.

        An empty list is returned for unknown or brand-new threads (it
        is not an error). Roles are reported in LangGraph's vocabulary
        (``human`` / ``ai``) -- the client converts to its own.
        """
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
        """Clear the current message list for a thread.

        Delegates to ``graph.update_state`` with ``RemoveMessage``
        entries so the SQLite checkpoint log still retains the prior
        transcript versions.
        """
        try:
            removed = reset_thread(thread_id, graph=graph)
        except Exception as exc:  # noqa: BLE001 -- surface as 500
            _logger.exception("Reset failed for %s", thread_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ResetResponse(thread_id=thread_id, removed=removed)

    # ---- LangServe routes --------------------------------------------------
    #
    # Registered lazily so the module remains importable even if
    # langserve is missing in some thin testing environment.
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
