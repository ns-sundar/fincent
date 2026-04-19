"""FastAPI ingress that exposes the compiled LangGraph via LangServe.

Endpoints:
    GET  /health           -> liveness probe.
    POST /query            -> typed convenience endpoint returning
                              ``QueryResponse``.
    *    {graph_path}/...  -> LangServe routes (invoke, stream, etc.)
                              for the raw graph runnable.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import AppConfig, get_config
from src.core.schemas import QueryRequest, QueryResponse
from src.utils.logging import configure_logging, get_logger
from src.workflow.graph import build_graph, run_query

_logger = get_logger(__name__)


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI app, including LangServe routes.

    Args:
        cfg: Optional pre-loaded config (mainly for tests).

    Returns:
        A configured ``FastAPI`` instance.
    """
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

    graph = build_graph()

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
        """Typed wrapper around the graph for the Streamlit UI."""
        try:
            return run_query(request, graph=graph)
        except Exception as exc:  # noqa: BLE001 -- surface as 500
            _logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ---- LangServe routes --------------------------------------------------
    #
    # We register lazily so the module remains importable even if
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
