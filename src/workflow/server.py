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

import json
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from markdown_it import MarkdownIt
from openai import OpenAIError
from pydantic import BaseModel, Field

from src.agents.llm_errors import (
    context_overflow_user_message,
    is_context_overflow_error,
)
from src.agents.market_research.mcp_tools import (
    start_market_research_mcp_sessions,
    stop_market_research_mcp_sessions,
)
from src.agents.goal_planning.mcp_tools import (
    start_goal_planning_mcp_sessions,
    stop_goal_planning_mcp_sessions,
)
from src.agents.portfolio.mcp_tools import (
    start_portfolio_mcp_sessions,
    stop_portfolio_mcp_sessions,
)
from src.agents.portfolio.loader import load_portfolio
from src.agents.portfolio.seed import seed_portfolio_if_needed
from src.core.llm import get_current_model, set_current_model
from src.core.config import AppConfig, get_config
from src.utils.async_shutdown import is_lifespan_shutdown_noise
from src.core.moderation import REJECTION_PREFIX, flagged_categories, moderate_query
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
_QUERY_PATH = "/query"
_MAX_MARKDOWN_LOG_CHARS = 8000
_MARKDOWN = MarkdownIt("commonmark").enable("table")
_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})")


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


class PortfolioRefreshResponse(BaseModel):
    """Response envelope for ``POST /portfolio/refresh``."""

    refreshed: bool = True


class ModelResponse(BaseModel):
    """Response envelope for ``GET /model`` and ``POST /model``."""

    model: str = Field(..., description="Active chat model name.")


class ModelSetRequest(BaseModel):
    """Request body for ``POST /model``."""

    model: str = Field(..., description="Model name to activate.")


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
# Request / response guardrails
# ---------------------------------------------------------------------


def _restore_request_body(request: Request, body: bytes) -> None:
    """Put a consumed request body back so FastAPI can parse it later."""

    async def _receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive  # type: ignore[attr-defined]  # noqa: SLF001


def _flagged_categories(moderation: Any) -> List[str]:
    """Delegate to the shared moderation helper."""
    return flagged_categories(moderation)


async def _moderate_query_text(query: str) -> List[str]:
    """Delegate to the shared moderation helper."""
    return await moderate_query(query)


def _response_media_type(response: Response) -> str:
    return (response.media_type or response.headers.get("content-type") or "").lower()


async def _read_response_body(response: Response) -> bytes:
    """Consume a Starlette response body into bytes."""
    return b"".join([chunk async for chunk in response.body_iterator])


def _unclosed_fence_error(markdown: str) -> Optional[str]:
    """Detect unclosed fenced code blocks before markdown-it normalises them."""
    opener: Optional[str] = None
    opener_char = ""
    opener_len = 0
    for line_no, line in enumerate(markdown.splitlines(), start=1):
        match = _FENCE_LINE.match(line)
        if not match:
            continue
        fence = match.group(1)
        if opener is None:
            opener = fence
            opener_char = fence[0]
            opener_len = len(fence)
            continue
        if fence[0] == opener_char and len(fence) >= opener_len:
            opener = None
            opener_char = ""
            opener_len = 0
    if opener is not None:
        return "Unclosed fenced code block."
    return None


def _table_error(markdown: str) -> Optional[str]:
    """Detect common malformed GitHub-style markdown tables."""
    lines = markdown.splitlines()
    for i in range(len(lines) - 1):
        header = lines[i].strip()
        separator = lines[i + 1].strip()
        if "|" not in header or "|" not in separator:
            continue
        if not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", separator):
            continue
        header_cols = [c.strip() for c in header.strip("|").split("|")]
        sep_cols = [c.strip() for c in separator.strip("|").split("|")]
        if len(header_cols) != len(sep_cols):
            return "Markdown table header and separator column counts differ."
        for j in range(i + 2, len(lines)):
            row = lines[j].strip()
            if not row or "|" not in row:
                break
            row_cols = [c.strip() for c in row.strip("|").split("|")]
            if len(row_cols) != len(header_cols):
                return "Markdown table row column count differs from the header."
    return None


def _markdown_validation_error(markdown: str) -> Optional[str]:
    """Return a human-readable issue if markdown is structurally malformed."""
    fence_error = _unclosed_fence_error(markdown)
    if fence_error:
        return fence_error
    table_error = _table_error(markdown)
    if table_error:
        return table_error
    try:
        _MARKDOWN.parse(markdown)
    except Exception as exc:  # noqa: BLE001
        return f"Markdown parser rejected the response: {exc}"
    return None


def _markdown_log_excerpt(markdown: str) -> str:
    """Return markdown text as seen by FastAPI, capped for log safety."""
    if len(markdown) <= _MAX_MARKDOWN_LOG_CHARS:
        return markdown
    return (
        markdown[:_MAX_MARKDOWN_LOG_CHARS]
        + f"\n... [truncated; total_chars={len(markdown)}]"
    )


# ---------------------------------------------------------------------
# Lifespan: run RAG ingestion before serving
# ---------------------------------------------------------------------


def _build_lifespan(cfg: AppConfig):
    """Return a lifespan context manager bound to a specific config."""

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Portfolio seeding is a fast, synchronous directory-copy. We
        # run it before RAG ingestion so the Portfolio agent + UI
        # graphics always see a populated data_path, and any seed
        # failure is logged before the (much slower) RAG step starts.
        try:
            seeded = seed_portfolio_if_needed(cfg)
            _logger.info("Portfolio seed ready at %s", seeded)
        except Exception:  # noqa: BLE001 -- never block startup
            _logger.exception("Portfolio seeding failed; continuing startup")

        try:
            await start_portfolio_mcp_sessions(cfg)
        except Exception:  # noqa: BLE001 -- never block RAG / serving
            _logger.exception("Portfolio MCP startup failed; continuing without tools")

        try:
            await start_market_research_mcp_sessions(cfg)
        except Exception:  # noqa: BLE001 -- never block RAG / serving
            _logger.exception(
                "Market Research MCP startup failed; continuing without tools"
            )

        try:
            await start_goal_planning_mcp_sessions(cfg)
        except Exception:  # noqa: BLE001 -- never block RAG / serving
            _logger.exception("Goal Planning MCP startup failed; continuing without tools")

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
        try:
            await stop_portfolio_mcp_sessions()
        except BaseException as exc:  # noqa: BLE001 -- CancelledError not Exception
            if is_lifespan_shutdown_noise(exc):
                _logger.debug("Portfolio MCP lifespan shutdown: %s", exc)
            else:
                _logger.exception("Portfolio MCP shutdown failed")
        try:
            await stop_market_research_mcp_sessions()
        except BaseException as exc:  # noqa: BLE001
            if is_lifespan_shutdown_noise(exc):
                _logger.debug("Market Research MCP lifespan shutdown: %s", exc)
            else:
                _logger.exception("Market Research MCP shutdown failed")
        try:
            await stop_goal_planning_mcp_sessions()
        except BaseException as exc:  # noqa: BLE001
            if is_lifespan_shutdown_noise(exc):
                _logger.debug("Goal Planning MCP lifespan shutdown: %s", exc)
            else:
                _logger.exception("Goal Planning MCP shutdown failed")

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

    @app.middleware("http")
    async def query_guardrails(request: Request, call_next: Any) -> Response:
        """Moderate incoming queries and validate outgoing markdown answers."""
        if request.url.path != _QUERY_PATH or request.method.upper() != "POST":
            return await call_next(request)

        body = await request.body()
        _restore_request_body(request, body)

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            # Let FastAPI's request model validation return the canonical error.
            return await call_next(request)

        query_text = payload.get("query")
        if isinstance(query_text, str) and query_text.strip():
            try:
                categories = await _moderate_query_text(query_text)
            except OpenAIError as exc:
                _logger.exception("OpenAI moderation failed")
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"Moderation check failed: {exc}"},
                )

            if categories:
                _logger.info("Moderation rejected query (categories=%s)", categories)
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": REJECTION_PREFIX,
                        "categories": categories,
                    },
                )

        response = await call_next(request)
        media_type = _response_media_type(response)
        if response.status_code >= 400 or "application/json" not in media_type:
            return response

        response_body = await _read_response_body(response)
        try:
            response_payload = json.loads(response_body or b"{}")
        except json.JSONDecodeError:
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        answer = response_payload.get("answer")
        if isinstance(answer, str):
            error = _markdown_validation_error(answer)
            if error:
                _logger.warning(
                    "Backend produced markdown with validation warning: %s\n"
                    "Markdown seen by FastAPI:\n%s",
                    error,
                    _markdown_log_excerpt(answer),
                )

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

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

    @app.post("/portfolio/refresh", response_model=PortfolioRefreshResponse)
    def portfolio_refresh() -> PortfolioRefreshResponse:
        """Clear the backend's portfolio LRU cache.

        Called by the Streamlit UI after a successful portfolio upload so
        that the Portfolio agent reads the freshly written JSON files on
        its next invocation instead of serving the stale cached snapshot.
        """
        load_portfolio(force_refresh=True)
        _logger.info("Portfolio cache cleared via /portfolio/refresh")
        return PortfolioRefreshResponse(refreshed=True)

    @app.get("/model", response_model=ModelResponse)
    def get_model() -> ModelResponse:
        """Return the name of the currently active chat model."""
        return ModelResponse(model=get_current_model())

    @app.post("/model", response_model=ModelResponse)
    def set_model(request: ModelSetRequest) -> ModelResponse:
        """Switch the active chat model without restarting the server.

        Clears the LRU cache so the next agent call uses the new model.
        """
        set_current_model(request.model)
        _logger.info("Chat model switched to %s", request.model)
        return ModelResponse(model=get_current_model())

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
            detail = (
                context_overflow_user_message("question")
                if is_context_overflow_error(exc)
                else "I hit an internal error while answering that question. Please try again."
            )
            raise HTTPException(status_code=500, detail=detail) from exc

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
