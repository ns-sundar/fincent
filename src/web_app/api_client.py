"""Thin HTTP client used by the Streamlit UI to call the FastAPI server."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import requests

from src.core.schemas import QueryRequest, QueryResponse


class FincentApiError(RuntimeError):
    """Raised on any non-2xx response from the Fincent API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or message
        self.categories = categories or []


def _api_error_from_response(resp: requests.Response, *, fallback: str) -> FincentApiError:
    """Preserve structured API error details for UI rendering."""
    detail = fallback
    categories: List[str] = []
    try:
        payload = resp.json() or {}
    except (ValueError, json.JSONDecodeError):
        payload = {}

    if isinstance(payload, dict):
        raw_detail = payload.get("detail")
        if isinstance(raw_detail, str) and raw_detail:
            detail = raw_detail
        raw_categories = payload.get("categories")
        if isinstance(raw_categories, list):
            categories = [str(c) for c in raw_categories]

    if categories:
        message = f"{detail} " + ", ".join(categories)
    else:
        message = f"{detail} ({resp.status_code})"
    return FincentApiError(
        message,
        status_code=resp.status_code,
        detail=detail,
        categories=categories,
    )


# LangGraph/LangChain -> Streamlit role mapping. Keep in sync with
# ``src.workflow.graph._LANGGRAPH_TO_STREAMLIT_ROLE``.
_LANGGRAPH_TO_STREAMLIT_ROLE: Dict[str, str] = {
    "human": "user",
    "ai": "assistant",
}


def _url(base_url: str, path: str) -> str:
    """Join a base URL with a path, trimming any trailing slash."""
    return base_url.rstrip("/") + path


def query_fincent(
    base_url: str,
    query: str,
    *,
    session_id: Optional[str] = None,
    intent_hint: Optional[str] = None,
    timeout: int = 120,
) -> QueryResponse:
    """POST a user query to the FastAPI ``/query`` endpoint.

    Args:
        base_url: Root URL of the running FastAPI server.
        query: Natural-language question from the user.
        session_id: Thread id used by the LangGraph checkpointer so
            that state persists across requests.
        intent_hint: Optional caller-pinned intent (e.g. ``"portfolio"``
            from the Portfolio tab). The server's planner will skip the
            LLM classifier and dispatch directly to the hinted
            specialist.
        timeout: Per-request timeout in seconds.

    Returns:
        A typed ``QueryResponse``.

    Raises:
        FincentApiError: If the server returns an error status.
    """
    url = _url(base_url, "/query")
    payload = QueryRequest(
        query=query,
        session_id=session_id,
        intent_hint=intent_hint,
    ).model_dump(mode="json")
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc

    if resp.status_code >= 400:
        raise _api_error_from_response(
            resp, fallback=f"Fincent API error {resp.status_code}: {resp.text[:500]}"
        )
    return QueryResponse.model_validate(resp.json())


def get_history(
    base_url: str,
    thread_id: str,
    *,
    timeout: int = 15,
) -> List[Dict[str, str]]:
    """Fetch the chat transcript for ``thread_id``, converting roles.

    Returns:
        A list of ``{"role": "user"|"assistant", "content": str}``
        dicts -- already in the format Streamlit expects.
    """
    url = _url(base_url, f"/history/{thread_id}")
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FincentApiError(
            f"History fetch failed ({resp.status_code}): {resp.text[:500]}"
        )
    payload = resp.json() or {}
    raw = payload.get("messages") or []

    out: List[Dict[str, str]] = []
    for m in raw:
        role = _LANGGRAPH_TO_STREAMLIT_ROLE.get(m.get("role", ""), None)
        if role is None:
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out


def reset_thread(
    base_url: str,
    thread_id: str,
    *,
    timeout: int = 15,
) -> int:
    """Invoke ``POST /reset/{thread_id}``.

    Returns:
        The number of messages that were removed from the current
        state (the SQLite checkpoint log still keeps prior versions).
    """
    url = _url(base_url, f"/reset/{thread_id}")
    try:
        resp = requests.post(url, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FincentApiError(
            f"Reset failed ({resp.status_code}): {resp.text[:500]}"
        )
    return int((resp.json() or {}).get("removed", 0))


def get_model(base_url: str, *, timeout: int = 5) -> str:
    """Return the name of the chat model currently active in the backend.

    Falls back to an empty string if the backend is unreachable so
    callers can display a sensible default without crashing.
    """
    url = _url(base_url, "/model")
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code < 400:
            return str((resp.json() or {}).get("model", ""))
    except requests.RequestException:
        pass
    return ""


def set_model(base_url: str, model: str, *, timeout: int = 10) -> str:
    """Tell the backend to switch to *model* and return the active model name.

    Raises:
        FincentApiError: If the backend is unreachable or returns an error.
    """
    url = _url(base_url, "/model")
    try:
        resp = requests.post(url, json={"model": model}, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FincentApiError(
            f"Model switch failed ({resp.status_code}): {resp.text[:500]}"
        )
    return str((resp.json() or {}).get("model", model))


def refresh_portfolio(base_url: str, *, timeout: int = 10) -> None:
    """Tell the FastAPI backend to clear its portfolio LRU cache.

    Should be called after a successful portfolio upload so the Portfolio
    agent reads the newly written files on its next query.

    Raises:
        FincentApiError: If the backend is unreachable or returns an error.
    """
    url = _url(base_url, "/portfolio/refresh")
    try:
        resp = requests.post(url, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FincentApiError(
            f"Portfolio refresh failed ({resp.status_code}): {resp.text[:500]}"
        )


def health(base_url: str, *, timeout: int = 5) -> bool:
    """Return True if the API ``/health`` endpoint is reachable and OK."""
    try:
        resp = requests.get(_url(base_url, "/health"), timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def rag_status(
    base_url: str, *, timeout: int = 5
) -> Dict[str, object]:
    """Fetch the RAG ingestion status from the backend.

    Returns:
        A dict with at least a ``state`` key. When the backend is
        unreachable or responds with an error, the dict is a synthetic
        ``{"state": "unknown", "detail": <reason>}`` payload so the UI
        can render a neutral banner rather than crashing.
    """
    url = _url(base_url, "/rag/status")
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return {"state": "unknown", "detail": f"Could not reach {url}: {exc}"}
    if resp.status_code >= 400:
        return {
            "state": "unknown",
            "detail": f"HTTP {resp.status_code} from {url}",
        }
    try:
        return dict(resp.json() or {})
    except ValueError:
        return {"state": "unknown", "detail": "Malformed JSON from /rag/status"}
