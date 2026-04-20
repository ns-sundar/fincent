"""Thin HTTP client used by the Streamlit UI to call the FastAPI server."""

from __future__ import annotations

from typing import Dict, List, Optional

import requests

from src.core.schemas import QueryRequest, QueryResponse


class FincentApiError(RuntimeError):
    """Raised on any non-2xx response from the Fincent API."""


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
    timeout: int = 120,
) -> QueryResponse:
    """POST a user query to the FastAPI ``/query`` endpoint.

    Args:
        base_url: Root URL of the running FastAPI server.
        query: Natural-language question from the user.
        session_id: Thread id used by the LangGraph checkpointer so
            that state persists across requests.
        timeout: Per-request timeout in seconds.

    Returns:
        A typed ``QueryResponse``.

    Raises:
        FincentApiError: If the server returns an error status.
    """
    url = _url(base_url, "/query")
    payload = QueryRequest(query=query, session_id=session_id).model_dump()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise FincentApiError(f"Network error talking to {url}: {exc}") from exc

    if resp.status_code >= 400:
        raise FincentApiError(
            f"Fincent API error {resp.status_code}: {resp.text[:500]}"
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


def health(base_url: str, *, timeout: int = 5) -> bool:
    """Return True if the API ``/health`` endpoint is reachable and OK."""
    try:
        resp = requests.get(_url(base_url, "/health"), timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False
