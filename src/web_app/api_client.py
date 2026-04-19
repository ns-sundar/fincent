"""Thin HTTP client used by the Streamlit UI to call the FastAPI server."""

from __future__ import annotations

from typing import Optional

import requests

from src.core.schemas import QueryRequest, QueryResponse


class FincentApiError(RuntimeError):
    """Raised on any non-2xx response from the Fincent API."""


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
        session_id: Optional opaque session id (reserved for future use).
        timeout: Per-request timeout in seconds.

    Returns:
        A typed ``QueryResponse``.

    Raises:
        FincentApiError: If the server returns an error status.
    """
    url = base_url.rstrip("/") + "/query"
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


def health(base_url: str, *, timeout: int = 5) -> bool:
    """Return True if the API ``/health`` endpoint is reachable and OK."""
    try:
        resp = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False
