"""Placeholder retriever for future RAG-enabled agents.

This intentionally returns no documents; specialist agents that need
RAG should depend on this interface and be swapped to a real
implementation (e.g. FAISS, Chroma) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedDoc:
    """A single retrieval result."""

    text: str
    source: str
    score: float = 0.0


def retrieve(query: str, *, k: int = 4) -> List[RetrievedDoc]:
    """Stub that returns an empty list.

    Args:
        query: The user query.
        k: Maximum number of documents to return.

    Returns:
        An empty list (the real implementation will land in a later
        iteration).
    """
    _ = (query, k)
    return []
