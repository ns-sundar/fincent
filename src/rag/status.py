"""In-process status tracker for the RAG ingestion pipeline.

The FastAPI lifespan hook runs ingestion at startup and writes the
outcome here. Both the ``/rag/status`` endpoint and the Q&A agent
read this state to decide how to react (serve with banner, skip
retrieval, etc.).
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Lifecycle states exposed to the UI / callers.
STATE_PENDING: str = "pending"
STATE_INGESTING: str = "ingesting"
STATE_READY: str = "ready"
STATE_SKIPPED: str = "skipped"  # index already present on disk
STATE_DISABLED: str = "disabled"  # rag.enabled=false
STATE_FAILED: str = "failed"


@dataclass
class RagStatus:
    """Snapshot of the RAG ingestion pipeline state."""

    state: str = STATE_PENDING
    # Human-readable detail (e.g. "62 chunks ingested from 59 docs").
    detail: str = ""
    # Non-empty only when ``state == STATE_FAILED``.
    error: Optional[str] = None
    # Number of Document chunks in the active vector store (0 if unknown).
    chunk_count: int = 0
    # Number of source articles successfully ingested during this run.
    ingested_articles: int = 0
    # Extra free-form metadata (e.g. per-URL failures).
    meta: Dict[str, Any] = field(default_factory=dict)


# Module-level singleton guarded by a lock because the lifespan writer
# and request-serving readers run on different threads.
_lock: threading.Lock = threading.Lock()
_status: RagStatus = RagStatus()


def get_status() -> RagStatus:
    """Return a snapshot of the current RAG status (thread-safe)."""
    with _lock:
        # Return a copy so callers can't mutate the singleton.
        return RagStatus(**asdict(_status))


def set_status(**fields: Any) -> RagStatus:
    """Update the RAG status singleton and return the new snapshot."""
    with _lock:
        for key, value in fields.items():
            if hasattr(_status, key):
                setattr(_status, key, value)
        return RagStatus(**asdict(_status))


def reset_status() -> None:
    """Reset the singleton back to ``pending`` (used by tests)."""
    with _lock:
        global _status
        _status = RagStatus()
