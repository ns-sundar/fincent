"""Retrieval-Augmented Generation pipeline.

Public surface:

* :mod:`src.rag.ingest` -- one-shot ingestion pipeline invoked by the
  FastAPI lifespan hook at startup.
* :mod:`src.rag.retriever` -- FAISS-backed retriever used at query
  time by the Q&A agent.
* :mod:`src.rag.status` -- thread-safe singleton describing the state
  of the ingestion pipeline (exposed via ``/rag/status``).
"""

from src.rag.ingest import ingest_if_needed  # noqa: F401  (re-export)
from src.rag.retriever import (  # noqa: F401  (re-export)
    RetrievedDoc,
    Retriever,
    get_default_retriever,
    retrieve,
)
from src.rag.status import RagStatus, get_status  # noqa: F401  (re-export)
