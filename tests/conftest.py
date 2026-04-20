"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``src`` importable when running pytest from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# A dummy key prevents the LLM factory from raising during accidental
# real-LLM construction; tests should always inject fake LLMs.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

# Default production path is /data/checkpoints.sqlite (outside the repo).
# Tests must not require that directory; use an in-memory SQLite DB.
os.environ.setdefault("FINCENT__CHECKPOINTER__PATH", ":memory:")

# RAG ingestion hits the network + OpenAI embeddings. Unit tests must
# never run it as a side effect of importing the FastAPI app; specific
# RAG tests re-enable it explicitly.
os.environ.setdefault("FINCENT__RAG__ENABLED", "false")


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Reset cached config between tests so env overrides take effect."""
    from src.core.config import reset_config_cache
    from src.rag.retriever import reset_default_retriever
    from src.rag.status import reset_status

    reset_config_cache()
    reset_default_retriever()
    reset_status()
    yield
    reset_config_cache()
    reset_default_retriever()
    reset_status()
