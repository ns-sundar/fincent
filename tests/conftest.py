"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
import tempfile
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

# The Portfolio agent loads MCP tools from the OpenBB and Fincent RAG
# MCP servers in stdio mode, which spawns real subprocesses. Tests must
# never spawn those subprocesses implicitly; individual tests that
# exercise the tool-calling path inject fake tools via the ``tools=``
# kwarg on ``answer()`` and must do so explicitly.
os.environ.setdefault("FINCENT__PORTFOLIO__TOOLS__OPENBB__ENABLED", "false")
os.environ.setdefault("FINCENT__PORTFOLIO__TOOLS__RAG__ENABLED", "false")
os.environ.setdefault("FINCENT__MARKET_RESEARCH__TOOLS__OPENBB__ENABLED", "false")
os.environ.setdefault(
    "FINCENT__MARKET_RESEARCH__TOOLS__ALPHA_VANTAGE__ENABLED", "false"
)
os.environ.setdefault("FINCENT__MARKET_RESEARCH__TOOLS__TAVILY__ENABLED", "false")
os.environ.setdefault("FINCENT__MARKET_RESEARCH__TOOLS__FMP__ENABLED", "false")

# Keep portfolio seeding off the real ``/data`` host mount during
# tests. The seed copier runs once per process into this dir and then
# every ``load_portfolio()`` call reads from it -- so tests exercise
# the same code path as production without touching the real
# ``/data/portfolio``.
_PORTFOLIO_TEST_DIR = Path(tempfile.mkdtemp(prefix="fincent-portfolio-test-"))
os.environ.setdefault("FINCENT__PORTFOLIO__DATA_PATH", str(_PORTFOLIO_TEST_DIR))


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Reset cached config between tests so env overrides take effect."""
    from src.agents.market_research.mcp_tools import reset_market_research_tools_cache
    from src.agents.portfolio.mcp_tools import reset_portfolio_tools_cache
    from src.core.config import reset_config_cache
    from src.rag.retriever import reset_default_retriever
    from src.rag.status import reset_status

    reset_config_cache()
    reset_default_retriever()
    reset_status()
    reset_market_research_tools_cache()
    reset_portfolio_tools_cache()
    yield
    reset_config_cache()
    reset_default_retriever()
    reset_status()
    reset_market_research_tools_cache()
    reset_portfolio_tools_cache()
