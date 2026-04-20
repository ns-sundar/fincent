"""Unit tests for the Q&A agent (now RAG-enabled)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.qna import answer
from src.core.schemas import AgentName
from src.rag.retriever import RetrievedDoc, Retriever


class _FakeRetriever(Retriever):
    """Retriever stub that returns a fixed list of chunks for every query."""

    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs
        self._default_k = len(docs) or 1
        # Fake a vector store with a matching ``.index.ntotal``.
        self._vs = SimpleNamespace(index=SimpleNamespace(ntotal=len(docs)))

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        *,
        k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        return list(self._docs[: (k or self._default_k)])


def test_qna_without_retriever_does_not_use_rag():
    """With no retriever available, the agent answers from the base prompt only."""
    fake = FakeListChatModel(responses=["An ETF is an exchange-traded fund."])
    response = answer("What is an ETF?", llm=fake, retriever=None)

    assert response.agent == AgentName.QNA
    assert "ETF" in response.content
    assert response.metadata.get("rag_used") is False
    assert response.metadata.get("rag_chunk_count") == 0
    assert "sources" not in response.metadata


def test_qna_with_retriever_uses_rag_and_emits_sources():
    """When the retriever yields hits, metadata lists the cited sources."""
    retriever = _FakeRetriever(
        [
            RetrievedDoc(
                text="An ETF is a basket of securities traded like a stock.",
                url="https://www.sec.gov/etf-guide",
                title="SEC: ETF Basics",
                tags={"source": "sec", "category": ["etf", "basics"]},
                score=0.12,
            )
        ]
    )
    fake = FakeListChatModel(
        responses=["An ETF is a basket of securities traded like a stock."]
    )
    response = answer("What is an ETF?", llm=fake, retriever=retriever)

    assert response.agent == AgentName.QNA
    assert response.metadata["rag_used"] is True
    assert response.metadata["rag_chunk_count"] == 1
    sources = response.metadata["sources"]
    assert sources[0]["url"] == "https://www.sec.gov/etf-guide"
    assert sources[0]["title"] == "SEC: ETF Basics"
    assert sources[0]["tags"]["source"] == "sec"
