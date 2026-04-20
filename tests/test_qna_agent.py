"""Unit tests for the Q&A agent (agentic RAG + source filtering)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.qna import answer
from src.core.schemas import AgentName
from src.rag.retriever import RetrievedDoc, Retriever


class _FakeRetriever(Retriever):
    """Retriever stub that replays a fixed list of chunks and records the call."""

    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs
        self._default_k = len(docs) or 1
        self._default_use_mmr = True
        self._default_fetch_k = 20
        self._default_lambda_mult = 0.5
        # Fake vector store with a matching ``.index.ntotal``.
        self._vs = SimpleNamespace(index=SimpleNamespace(ntotal=len(docs)))
        self.last_call: dict = {}

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        *,
        k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        self.last_call = {
            "query": query,
            "k": k,
            "use_mmr": use_mmr,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "source_filter": source_filter,
        }
        if source_filter:
            filtered = [
                d
                for d in self._docs
                if str((d.tags or {}).get("source") or "").lower()
                == source_filter.lower()
            ]
        else:
            filtered = list(self._docs)
        return filtered[: (k or self._default_k)]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _irs_doc() -> RetrievedDoc:
    return RetrievedDoc(
        text="An IRA is a retirement savings vehicle with tax advantages.",
        url="https://www.irs.gov/retirement-plans/iras",
        title="IRS: Individual Retirement Arrangements (IRAs)",
        tags={"source": "irs", "category": ["tax", "retirement"]},
        score=0.11,
    )


def _sec_doc() -> RetrievedDoc:
    return RetrievedDoc(
        text="An ETF is a pooled investment vehicle that trades on an exchange.",
        url="https://www.sec.gov/etf-guide",
        title="SEC: ETF Basics",
        tags={"source": "sec", "category": ["etf", "basics"]},
        score=0.12,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_qna_without_retriever_does_not_use_rag(monkeypatch):
    """With no retriever available, the agent answers from the base prompt only."""
    # Ensure ``get_default_retriever`` also returns None.
    from src.rag import tool as tool_mod

    monkeypatch.setattr(tool_mod, "get_default_retriever", lambda: None)

    fake = FakeListChatModel(responses=["An ETF is an exchange-traded fund."])
    response = answer("What is an ETF?", llm=fake, retriever=None)

    assert response.agent == AgentName.QNA
    assert "ETF" in response.content
    assert response.metadata.get("rag_used") is False
    assert response.metadata.get("rag_chunk_count") == 0
    assert "sources" not in response.metadata
    assert "## Sources" not in response.content


def _llm_answer_with_sources(body: str, *, url: str) -> str:
    """Simulate a compliant model: prose + ## Sources with URL on its own line."""
    return (
        f"{body}\n\n"
        "## Sources\n\n"
        "[1] …\n"
        f"{url}"
    )


def test_qna_with_retriever_uses_rag_and_emits_sources():
    """Retriever hits + [1] in prose -> metadata.sources; model adds ## Sources."""
    retriever = _FakeRetriever([_sec_doc()])
    fake = FakeListChatModel(
        responses=[
            _llm_answer_with_sources(
                "An ETF is a basket of securities traded like a stock [1].",
                url="https://www.sec.gov/etf-guide",
            )
        ]
    )
    response = answer("What is an ETF?", llm=fake, retriever=retriever)

    assert response.agent == AgentName.QNA
    assert response.metadata["rag_used"] is True
    assert response.metadata["rag_chunk_count"] == 1
    sources = response.metadata["sources"]
    assert sources[0]["url"] == "https://www.sec.gov/etf-guide"
    assert sources[0]["title"] == "SEC: ETF Basics"
    assert sources[0]["tags"]["source"] == "sec"
    assert "## Sources" in response.content
    assert "https://www.sec.gov/etf-guide" in response.content
    assert "[1]" in response.content
    assert "\nhttps://www.sec.gov/etf-guide" in response.content


def test_qna_auto_detects_source_filter_from_query():
    """'Cite only IRS documents' restricts retrieval to source=irs."""
    retriever = _FakeRetriever([_irs_doc(), _sec_doc()])
    fake = FakeListChatModel(
        responses=[
            _llm_answer_with_sources(
                "An IRA is a retirement account [1].",
                url="https://www.irs.gov/retirement-plans/iras",
            )
        ]
    )

    response = answer(
        "Cite only IRS documents: what is an IRA?",
        llm=fake,
        retriever=retriever,
    )

    assert retriever.last_call["source_filter"] == "irs"
    assert response.metadata["source_filter"] == "irs"
    assert response.metadata["rag_chunk_count"] == 1
    assert response.metadata["sources"][0]["tags"]["source"] == "irs"
    # No SEC source leaked into the trailer.
    assert "sec.gov" not in response.content.lower()


def test_qna_explicit_source_filter_overrides_detection():
    """An explicit source_filter argument wins over query parsing."""
    retriever = _FakeRetriever([_irs_doc(), _sec_doc()])
    fake = FakeListChatModel(
        responses=[
            _llm_answer_with_sources(
                "Answer based on SEC only [1].",
                url="https://www.sec.gov/etf-guide",
            )
        ]
    )

    response = answer(
        "Based on IRS documents, what is an ETF?",  # would auto-detect 'irs'
        llm=fake,
        retriever=retriever,
        source_filter="sec",
    )

    assert retriever.last_call["source_filter"] == "sec"
    assert response.metadata["source_filter"] == "sec"
    assert response.metadata["sources"][0]["tags"]["source"] == "sec"


def test_qna_top_k_override_is_respected():
    """The ``top_k`` kwarg flows into the retriever call."""
    retriever = _FakeRetriever([_irs_doc(), _sec_doc()])
    fake = FakeListChatModel(responses=["answer"])
    answer("What is an ETF?", llm=fake, retriever=retriever, top_k=1)
    assert retriever.last_call["k"] == 1


def test_qna_sources_only_lists_cited_indices():
    """Retrieved two chunks but only [1] in the answer -> one Sources entry."""
    # Context [1] = SEC, [2] = IRS — citing only [1] must not list IRS.
    retriever = _FakeRetriever([_sec_doc(), _irs_doc()])
    fake = FakeListChatModel(
        responses=[
            _llm_answer_with_sources(
                "The SEC covers ETFs [1].",
                url="https://www.sec.gov/etf-guide",
            )
        ]
    )
    response = answer("What is an ETF?", llm=fake, retriever=retriever)

    assert response.metadata["rag_chunk_count"] == 2
    assert response.metadata["cited_chunk_indices"] == [1]
    assert len(response.metadata["sources"]) == 1
    assert response.metadata["sources"][0]["tags"]["source"] == "sec"
    assert "irs.gov" not in response.content.lower()
    assert "[2]" not in response.content


def test_qna_no_inline_citations_yields_no_sources_trailer():
    """No [n] markers -> empty sources metadata; model should omit ## Sources."""
    retriever = _FakeRetriever([_sec_doc()])
    fake = FakeListChatModel(
        responses=["ETFs trade on exchanges like stocks."]
    )
    response = answer("What is an ETF?", llm=fake, retriever=retriever)

    assert response.metadata["sources"] == []
    assert response.metadata.get("cited_chunk_indices") == []
    assert "## Sources" not in response.content


def test_qna_llm_output_preserved_including_sources_heading():
    """Model-written ## Citations / ## Sources text is not stripped."""
    retriever = _FakeRetriever([_sec_doc()])
    raw = (
        "ETFs pool assets [1].\n\n"
        "## Citations\n\n"
        "- legacy heading per old habit\n\n"
        "## Sources\n\n"
        "[1] SEC: ETF Basics (SEC)\n"
        "https://www.sec.gov/etf-guide"
    )
    fake = FakeListChatModel(responses=[raw])
    response = answer("What is an ETF?", llm=fake, retriever=retriever)

    assert "## Citations" in response.content
    assert "legacy heading" in response.content
    assert "## Sources" in response.content
    assert response.metadata["sources"][0]["tags"]["source"] == "sec"
