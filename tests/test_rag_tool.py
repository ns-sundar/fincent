"""Tests for the canonical RAG search tool and source-filter detection."""

from __future__ import annotations

from typing import List, Optional

from src.rag.retriever import RetrievedDoc, Retriever
from src.rag.tool import detect_source_filter, rag_search, to_wire


# ---------------------------------------------------------------------
# detect_source_filter
# ---------------------------------------------------------------------


def test_detect_only_irs():
    assert detect_source_filter("Cite only IRS documents") == "irs"


def test_detect_based_on_sec():
    assert detect_source_filter("Based on SEC documents, what is Reg T?") == "sec"


def test_detect_per_finra():
    assert detect_source_filter("Per FINRA rules, explain a margin call.") == "finra"


def test_detect_alias_federal_reserve():
    assert detect_source_filter("According to the Federal Reserve, ...") == "fed"


def test_detect_none_when_absent():
    assert detect_source_filter("What is an ETF?") is None


def test_detect_unknown_source_returns_none():
    # "Wikipedia" is not in the ingested catalog -- no filter applied.
    assert detect_source_filter("Based on Wikipedia, what is an ETF?") is None


# ---------------------------------------------------------------------
# rag_search -- uses a fake retriever injection
# ---------------------------------------------------------------------


class _RecordingRetriever(Retriever):
    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs
        self._default_k = 5
        self._default_use_mmr = True
        self._default_fetch_k = 20
        self._default_lambda_mult = 0.5
        self._vs = None
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
            return [
                d
                for d in self._docs
                if (d.tags or {}).get("source") == source_filter
            ][: (k or self._default_k)]
        return list(self._docs)[: (k or self._default_k)]


def _fixture_docs() -> List[RetrievedDoc]:
    return [
        RetrievedDoc(
            text="IRS guidance on IRAs.",
            url="https://irs.gov/ira",
            title="IRS IRAs",
            tags={"source": "irs", "category": ["tax"]},
            score=0.1,
        ),
        RetrievedDoc(
            text="SEC ETF primer.",
            url="https://sec.gov/etf",
            title="SEC ETFs",
            tags={"source": "sec", "category": ["etf"]},
            score=0.2,
        ),
    ]


def test_rag_search_defaults_come_from_cfg(monkeypatch):
    """Unspecified knobs fall through to ``cfg.rag``."""
    monkeypatch.setenv("FINCENT__RAG__ENABLED", "true")
    monkeypatch.setenv("FINCENT__RAG__TOP_K", "5")
    monkeypatch.setenv("FINCENT__RAG__USE_MMR", "true")
    monkeypatch.setenv("FINCENT__RAG__MMR_FETCH_K", "20")
    monkeypatch.setenv("FINCENT__RAG__MMR_LAMBDA", "0.5")
    from src.core.config import reset_config_cache

    reset_config_cache()

    ret = _RecordingRetriever(_fixture_docs())
    hits = rag_search("what is an etf?", retriever=ret)

    assert len(hits) == 2
    assert ret.last_call["k"] == 5
    assert ret.last_call["use_mmr"] is True
    assert ret.last_call["fetch_k"] == 20
    assert ret.last_call["lambda_mult"] == 0.5
    assert ret.last_call["source_filter"] is None


def test_rag_search_applies_source_alias():
    """An alias like ``"Federal Reserve"`` normalises to the canonical tag."""
    ret = _RecordingRetriever(_fixture_docs())
    rag_search("anything", retriever=ret, source="Federal Reserve")
    assert ret.last_call["source_filter"] == "fed"


def test_rag_search_respects_explicit_topk_and_mmr_knobs():
    ret = _RecordingRetriever(_fixture_docs())
    rag_search(
        "query",
        retriever=ret,
        top_k=1,
        use_mmr=False,
        mmr_fetch_k=7,
        mmr_lambda=0.9,
    )
    assert ret.last_call["k"] == 1
    assert ret.last_call["use_mmr"] is False
    assert ret.last_call["fetch_k"] == 7
    assert ret.last_call["lambda_mult"] == 0.9


def test_rag_search_returns_empty_when_no_retriever(monkeypatch):
    """Missing retriever -> empty list, never exception."""
    from src.rag import tool as tool_mod

    monkeypatch.setattr(tool_mod, "get_default_retriever", lambda: None)
    assert rag_search("q") == []


def test_to_wire_round_trip():
    wire = to_wire(_fixture_docs()[0])
    assert wire["url"] == "https://irs.gov/ira"
    assert wire["title"] == "IRS IRAs"
    assert wire["tags"]["source"] == "irs"
    assert isinstance(wire["score"], float)
