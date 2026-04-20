"""Canonical RAG search function shared by the Q&A agent and MCP server.

Both entry points (the in-process Q&A agent and the external MCP
server at ``src.rag.mcp_server``) delegate to :func:`rag_search` so
that there is exactly one retrieval code path to reason about.

Capabilities:

* Top-k with MMR re-ranking (defaults from ``cfg.rag``).
* Optional ``source`` filter keyed on ``metadata.tags.source`` stamped
  at ingestion (e.g. ``"irs"``, ``"sec"``, ``"finra"``).
* A natural-language detector (:func:`detect_source_filter`) that
  extracts the intended source from phrases such as
  "cite only IRS documents", "based on SEC documents", etc. The Q&A
  agent calls this to honour the user's narrowing instruction
  automatically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.config import AppConfig, get_config
from src.rag.retriever import (
    Retriever,
    RetrievedDoc,
    get_default_retriever,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Source vocabulary -- keep aligned with the values used in
# ``rag/fincent_rag_articles.json``. Aliases on the right-hand side map
# colloquial/user-facing names to the canonical metadata tag.
# ---------------------------------------------------------------------

_SOURCE_ALIASES: Dict[str, str] = {
    # Canonical identifiers (lowercase) used in the catalog.
    "irs": "irs",
    "sec": "sec",
    "finra": "finra",
    "fed": "fed",
    "fdic": "fdic",
    "occ": "occ",
    "treasury": "treasury",
    "cbp": "cbp",
    "nyse": "nyse",
    "investopedia": "investopedia",
    "bogleheads": "bogleheads",
    "fidelity": "fidelity",
    "tax-foundation": "tax-foundation",
    # Common aliases.
    "federal reserve": "fed",
    "us treasury": "treasury",
    "u.s. treasury": "treasury",
    "customs": "cbp",
    "new york stock exchange": "nyse",
    "tax foundation": "tax-foundation",
}

# The set of canonical source values the tool will expose to clients.
KNOWN_SOURCES: List[str] = sorted(set(_SOURCE_ALIASES.values()))

# Phrases like "only IRS", "based on SEC documents", "cite IRS only",
# "ground on FINRA", "using SEC", "per IRS", "from the IRS".
_SOURCE_FILTER_PATTERN = re.compile(
    r"""
    \b(?:
        only\s+|
        cite\s+(?:only\s+)?|
        based\s+(?:only\s+)?on\s+|
        grounded?\s+(?:only\s+)?on\s+|
        restricted?\s+to\s+|
        using\s+only\s+|
        from\s+(?:the\s+)?|
        per\s+(?:the\s+)?|
        according\s+to\s+(?:the\s+)?
    )
    (?P<source>[A-Za-z][A-Za-z .\-]{1,24}?)
    (?:\s+(?:docs?|documents?|publications?|materials?|sources?))?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Pre-compute multi-word aliases (they need substring fallback because
# the main regex captures a single \b-delimited token).
_MULTIWORD_ALIASES = [
    (alias, canonical)
    for alias, canonical in _SOURCE_ALIASES.items()
    if " " in alias or "-" in alias or "." in alias
]


def detect_source_filter(query: str) -> Optional[str]:
    """Parse a user query for "ground on X" style narrowing directives.

    Returns a canonical source tag (matching
    ``metadata.tags.source`` in the ingested catalog) or ``None`` if no
    recognised source is mentioned.

    Examples:
        >>> detect_source_filter("Cite only IRS documents")
        'irs'
        >>> detect_source_filter("Based on SEC documents, what is Reg T?")
        'sec'
        >>> detect_source_filter("What is an ETF?") is None
        True
    """
    if not query:
        return None

    # First pass: the structured narrowing phrase ("only IRS",
    # "based on SEC", "per the Federal Reserve", ...).
    for match in _SOURCE_FILTER_PATTERN.finditer(query):
        raw = match.group("source").strip().lower()
        if raw in _SOURCE_ALIASES:
            return _SOURCE_ALIASES[raw]

        # Extend the capture to cover multi-word aliases like
        # "Federal Reserve" or "Tax Foundation" that get truncated at
        # the first \b by the non-greedy capture group.
        tail = query[match.start("source"):].lower()
        for alias, canonical in _MULTIWORD_ALIASES:
            if tail.startswith(alias):
                return canonical

        # Fall back to single-word tokens from within the capture.
        for token in re.split(r"[\s.\-]+", raw):
            token = token.strip().lower()
            if token in _SOURCE_ALIASES:
                return _SOURCE_ALIASES[token]

    return None


# ---------------------------------------------------------------------
# Canonical search function
# ---------------------------------------------------------------------


def rag_search(
    query: str,
    *,
    source: Optional[str] = None,
    top_k: Optional[int] = None,
    use_mmr: Optional[bool] = None,
    mmr_fetch_k: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    retriever: Optional[Retriever] = None,
    cfg: Optional[AppConfig] = None,
) -> List[RetrievedDoc]:
    """Search the FAISS vector_db and return the top chunks.

    The defaults for ``top_k`` / MMR knobs come from ``cfg.rag`` so the
    operator can tune them in ``config.yaml`` / via env vars without
    code changes.

    Args:
        query:          User query text.
        source:         Optional canonical source tag (e.g. ``"irs"``)
                        to restrict results to one ``tags.source``
                        family. Unknown values are normalised via the
                        alias table (:data:`_SOURCE_ALIASES`).
        top_k:          Number of results to return.
        use_mmr:        Toggle MMR re-ranking.
        mmr_fetch_k:    Candidate pool size for MMR.
        mmr_lambda:     MMR diversity/relevance weight.
        retriever:      Optional retriever override (mostly for tests).
        cfg:            Optional pre-loaded config.

    Returns:
        A list of :class:`src.rag.retriever.RetrievedDoc`. Empty if
        RAG is disabled, the index is missing, or retrieval failed.
    """
    cfg = cfg or get_config()
    rag_cfg = cfg.rag

    normalised_source: Optional[str] = None
    if source:
        needle = source.strip().lower()
        normalised_source = _SOURCE_ALIASES.get(needle, needle)

    active = retriever if retriever is not None else get_default_retriever()
    if active is None:
        _logger.debug("rag_search: no retriever available; returning []")
        return []

    return active.retrieve(
        query,
        k=top_k if top_k is not None else rag_cfg.top_k,
        use_mmr=use_mmr if use_mmr is not None else rag_cfg.use_mmr,
        fetch_k=mmr_fetch_k if mmr_fetch_k is not None else rag_cfg.mmr_fetch_k,
        lambda_mult=mmr_lambda if mmr_lambda is not None else rag_cfg.mmr_lambda,
        source_filter=normalised_source,
    )


# ---------------------------------------------------------------------
# Lightweight serialisation helpers -- same shape over the MCP wire and
# inside AgentResponse.metadata["sources"] so the UI/tool-clients see
# identical citation records.
# ---------------------------------------------------------------------


def to_wire(doc: RetrievedDoc) -> Dict[str, Any]:
    """Serialise a :class:`RetrievedDoc` to a plain JSON-safe dict."""
    return {
        "text": doc.text,
        "url": doc.url,
        "title": doc.title,
        "tags": dict(doc.tags or {}),
        "score": float(doc.score),
    }


def to_wire_many(docs: List[RetrievedDoc]) -> List[Dict[str, Any]]:
    """Batch version of :func:`to_wire`."""
    return [to_wire(d) for d in docs]


__all__ = [
    "KNOWN_SOURCES",
    "detect_source_filter",
    "rag_search",
    "to_wire",
    "to_wire_many",
]
