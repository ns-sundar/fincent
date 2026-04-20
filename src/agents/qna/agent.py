"""Generic financial Q&A agent with agentic RAG.

Upgrades over the original skeleton:

* Retrieval goes through :func:`src.rag.tool.rag_search`, the same
  canonical function the MCP server exposes. Swapping the transport
  (in-process vs. MCP client) does not change behaviour.
* Top-k / MMR / diversity knobs come from ``cfg.rag`` and are
  overridable per-call.
* Natural-language source filtering: phrases like "cite only IRS
  documents" or "based on SEC documents" are detected from the user
  query (:func:`src.rag.tool.detect_source_filter`) and restrict
  retrieval to the requested source family via the
  ``metadata.tags.source`` tag stamped at ingestion.
* Citations: inline ``[n]`` markers in the prose; the model is
  instructed to end with ``## Sources`` listing **only** cited
  context entries (no uncited retrieval noise). ``metadata.sources``
  is derived from the same ``[n]`` markers for programmatic use.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.qna.prompts import (
    QNA_CONTEXT_TEMPLATE,
    QNA_SOURCE_FILTER_CLAUSE,
    QNA_SYSTEM_PROMPT,
)
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.rag.retriever import RetrievedDoc, Retriever
from src.rag.tool import detect_source_filter, rag_search
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def _ordered_unique_citation_indices(body: str, *, max_n: int) -> List[int]:
    """Parse ``[1]``, ``[12]``-style markers; ignore out-of-range indices."""
    out: List[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", body):
        idx = int(m.group(1))
        if 1 <= idx <= max_n and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------


def _format_context_block(docs: List[RetrievedDoc]) -> str:
    """Render retrieved chunks as a numbered list for the LLM.

    Each entry includes title, URL, and (when present) the source tag
    and category list so the model can cite inline using ``[n]``.
    """
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.title.strip() or "(untitled)"
        url = d.url.strip() or "(no url)"
        source = str((d.tags or {}).get("source") or "").strip()
        categories = (d.tags or {}).get("category") or []
        if isinstance(categories, (list, tuple)):
            cat_str = ", ".join(str(c) for c in categories)
        else:
            cat_str = str(categories)
        meta_line_parts = []
        if source:
            meta_line_parts.append(f"source={source}")
        if cat_str:
            meta_line_parts.append(f"categories=[{cat_str}]")
        meta_line = " | ".join(meta_line_parts) or "(no tags)"
        excerpt = d.text.strip()
        lines.append(
            f"[{i}] {title}\nURL: {url}\nTags: {meta_line}\n{excerpt}"
        )
    return "\n\n".join(lines)


def _citation_metadata(docs: List[RetrievedDoc]) -> List[dict]:
    """Compact, JSON-safe representation of the cited sources."""
    return [
        {
            "url": d.url,
            "title": d.title,
            "tags": dict(d.tags or {}),
            "score": float(d.score),
        }
        for d in docs
    ]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def answer(
    query: str,
    *,
    llm: Optional[BaseChatModel] = None,
    retriever: Optional[Retriever] = None,
    top_k: Optional[int] = None,
    source_filter: Optional[str] = None,
) -> AgentResponse:
    """Answer a generic (non-personal) financial question using agentic RAG.

    Args:
        query:          The user's question (routed here by the central
                        agent).
        llm:            Optional chat model (mainly for tests).
        retriever:      Optional retriever override. When ``None`` the
                        canonical :func:`src.rag.tool.rag_search` path
                        resolves to the cached FAISS retriever.
        top_k:          Optional override for the number of chunks to
                        retrieve (defaults to ``cfg.rag.top_k``).
        source_filter:  Optional explicit source tag (e.g. ``"irs"``)
                        used to restrict retrieval to one source
                        family. When ``None`` the agent auto-detects
                        "cite only IRS", "based on SEC", etc. from
                        ``query``.

    Returns:
        An ``AgentResponse`` attributed to the Q&A agent. When RAG ran,
        ``metadata["sources"]`` lists **only** context entries cited via
        ``[n]`` in the answer (see ``metadata["cited_chunk_indices"]``).
        The visible ``## Sources`` section is produced by the model
        following instructions; this metadata mirrors cited indices.
    """
    _logger.debug("QnA agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()

    # Auto-detect "cite only X" style narrowing from the query unless
    # the caller pinned a filter explicitly.
    effective_source = source_filter or detect_source_filter(query)
    if effective_source:
        _logger.info("QnA RAG: restricting to source=%s", effective_source)

    retrieved: List[RetrievedDoc] = []
    try:
        retrieved = rag_search(
            query,
            source=effective_source,
            top_k=top_k,
            retriever=retriever,
        )
    except Exception as exc:  # noqa: BLE001 -- never break the agent
        _logger.warning("rag_search raised during query: %s", exc)
        retrieved = []

    # Build the message list. When we have context, add a second
    # system message carrying the retrieved block and an optional
    # source-filter clause.
    messages = [SystemMessage(content=QNA_SYSTEM_PROMPT)]
    if retrieved:
        source_clause = ""
        if effective_source:
            source_clause = QNA_SOURCE_FILTER_CLAUSE.format(
                source_label=effective_source.upper()
            )
        messages.append(
            SystemMessage(
                content=QNA_CONTEXT_TEMPLATE.format(
                    context=_format_context_block(retrieved),
                    source_clause=source_clause,
                )
            )
        )
    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    body = str(response.content).strip()

    metadata: dict = {
        "rag_used": bool(retrieved),
        "rag_chunk_count": len(retrieved),
    }
    if effective_source:
        metadata["source_filter"] = effective_source
    if retrieved:
        cite_indices = _ordered_unique_citation_indices(body, max_n=len(retrieved))
        cited_docs = [retrieved[i - 1] for i in cite_indices]
        metadata["cited_chunk_indices"] = cite_indices
        metadata["sources"] = _citation_metadata(cited_docs)

    return AgentResponse(
        agent=AgentName.QNA,
        content=body.strip(),
        metadata=metadata,
    )
