"""Generic financial Q&A agent (now RAG-enabled).

Upgraded from the original skeleton: when the FAISS-backed retriever
built at startup is available, the agent grounds its answer with the
top-k chunks. Metadata (url, title, tags) is propagated from
ingestion through to the agent response so the UI can show sources.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.qna.prompts import QNA_CONTEXT_TEMPLATE, QNA_SYSTEM_PROMPT
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.rag.retriever import (
    RetrievedDoc,
    Retriever,
    get_default_retriever,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _format_context_block(docs: List[RetrievedDoc]) -> str:
    """Render retrieved chunks as a numbered list for the LLM.

    Each entry includes the source URL so the model can cite inline.
    """
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.title.strip() or "(untitled)"
        url = d.url.strip() or "(no url)"
        # Keep each chunk compact: title + URL + excerpt. The chunker
        # already capped us near ~1000 tokens per entry.
        excerpt = d.text.strip()
        lines.append(f"[{i}] {title}\nSource: {url}\n{excerpt}")
    return "\n\n".join(lines)


def _citation_metadata(docs: List[RetrievedDoc]) -> List[dict]:
    """Compact, JSON-safe representation of the cited sources."""
    return [
        {
            "url": d.url,
            "title": d.title,
            "tags": d.tags,
            "score": d.score,
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
) -> AgentResponse:
    """Answer a generic (non-personal) financial question.

    Args:
        query:     The user's question (routed here by the central
                   agent).
        llm:       Optional chat model (mainly for tests).
        retriever: Optional retriever override. When ``None`` (default)
                   the cached FAISS retriever from
                   :func:`src.rag.retriever.get_default_retriever` is
                   used; when that is itself ``None`` (RAG disabled /
                   ingestion failed) the agent degrades to
                   non-retrieval behaviour.
        top_k:     Optional override for the number of chunks to
                   retrieve. Defaults to ``cfg.rag.top_k``.

    Returns:
        An ``AgentResponse`` attributed to the Q&A agent with citation
        metadata populated when retrieval succeeded.
    """
    _logger.debug("QnA agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()

    # ``get_default_retriever`` returns ``None`` when RAG is disabled
    # or the index is missing; we tolerate both silently.
    active_retriever = retriever if retriever is not None else get_default_retriever()

    retrieved: List[RetrievedDoc] = []
    if active_retriever is not None:
        try:
            retrieved = active_retriever.retrieve(query, k=top_k)
        except Exception as exc:  # noqa: BLE001 -- never break the agent
            _logger.warning("Retriever raised during query: %s", exc)
            retrieved = []

    # Build the message list. When we have context, prepend a second
    # system message carrying the retrieved block; otherwise stick to
    # the base persona only.
    messages = [SystemMessage(content=QNA_SYSTEM_PROMPT)]
    if retrieved:
        messages.append(
            SystemMessage(
                content=QNA_CONTEXT_TEMPLATE.format(
                    context=_format_context_block(retrieved),
                )
            )
        )
    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)

    metadata: dict = {
        "rag_used": bool(retrieved),
        "rag_chunk_count": len(retrieved),
    }
    if retrieved:
        metadata["sources"] = _citation_metadata(retrieved)

    return AgentResponse(
        agent=AgentName.QNA,
        content=str(response.content).strip(),
        metadata=metadata,
    )
