"""Generic financial Q&A agent.

This is a *skeleton* implementation: the routing and prompt scaffolding
is in place, but richer behaviour (RAG retrieval, citation, tool use)
will be added in a later iteration.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.qna.prompts import QNA_SYSTEM_PROMPT
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def answer(
    query: str,
    *,
    llm: Optional[BaseChatModel] = None,
) -> AgentResponse:
    """Answer a generic (non-personal) financial question.

    Args:
        query: The user's question (already routed here by the central
            agent).
        llm: Optional pre-built chat model (mainly for tests).

    Returns:
        An ``AgentResponse`` attributed to the Q&A agent.
    """
    _logger.debug("QnA agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()

    response = llm.invoke(
        [
            SystemMessage(content=QNA_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )
    return AgentResponse(
        agent=AgentName.QNA,
        content=str(response.content).strip(),
        metadata={"skeleton": True},
    )
