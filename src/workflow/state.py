"""Typed state container shared across LangGraph nodes."""

from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.core.schemas import AgentResponse, Intent, RoutingPlan


def _reset_or_extend(
    existing: Optional[List[AgentResponse]],
    new: Optional[List[AgentResponse]],
) -> List[AgentResponse]:
    """Custom reducer for ``agent_responses``.

    * ``None`` from a node resets the list to ``[]`` -- this is how the
      planner clears responses from previous turns that a SQLite
      checkpoint may have restored.
    * A list from a node is appended (so parallel specialists can each
      add their own reply without overwriting peers).
    """
    if new is None:
        return []
    return list(existing or []) + list(new)


class GraphState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes.

    Fields:
        query:           Latest user query (overwritten each turn).
        messages:        Full conversation transcript, persisted across
                         turns via the checkpointer. Uses LangGraph's
                         ``add_messages`` reducer.
        plan:            Current turn's routing plan (overwritten).
        agent_responses: Responses produced by agents during the
                         current turn. Uses ``_reset_or_extend`` so the
                         planner can clear stale checkpoint data.
        final_answer:    Final user-facing answer for the current turn.
        session_id:      Opaque thread identifier (informational).
        intent_hint:     Optional caller-pinned intent (e.g. a
                         specialist Streamlit tab). When set
                         the planner skips LLM classification and
                         dispatches directly to the matching
                         specialist.
    """

    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[RoutingPlan]
    agent_responses: Annotated[List[AgentResponse], _reset_or_extend]
    final_answer: Optional[str]
    session_id: Optional[str]
    intent_hint: Optional[Intent]
