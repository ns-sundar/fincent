"""Typed state container shared across LangGraph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from src.core.schemas import AgentResponse, RoutingPlan


class GraphState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes.

    Notes:
        ``agent_responses`` is annotated with ``operator.add`` so that
        multiple specialist nodes running in parallel can each append
        their reply without overwriting peers.
    """

    query: str
    plan: Optional[RoutingPlan]
    agent_responses: Annotated[List[AgentResponse], operator.add]
    final_answer: Optional[str]
    session_id: Optional[str]
