"""Shared Pydantic schemas used across the workflow.

These types form the public contract between the FastAPI layer, the
LangGraph workflow, and the Streamlit UI.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    """Canonical identifiers for every agent that can run in the graph."""

    CENTRAL = "central"
    QNA = "qna"
    AGENT_TWO = "agent_two"
    AGENT_THREE = "agent_three"
    AGENT_FOUR = "agent_four"


class Intent(str, Enum):
    """High-level intent buckets recognised by the central router.

    ``APP_INFO`` and ``USER_GENERIC`` are answered by the central agent
    itself; everything else is delegated to one or more specialists.
    """

    APP_INFO = "app_info"
    USER_GENERIC = "user_generic"
    QNA = "qna"
    AGENT_TWO = "agent_two"
    AGENT_THREE = "agent_three"
    AGENT_FOUR = "agent_four"
    UNKNOWN = "unknown"


class RoutingPlan(BaseModel):
    """The structured plan emitted by the central planner.

    ``intents`` is ordered by priority. ``handled_by_central`` tells the
    graph that the central agent will produce the user-facing answer
    itself (no specialist dispatch required).
    """

    intents: List[Intent] = Field(default_factory=list)
    handled_by_central: bool = False
    rationale: str = ""


class AgentResponse(BaseModel):
    """A single response produced by one agent."""

    agent: AgentName
    content: str
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """User-facing input payload."""

    query: str = Field(..., min_length=1, description="The user's question.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional opaque session id (reserved for future memory).",
    )


class QueryResponse(BaseModel):
    """User-facing output payload."""

    answer: str
    plan: RoutingPlan
    agent_responses: List[AgentResponse] = Field(default_factory=list)
