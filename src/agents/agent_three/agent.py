"""Placeholder implementation for the third specialist agent."""

from __future__ import annotations

from src.core.schemas import AgentName, AgentResponse


def answer(query: str) -> AgentResponse:
    """Return a 'not implemented yet' message."""
    _ = query
    return AgentResponse(
        agent=AgentName.AGENT_THREE,
        content="agent_three is not implemented yet.",
        metadata={"placeholder": True},
    )
