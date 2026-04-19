"""Placeholder implementation for the second specialist agent.

Replace ``answer`` with a real implementation when this agent is wired
in. The signature is intentionally identical to other agents so the
LangGraph workflow can call it polymorphically.
"""

from __future__ import annotations

from src.core.schemas import AgentName, AgentResponse


def answer(query: str) -> AgentResponse:
    """Return a 'not implemented yet' message."""
    _ = query  # reserved
    return AgentResponse(
        agent=AgentName.AGENT_TWO,
        content="agent_two is not implemented yet.",
        metadata={"placeholder": True},
    )
