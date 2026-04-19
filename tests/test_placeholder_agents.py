"""Sanity tests for the three placeholder specialist agents."""

from __future__ import annotations

from src.agents import agent_four, agent_three, agent_two
from src.core.schemas import AgentName


def test_agent_two_placeholder():
    """agent_two should return a placeholder AgentResponse."""
    r = agent_two.answer("anything")
    assert r.agent == AgentName.AGENT_TWO
    assert r.metadata.get("placeholder") is True


def test_agent_three_placeholder():
    """agent_three should return a placeholder AgentResponse."""
    r = agent_three.answer("anything")
    assert r.agent == AgentName.AGENT_THREE
    assert r.metadata.get("placeholder") is True


def test_agent_four_placeholder():
    """agent_four should return a placeholder AgentResponse."""
    r = agent_four.answer("anything")
    assert r.agent == AgentName.AGENT_FOUR
    assert r.metadata.get("placeholder") is True
