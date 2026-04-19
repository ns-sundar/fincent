"""Tests for the shared Pydantic schemas."""

from __future__ import annotations

import pytest

from src.core.schemas import (
    AgentName,
    AgentResponse,
    Intent,
    QueryRequest,
    QueryResponse,
    RoutingPlan,
)


def test_query_request_requires_non_empty():
    """An empty query should fail validation."""
    with pytest.raises(Exception):
        QueryRequest(query="")


def test_routing_plan_default_empty():
    """A default RoutingPlan should be empty and not central-handled."""
    plan = RoutingPlan()
    assert plan.intents == []
    assert plan.handled_by_central is False


def test_query_response_round_trip():
    """QueryResponse should round-trip through model_dump / model_validate."""
    payload = QueryResponse(
        answer="hello",
        plan=RoutingPlan(intents=[Intent.QNA], handled_by_central=False),
        agent_responses=[AgentResponse(agent=AgentName.QNA, content="hi")],
    )
    rebuilt = QueryResponse.model_validate(payload.model_dump())
    assert rebuilt.answer == "hello"
    assert rebuilt.plan.intents == [Intent.QNA]
    assert rebuilt.agent_responses[0].agent == AgentName.QNA
