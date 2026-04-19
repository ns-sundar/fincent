"""Central orchestrator/router agent.

Public entry points:
    - ``plan_route``: classify a user query into a ``RoutingPlan``.
    - ``answer_directly``: produce a final answer for queries the
      central agent handles itself (app info / generic user chit-chat).
    - ``aggregate``: combine specialist responses into a single answer.
"""

from src.agents.central.agent import (
    aggregate,
    answer_directly,
    plan_route,
)

__all__ = ["plan_route", "answer_directly", "aggregate"]
