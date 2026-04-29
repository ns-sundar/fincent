"""LangGraph node implementations.

Each node is a pure function (state in -> partial state out). The
graph wiring lives in ``src.workflow.graph``.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from src.agents import portfolio, qna
from src.agents.central import aggregate, answer_directly, plan_route
from src.core.schemas import AgentName, AgentResponse, Intent
from src.utils.logging import get_logger
from src.workflow.state import GraphState

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Static dispatch table -- intent -> specialist callable
# ---------------------------------------------------------------------

# Each specialist exposes ``answer(query: str) -> AgentResponse``.
SPECIALIST_DISPATCH: Dict[Intent, callable] = {
    Intent.QNA: qna.answer,
    Intent.PORTFOLIO: portfolio.answer,
}


# Graph-node names that map 1:1 to specialist intents.
SPECIALIST_NODE_FOR: Dict[Intent, str] = {
    Intent.QNA: "qna_node",
    Intent.PORTFOLIO: "portfolio_node",
}


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------


def planner_node(state: GraphState) -> GraphState:
    """Run the central planner and store the resulting ``RoutingPlan``.

    Also resets ``agent_responses`` to ``[]`` at the start of a turn
    (the custom reducer treats ``None`` as a reset signal) so that
    responses recovered from a SQLite checkpoint do not leak into the
    current turn. When an ``intent_hint`` is present in the state
    (set by the HTTP layer from ``QueryRequest.intent_hint``) the
    planner short-circuits the LLM classifier and dispatches directly
    to that specialist.
    """
    query = state["query"]
    msgs = list(state.get("messages") or [])
    # Prior turns only — the last message is the current HumanMessage.
    history = msgs[:-1] if msgs and isinstance(msgs[-1], HumanMessage) else msgs
    plan = plan_route(query, history=history, intent_hint=state.get("intent_hint"))
    _logger.info(
        "Plan: handled_by_central=%s intents=%s",
        plan.handled_by_central,
        [i.value for i in plan.intents],
    )
    return {"plan": plan, "agent_responses": None}


def central_direct_node(state: GraphState) -> GraphState:
    """Central agent answers directly (app-info / user-generic)."""
    query = state["query"]
    response = answer_directly(query)
    return {"agent_responses": [response]}


def _make_specialist_node(intent: Intent):
    """Build a node that runs a single specialist for the current query."""

    def _node(state: GraphState) -> GraphState:
        query = state["query"]
        msgs = list(state.get("messages") or [])
        # Prior turns only — the last message is the current HumanMessage.
        history = msgs[:-1] if msgs and isinstance(msgs[-1], HumanMessage) else msgs
        try:
            response: AgentResponse = SPECIALIST_DISPATCH[intent](query, history=history)
        except Exception as exc:  # noqa: BLE001  -- surface as agent error
            _logger.exception("Specialist %s failed", intent.value)
            response = AgentResponse(
                agent=AgentName(intent.value),
                content=(
                    f"The {intent.value} agent failed to produce a "
                    f"response: {exc}"
                ),
                metadata={"error": True},
            )
        return {"agent_responses": [response]}

    _node.__name__ = f"{intent.value}_node"
    return _node


qna_node = _make_specialist_node(Intent.QNA)
portfolio_node = _make_specialist_node(Intent.PORTFOLIO)


def aggregator_node(state: GraphState) -> GraphState:
    """Combine collected specialist responses into a final answer.

    Also appends the final answer as an ``AIMessage`` to the running
    ``messages`` transcript so the checkpointer can rehydrate the UI.
    """
    query = state["query"]
    responses: List[AgentResponse] = list(state.get("agent_responses") or [])
    final = aggregate(query, responses)
    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }
