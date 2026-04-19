"""LangGraph assembly for the Fincent multi-agent workflow.

Topology:

    START
      |
      v
    planner ----------------------------+
      |                                  |
      | (handled_by_central?)            |
      |                                  |
   yes|                                  |no
      v                                  v
    central_direct                fan-out to one or more
      |                           specialist nodes (Send API)
      |                                  |
      +--------------+-------------------+
                     |
                     v
                 aggregator
                     |
                     v
                    END
"""

from __future__ import annotations

from typing import List, Union

from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.core.schemas import Intent, QueryRequest, QueryResponse, RoutingPlan
from src.workflow.nodes import (
    SPECIALIST_NODE_FOR,
    aggregator_node,
    agent_four_node,
    agent_three_node,
    agent_two_node,
    central_direct_node,
    planner_node,
    qna_node,
)
from src.workflow.state import GraphState


def _route_after_planner(state: GraphState) -> Union[str, List[Send]]:
    """Conditional edge: decide what runs after the planner.

    Returns:
        Either the name of the next node (string) for the central
        direct-answer path, or a list of ``Send`` objects to fan out
        in parallel to one or more specialist nodes.
    """
    plan: RoutingPlan = state["plan"]  # planner always sets this

    if plan.handled_by_central:
        return "central_direct"

    sends: List[Send] = []
    for intent in plan.intents:
        node_name = SPECIALIST_NODE_FOR.get(intent)
        if node_name is None:
            continue
        sends.append(Send(node_name, {"query": state["query"]}))

    # Defensive: if nothing matched, fall through to the aggregator,
    # which will emit a polite "no answer" message.
    if not sends:
        return "aggregator"
    return sends


def build_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph workflow."""
    g: StateGraph = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("central_direct", central_direct_node)
    g.add_node("qna_node", qna_node)
    g.add_node("agent_two_node", agent_two_node)
    g.add_node("agent_three_node", agent_three_node)
    g.add_node("agent_four_node", agent_four_node)
    g.add_node("aggregator", aggregator_node)

    g.add_edge(START, "planner")

    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        # Explicit destination set so static analyzers (and the
        # runtime) know every node the conditional may target.
        {
            "central_direct": "central_direct",
            "aggregator": "aggregator",
            "qna_node": "qna_node",
            "agent_two_node": "agent_two_node",
            "agent_three_node": "agent_three_node",
            "agent_four_node": "agent_four_node",
        },
    )

    # All branches converge on the aggregator, which then ends.
    g.add_edge("central_direct", "aggregator")
    g.add_edge("qna_node", "aggregator")
    g.add_edge("agent_two_node", "aggregator")
    g.add_edge("agent_three_node", "aggregator")
    g.add_edge("agent_four_node", "aggregator")
    g.add_edge("aggregator", END)

    return g.compile()


# ---------------------------------------------------------------------
# Public, typed convenience wrapper
# ---------------------------------------------------------------------


def run_query(
    request: QueryRequest,
    *,
    graph: CompiledStateGraph = None,
) -> QueryResponse:
    """Execute the full graph for a single user query.

    Args:
        request: The user query payload.
        graph: Optional pre-compiled graph (mainly to avoid the
            cost of recompilation in tests).

    Returns:
        A typed ``QueryResponse``.
    """
    compiled = graph or build_graph()
    final_state: GraphState = compiled.invoke(
        {
            "query": request.query,
            "session_id": request.session_id,
            "agent_responses": [],
        }
    )
    plan: RoutingPlan = final_state.get("plan") or RoutingPlan()
    return QueryResponse(
        answer=final_state.get("final_answer") or "",
        plan=plan,
        agent_responses=list(final_state.get("agent_responses") or []),
    )
