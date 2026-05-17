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

The compiled graph is wired up with a SQLite ``Checkpointer`` so that
per-thread state (messages, plan, etc.) persists across HTTP requests
and Streamlit reloads.
"""

from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.core.config import AppConfig, get_config
from src.core.llm import get_current_model
from src.core.schemas import Intent, QueryRequest, QueryResponse, RoutingPlan
from src.utils.logging import get_logger
from src.workflow.nodes import (
    SPECIALIST_NODE_FOR,
    aggregator_node,
    central_direct_node,
    goal_planning_node,
    market_research_node,
    planner_node,
    portfolio_node,
    qna_node,
)
from src.workflow.state import GraphState

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------


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
        sends.append(Send(node_name, {"query": state["query"], "messages": list(state.get("messages") or [])}))

    # Defensive: if nothing matched, fall through to the aggregator,
    # which will emit a polite "no answer" message.
    if not sends:
        return "aggregator"
    return sends


def build_graph(
    *,
    checkpointer: Optional[object] = None,
) -> CompiledStateGraph:
    """Construct and compile the LangGraph workflow.

    Args:
        checkpointer: Optional LangGraph-compatible checkpointer. Pass
            ``None`` (default) to compile an ephemeral graph, which is
            convenient for tests. The server uses ``default_graph()``
            which attaches a SQLite checkpointer.
    """
    g: StateGraph = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("central_direct", central_direct_node)
    g.add_node("qna_node", qna_node)
    g.add_node("portfolio_node", portfolio_node)
    g.add_node("market_research_node", market_research_node)
    g.add_node("goal_planning_node", goal_planning_node)
    g.add_node("aggregator", aggregator_node)

    g.add_edge(START, "planner")

    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "central_direct": "central_direct",
            "aggregator": "aggregator",
            "qna_node": "qna_node",
            "portfolio_node": "portfolio_node",
            "market_research_node": "market_research_node",
            "goal_planning_node": "goal_planning_node",
        },
    )

    g.add_edge("central_direct", "aggregator")
    g.add_edge("qna_node", "aggregator")
    g.add_edge("portfolio_node", "aggregator")
    g.add_edge("market_research_node", "aggregator")
    g.add_edge("goal_planning_node", "aggregator")
    g.add_edge("aggregator", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


# ---------------------------------------------------------------------
# Default (checkpointed) graph singleton used by the server
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _default_checkpointer() -> SqliteSaver:
    """Create the process-wide SQLite checkpointer from config."""
    cfg: AppConfig = get_config()
    backend = cfg.checkpointer.backend.lower()
    if backend != "sqlite":
        raise ValueError(
            f"Unsupported checkpointer backend '{backend}'. Only 'sqlite' is wired."
        )
    path = cfg.checkpointer.path
    if path == ":memory:":
        db_path: Path | str = ":memory:"
    else:
        db_path = Path(path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
    _logger.info("Opening SQLite checkpointer at %s", db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


@lru_cache(maxsize=1)
def default_graph() -> CompiledStateGraph:
    """Compile the graph once, attached to the default checkpointer."""
    return build_graph(checkpointer=_default_checkpointer())


# ---------------------------------------------------------------------
# Invocation helpers
# ---------------------------------------------------------------------


def _thread_config(thread_id: str) -> Dict:
    """Shape the LangGraph ``config`` dict for a given thread."""
    return {"configurable": {"thread_id": thread_id}}


def run_query(
    request: QueryRequest,
    *,
    graph: Optional[CompiledStateGraph] = None,
    thread_id: Optional[str] = None,
) -> QueryResponse:
    """Execute the full graph for a single user query.

    Args:
        request: The user query payload. ``request.session_id`` is used
            as the LangGraph thread id when ``thread_id`` is not
            supplied explicitly.
        graph: Optional pre-compiled graph (mainly to avoid the cost
            of recompilation in tests). Defaults to ``default_graph()``.
        thread_id: Optional override for the thread id. When missing, a
            fresh UUID is generated (the checkpoint is effectively
            ephemeral for that call).

    Returns:
        A typed ``QueryResponse``.
    """
    compiled = graph or default_graph()
    tid = thread_id or request.session_id or str(uuid.uuid4())
    config = _thread_config(tid)
    active_model = get_current_model()

    final_state: GraphState = compiled.invoke(
        {
            "query": request.query,
            "messages": [HumanMessage(content=request.query)],
            "session_id": tid,
            "intent_hint": request.intent_hint,
        },
        config=config,
    )
    plan: RoutingPlan = final_state.get("plan") or RoutingPlan()
    return QueryResponse(
        answer=final_state.get("final_answer") or "",
        plan=plan,
        agent_responses=list(final_state.get("agent_responses") or []),
        model=active_model,
    )


# ---------------------------------------------------------------------
# History / reset helpers
# ---------------------------------------------------------------------


# LangChain's BaseMessage exposes ``.type``; Streamlit expects the more
# conventional chat role strings.
_LANGGRAPH_TO_STREAMLIT_ROLE: Dict[str, str] = {
    "human": "user",
    "ai": "assistant",
}


def _messages_from_snapshot(graph: CompiledStateGraph, thread_id: str) -> List[BaseMessage]:
    """Return the raw ``BaseMessage`` list for a thread (or empty)."""
    snap = graph.get_state(_thread_config(thread_id))
    values = getattr(snap, "values", None) or {}
    return list(values.get("messages") or [])


def get_history(
    thread_id: str,
    *,
    graph: Optional[CompiledStateGraph] = None,
) -> List[Dict[str, str]]:
    """Return the transcript for ``thread_id`` as chat-style dicts.

    Each item has the shape ``{"role": "human"|"ai", "content": str}``.
    Clients are responsible for mapping to their own role vocabulary
    (e.g. Streamlit's "user"/"assistant"). Non-chat messages (system,
    tool, etc.) are filtered out.
    """
    compiled = graph or default_graph()
    out: List[Dict[str, str]] = []
    for m in _messages_from_snapshot(compiled, thread_id):
        mtype = getattr(m, "type", None)
        if mtype not in _LANGGRAPH_TO_STREAMLIT_ROLE:
            continue
        out.append({"role": mtype, "content": str(m.content)})
    return out


def reset_thread(
    thread_id: str,
    *,
    graph: Optional[CompiledStateGraph] = None,
) -> int:
    """Clear the message history for ``thread_id``.

    Uses ``update_state`` with ``RemoveMessage`` for every existing
    message, so the SQLite log retains the prior checkpoints while the
    *current* state for the thread now has an empty message list.

    Returns:
        Number of messages that were removed.
    """
    compiled = graph or default_graph()
    config = _thread_config(thread_id)
    existing = _messages_from_snapshot(compiled, thread_id)
    removals = [
        RemoveMessage(id=m.id)
        for m in existing
        if getattr(m, "id", None) is not None
    ]
    if not removals:
        return 0
    compiled.update_state(config, {"messages": removals})
    return len(removals)


__all__ = [
    "AIMessage",
    "HumanMessage",
    "build_graph",
    "default_graph",
    "get_history",
    "reset_thread",
    "run_query",
]
