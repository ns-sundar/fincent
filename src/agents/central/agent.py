"""Central orchestrator agent: planning, direct answers, aggregation."""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.central.prompts import (
    AGGREGATOR_SYSTEM_PROMPT,
    AGGREGATOR_USER_TEMPLATE,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
)
from src.core.config import AppConfig, get_config
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse, Intent, RoutingPlan
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Intents the central agent answers itself.
_SELF_INTENTS: set[Intent] = {Intent.APP_INFO, Intent.USER_GENERIC}


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` style fences if the model added them."""
    fence = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return fence.group(1) if fence else text.strip()


def _coerce_intents(raw_intents: object) -> List[Intent]:
    """Convert untrusted strings into a deduplicated list of ``Intent``."""
    if not isinstance(raw_intents, list):
        return []
    seen: set[Intent] = set()
    out: List[Intent] = []
    for raw in raw_intents:
        if not isinstance(raw, str):
            continue
        try:
            intent = Intent(raw.strip().lower())
        except ValueError:
            continue
        if intent in seen:
            continue
        seen.add(intent)
        out.append(intent)
    return out


def _enabled_specialist_intents(cfg: AppConfig) -> set[Intent]:
    """Compute the set of specialist intents currently enabled in config."""
    mapping = {
        Intent.QNA: cfg.agents.qna.enabled,
        Intent.PORTFOLIO: cfg.agents.portfolio.enabled,
    }
    return {intent for intent, enabled in mapping.items() if enabled}


def _fallback_plan(reason: str) -> RoutingPlan:
    """Return a safe fallback routing plan when parsing fails."""
    _logger.warning("Falling back to qna plan: %s", reason)
    return RoutingPlan(
        intents=[Intent.QNA],
        handled_by_central=False,
        rationale=f"Fallback ({reason})",
    )


# ---------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------


def plan_route(
    query: str,
    *,
    history: Optional[List[Any]] = None,
    llm: Optional[BaseChatModel] = None,
    cfg: Optional[AppConfig] = None,
    intent_hint: Optional[Intent] = None,
) -> RoutingPlan:
    """Classify a user query into a structured ``RoutingPlan``.

    The central LLM is asked to emit a strict JSON object describing
    which downstream agents should run. We parse defensively and fall
    back to the Q&A agent if anything looks malformed.

    Args:
        query: The raw user message.
        llm: Optional pre-built chat model (mainly for tests).
        cfg: Optional pre-loaded ``AppConfig`` (mainly for tests).
        intent_hint: Optional caller-pinned intent (e.g. the Portfolio
            tab in the Streamlit UI pins ``Intent.PORTFOLIO``). When
            provided and the hinted specialist is enabled, the LLM
            classifier is skipped entirely and the plan is built from
            the hint. Ignored for ``app_info`` / ``user_generic`` /
            ``unknown`` hints and for disabled specialists.

    Returns:
        A validated ``RoutingPlan``.
    """
    cfg = cfg or get_config()

    if intent_hint is not None:
        enabled = _enabled_specialist_intents(cfg)
        if intent_hint in enabled:
            return RoutingPlan(
                intents=[intent_hint],
                handled_by_central=False,
                rationale=f"Caller pinned intent '{intent_hint.value}'.",
            )
        _logger.info(
            "Ignoring intent_hint=%s (not an enabled specialist).",
            intent_hint.value,
        )

    llm = llm or get_default_chat_model()

    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            *list((history or [])[-6:]),
            HumanMessage(content=ROUTER_USER_TEMPLATE.format(query=query)),
        ]
    )
    text = _strip_code_fence(str(response.content))

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _fallback_plan(f"router JSON parse failed: {exc}")

    if not isinstance(payload, dict):
        return _fallback_plan("router payload was not an object")

    intents = _coerce_intents(payload.get("intents"))
    handled_by_central = bool(payload.get("handled_by_central", False))
    rationale = str(payload.get("rationale", "")).strip()

    # If the router claims central-handling, restrict intents to the
    # self-handled set; if it picked specialists, drop any that are
    # disabled in config.
    if handled_by_central:
        intents = [i for i in intents if i in _SELF_INTENTS] or [Intent.USER_GENERIC]
    else:
        enabled = _enabled_specialist_intents(cfg)
        intents = [i for i in intents if i in enabled]
        if not intents:
            return _fallback_plan("no enabled specialist matched")

    # Cap fan-out to the configured limit.
    max_fanout = max(1, cfg.agents.central.max_fanout)
    if not handled_by_central and len(intents) > max_fanout:
        intents = intents[:max_fanout]

    return RoutingPlan(
        intents=intents,
        handled_by_central=handled_by_central,
        rationale=rationale,
    )


# ---------------------------------------------------------------------
# 2. Direct answer (when the central agent answers itself)
# ---------------------------------------------------------------------


def answer_directly(
    query: str,
    *,
    llm: Optional[BaseChatModel] = None,
    cfg: Optional[AppConfig] = None,
) -> AgentResponse:
    """Produce a final answer for app-info / user-generic queries.

    Args:
        query: The raw user message.
        llm: Optional chat model (for tests).
        cfg: Optional config (for tests).

    Returns:
        An ``AgentResponse`` attributed to the central agent.
    """
    cfg = cfg or get_config()
    llm = llm or get_default_chat_model()

    system = DIRECT_ANSWER_SYSTEM_PROMPT.format(
        app_name=cfg.app.name,
        app_version=cfg.app.version,
        app_description=cfg.app.description,
        app_about=cfg.app.about,
    )
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=query)]
    )
    return AgentResponse(
        agent=AgentName.CENTRAL,
        content=str(response.content).strip(),
        metadata={"mode": "direct"},
    )


# ---------------------------------------------------------------------
# 3. Aggregator (combine specialist replies)
# ---------------------------------------------------------------------


def _format_responses_block(responses: List[AgentResponse]) -> str:
    """Render specialist replies as a human-readable block for the LLM."""
    parts: List[str] = []
    for resp in responses:
        parts.append(f"[{resp.agent.value}]\n{resp.content.strip()}")
    return "\n\n".join(parts) if parts else "(no specialist responses)"


def aggregate(
    query: str,
    responses: List[AgentResponse],
    *,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Merge multiple specialist responses into one final answer.

    Short-circuits when only a single response is present (in which
    case we return that response verbatim).

    Args:
        query: The original user query.
        responses: Specialist responses collected by the workflow.
        llm: Optional chat model (for tests).

    Returns:
        The final answer string for the user.
    """
    if not responses:
        return (
            "I'm sorry -- I wasn't able to produce an answer for that "
            "question. Please try rephrasing it."
        )
    if len(responses) == 1:
        return responses[0].content.strip()

    llm = llm or get_default_chat_model()
    user_prompt = AGGREGATOR_USER_TEMPLATE.format(
        query=query,
        responses_block=_format_responses_block(responses),
    )
    response = llm.invoke(
        [
            SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content).strip()
