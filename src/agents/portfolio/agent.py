"""Portfolio specialist agent.

Given a natural-language question about the user's own portfolio, the
agent renders a compact JSON/markdown snapshot of the data (accounts
sorted by balance, asset-class allocation, ten most recent
transactions) into a system prompt and asks the chat LLM to produce a
grounded answer.

The same ``PortfolioSnapshot`` object feeds the Streamlit right-hand
graphics panel, so chat answers cannot drift from what the user sees.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.portfolio.loader import (
    AccountSummary,
    PortfolioSnapshot,
    load_portfolio,
)
from src.agents.portfolio.prompts import (
    PORTFOLIO_CONTEXT_TEMPLATE,
    PORTFOLIO_SYSTEM_PROMPT,
)
from src.core.config import AppConfig
from src.core.llm import get_default_chat_model
from src.core.schemas import AgentName, AgentResponse
from src.utils.logging import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Snapshot -> LLM context
# ---------------------------------------------------------------------


def _account_to_dict(acc: AccountSummary) -> Dict[str, Any]:
    """Render a single account summary as JSON-friendly data."""
    return {
        "account_id": acc.account_id,
        "name": acc.name,
        "type": acc.type,
        "broker": acc.broker,
        "currency": acc.currency,
        "balance": acc.balance,
        "holdings": acc.holdings,
    }


def _build_portfolio_block(snapshot: PortfolioSnapshot) -> str:
    """Render the snapshot as a JSON block for the LLM to ground on.

    The agent receives **every** transaction (not just the 10 shown in
    the UI) plus the total count, so questions like "how many
    transactions do I have?" or "what was my first deposit?" can be
    answered correctly.
    """
    payload: Dict[str, Any] = {
        "as_of": snapshot.as_of,
        "total_balance": snapshot.total_balance,
        "allocation": snapshot.allocation,
        "accounts_sorted_by_balance_desc": [
            _account_to_dict(a) for a in snapshot.accounts
        ],
        "transaction_count": snapshot.transaction_count,
        "transactions_newest_first": snapshot.all_transactions,
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def answer(
    query: str,
    *,
    llm: Optional[BaseChatModel] = None,
    cfg: Optional[AppConfig] = None,
    snapshot: Optional[PortfolioSnapshot] = None,
) -> AgentResponse:
    """Answer a portfolio question grounded in the static snapshot.

    Args:
        query:    User question (routed here by the central planner or
                  pinned by the Portfolio tab in the Streamlit UI).
        llm:      Optional chat model (tests).
        cfg:      Optional config override (tests).
        snapshot: Optional pre-loaded snapshot (tests). When ``None``
                  the loader's on-disk cache is used.

    Returns:
        An ``AgentResponse`` attributed to the portfolio agent with
        rollup metadata so the UI can surface "which data was shown
        to the agent".
    """
    _logger.debug("Portfolio agent invoked with query: %s", query)
    llm = llm or get_default_chat_model()
    snap = snapshot or load_portfolio(cfg)

    context = PORTFOLIO_CONTEXT_TEMPLATE.format(
        portfolio_block=_build_portfolio_block(snap)
    )
    messages: List = [
        SystemMessage(content=PORTFOLIO_SYSTEM_PROMPT),
        SystemMessage(content=context),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    body = str(response.content).strip()

    metadata: Dict[str, Any] = {
        "account_count": len(snap.accounts),
        "total_balance": snap.total_balance,
        "allocation": snap.allocation,
        "as_of": snap.as_of,
        "transaction_count": snap.transaction_count,
    }
    return AgentResponse(
        agent=AgentName.PORTFOLIO,
        content=body,
        metadata=metadata,
    )
