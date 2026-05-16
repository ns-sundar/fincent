"""Portfolio context helpers for the Goal Planning agent."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.agents.portfolio.loader import PortfolioSnapshot, load_portfolio
from src.core.config import AppConfig


def build_portfolio_summary(snapshot: PortfolioSnapshot) -> Dict[str, Any]:
    """Return a compact JSON-safe portfolio summary for planning."""

    accounts = []
    for account in snapshot.accounts:
        accounts.append(
            {
                "account_id": account.account_id,
                "name": account.name,
                "type": account.type,
                "balance": account.balance,
                "currency": account.currency,
                "holdings": account.holdings,
            }
        )
    return {
        "as_of": snapshot.as_of,
        "total_balance": snapshot.total_balance,
        "allocation": snapshot.allocation,
        "accounts": accounts,
        "transaction_count": snapshot.transaction_count,
        "recent_transactions": snapshot.all_transactions[:25],
    }


def load_portfolio_summary(
    cfg: Optional[AppConfig] = None,
    *,
    snapshot: Optional[PortfolioSnapshot] = None,
) -> Dict[str, Any]:
    """Load and summarize the current portfolio snapshot."""

    snap = snapshot or load_portfolio(cfg)
    return build_portfolio_summary(snap)


def portfolio_summary_json(
    cfg: Optional[AppConfig] = None,
    *,
    snapshot: Optional[PortfolioSnapshot] = None,
) -> str:
    """Render the portfolio summary as formatted JSON."""

    return json.dumps(load_portfolio_summary(cfg, snapshot=snapshot), indent=2, default=str)

