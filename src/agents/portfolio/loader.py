"""Static-file loader + derived views of the user's portfolio.

The Portfolio agent and the Streamlit UI both need the same three
views of the data:

* a list of accounts sorted by current balance (descending),
* an asset-class allocation split (``stock`` / ``bond`` / ``cash``),
* the ten most recent transactions.

Both the LLM agent (text context) and the UI (plotly graphics) render
these from the same ``PortfolioSnapshot`` object, so the two stay in
lock-step automatically.

Data source: ``cfg.portfolio.data_path`` (e.g. ``/data/portfolio`` in
production), which must contain ``accounts.json`` and
``transactions.json`` in the schema documented in those files. On
first load the directory is seeded from ``cfg.portfolio.seed_path``
(see :mod:`src.agents.portfolio.seed`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.portfolio.seed import seed_portfolio_if_needed
from src.core.config import AppConfig, get_config
from src.utils.logging import get_logger

_logger = get_logger(__name__)

# Canonical asset-class buckets used by the allocation pie chart and
# the agent's grounded summary. ``unknown`` catches any account whose
# ``type`` does not match one of the three recognised classes.
_ASSET_CLASSES: tuple[str, ...] = ("stock", "bond", "cash")


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AccountSummary:
    """A single account with its derived current balance.

    ``balance`` is computed as ``sum(shares * current_price)`` across
    all holdings so the cash-money-market account (``shares=22k`` at
    ``price=1.0``) produces the expected dollar figure.
    """

    account_id: str
    name: str
    type: str
    broker: str
    currency: str
    balance: float
    holdings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Everything the agent + UI need about the portfolio.

    ``accounts`` is pre-sorted by descending balance.
    ``allocation`` maps canonical asset class -> dollar total.
    ``all_transactions`` is every transaction on file, sorted newest
    first. ``recent_transactions`` is just the first 10 of that list
    (kept as a convenience for the UI; the agent sees the full list
    and the total count so questions like "how many transactions do I
    have?" answer correctly).
    """

    accounts: List[AccountSummary]
    allocation: Dict[str, float]
    all_transactions: List[Dict[str, Any]]
    recent_transactions: List[Dict[str, Any]]
    transaction_count: int
    total_balance: float
    as_of: str = ""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _resolve_data_dir(cfg: AppConfig) -> Path:
    """Resolve the runtime portfolio directory, seeding it if needed.

    Delegates to :func:`seed_portfolio_if_needed`, which returns either
    the fully-seeded ``data_path`` or, if ``data_path`` cannot be
    created (e.g. ``/data`` is not mounted in a dev environment), the
    read-only ``seed_path`` as a fallback.
    """
    return seed_portfolio_if_needed(cfg)


def _read_json(path: Path) -> Any:
    """Read a JSON file with a helpful error message on failure."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Portfolio data file is missing: {path}. "
            f"Expected JSON at this path."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _account_balance(account: Dict[str, Any]) -> float:
    """Compute ``sum(shares * current_price)`` across holdings."""
    total = 0.0
    for holding in account.get("holdings") or []:
        shares = float(holding.get("shares") or 0)
        price = float(holding.get("current_price") or 0)
        total += shares * price
    return round(total, 2)


def _allocation(accounts: List[AccountSummary]) -> Dict[str, float]:
    """Bucket account balances by asset class, rounded to cents."""
    buckets: Dict[str, float] = {cls: 0.0 for cls in _ASSET_CLASSES}
    for acc in accounts:
        key = acc.type if acc.type in _ASSET_CLASSES else "unknown"
        buckets[key] = buckets.get(key, 0.0) + acc.balance
    return {k: round(v, 2) for k, v in buckets.items() if v > 0 or k in _ASSET_CLASSES}


def _sort_newest_first(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return every transaction sorted newest first (stable on date)."""
    def _key(t: Dict[str, Any]) -> str:
        return str(t.get("date") or "")
    return sorted(transactions, key=_key, reverse=True)


# ---------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------


def load_portfolio(
    cfg: Optional[AppConfig] = None,
    *,
    force_refresh: bool = False,
) -> PortfolioSnapshot:
    """Load the static portfolio from disk into a ``PortfolioSnapshot``.

    The result is memoised per-process; pass ``force_refresh=True`` to
    re-read the JSON files (handy during development / tests).
    """
    if force_refresh:
        _load_portfolio_cached.cache_clear()
    cfg = cfg or get_config()
    return _load_portfolio_cached(_resolve_data_dir(cfg))


@lru_cache(maxsize=4)
def _load_portfolio_cached(data_dir: Path) -> PortfolioSnapshot:
    """Cached worker behind :func:`load_portfolio`.

    Keyed by the resolved on-disk directory so different config paths
    produce independent caches.
    """
    _logger.info("Loading portfolio snapshot from %s", data_dir)
    accounts_raw = _read_json(data_dir / "accounts.json")
    transactions_raw = _read_json(data_dir / "transactions.json")

    if not isinstance(accounts_raw, list):
        raise ValueError("accounts.json must be a JSON array of accounts.")
    if not isinstance(transactions_raw, list):
        raise ValueError("transactions.json must be a JSON array of transactions.")

    summaries: List[AccountSummary] = []
    for account in accounts_raw:
        summaries.append(
            AccountSummary(
                account_id=str(account.get("account_id") or ""),
                name=str(account.get("name") or ""),
                type=str(account.get("type") or "unknown").lower(),
                broker=str(account.get("broker") or ""),
                currency=str(account.get("currency") or "USD"),
                balance=_account_balance(account),
                holdings=list(account.get("holdings") or []),
            )
        )
    summaries.sort(key=lambda a: a.balance, reverse=True)

    total = round(sum(a.balance for a in summaries), 2)
    allocation = _allocation(summaries)
    all_sorted = _sort_newest_first(transactions_raw)
    recent = all_sorted[:10]
    most_recent_date = str(recent[0].get("date") or "") if recent else ""

    return PortfolioSnapshot(
        accounts=summaries,
        allocation=allocation,
        all_transactions=all_sorted,
        recent_transactions=recent,
        transaction_count=len(all_sorted),
        total_balance=total,
        as_of=most_recent_date,
    )
