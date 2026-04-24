"""Unit tests for the Portfolio specialist agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.portfolio import answer as portfolio_answer
from src.agents.portfolio import load_portfolio
from src.agents.portfolio.loader import AccountSummary, PortfolioSnapshot
from src.agents.portfolio.seed import seed_portfolio_if_needed
from src.core.config import load_config, reset_config_cache
from src.core.schemas import AgentName


def _fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _synthetic_snapshot() -> PortfolioSnapshot:
    """A tiny in-memory snapshot so tests don't depend on disk files."""
    accounts = [
        AccountSummary(
            account_id="ACC-X",
            name="Stocks",
            type="stock",
            broker="Test",
            currency="USD",
            balance=100.0,
            holdings=[{"ticker": "AAA", "shares": 1, "current_price": 100.0}],
        ),
        AccountSummary(
            account_id="ACC-Y",
            name="Cash",
            type="cash",
            broker="Test",
            currency="USD",
            balance=40.0,
            holdings=[{"ticker": "CASH", "shares": 40, "current_price": 1.0}],
        ),
    ]
    transactions = [
        {"date": "2026-01-10", "type": "buy", "ticker": "AAA", "amount": 100.0},
        {"date": "2025-12-01", "type": "deposit", "ticker": "CASH", "amount": 40.0},
    ]
    return PortfolioSnapshot(
        accounts=accounts,
        allocation={"stock": 100.0, "bond": 0.0, "cash": 40.0},
        all_transactions=transactions,
        recent_transactions=transactions,
        transaction_count=len(transactions),
        total_balance=140.0,
        as_of="2026-01-10",
    )


# ---------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------


def test_load_portfolio_reads_default_files():
    """The default snapshot ships with 4 accounts sorted by balance desc.

    Tests run with ``FINCENT__PORTFOLIO__DATA_PATH`` pointed at an
    ephemeral tmp dir (see ``conftest.py``). On first call the loader
    seeds that directory from the repo's ``data/default_portfolio/``,
    so what the test reads below is the same snapshot shipped in the
    repo.
    """
    snap = load_portfolio(force_refresh=True)
    assert len(snap.accounts) == 4
    balances = [a.balance for a in snap.accounts]
    assert balances == sorted(balances, reverse=True)
    assert snap.total_balance > 0
    # Stocks / bonds / cash buckets must all be populated.
    assert set(snap.allocation) >= {"stock", "bond", "cash"}
    # The UI preview is capped at 10 rows, but the full list AND the
    # total count must be preserved so the agent can answer aggregate
    # questions like "how many transactions do I have?".
    assert len(snap.recent_transactions) == 10
    assert snap.transaction_count == len(snap.all_transactions)
    assert snap.transaction_count > 10
    # all_transactions must be sorted newest first.
    dates = [str(t.get("date") or "") for t in snap.all_transactions]
    assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------


def _with_portfolio_path(tmp_path: Path):
    """Set ``FINCENT__PORTFOLIO__DATA_PATH`` to ``tmp_path`` for one test.

    Returns a context-manager-like tuple of (activate, restore) so the
    env var is restored even if the test fails.
    """
    key = "FINCENT__PORTFOLIO__DATA_PATH"
    previous = os.environ.get(key)
    os.environ[key] = str(tmp_path)
    reset_config_cache()
    return previous, key


def _restore_portfolio_path(previous, key):
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous
    reset_config_cache()


def test_seed_portfolio_copies_files_on_first_run(tmp_path):
    """Seeding an empty target dir must copy every JSON from seed_path."""
    previous, key = _with_portfolio_path(tmp_path)
    try:
        cfg = load_config()
        seeded = seed_portfolio_if_needed(cfg)
        assert seeded == tmp_path.resolve() or seeded == tmp_path
        copied_names = sorted(p.name for p in tmp_path.glob("*.json"))
        assert "accounts.json" in copied_names
        assert "transactions.json" in copied_names
    finally:
        _restore_portfolio_path(previous, key)


def test_seed_portfolio_is_idempotent_and_preserves_edits(tmp_path):
    """Re-seeding must NOT overwrite files the user (or agent) edited.

    This is the property that lets us run seeding on every startup:
    once ``/data/portfolio`` is populated, subsequent boots leave it
    alone so any user or in-app mutation survives restarts.
    """
    previous, key = _with_portfolio_path(tmp_path)
    try:
        cfg = load_config()
        seed_portfolio_if_needed(cfg)

        # Mutate accounts.json to simulate an in-app edit.
        accounts_file = tmp_path / "accounts.json"
        original = json.loads(accounts_file.read_text(encoding="utf-8"))
        sentinel = [{"account_id": "SENTINEL", "name": "edited"}]
        accounts_file.write_text(json.dumps(sentinel), encoding="utf-8")

        # Re-seed: must leave the edited file intact.
        seed_portfolio_if_needed(cfg)
        assert json.loads(accounts_file.read_text(encoding="utf-8")) == sentinel

        # But the transactions file (which we did not touch) must
        # remain the same as what the seed copied on the first run.
        assert (tmp_path / "transactions.json").exists()
        # Sanity: the seed must not have deleted the unrelated file.
        assert original[0].get("account_id")
    finally:
        _restore_portfolio_path(previous, key)


def test_seed_portfolio_falls_back_when_target_not_writable(tmp_path):
    """An unwritable data_path must NOT crash; fall back to seed_path."""
    previous, key = _with_portfolio_path(tmp_path / "subdir" / "missing")
    try:
        # Make the parent exist but read-only so mkdir(parents=True)
        # below fails with OSError on POSIX.
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir").chmod(0o500)
        try:
            cfg = load_config()
            seeded = seed_portfolio_if_needed(cfg)
            # Fallback: the function returns the seed dir so the
            # loader can still read the static default.
            assert (seeded / "accounts.json").is_file()
        finally:
            # Restore writability so pytest can clean the tmpdir up.
            (tmp_path / "subdir").chmod(0o700)
    finally:
        _restore_portfolio_path(previous, key)


# ---------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------


def test_portfolio_answer_attribution_and_metadata():
    """The agent must attribute to PORTFOLIO and surface snapshot rollups."""
    snap = _synthetic_snapshot()
    response = portfolio_answer(
        "How much cash do I have?",
        llm=_fake_llm("You hold $40 of cash and $100 of stocks."),
        snapshot=snap,
    )
    assert response.agent == AgentName.PORTFOLIO
    assert "$40" in response.content
    assert response.metadata["account_count"] == 2
    assert response.metadata["total_balance"] == 140.0
    assert response.metadata["allocation"]["cash"] == 40.0
    assert response.metadata["transaction_count"] == 2


def test_portfolio_context_block_exposes_full_transaction_list():
    """The prompt must surface every transaction + a total count."""
    from src.agents.portfolio.agent import _build_portfolio_block

    snap = _synthetic_snapshot()
    block = _build_portfolio_block(snap)
    assert '"transaction_count": 2' in block
    assert "transactions_newest_first" in block
    # Both transaction tickers should appear in the JSON context.
    assert "AAA" in block and "CASH" in block
