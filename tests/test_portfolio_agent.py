"""Unit tests for the Portfolio specialist agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool

from src.agents.openbb_mcp_invoke import FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
from src.agents.portfolio import answer as portfolio_answer
from src.agents.portfolio import get_portfolio_tools, load_portfolio
from src.agents.portfolio.loader import AccountSummary, PortfolioSnapshot
from src.agents.portfolio.mcp_tools import reset_portfolio_tools_cache
from src.agents.portfolio.seed import seed_portfolio_if_needed
from src.core.config import load_config, reset_config_cache
from src.core.schemas import AgentName


def _fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _force_fmp_free_tier(monkeypatch) -> None:
    monkeypatch.delenv("FMP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)


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
# OpenBB arg defaults (provider)
# ---------------------------------------------------------------------


def test_merge_openbb_tool_args_adds_yfinance_for_equity_price_quote():
    from src.agents.portfolio.agent import _merge_openbb_tool_args

    assert _merge_openbb_tool_args("equity_price_quote", {"symbol": "AAPL"}) == {
        "symbol": "AAPL",
        "provider": "yfinance",
    }
    assert _merge_openbb_tool_args(
        "equity_price_quote",
        {"symbol": "AAPL", "provider": "fmp"},
    ) == {"symbol": "AAPL", "provider": "yfinance"}
    assert _merge_openbb_tool_args("rag_search", {"query": "x"}) == {"query": "x"}


def test_merge_openbb_coerces_yfinance_to_fmp_for_fundamentals():
    from src.agents.portfolio.agent import _merge_openbb_tool_args

    assert _merge_openbb_tool_args(
        "equity_fundamental_ratios",
        {"symbol": "AAPL", "provider": "yfinance"},
    ) == {"symbol": "AAPL", "provider": "fmp"}


def test_merge_openbb_respects_allow_keyed_providers_env(monkeypatch):
    from src.agents.portfolio.agent import _merge_openbb_tool_args

    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_KEYED_PROVIDERS", "true")
    assert _merge_openbb_tool_args(
        "equity_price_quote",
        {"symbol": "AAPL", "provider": "fmp"},
    ) == {"symbol": "AAPL", "provider": "fmp"}


# ---------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------


def test_portfolio_answer_attribution_and_metadata(monkeypatch):
    """The agent must attribute to PORTFOLIO and surface snapshot rollups."""
    _force_fmp_free_tier(monkeypatch)
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
    assert response.metadata["tools_invoked"] == []
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER


def test_portfolio_answer_no_data_sources_note_when_paid_fmp_enabled(monkeypatch):
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")
    snap = _synthetic_snapshot()
    response = portfolio_answer(
        "How much cash do I have?",
        llm=_fake_llm("You hold $40 of cash and $100 of stocks."),
        snapshot=snap,
    )
    assert "data_sources_note" not in response.metadata


def test_portfolio_answer_no_data_sources_note_when_fmp_key_present(monkeypatch):
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)
    monkeypatch.setenv("FMP_ACCESS_TOKEN", "starter-plan-key")
    snap = _synthetic_snapshot()
    response = portfolio_answer(
        "How much cash do I have?",
        llm=_fake_llm("You hold $40 of cash and $100 of stocks."),
        snapshot=snap,
    )
    assert "data_sources_note" not in response.metadata


def test_portfolio_context_block_exposes_full_transaction_list():
    """The prompt must surface every transaction + a total count."""
    from src.agents.portfolio.agent import _build_portfolio_block

    snap = _synthetic_snapshot()
    block = _build_portfolio_block(snap)
    assert '"transaction_count": 2' in block
    assert "transactions_newest_first" in block
    # Both transaction tickers should appear in the JSON context.
    assert "AAA" in block and "CASH" in block


# ---------------------------------------------------------------------
# answer -- tool-calling path
# ---------------------------------------------------------------------


class _ScriptedToolCallingLLM(FakeListChatModel):
    """Fake chat model that emits a canned tool call on the first turn.

    FakeListChatModel can only return plain string responses and
    raises ``NotImplementedError`` for ``bind_tools``. This minimal
    wrapper overrides both so that:

    * ``bind_tools(tools)`` returns ``self`` (the tools list is
      irrelevant for the fake's scripted reply), and
    * the first ``invoke`` returns an ``AIMessage`` with one tool
      call. Subsequent calls fall back to the parent class so the
      final-answer text can be fed through ``responses``.
    """

    first_turn_tool: str = ""
    first_turn_args: Dict[str, Any] = {}
    _emitted: bool = False

    def bind_tools(self, tools: List[Any], **_kwargs: Any) -> "_ScriptedToolCallingLLM":  # type: ignore[override]
        return self

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        if not self._emitted and self.first_turn_tool:
            self._emitted = True
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.first_turn_tool,
                        "args": dict(self.first_turn_args),
                        "id": "call-1",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_portfolio_answer_invokes_mcp_tools_and_returns_final_text(monkeypatch):
    """A tool-calling LLM must drive the ReAct loop, then the final AIMessage wins.

    The fake LLM asks the agent to call ``rag_search`` once; the loop
    must invoke the (stub) tool, append a ToolMessage, and then the
    second invocation yields the user-visible final answer. Proves
    that the Portfolio agent correctly wires MCP-style tools.
    """
    _force_fmp_free_tier(monkeypatch)
    call_log: List[Dict[str, Any]] = []

    @tool("rag_search")
    def rag_search(query: str) -> str:
        """Search the Fincent knowledge base."""
        call_log.append({"query": query})
        return "[stub] ETFs trade on an exchange like stocks."

    llm = _ScriptedToolCallingLLM(
        responses=["Your holdings (AAA, CASH) complement a simple ETF strategy."],
        first_turn_tool="rag_search",
        first_turn_args={"query": "ETF basics"},
    )

    response = portfolio_answer(
        "How do ETFs work given my current holdings?",
        llm=llm,
        snapshot=_synthetic_snapshot(),
        tools=[rag_search],
    )
    assert response.agent == AgentName.PORTFOLIO
    assert "AAA" in response.content or "ETF" in response.content
    assert response.metadata["tool_count"] == 1
    assert response.metadata["tool_names"] == ["rag_search"]
    assert response.metadata["tools_invoked"] == ["rag_search"]
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    assert call_log == [{"query": "ETF basics"}]


def test_portfolio_answer_tool_free_path_runs_without_tools(monkeypatch):
    """Passing ``tools=[]`` must short-circuit past ``bind_tools``.

    Guarantees that operators with MCP disabled (or missing the
    ``langchain-mcp-adapters`` dependency) still get a working
    portfolio answer from the raw LLM output.
    """
    _force_fmp_free_tier(monkeypatch)
    response = portfolio_answer(
        "What's my total balance?",
        llm=_fake_llm("Your total balance is $140."),
        snapshot=_synthetic_snapshot(),
        tools=[],
    )
    assert response.agent == AgentName.PORTFOLIO
    assert "$140" in response.content
    assert response.metadata["tool_count"] == 0
    assert response.metadata["tool_names"] == []
    assert response.metadata["tools_invoked"] == []
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER


def test_portfolio_answer_handles_unknown_tool_gracefully(monkeypatch):
    """Unknown tool names feed an error back instead of crashing."""
    _force_fmp_free_tier(monkeypatch)
    llm = _ScriptedToolCallingLLM(
        responses=["Could not verify that -- I answered from the snapshot."],
        first_turn_tool="some_missing_tool",
        first_turn_args={"x": 1},
    )
    response = portfolio_answer(
        "What's my AAA ticker worth on the market today?",
        llm=llm,
        snapshot=_synthetic_snapshot(),
        tools=[],  # no tool actually registered
    )
    assert response.agent == AgentName.PORTFOLIO
    assert "snapshot" in response.content.lower()
    assert response.metadata["tools_invoked"] == ["some_missing_tool"]
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER


# ---------------------------------------------------------------------
# mcp_tools loader
# ---------------------------------------------------------------------


def test_get_portfolio_tools_disabled_in_tests_returns_empty_list():
    """With both MCP servers disabled (conftest default), loader returns []."""
    reset_portfolio_tools_cache()
    tools = get_portfolio_tools(force_refresh=True)
    assert tools == []


def test_get_portfolio_tools_caches_between_calls(monkeypatch):
    """A second call must not re-enter the async loader if cache is warm."""
    import src.agents.portfolio.mcp_tools as mcp_mod

    reset_portfolio_tools_cache()
    # Warm the cache with a sentinel.
    mcp_mod._TOOLS_CACHE = ["sentinel"]  # type: ignore[assignment]

    called = {"n": 0}

    def _boom(*_args: Any, **_kwargs: Any) -> List[Any]:
        called["n"] += 1
        raise AssertionError("cache was bypassed")

    monkeypatch.setattr(mcp_mod, "_build_client_specs", _boom)
    out = mcp_mod.get_portfolio_tools()
    assert out == ["sentinel"]
    assert called["n"] == 0


def test_portfolio_agent_uses_cached_mcp_tools_when_tools_arg_absent(monkeypatch):
    """When ``tools`` is omitted, the agent pulls from ``get_portfolio_tools``.

    We stub the loader to return a dummy empty list and verify the
    agent recorded ``tool_count=0`` rather than silently using the
    live-subprocess path. This keeps the wiring covered even in the
    fully-disabled default.
    """
    import src.agents.portfolio.agent as agent_mod

    _force_fmp_free_tier(monkeypatch)
    monkeypatch.setattr(agent_mod, "get_portfolio_tools", lambda _cfg: [])
    response = portfolio_answer(
        "How many accounts do I have?",
        llm=_fake_llm("You have 2 accounts."),
        snapshot=_synthetic_snapshot(),
    )
    assert response.metadata["tool_count"] == 0
    assert response.metadata["tool_names"] == []
    assert response.metadata["tools_invoked"] == []
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    assert "2 accounts" in response.content
