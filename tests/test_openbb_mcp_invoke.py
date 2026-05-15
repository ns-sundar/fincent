"""Tests for OpenBB MCP invoke helpers (FMP free-tier and paid providers)."""

from __future__ import annotations

import pytest

from src.agents.openbb_mcp_invoke import (
    ainvoke_openbb_tool,
    FMP_FOOTPRINT_FREE_DATA_DISCLAIMER,
    FMP_FOOTPRINT_RUNTIME_PAYWALL_DISCLAIMER,
    fmp_plan_or_symbol_denied,
    fmp_sources_footprint_note,
    fundamental_tool_fmp_hint,
    fmp_api_key_configured,
    intrinio_provider_allowed,
    openbb_tool_text_suggests_fmp_paywall_fallback,
    compact_tool_result_for_llm,
)


def _clear_fmp_env(monkeypatch):
    monkeypatch.delenv("FMP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)


def test_openbb_tool_text_detects_skipped_fmp_and_402_errors():
    assert openbb_tool_text_suggests_fmp_paywall_fallback(
        "[tool skipped] OpenBB FMP fundamental tool omitted in free-tier mode."
    )
    assert openbb_tool_text_suggests_fmp_paywall_fallback(
        "[tool error] HTTPError: 402 something"
    )
    assert openbb_tool_text_suggests_fmp_paywall_fallback(
        "[tool unavailable] FMP data for this request is not available."
    )
    assert not openbb_tool_text_suggests_fmp_paywall_fallback(
        "[tool skipped] Intrinio provider is disabled"
    )


def test_fmp_sources_footprint_note_free_tier_vs_runtime(monkeypatch):
    _clear_fmp_env(monkeypatch)
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)
    assert fmp_sources_footprint_note(saw_runtime_fmp_paywall_signal=False) == (
        FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    )

    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")
    assert fmp_sources_footprint_note(saw_runtime_fmp_paywall_signal=False) is None
    assert fmp_sources_footprint_note(saw_runtime_fmp_paywall_signal=True) == (
        FMP_FOOTPRINT_RUNTIME_PAYWALL_DISCLAIMER
    )


def test_fmp_key_enables_paid_fmp_by_default(monkeypatch):
    _clear_fmp_env(monkeypatch)
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)
    monkeypatch.setenv("FMP_ACCESS_TOKEN", "starter-plan-key")

    assert fmp_api_key_configured()
    assert fmp_sources_footprint_note(saw_runtime_fmp_paywall_signal=False) is None

    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "false")
    assert fmp_sources_footprint_note(saw_runtime_fmp_paywall_signal=False) == (
        FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    )


def test_compact_tool_result_for_llm_caps_large_payloads():
    text = "a" * 50 + "MIDDLE" + "z" * 50
    out = compact_tool_result_for_llm(text, max_chars=40)
    assert "tool output truncated" in out
    assert out.startswith("a" * 20)
    assert out.endswith("z" * 20)
    assert "MIDDLE" not in out


def test_filter_drops_paid_fmp_and_reported_financials_by_default(monkeypatch):
    from types import SimpleNamespace

    from src.agents.openbb_mcp_invoke import filter_openbb_mcp_tools_for_available_credentials

    _clear_fmp_env(monkeypatch)
    monkeypatch.delenv("INTRINIO_API_KEY", raising=False)
    monkeypatch.delenv("API_INTRINIO_KEY", raising=False)
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)
    tools = [
        SimpleNamespace(name="equity_price_quote"),
        SimpleNamespace(name="equity_estimates_consensus"),
        SimpleNamespace(name="equity_fundamental_income"),
        SimpleNamespace(name="equity_fundamental_reported_financials"),
    ]
    out = filter_openbb_mcp_tools_for_available_credentials(tools)
    assert [t.name for t in out] == ["equity_price_quote"]


def test_filter_keeps_paid_fmp_tools_when_explicitly_enabled(monkeypatch):
    from types import SimpleNamespace

    from src.agents.openbb_mcp_invoke import filter_openbb_mcp_tools_for_available_credentials

    monkeypatch.delenv("INTRINIO_API_KEY", raising=False)
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")
    tools = [
        SimpleNamespace(name="equity_fundamental_income"),
        SimpleNamespace(name="equity_fundamental_reported_financials"),
    ]
    out = filter_openbb_mcp_tools_for_available_credentials(tools)
    assert [t.name for t in out] == ["equity_fundamental_income"]


def test_filter_keeps_intrinio_tool_only_when_explicitly_enabled(monkeypatch):
    from types import SimpleNamespace

    from src.agents.openbb_mcp_invoke import filter_openbb_mcp_tools_for_available_credentials

    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_INTRINIO", "true")
    monkeypatch.setenv("INTRINIO_API_KEY", "intrinio-test-key")
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")
    tools = [SimpleNamespace(name="equity_fundamental_reported_financials")]
    out = filter_openbb_mcp_tools_for_available_credentials(tools)
    assert [t.name for t in out] == ["equity_fundamental_reported_financials"]


def test_fundamental_hint_for_missing_intrinio_credential():
    from src.agents.openbb_mcp_invoke import fundamental_tool_fmp_hint

    h = fundamental_tool_fmp_hint(
        "equity_fundamental_reported_financials",
        "Missing credential 'intrinio_api_key'",
    )
    assert "disabled by default" in h
    assert "non-Intrinio sources" in h
    assert fmp_plan_or_symbol_denied(
        "Unauthorized FMP request -> 402 -> Premium Query Parameter"
    )
    assert fmp_plan_or_symbol_denied(
        "not available under your current subscription"
    )
    assert not fmp_plan_or_symbol_denied("HTTP 500: upstream timeout")


def test_fundamental_hint_only_for_fundamentals_and_fmp():
    msg = "402 Premium"
    h = fundamental_tool_fmp_hint("equity_fundamental_ratios", msg)
    assert "SEC/Tavily" in h
    assert fundamental_tool_fmp_hint("equity_price_quote", msg) == ""


class _FakeTool:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict):
        self.calls.append(dict(args))
        if args.get("provider") == "intrinio":
            return {"ok": True, "provider": "intrinio"}
        if len(self.calls) == 1:
            raise RuntimeError(
                "HTTP error 502: Bad Gateway - {'detail': 'Unauthorized FMP request -> 402'}"
            )
        return {"ok": True, "provider": args.get("provider")}


@pytest.mark.asyncio
async def test_intrinio_not_used_without_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("INTRINIO_API_KEY", "intrinio-test-key")
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_INTRINIO", raising=False)
    assert not intrinio_provider_allowed()

    class _Log:
        def warning(self, *args):
            pass

    tool = _FakeTool()
    out = await ainvoke_openbb_tool(
        tool,
        {"symbol": "AAPL", "provider": "intrinio"},
        tool_name="equity_fundamental_reported_financials",
        logger=_Log(),
        log_context="test",
    )
    assert len(tool.calls) == 0
    assert "[tool skipped]" in out


@pytest.mark.asyncio
async def test_intrinio_explicit_opt_in_allows_provider_call(monkeypatch):
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_INTRINIO", "true")
    monkeypatch.setenv("INTRINIO_API_KEY", "intrinio-test-key")
    assert intrinio_provider_allowed()

    class _Log:
        def warning(self, *args):
            pass

    tool = _FakeTool()
    out = await ainvoke_openbb_tool(
        tool,
        {"symbol": "AAPL", "provider": "intrinio"},
        tool_name="equity_fundamental_reported_financials",
        logger=_Log(),
        log_context="test",
    )
    assert len(tool.calls) == 1
    assert tool.calls[0]["provider"] == "intrinio"
    assert "intrinio" in out.lower() or "true" in out.lower()


@pytest.mark.asyncio
async def test_no_retry_without_intrinio(monkeypatch):
    _clear_fmp_env(monkeypatch)
    monkeypatch.delenv("INTRINIO_API_KEY", raising=False)
    monkeypatch.delenv("API_INTRINIO_KEY", raising=False)

    class _Log:
        def warning(self, *args):
            pass

    tool = _FakeTool()

    out = await ainvoke_openbb_tool(
        tool,
        {"symbol": "WEIRD", "provider": "fmp"},
        tool_name="equity_fundamental_ratios",
        logger=_Log(),
        log_context="test",
    )
    assert len(tool.calls) == 0
    assert "[tool skipped]" in out
    assert "FMP Starter-plan access" in out


@pytest.mark.asyncio
async def test_fmp_paywall_errors_are_user_safe(monkeypatch):
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")

    class _Log:
        def warning(self, *args):
            pass

    tool = _FakeTool()
    out = await ainvoke_openbb_tool(
        tool,
        {"symbol": "WEIRD", "provider": "fmp"},
        tool_name="equity_estimates_consensus",
        logger=_Log(),
        log_context="test",
    )
    assert len(tool.calls) == 1
    assert "[tool unavailable]" in out
    assert "FMP data" in out
    assert "HTTP error 502" not in out
    assert "Premium Query Parameter" not in out
