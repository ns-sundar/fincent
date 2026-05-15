"""Tests for optional FMP MCP tool name filtering."""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.market_research.mcp_tools import _filter_fmp_tools_by_name_substrings


def test_filter_drops_news_company_when_substring_news():
    tools = [
        SimpleNamespace(name="get_income_statement"),
        SimpleNamespace(name="news_company"),
        SimpleNamespace(name="CompanyNews"),  # case-insensitive match
    ]
    out = _filter_fmp_tools_by_name_substrings(
        tools, ["news"], server_key="fmp"
    )
    assert [t.name for t in out] == ["get_income_statement"]


def test_filter_empty_substrings_keeps_all():
    tools = [SimpleNamespace(name="news_company")]
    out = _filter_fmp_tools_by_name_substrings(
        tools, [], server_key="fmp"
    )
    assert len(out) == 1
