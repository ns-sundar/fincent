"""Tests for Goal Planning financial math and MCP wrappers."""

from __future__ import annotations

from src.agents.goal_planning.financial_math import calculate_fv, run_monte_carlo
from src.agents.goal_planning.mcp_server import calculate_fv as mcp_calculate_fv
from src.agents.goal_planning.mcp_server import run_monte_carlo as mcp_run_monte_carlo


def test_calculate_fv_compounds_pv_and_payments():
    out = calculate_fv(pv=1000, rate=0.01, nper=12, pmt=100)
    assert round(out, 2) == 2395.08


def test_mcp_calculate_fv_returns_payload():
    out = mcp_calculate_fv(pv=1000, rate=0.0, nper=3, pmt=100)
    assert out["future_value"] == 1300
    assert out["pmt"] == 100


def test_run_monte_carlo_returns_probability_distribution():
    out = run_monte_carlo(
        portfolio_mix={"stock": 0.6, "bond": 0.3, "cash": 0.1},
        years=10,
        monthly_contribution=1000,
        target=200000,
        starting_balance=50000,
        simulations=500,
        seed=7,
    )
    assert 0.0 <= out["probability_of_success"] <= 1.0
    assert out["simulations"] == 500
    assert out["percentiles"]["p05"] <= out["percentiles"]["p50"]
    assert out["percentiles"]["p50"] <= out["percentiles"]["p95"]


def test_mcp_run_monte_carlo_defaults_to_portfolio_mix_when_missing():
    out = mcp_run_monte_carlo(
        years=1,
        monthly_contribution=1000,
        target=12000,
        simulations=100,
        seed=42,
    )
    assert out["simulations"] == 100
    assert out["starting_balance"] > 0
    assert out["portfolio_mix"]

