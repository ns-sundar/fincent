"""Deterministic and stochastic helpers for financial goal planning."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

_ASSET_ASSUMPTIONS: Dict[str, Dict[str, float]] = {
    "stock": {"return": 0.075, "volatility": 0.18},
    "equity": {"return": 0.075, "volatility": 0.18},
    "bond": {"return": 0.04, "volatility": 0.06},
    "fixed_income": {"return": 0.04, "volatility": 0.06},
    "cash": {"return": 0.025, "volatility": 0.01},
    "unknown": {"return": 0.05, "volatility": 0.12},
}


def calculate_fv(pv: float, rate: float, nper: int, pmt: float = 0.0) -> float:
    """Calculate future value with end-of-period contributions."""

    periods = int(nper)
    if periods < 0:
        raise ValueError("nper must be non-negative")
    pv_f = float(pv)
    rate_f = float(rate)
    pmt_f = float(pmt)
    if periods == 0:
        return pv_f
    if abs(rate_f) < 1e-12:
        return pv_f + pmt_f * periods
    growth = (1.0 + rate_f) ** periods
    return pv_f * growth + pmt_f * ((growth - 1.0) / rate_f)


def required_payment(target: float, pv: float, rate: float, nper: int) -> float:
    """Return required end-of-period contribution to reach a target."""

    periods = int(nper)
    if periods <= 0:
        raise ValueError("nper must be positive")
    target_f = float(target)
    pv_f = float(pv)
    rate_f = float(rate)
    if abs(rate_f) < 1e-12:
        return (target_f - pv_f) / periods
    growth = (1.0 + rate_f) ** periods
    return (target_f - pv_f * growth) * rate_f / (growth - 1.0)


def _normalise_mix(portfolio_mix: Mapping[str, Any]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    total = 0.0
    for raw_key, raw_value in portfolio_mix.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        key = str(raw_key).strip().lower() or "unknown"
        weights[key] = weights.get(key, 0.0) + value
        total += value
    if total <= 0:
        return {"unknown": 1.0}
    return {key: value / total for key, value in weights.items()}


def portfolio_return_volatility(portfolio_mix: Mapping[str, Any]) -> Dict[str, Any]:
    """Estimate nominal annual return and volatility from asset-class weights."""

    weights = _normalise_mix(portfolio_mix)
    expected_return = 0.0
    variance = 0.0
    for key, weight in weights.items():
        assumption = _ASSET_ASSUMPTIONS.get(key, _ASSET_ASSUMPTIONS["unknown"])
        expected_return += weight * assumption["return"]
        # Simple independent-asset approximation; intentionally conservative
        # enough for educational planning without a full covariance matrix.
        variance += (weight * assumption["volatility"]) ** 2
    return {
        "weights": weights,
        "expected_return": expected_return,
        "volatility": float(np.sqrt(variance)),
        "assumptions": _ASSET_ASSUMPTIONS,
    }


def run_monte_carlo(
    portfolio_mix: Mapping[str, Any],
    years: float,
    monthly_contribution: float,
    target: float,
    *,
    starting_balance: float = 0.0,
    simulations: int = 5000,
    seed: int | None = 42,
) -> Dict[str, Any]:
    """Simulate portfolio paths and return a compact distribution profile."""

    years_f = max(0.0, float(years))
    months = int(round(years_f * 12))
    if months <= 0:
        ending = np.full(max(1, int(simulations)), float(starting_balance))
    else:
        sims = max(1, int(simulations))
        stats = portfolio_return_volatility(portfolio_mix)
        annual_return = float(stats["expected_return"])
        annual_vol = float(stats["volatility"])
        monthly_mean = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
        monthly_vol = annual_vol / np.sqrt(12.0)
        rng = np.random.default_rng(seed)
        monthly_returns = rng.normal(monthly_mean, monthly_vol, size=(sims, months))
        monthly_returns = np.clip(monthly_returns, -0.95, 1.0)
        paths = np.empty((sims, months + 1), dtype=float)
        paths[:, 0] = float(starting_balance)
        contribution = float(monthly_contribution)
        for idx in range(months):
            paths[:, idx + 1] = paths[:, idx] * (1.0 + monthly_returns[:, idx]) + contribution
        ending = paths[:, -1]
    target_f = float(target)
    percentiles = np.percentile(ending, [5, 10, 25, 50, 75, 90, 95])
    success_probability = float(np.mean(ending >= target_f)) if target_f > 0 else 1.0
    stats = portfolio_return_volatility(portfolio_mix)
    return {
        "years": years_f,
        "months": months,
        "starting_balance": float(starting_balance),
        "monthly_contribution": float(monthly_contribution),
        "target": target_f,
        "simulations": int(len(ending)),
        "portfolio_mix": stats["weights"],
        "assumed_annual_return": stats["expected_return"],
        "assumed_annual_volatility": stats["volatility"],
        "probability_of_success": success_probability,
        "median_ending_value": float(percentiles[3]),
        "mean_ending_value": float(np.mean(ending)),
        "shortfall_probability": 1.0 - success_probability,
        "percentiles": {
            "p05": float(percentiles[0]),
            "p10": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p90": float(percentiles[5]),
            "p95": float(percentiles[6]),
        },
    }

