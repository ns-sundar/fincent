"""Internal MCP server for Goal Planning calculations and context."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.agents.goal_planning.context import load_portfolio_summary
from src.agents.goal_planning.financial_math import (
    calculate_fv as _calculate_fv,
    run_monte_carlo as _run_monte_carlo,
)
from src.core.config import AppConfig, get_config
from src.utils.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def calculate_fv(pv: float, rate: float, nper: int, pmt: float = 0.0) -> Dict[str, Any]:
    """MCP-friendly wrapper for future value calculations."""

    fv = _calculate_fv(pv=pv, rate=rate, nper=nper, pmt=pmt)
    return {"future_value": fv, "pv": pv, "rate": rate, "nper": nper, "pmt": pmt}


def _portfolio_defaults() -> Dict[str, Any]:
    summary = load_portfolio_summary()
    return {
        "portfolio_mix": summary.get("allocation") or {"unknown": 1.0},
        "starting_balance": float(summary.get("total_balance") or 0.0),
    }


def run_monte_carlo(
    years: float,
    monthly_contribution: float,
    target: float,
    portfolio_mix: Optional[Mapping[str, Any]] = None,
    starting_balance: float = 0.0,
    simulations: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """MCP-friendly wrapper for goal success simulations."""

    cfg = get_config()
    sim_count = int(simulations or cfg.goal_planning.monte_carlo_simulations)
    sim_count = max(1, min(sim_count, cfg.goal_planning.max_monte_carlo_simulations))
    years = max(0.0, min(float(years), float(cfg.goal_planning.max_projection_years)))
    if not portfolio_mix:
        defaults = _portfolio_defaults()
        portfolio_mix = defaults["portfolio_mix"]
        if not starting_balance:
            starting_balance = defaults["starting_balance"]
    return _run_monte_carlo(
        portfolio_mix=portfolio_mix,
        years=years,
        monthly_contribution=monthly_contribution,
        target=target,
        starting_balance=starting_balance,
        simulations=sim_count,
        seed=seed,
    )


def portfolio_summary() -> Dict[str, Any]:
    """Return the current user's compact portfolio summary."""

    return load_portfolio_summary()


def build_server(cfg: Optional[AppConfig] = None) -> Any:
    """Construct a FastMCP server exposing goal-planning helpers."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The 'mcp' package is required for Goal Planning MCP.") from exc

    cfg = cfg or get_config()
    server = FastMCP(
        name="fincent-goal-planning",
        instructions=(
            "Deterministic and stochastic planning tools for Fincent. "
            "Use these for TVM calculations, Monte Carlo goal success "
            "analysis, and portfolio summary context."
        ),
    )

    @server.tool(name="calculate_fv")
    def _tool_calculate_fv(pv: float, rate: float, nper: int, pmt: float = 0.0) -> Dict[str, Any]:
        """Calculate future value using end-of-period contributions."""

        return calculate_fv(pv=pv, rate=rate, nper=nper, pmt=pmt)

    @server.tool(name="run_monte_carlo")
    def _tool_run_monte_carlo(
        years: float,
        monthly_contribution: float,
        target: float,
        portfolio_mix: Optional[Dict[str, Any]] = None,
        starting_balance: float = 0.0,
        simulations: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Simulate portfolio paths and return probability of success."""

        return run_monte_carlo(
            portfolio_mix=portfolio_mix,
            years=years,
            monthly_contribution=monthly_contribution,
            target=target,
            starting_balance=starting_balance,
            simulations=simulations,
            seed=seed,
        )

    @server.tool(name="portfolio_summary")
    def _tool_portfolio_summary() -> Dict[str, Any]:
        """Return Fincent's compact portfolio summary JSON."""

        return portfolio_summary()

    resource = getattr(server, "resource", None)
    if callable(resource):
        try:

            @resource("fincent://portfolio/summary")
            def _portfolio_summary_resource() -> Dict[str, Any]:
                return portfolio_summary()

        except Exception as exc:  # pragma: no cover -- depends on MCP version
            _logger.debug("Goal Planning MCP resource registration skipped: %s", exc)

    return server


def main() -> None:
    """Run the Goal Planning MCP server over stdio."""

    cfg = get_config()
    configure_logging(level=cfg.logging.level, log_file=cfg.logging.file or None)
    _logger.info("Starting Fincent Goal Planning MCP server")
    build_server(cfg).run(transport="stdio")


if __name__ == "__main__":
    main()

