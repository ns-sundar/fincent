"""OpenBB MCP tool invocation helpers (FMP plan limits, paid providers)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.utils.logging import get_logger

_logger = get_logger(__name__)

# Shown in UI footprint (Streamlit "Under the hood") when answers rely on free or
# non-paywalled routes instead of FMP endpoints that require a paid plan.
FMP_FOOTPRINT_FREE_DATA_DISCLAIMER = (
    "Free / non-paywalled data was used instead of paywalled Financial Modeling "
    "Prep (FMP) OpenBB fundamentals. Responses may draw on yfinance, SEC filings, "
    "web search, or other sources. Configure FMP_ACCESS_TOKEN or FMP_API_KEY to "
    "enable FMP Starter-plan endpoints."
)

FMP_FOOTPRINT_RUNTIME_PAYWALL_DISCLAIMER = (
    "Some OpenBB requests hit FMP paywall or plan limits (HTTP 402 or similar); "
    "the agent may have relied on other free or non-FMP sources for that turn."
)

_FMP_TOOL_UNAVAILABLE_MESSAGE = (
    "[tool unavailable] FMP data for this request is not available on the "
    "configured plan. Use other available sources such as yfinance quotes, SEC "
    "filings/context, web search, or non-FMP OpenBB data instead."
)

_MAX_TOOL_RESULT_CHARS = 12_000

# OpenBB tool names that map only to Intrinio in the default extension set.
# Intrinio is paid-only, so these are omitted unless explicitly enabled.
OPENBB_TOOL_NAMES_REQUIRING_INTRINIO: frozenset[str] = frozenset(
    {
        "equity_fundamental_reported_financials",
    },
)

# These OpenBB fundamental/estimate tools route to FMP or another paid/keyed
# provider in the installed OpenBB bundle. They return HTTP 402 on free tiers,
# so expose them only when an FMP key is configured (Starter or higher) or when
# explicitly enabled with FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS=true.
OPENBB_TOOL_NAMES_REQUIRING_PAID_FMP: frozenset[str] = frozenset(
    {
        "equity_estimates_consensus",
        "equity_fundamental_balance",
        "equity_fundamental_balance_growth",
        "equity_fundamental_cash",
        "equity_fundamental_cash_growth",
        "equity_fundamental_income",
        "equity_fundamental_income_growth",
        "equity_fundamental_metrics",
        "equity_fundamental_ratios",
    },
)


def paid_fmp_tool(tool_name: str) -> bool:
    """Return True when an OpenBB tool commonly routes to paid FMP endpoints."""

    name = str(tool_name)
    return name in OPENBB_TOOL_NAMES_REQUIRING_PAID_FMP or name.startswith(
        "equity_estimates_"
    )


def paid_fmp_fundamentals_allowed() -> bool:
    """Return True when paid/keyed FMP fundamentals should be exposed.

    An FMP Starter (or higher) key is enough to opt in. Operators can still
    force free-tier behavior with FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS=false.
    """

    raw = os.environ.get("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes"}
    return fmp_api_key_configured()


def intrinio_provider_allowed() -> bool:
    """Return True only when Intrinio is explicitly enabled and credentialed."""

    enabled = os.environ.get("FINCENT_OPENBB_ALLOW_INTRINIO", "").lower() in {
        "1",
        "true",
        "yes",
    }
    return enabled and intrinio_api_key_configured()


def fmp_api_key_configured() -> bool:
    """Return True if an FMP key is present in the process environment."""

    for key in (
        "FMP_ACCESS_TOKEN",
        "FMP_API_KEY",
        "FINANCIAL_MODELING_PREP_API_KEY",
    ):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return True
    return False


def filter_openbb_mcp_tools_without_intrinio(tools: List[Any]) -> List[Any]:
    """Drop Intrinio-only OpenBB tools unless the paid provider is enabled."""

    if intrinio_provider_allowed():
        return list(tools)
    kept: List[Any] = []
    dropped = 0
    for t in tools:
        name = str(getattr(t, "name", "") or "")
        if name in OPENBB_TOOL_NAMES_REQUIRING_INTRINIO:
            dropped += 1
            continue
        kept.append(t)
    if dropped:
        _logger.info(
            "OpenBB MCP: excluded %d Intrinio-only tool(s) %s "
            "(set FINCENT_OPENBB_ALLOW_INTRINIO=true and INTRINIO_API_KEY to enable).",
            dropped,
            sorted(OPENBB_TOOL_NAMES_REQUIRING_INTRINIO),
        )
    return kept


def filter_openbb_mcp_tools_for_fmp_free_tier(tools: List[Any]) -> List[Any]:
    """Drop OpenBB tools that repeatedly hit paid FMP endpoints on free plans."""

    if paid_fmp_fundamentals_allowed():
        return list(tools)
    kept: List[Any] = []
    dropped = 0
    for t in tools:
        name = str(getattr(t, "name", "") or "")
        if paid_fmp_tool(name):
            dropped += 1
            continue
        kept.append(t)
    if dropped:
        _logger.info(
            "OpenBB MCP: excluded %d paid-FMP fundamental tool(s) for free-tier mode "
            "(configure FMP_ACCESS_TOKEN/FMP_API_KEY or set "
            "FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS=true to enable).",
            dropped,
        )
    return kept


def filter_openbb_mcp_tools_for_available_credentials(tools: List[Any]) -> List[Any]:
    """Apply OpenBB tool filters based on configured free/paid credentials."""

    return filter_openbb_mcp_tools_for_fmp_free_tier(
        filter_openbb_mcp_tools_without_intrinio(tools)
    )


def equity_fundamental_tool(tool_name: str) -> bool:
    return str(tool_name).startswith("equity_fundamental_")


def fmp_plan_or_symbol_denied(message: str) -> bool:
    """True when FMP/OpenBB error text indicates tier, symbol, or endpoint limits."""

    text = str(message)
    tl = text.lower()
    if "402" in text:
        return True
    if "unauthorized fmp" in tl:
        return True
    if "restricted endpoint" in tl:
        return True
    if "not available under your current subscription" in tl:
        return True
    if "premium query parameter" in tl:
        return True
    if "special endpoint" in tl and "subscription" in tl:
        return True
    return False


def openbb_tool_text_suggests_fmp_paywall_fallback(text: str) -> bool:
    """True when tool output indicates FMP was skipped or blocked at runtime."""

    t = str(text)
    tl = t.lower()
    if "[tool skipped]" in t and "fmp" in tl and "free-tier" in tl:
        return True
    if "[tool unavailable]" in t and "fmp" in tl:
        return True
    if "[tool error]" in t and fmp_plan_or_symbol_denied(t):
        return True
    return False


def fmp_sources_footprint_note(*, saw_runtime_fmp_paywall_signal: bool) -> str | None:
    """Human-readable disclaimer for Streamlit footprint when FMP paywalled paths were not used."""

    if not paid_fmp_fundamentals_allowed():
        return FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    if saw_runtime_fmp_paywall_signal:
        return FMP_FOOTPRINT_RUNTIME_PAYWALL_DISCLAIMER
    return None


def _openbb_provider_token(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str):
        return str(raw).strip().lower()
    return raw.strip().lower()


def intrinio_api_key_configured() -> bool:
    """Return True if an Intrinio key is present in the process environment."""

    for key in ("INTRINIO_API_KEY", "API_INTRINIO_KEY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return True
    return False


def fundamental_tool_fmp_hint(tool_name: str, error_snippet: str) -> str:
    """Append guidance when OpenBB fundamentals failed due to paid data limits."""

    if not equity_fundamental_tool(tool_name):
        return ""
    es = str(error_snippet).lower()
    if "missing credential" in es and "intrinio" in es:
        return (
            " Hint: This OpenBB tool requires Intrinio, which is disabled by default "
            "because it is paid-only. Use available SEC filings/context, quotes, news, "
            "or other non-Intrinio sources instead."
        )
    if not fmp_plan_or_symbol_denied(error_snippet):
        return ""
    return (
        " Hint: FMP blocked this OpenBB request (plan, symbol, or endpoint). "
        "Options: use SEC/Tavily/OpenBB non-FMP sources, try a plain US common-stock "
        "ticker, or explicitly enable paid FMP fundamentals if your subscription "
        "supports them."
    )


def _result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def compact_tool_result_for_llm(text: str, *, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """Keep tool messages small enough that one large payload cannot overflow context."""

    raw = str(text)
    if len(raw) <= max_chars:
        return raw
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(raw) - max_chars
    return (
        raw[:head_chars]
        + f"\n\n[tool output truncated: {omitted} characters omitted]\n\n"
        + raw[-tail_chars:]
    )


async def ainvoke_openbb_tool(
    tool: Any,
    call_args: Dict[str, Any],
    *,
    tool_name: str,
    logger: Any,
    log_context: str,
) -> str:
    """Invoke an OpenBB MCP tool while avoiding paid providers by default."""

    if (
        _openbb_provider_token(call_args.get("provider")) == "intrinio"
        and not intrinio_provider_allowed()
    ):
        return (
            "[tool skipped] Intrinio provider is disabled by default because it is "
            "paid-only. Set FINCENT_OPENBB_ALLOW_INTRINIO=true and INTRINIO_API_KEY "
            "only if you want to enable Intrinio calls."
        )

    if (
        paid_fmp_tool(tool_name)
        and not paid_fmp_fundamentals_allowed()
    ):
        return (
            "[tool skipped] OpenBB FMP fundamental tool omitted in free-tier mode. "
            "Configure FMP_ACCESS_TOKEN/FMP_API_KEY for FMP Starter-plan access, "
            "or set FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS=true to force-enable."
        )

    try:
        result = await tool.ainvoke(call_args)
    except Exception as exc:  # noqa: BLE001
        exc_str = f"{type(exc).__name__}: {exc}"
        logger.warning("%s tool %s failed: %s", log_context, tool_name or "?", exc)
        if fmp_plan_or_symbol_denied(exc_str):
            return _FMP_TOOL_UNAVAILABLE_MESSAGE
        hint = fundamental_tool_fmp_hint(tool_name, exc_str)
        return f"[tool error] {exc_str}" + hint
    return compact_tool_result_for_llm(_result_to_text(result))
