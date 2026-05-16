"""Plotly-powered graphics for the Streamlit Portfolio tab.

The three graphics are driven by the same ``PortfolioSnapshot`` the
agent sees, so the right-hand panel and the chat stay in sync.

Graphics:

1. **Accounts table**   -- one row per account, sorted by current
   balance (desc).
2. **Allocation pie**   -- stocks / bonds / cash split by dollar total.
3. **Recent activity**  -- the ten most recent transactions (newest
   first).
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Set

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.portfolio.loader import PortfolioSnapshot


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Stable, colour-blind-friendly palette for the asset-class pie.
_ALLOCATION_COLORS: Dict[str, str] = {
    "stock": "#1f77b4",
    "bond": "#2ca02c",
    "cash": "#ff7f0e",
    "unknown": "#8c8c8c",
}


def _fmt_currency(value: float, currency: str = "USD") -> str:
    """Render a number as ``$12,345.67``-style currency."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "-"
    symbol = "$" if currency.upper() == "USD" else f"{currency} "
    return f"{symbol}{amount:,.2f}"


def _accounts_dataframe(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """Build the accounts table already sorted by balance (desc)."""
    rows: List[Dict[str, Any]] = []
    for acc in snapshot.accounts:
        rows.append(
            {
                "Account": acc.name,
                "Type": acc.type.capitalize(),
                "Broker": acc.broker,
                "Balance": acc.balance,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Balance", ascending=False, ignore_index=True)
    return df


def _allocation_dataframe(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """Build the allocation table in a predictable class order."""
    rows: List[Dict[str, Any]] = []
    for asset_class in ("stock", "bond", "cash", "unknown"):
        amount = snapshot.allocation.get(asset_class, 0.0)
        if amount <= 0:
            continue
        rows.append({"Asset class": asset_class.capitalize(), "Balance": amount})
    return pd.DataFrame(rows)


def _transactions_dataframe(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """Normalise recent transactions into a tidy DataFrame."""
    rows: List[Dict[str, Any]] = []
    for txn in snapshot.recent_transactions:
        rows.append(
            {
                "Date": str(txn.get("date") or ""),
                "Type": str(txn.get("type") or "").capitalize(),
                "Ticker": str(txn.get("ticker") or "") or "-",
                "Qty": txn.get("quantity"),
                "Price": txn.get("price"),
                "Amount": txn.get("amount"),
                "Account": str(txn.get("account_id") or ""),
                "Note": str(txn.get("note") or ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# HTML table helpers (replaces go.Table to avoid Plotly fill_color bugs)
# ---------------------------------------------------------------------

_TABLE_CSS = """
<style>
.fincent-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: sans-serif;
}
.fincent-table thead tr th {
    background-color: #1f2937;
    color: #ffffff;
    font-weight: 600;
    padding: 6px 10px;
    vertical-align: middle;
    text-align: left;
    white-space: nowrap;
}
.fincent-table thead tr th.right {
    text-align: right;
}
.fincent-table tbody tr:nth-child(odd)  { background-color: #f9fafb; }
.fincent-table tbody tr:nth-child(even) { background-color: #ffffff; }
.fincent-table tbody tr td {
    padding: 5px 10px;
    color: #111827;
    vertical-align: middle;
}
.fincent-table tbody tr td.right { text-align: right; }
</style>
"""
def _inject_table_css() -> None:
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)


def _html_table(
    headers: List[str],
    rows: List[List[str]],
    right_align_cols: Set[int] | None = None,
) -> str:
    """Render *headers* + *rows* as a styled HTML table string."""
    right_align_cols = right_align_cols or set()

    th_cells = "".join(
        f'<th class="{"right" if i in right_align_cols else ""}">{h}</th>'
        for i, h in enumerate(headers)
    )
    body_rows = ""
    for row in rows:
        td_cells = "".join(
            f'<td class="{"right" if i in right_align_cols else ""}">'
            f'{html.escape(str(v))}</td>'
            for i, v in enumerate(row)
        )
        body_rows += f"<tr>{td_cells}</tr>"

    return (
        f'<table class="fincent-table">'
        f"<thead><tr>{th_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        f"</table>"
    )


# ---------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------


def _render_accounts_table(df: pd.DataFrame) -> None:
    """Render accounts sorted by balance as an HTML table."""
    _inject_table_css()
    rows = [
        [row["Account"], row["Type"], row["Broker"], _fmt_currency(row["Balance"])]
        for _, row in df.iterrows()
    ]
    st.markdown(
        _html_table(
            headers=["Account", "Type", "Broker", "Balance"],
            rows=rows,
            right_align_cols={3},
        ),
        unsafe_allow_html=True,
    )


def _allocation_pie_figure(df: pd.DataFrame) -> go.Figure:
    """Stocks / bonds / cash allocation pie chart."""
    fig = px.pie(
        df,
        names="Asset class",
        values="Balance",
        color="Asset class",
        color_discrete_map={k.capitalize(): v for k, v in _ALLOCATION_COLORS.items()},
        hole=0.45,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
        sort=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        height=225,
    )
    return fig


def _render_transactions_table(df: pd.DataFrame) -> None:
    """Render the ten most recent transactions as an HTML table."""
    _inject_table_css()

    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        try:
            return f"{float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)

    rows = [
        [
            row["Date"],
            row["Type"],
            row["Ticker"],
            _fmt(row["Qty"]),
            _fmt(row["Price"]),
            _fmt(row["Amount"]),
            row["Account"],
        ]
        for _, row in df.iterrows()
    ]
    st.markdown(
        _html_table(
            headers=["Date", "Type", "Ticker", "Qty", "Price", "Amount", "Account"],
            rows=rows,
            right_align_cols={3, 4, 5},
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Public: render the full right-hand panel
# ---------------------------------------------------------------------


def render_portfolio_panel(snapshot: PortfolioSnapshot) -> None:
    """Render the static portfolio graphics at the top of the Portfolio tab.

    Layout (full page width):

    * Row 1 -- total-balance metric.
    * Row 2 -- accounts table (2/3) + asset-allocation pie (1/3).
    * Row 3 -- ten most recent transactions (full width).
    """
    st.subheader("Portfolio overview")
    total = _fmt_currency(snapshot.total_balance)
    st.metric(label="Total balance", value=total, help=f"As of {snapshot.as_of or '—'}")

    accounts_col, allocation_col = st.columns([2, 1], gap="large")

    with accounts_col:
        st.markdown("**Accounts (by balance)**")
        accounts_df = _accounts_dataframe(snapshot)
        if accounts_df.empty:
            st.info("No accounts on file.")
        else:
            _render_accounts_table(accounts_df)

    with allocation_col:
        st.markdown("**Asset allocation**")
        allocation_df = _allocation_dataframe(snapshot)
        if allocation_df.empty:
            st.info("No balances to allocate.")
        else:
            st.plotly_chart(
                _allocation_pie_figure(allocation_df),
                width="stretch",
                config={"displayModeBar": False},
            )

    st.markdown("**10 most recent transactions**")
    transactions_df = _transactions_dataframe(snapshot)
    if transactions_df.empty:
        st.info("No transactions on file.")
    else:
        txn_col, _ = st.columns([2, 1], gap="large")
        with txn_col:
            _render_transactions_table(transactions_df)
