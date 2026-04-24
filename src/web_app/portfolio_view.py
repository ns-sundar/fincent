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

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Plotly figure builders
# ---------------------------------------------------------------------


def _accounts_table_figure(df: pd.DataFrame) -> go.Figure:
    """Plotly table of accounts sorted by balance (desc)."""
    balances = [_fmt_currency(v) for v in df["Balance"].tolist()]
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[140, 50, 65, 85],
                header=dict(
                    values=[
                        "<b>Account</b>",
                        "<b>Type</b>",
                        "<b>Broker</b>",
                        "<b>Balance</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=11),
                    align="left",
                    height=26,
                ),
                cells=dict(
                    values=[
                        df["Account"].tolist(),
                        df["Type"].tolist(),
                        df["Broker"].tolist(),
                        balances,
                    ],
                    fill_color=[["#f9fafb", "#ffffff"] * (len(df) // 2 + 1)],
                    align=["left", "left", "left", "right"],
                    font=dict(size=11),
                    height=22,
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=420,
        height=max(90, 30 + 24 * len(df)),
    )
    return fig


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
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        height=300,
    )
    return fig


def _transactions_table_figure(df: pd.DataFrame) -> go.Figure:
    """Plotly table for the ten most recent transactions."""
    def _fmt_num(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        try:
            return f"{float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)

    qty = [_fmt_num(v) for v in df["Qty"].tolist()]
    price = [_fmt_num(v) for v in df["Price"].tolist()]
    amount = [_fmt_num(v) for v in df["Amount"].tolist()]
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[60, 45, 40, 35, 50, 65, 75],
                header=dict(
                    values=[
                        "<b>Date</b>",
                        "<b>Type</b>",
                        "<b>Ticker</b>",
                        "<b>Qty</b>",
                        "<b>Price</b>",
                        "<b>Amount</b>",
                        "<b>Account</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=11),
                    align="left",
                    height=24,
                ),
                cells=dict(
                    values=[
                        df["Date"].tolist(),
                        df["Type"].tolist(),
                        df["Ticker"].tolist(),
                        qty,
                        price,
                        amount,
                        df["Account"].tolist(),
                    ],
                    fill_color=[["#f9fafb", "#ffffff"] * (len(df) // 2 + 1)],
                    align=[
                        "left",
                        "left",
                        "left",
                        "right",
                        "right",
                        "right",
                        "left",
                    ],
                    font=dict(size=10),
                    height=20,
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=520,
        height=max(90, 28 + 22 * len(df)),
    )
    return fig


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
            st.plotly_chart(
                _accounts_table_figure(accounts_df),
                use_container_width=False,
                config={"displayModeBar": False},
            )

    with allocation_col:
        st.markdown("**Asset allocation**")
        allocation_df = _allocation_dataframe(snapshot)
        if allocation_df.empty:
            st.info("No balances to allocate.")
        else:
            st.plotly_chart(
                _allocation_pie_figure(allocation_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown("**10 most recent transactions**")
    transactions_df = _transactions_dataframe(snapshot)
    if transactions_df.empty:
        st.info("No transactions on file.")
    else:
        st.plotly_chart(
            _transactions_table_figure(transactions_df),
            use_container_width=False,
            config={"displayModeBar": False},
        )
