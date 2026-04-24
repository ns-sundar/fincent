"""Streamlit chat UI for Fincent.

Run as a separate process:

    streamlit run src/web_app/streamlit_app.py

The UI talks to the FastAPI server over HTTP (URL configured via
``ui.api_base_url`` in ``config.yaml`` or the
``FINCENT__UI__API_BASE_URL`` env var).

The conversation is identified by a ``session_id`` query parameter
(``?session_id=<id>``). If none is present, the UI falls back to
``DEFAULT_SESSION_ID``.

The UI is organised as two tabs:

* **QnA**       -- routes questions through the central planner
                   (typically into the Q&A agent).
* **Portfolio** -- pins ``intent_hint="portfolio"`` so every message
                   goes straight to the Portfolio agent. A right-hand
                   sidebar renders plotly graphics for the user's
                   static portfolio snapshot.

Each tab uses its own LangGraph thread (``<session>-qna`` /
``<session>-portfolio``) so the two transcripts are independent and
both survive reloads.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is on sys.path regardless of how Streamlit was
# launched (CLI, Docker, HuggingFace Spaces, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.agents.portfolio import load_portfolio
from src.core.config import get_config
from src.web_app.api_client import (
    FincentApiError,
    get_history,
    health,
    query_fincent,
    rag_status,
    reset_thread,
)
from src.web_app.portfolio_view import render_portfolio_panel

# Hardcoded fallback when the visitor does not supply ``?session_id=``.
DEFAULT_SESSION_ID: str = "default-session"


# Per-tab thread-id suffixes so the QnA and Portfolio transcripts stay
# independent even though they share a base ``?session_id=``.
_QNA_SUFFIX: str = "qna"
_PORTFOLIO_SUFFIX: str = "portfolio"


# ---------------------------------------------------------------------
# Session / history helpers
# ---------------------------------------------------------------------


def _resolve_api_base_url() -> str:
    """Pick the API base URL from env/config, with env taking precedence."""
    env = os.environ.get("FINCENT__UI__API_BASE_URL")
    if env:
        return env
    return get_config().ui.api_base_url


def _resolve_session_id() -> str:
    """Read ``?session_id=`` from the URL, falling back to the default."""
    try:
        qp = st.query_params
    except AttributeError:
        qp = st.experimental_get_query_params()
    raw = qp.get("session_id") if qp else None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    candidate = (raw or "").strip()
    return candidate or DEFAULT_SESSION_ID


def _thread_id_for(session_id: str, suffix: str) -> str:
    """Compose a per-tab LangGraph thread id."""
    return f"{session_id}-{suffix}"


def _history_state_key(suffix: str) -> str:
    return f"history_{suffix}"


def _session_state_key(suffix: str) -> str:
    return f"session_id_{suffix}"


def _rehydrate_tab(api_base_url: str, session_id: str, suffix: str) -> None:
    """Populate ``st.session_state[history_<suffix>]`` from the backend.

    Runs once per tab the first time it is viewed for the active
    ``session_id``, or whenever the visitor switches to a different
    ``session_id`` via the URL.
    """
    active = st.session_state.get(_session_state_key(suffix))
    if _history_state_key(suffix) in st.session_state and active == session_id:
        return
    st.session_state[_session_state_key(suffix)] = session_id
    try:
        remote = get_history(api_base_url, _thread_id_for(session_id, suffix))
    except FincentApiError:
        remote = []
    st.session_state[_history_state_key(suffix)] = remote


# ---------------------------------------------------------------------
# Sidebar / banners
# ---------------------------------------------------------------------


def _render_sidebar(api_base_url: str, session_id: str) -> None:
    """Render the app-wide sidebar: reset buttons, backend status."""
    cfg = get_config()
    ok = health(api_base_url)
    api_line = "reachable" if ok else "unreachable"
    url_safe = html.escape(api_base_url, quote=True)
    sid_safe = html.escape(session_id, quote=True)

    with st.sidebar:
        st.header(cfg.app.name)

        if st.button("Clear QnA conversation", key="reset_qna_btn"):
            try:
                reset_thread(api_base_url, _thread_id_for(session_id, _QNA_SUFFIX))
            except FincentApiError as exc:
                st.error(f"Reset failed: {exc}")
            else:
                st.session_state[_history_state_key(_QNA_SUFFIX)] = []
                st.rerun()

        if st.button("Clear Portfolio conversation", key="reset_portfolio_btn"):
            try:
                reset_thread(
                    api_base_url, _thread_id_for(session_id, _PORTFOLIO_SUFFIX)
                )
            except FincentApiError as exc:
                st.error(f"Reset failed: {exc}")
            else:
                st.session_state[_history_state_key(_PORTFOLIO_SUFFIX)] = []
                st.rerun()

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.75rem;color:#6b7280;line-height:1.5;margin:0;'>"
            f"<span style='color:#374151;font-weight:600;'>Backend</span> · "
            f"<span style='word-break:break-all;'>{url_safe}</span><br/>"
            f"API {api_line}<br/>"
            f"<span style='color:#374151;font-weight:600;'>Session</span> · "
            f"<span style='word-break:break-all;'>{sid_safe}</span></p>",
            unsafe_allow_html=True,
        )


def _render_rag_banner(api_base_url: str) -> None:
    """Surface RAG ingestion failures without blocking the chat UI."""
    status = rag_status(api_base_url)
    state = str(status.get("state", "unknown")).lower()
    detail = str(status.get("detail") or "")
    error = str(status.get("error") or "")

    if state in {"ready", "skipped", "disabled"}:
        return

    if state in {"pending", "ingesting"}:
        st.info(
            f"Knowledge base ingestion is in progress ({state}). "
            "Answers may be less grounded until it completes."
            + (f"\n\n{detail}" if detail else "")
        )
        return

    msg = (
        "Knowledge base is unavailable; answers will fall back to the "
        "model's general knowledge without retrieval context."
    )
    if detail:
        msg += f"\n\nDetails: {detail}"
    if error:
        msg += f"\n\nError: {error}"
    st.error(msg)


# ---------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------


def _render_plan_expander(plan_payload: Dict, agent_payloads: List[Dict]) -> None:
    """Show routing details for the most recent assistant turn."""
    with st.expander("Routing details", expanded=False):
        st.json(plan_payload)
        st.markdown("**Agent responses**")
        for ar in agent_payloads:
            st.markdown(f"- **{ar.get('agent', '?')}** -- {ar.get('content', '')}")


def _render_history(history: List[Dict[str, str]]) -> None:
    """Replay past turns in chronological order (oldest first)."""
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


def _run_chat_turn(
    *,
    api_base_url: str,
    session_id: str,
    suffix: str,
    user_input: str,
    intent_hint: Optional[str],
) -> None:
    """Render one new turn at the bottom of the transcript.

    Renders the user bubble, a ``thinking...`` placeholder assistant
    bubble, and finally the real answer once the backend replies. Both
    bubbles render at the current Streamlit cursor -- i.e. directly
    below the previously-rendered history -- so the newest question
    sits at the bottom and its answer appears just below it.
    """
    history_key = _history_state_key(suffix)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_thinking..._")
        try:
            response = query_fincent(
                api_base_url,
                user_input,
                session_id=_thread_id_for(session_id, suffix),
                intent_hint=intent_hint,
            )
        except FincentApiError as exc:
            placeholder.error(str(exc))
            return

        placeholder.markdown(response.answer or "(no answer produced)")
        _render_plan_expander(
            response.plan.model_dump(),
            [ar.model_dump() for ar in response.agent_responses],
        )

    st.session_state[history_key].append({"role": "user", "content": user_input})
    st.session_state[history_key].append(
        {"role": "assistant", "content": response.answer or ""}
    )


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------


def _render_chat_tab(
    *,
    api_base_url: str,
    session_id: str,
    suffix: str,
    placeholder: str,
    input_key: str,
    intent_hint: Optional[str],
    intro: Optional[str] = None,
) -> None:
    """Render a chat tab: history on top, ``st.chat_input`` pinned at bottom.

    ``st.chat_input`` is called at the tab's top level (no columns, no
    fixed-height containers), which is what tells Streamlit to pin it
    to the bottom of the page -- the standard chat layout.
    """
    _rehydrate_tab(api_base_url, session_id, suffix)
    history = st.session_state[_history_state_key(suffix)]

    if intro:
        st.markdown(intro)

    _render_history(history)

    user_input = st.chat_input(placeholder, key=input_key)
    if not user_input:
        return
    _run_chat_turn(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=suffix,
        user_input=user_input,
        intent_hint=intent_hint,
    )


def _render_qna_tab(api_base_url: str, session_id: str) -> None:
    """QnA tab: full-width chat routed through the central planner."""
    _render_rag_banner(api_base_url)
    _render_chat_tab(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=_QNA_SUFFIX,
        placeholder="Ask a general financial question...",
        input_key="qna_chat_input",
        intent_hint=None,
    )


def _render_portfolio_tab(api_base_url: str, session_id: str) -> None:
    """Portfolio tab: graphics on top, chat below (same flow as QnA).

    Laying the graphics above the chat lets ``st.chat_input`` live at
    the tab's top level, which means Streamlit pins it to the bottom
    of the page -- exactly the same flow as the single-pane QnA tab.
    No columns, no fixed-height containers, no ``st.rerun()`` dance.
    """
    try:
        snapshot = load_portfolio()
    except Exception as exc:  # noqa: BLE001 -- always render something
        st.error(f"Could not load portfolio data: {exc}")
        snapshot = None

    if snapshot is not None:
        render_portfolio_panel(snapshot)
        st.divider()

    _render_chat_tab(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=_PORTFOLIO_SUFFIX,
        placeholder="Ask about your portfolio...",
        input_key="portfolio_chat_input",
        intent_hint="portfolio",
        intro=(
            "Ask the Portfolio agent about your accounts, balances, "
            "asset allocation, or recent transactions. The graphics "
            "above show the same snapshot the agent sees."
        ),
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main() -> None:
    """Streamlit entry point."""
    cfg = get_config()
    st.set_page_config(
        page_title=cfg.ui.page_title,
        page_icon=cfg.ui.page_icon,
        layout="wide",
    )

    api_base_url = _resolve_api_base_url()
    session_id = _resolve_session_id()

    _render_sidebar(api_base_url, session_id)

    st.title(cfg.ui.page_title)
    subtitle = (cfg.app.description or "").strip()
    if subtitle:
        st.caption(subtitle)

    qna_tab, portfolio_tab = st.tabs(["QnA", "Portfolio"])
    with qna_tab:
        _render_qna_tab(api_base_url, session_id)
    with portfolio_tab:
        _render_portfolio_tab(api_base_url, session_id)


if __name__ == "__main__":
    main()
