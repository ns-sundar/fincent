"""Streamlit chat UI for Fincent.

Run as a separate process:

    streamlit run src/web_app/streamlit_app.py

The UI talks to the FastAPI server over HTTP (URL configured via
``ui.api_base_url`` in ``config.yaml`` or the
``FINCENT__UI__API_BASE_URL`` env var).

The conversation is identified by a ``session_id`` query parameter
(``?session_id=<id>``). If none is present, the UI falls back to
``DEFAULT_SESSION_ID``. The backend uses the session id as a LangGraph
thread id so the transcript survives reloads and tab switches.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure the project root is on sys.path regardless of how Streamlit was
# launched (CLI, Docker, HuggingFace Spaces, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.core.config import get_config
from src.web_app.api_client import (
    FincentApiError,
    get_history,
    health,
    query_fincent,
    reset_thread,
)

# Hardcoded fallback when the visitor does not supply ``?session_id=``.
DEFAULT_SESSION_ID: str = "default-session"


# ---------------------------------------------------------------------
# Helpers
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
        # Streamlit < 1.30 compatibility (unlikely given requirements).
        qp = st.experimental_get_query_params()
    raw = qp.get("session_id") if qp else None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    candidate = (raw or "").strip()
    return candidate or DEFAULT_SESSION_ID


def _rehydrate_if_needed(api_base_url: str, session_id: str) -> None:
    """Populate ``st.session_state.history`` from the backend when needed.

    Runs when the app is loaded for the first time, or when the
    visitor switches to a different ``session_id`` via the URL.
    """
    active = st.session_state.get("session_id")
    if "history" in st.session_state and active == session_id:
        return
    st.session_state["session_id"] = session_id
    try:
        remote = get_history(api_base_url, session_id)
    except FincentApiError:
        # Treat a failed fetch as "no history" -- the UI must still load.
        remote = []
    st.session_state["history"] = remote


def _render_sidebar(api_base_url: str, session_id: str) -> None:
    """Render sidebar: primary actions at top, backend status as a quiet footer."""
    cfg = get_config()
    ok = health(api_base_url)
    api_line = "reachable" if ok else "unreachable"
    url_safe = html.escape(api_base_url, quote=True)
    sid_safe = html.escape(session_id, quote=True)

    with st.sidebar:
        st.header(cfg.app.name)
        if st.button("Clear conversation"):
            # Clear the server-side checkpoint BEFORE clearing the
            # local history so a failure doesn't desync the two.
            try:
                reset_thread(api_base_url, session_id)
            except FincentApiError as exc:
                st.error(f"Reset failed: {exc}")
                return
            st.session_state["history"] = []
            st.rerun()

        st.markdown("---")
        # Muted footer (less intrusive than success/error callouts).
        st.markdown(
            f"<p style='font-size:0.75rem;color:#6b7280;line-height:1.5;margin:0;'>"
            f"<span style='color:#374151;font-weight:600;'>Backend</span> · "
            f"<span style='word-break:break-all;'>{url_safe}</span><br/>"
            f"API {api_line}<br/>"
            f"<span style='color:#374151;font-weight:600;'>Session</span> · "
            f"<span style='word-break:break-all;'>{sid_safe}</span></p>",
            unsafe_allow_html=True,
        )


def _render_history() -> None:
    """Replay previous chat turns."""
    for turn in st.session_state.get("history", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


def _render_plan_expander(plan_payload: Dict, agent_payloads: List[Dict]) -> None:
    """Show routing details for the most recent assistant turn."""
    with st.expander("Routing details", expanded=False):
        st.json(plan_payload)
        st.markdown("**Agent responses**")
        for ar in agent_payloads:
            st.markdown(f"- **{ar.get('agent', '?')}** -- {ar.get('content', '')}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main() -> None:
    """Streamlit entry point."""
    cfg = get_config()
    st.set_page_config(
        page_title=cfg.ui.page_title,
        page_icon=cfg.ui.page_icon,
        layout="centered",
    )

    api_base_url = _resolve_api_base_url()
    session_id = _resolve_session_id()

    _rehydrate_if_needed(api_base_url, session_id)
    _render_sidebar(api_base_url, session_id)

    st.title(cfg.ui.page_title)
    subtitle = (cfg.app.description or "").strip()
    if subtitle:
        st.caption(subtitle)

    _render_history()

    user_input = st.chat_input("Ask a financial question...")
    if not user_input:
        return

    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_thinking..._")
        try:
            response = query_fincent(
                api_base_url,
                user_input,
                session_id=session_id,
            )
        except FincentApiError as exc:
            placeholder.error(str(exc))
            return

        placeholder.markdown(response.answer or "(no answer produced)")
        _render_plan_expander(
            response.plan.model_dump(),
            [ar.model_dump() for ar in response.agent_responses],
        )

    st.session_state["history"].append(
        {"role": "assistant", "content": response.answer or ""}
    )


if __name__ == "__main__":
    main()
