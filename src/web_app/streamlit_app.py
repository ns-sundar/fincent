"""Streamlit chat UI for Fincent.

Run as a separate process:

    streamlit run src/web_app/streamlit_app.py

The UI talks to the FastAPI server over HTTP (URL configured via
``ui.api_base_url`` in ``config.yaml`` or the
``FINCENT__UI__API_BASE_URL`` env var).
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
from src.web_app.api_client import FincentApiError, health, query_fincent


def _resolve_api_base_url() -> str:
    """Pick the API base URL from env/config, with env taking precedence."""
    env = os.environ.get("FINCENT__UI__API_BASE_URL")
    if env:
        return env
    return get_config().ui.api_base_url


def _init_session_state() -> None:
    """Ensure per-session containers exist in ``st.session_state``."""
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, str]] = []


def _render_sidebar(api_base_url: str) -> None:
    """Render sidebar: primary actions at top, backend status as a quiet footer."""
    cfg = get_config()
    ok = health(api_base_url)
    api_line = "reachable" if ok else "unreachable"
    url_safe = html.escape(api_base_url, quote=True)
    with st.sidebar:
        st.header(cfg.app.name)
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
        # Muted footer (less intrusive than success/error callouts).
        st.markdown(
            f"<p style='font-size:0.75rem;color:#6b7280;line-height:1.5;margin:0;'>"
            f"<span style='color:#374151;font-weight:600;'>Backend</span> · "
            f"<span style='word-break:break-all;'>{url_safe}</span><br/>"
            f"API {api_line}</p>",
            unsafe_allow_html=True,
        )


def _render_history() -> None:
    """Replay previous chat turns."""
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


def _render_plan_expander(plan_payload: Dict, agent_payloads: List[Dict]) -> None:
    """Show routing details for the most recent assistant turn."""
    with st.expander("Routing details", expanded=False):
        st.json(plan_payload)
        st.markdown("**Agent responses**")
        for ar in agent_payloads:
            st.markdown(f"- **{ar.get('agent', '?')}** -- {ar.get('content', '')}")


def main() -> None:
    """Streamlit entry point."""
    cfg = get_config()
    st.set_page_config(
        page_title=cfg.ui.page_title,
        page_icon=cfg.ui.page_icon,
        layout="centered",
    )
    api_base_url = _resolve_api_base_url()

    _init_session_state()
    _render_sidebar(api_base_url)
    st.title(cfg.ui.page_title)
    subtitle = (cfg.app.description or "").strip()
    if subtitle:
        st.caption(subtitle)

    _render_history()

    user_input = st.chat_input("Ask a financial question...")
    if not user_input:
        return

    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_thinking..._")
        try:
            response = query_fincent(api_base_url, user_input)
        except FincentApiError as exc:
            placeholder.error(str(exc))
            return

        placeholder.markdown(response.answer or "(no answer produced)")
        _render_plan_expander(
            response.plan.model_dump(),
            [ar.model_dump() for ar in response.agent_responses],
        )

    st.session_state.history.append(
        {"role": "assistant", "content": response.answer or ""}
    )


if __name__ == "__main__":
    main()
