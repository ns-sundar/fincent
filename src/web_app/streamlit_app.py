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

* **QnA**       -- routes questions through the central planner.
                   Generic non-financial chit-chat is answered by
                   the central agent; generic financial questions
                   route to the Q&A agent; anything that touches
                   the user's own portfolio routes to the Portfolio
                   agent.
* **Portfolio** -- same central-planner routing as QnA, plus a
                   right-hand graphics panel rendering plotly
                   charts / tables for the user's static portfolio
                   snapshot. This tab is intended for portfolio
                   deep-dives but will still correctly fall back to
                   the Q&A agent if the user asks a purely generic
                   financial question here.

Each tab uses its own LangGraph thread (``<session>-qna`` /
``<session>-portfolio``) so the two transcripts are independent and
both survive reloads.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

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
from src.web_app.markdownutil import sanitize_streamlit_markdown
from src.web_app.portfolio_view import render_portfolio_panel

# Hardcoded fallback when the visitor does not supply ``?session_id=``.
DEFAULT_SESSION_ID: str = "default-session"


# Per-tab thread-id suffixes so the QnA and Portfolio transcripts stay
# independent even though they share a base ``?session_id=``.
_QNA_SUFFIX: str = "qna"
_PORTFOLIO_SUFFIX: str = "portfolio"

_QNA_SUGGESTIONS: List[str] = [
    "What can you do for me?",
    "Compare ETFs with mutual funds",
    "How does the New York Stock Exchange (NYSE) work?",
    "What exactly is a tariff and how does it affect prices?",
    "Explain options trading",
    "What is Adjusted Gross Income (AGI) in tax forms?",
]


# CSS overrides applied once per page load. Streamlit's ``st.tabs``
# renders labels inside a BaseWeb tab list; bumping the ``p`` font
# size and colour inside ``data-testid="stMarkdownContainer"`` is the
# documented, version-stable way to restyle the labels without
# reaching into internal class names. Royal blue (#4169E1) keeps the
# two tab headers visually prominent at the top of the page.
_TAB_LABEL_CSS: str = """
<style>
/* ── Tab bar: flush bottom border that active tab "sits on" ──────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #4169E1;
    padding-bottom: 0;
}

/* ── Every tab button: folder-tab shape ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] button {
    border: 2px solid transparent !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 24px !important;
    margin-right: 4px !important;
    margin-bottom: -2px !important;   /* overlap the bar border */
    background-color: #F1F5F9 !important;
    transition: background-color 0.15s ease;
}

/* ── Inactive tab label ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem;
    color: #4169E1;
    font-weight: 600;
}

/* ── Active tab: white background, coloured border, no bottom border ─── */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-color: #4169E1 !important;
    border-bottom-color: transparent !important;
    background-color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]
    [data-testid="stMarkdownContainer"] p {
    color: #1E3A8A;
}

/* ── Hover on inactive tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] button:not([aria-selected="true"]):hover {
    background-color: #E0E7FF !important;
    border-color: #93C5FD !important;
}

/* ── Canned / suggestion question buttons ────────────────────────────── */
div[data-testid="stButton"] > button {
    border: 2px solid #4169E1 !important;
    border-radius: 8px !important;
    color: #4169E1 !important;
    background-color: #EEF2FF !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease, color 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background-color: #4169E1 !important;
    color: #FFFFFF !important;
}
</style>
"""


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

# JSON Schema files are stored alongside the seed portfolio data.
_SCHEMA_DIR = _PROJECT_ROOT / "data" / "default_portfolio"
_ACCOUNTS_SCHEMA: Dict[str, Any] = json.loads(
    (_SCHEMA_DIR / "accounts.schema.json").read_text(encoding="utf-8")
)
_TRANSACTIONS_SCHEMA: Dict[str, Any] = json.loads(
    (_SCHEMA_DIR / "transactions.schema.json").read_text(encoding="utf-8")
)


def _validate_portfolio_json(
    data: Any, schema: Dict[str, Any], label: str
) -> Optional[str]:
    """Return an error string if *data* violates *schema*, else ``None``."""
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        path = " → ".join(str(p) for p in exc.absolute_path) or "(root)"
        return f"{label}: {exc.message} (at {path})"
    except jsonschema.SchemaError as exc:
        return f"{label} schema is invalid: {exc.message}"
    return None
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
        _render_portfolio_upload(cfg)

        st.markdown("---")
        qna_tid = html.escape(_thread_id_for(session_id, _QNA_SUFFIX), quote=True)
        port_tid = html.escape(
            _thread_id_for(session_id, _PORTFOLIO_SUFFIX), quote=True
        )
        st.markdown(
            f"<p style='font-size:0.75rem;color:#6b7280;line-height:1.5;margin:0;'>"
            f"<span style='color:#374151;font-weight:600;'>Backend</span> · "
            f"<span style='word-break:break-all;'>{url_safe}</span><br/>"
            f"API {api_line}<br/>"
            f"<span style='color:#374151;font-weight:600;'>URL session</span> · "
            f"<span style='word-break:break-all;'>{sid_safe}</span><br/>"
            f"<span style='color:#374151;font-weight:600;'>QnA thread</span> · "
            f"<span style='word-break:break-all;'>{qna_tid}</span><br/>"
            f"<span style='color:#374151;font-weight:600;'>Portfolio thread</span> · "
            f"<span style='word-break:break-all;'>{port_tid}</span></p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Tip: open `?session_id=my-id` in the URL for a separate conversation pair."
        )


def _render_portfolio_upload(cfg: Any) -> None:  # noqa: ANN001
    """Sidebar section: upload custom accounts.json + transactions.json."""
    with st.expander("Upload your portfolio", expanded=False):
        data_path = Path(cfg.portfolio.data_path)

        st.markdown(
            "Upload your portfolio as an accounts file and a transactions file, "
            "both in JSON format. Use the sample documents below to understand "
            "the expected schema and content format for each."
        )

        # Download buttons for the read-only sample files.
        sample_accounts = data_path / "sample-accounts.json"
        sample_txns = data_path / "sample-transactions.json"
        col_a, col_b = st.columns(2)
        with col_a:
            if sample_accounts.exists():
                st.download_button(
                    label="sample-accounts.json",
                    data=sample_accounts.read_bytes(),
                    file_name="sample-accounts.json",
                    mime="application/json",
                    key="dl_sample_accounts",
                    use_container_width=True,
                )
            else:
                st.caption("_(sample-accounts.json not yet available)_")
        with col_b:
            if sample_txns.exists():
                st.download_button(
                    label="sample-transactions.json",
                    data=sample_txns.read_bytes(),
                    file_name="sample-transactions.json",
                    mime="application/json",
                    key="dl_sample_txns",
                    use_container_width=True,
                )
            else:
                st.caption("_(sample-transactions.json not yet available)_")

        st.markdown("---")
        acc_file = st.file_uploader(
            "accounts.json",
            type="json",
            key="upload_accounts",
            help="Array of account objects; see data/default_portfolio/accounts.schema.json",
        )
        txn_file = st.file_uploader(
            "transactions.json",
            type="json",
            key="upload_transactions",
            help="Array of transaction objects; see data/default_portfolio/transactions.schema.json",
        )

        if st.button("Apply upload", key="apply_portfolio_upload"):
            if acc_file is None or txn_file is None:
                st.error(
                    "Both accounts.json and transactions.json must be uploaded together. "
                    "Missing: "
                    + ", ".join(
                        n
                        for n, f in [("accounts.json", acc_file), ("transactions.json", txn_file)]
                        if f is None
                    )
                )
                return

            # Parse JSON
            try:
                acc_data = json.loads(acc_file.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"accounts.json is not valid JSON: {exc}")
                return
            try:
                txn_data = json.loads(txn_file.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"transactions.json is not valid JSON: {exc}")
                return

            # Validate against schemas
            acc_err = _validate_portfolio_json(acc_data, _ACCOUNTS_SCHEMA, "accounts.json")
            if acc_err:
                st.error(acc_err)
                return
            txn_err = _validate_portfolio_json(txn_data, _TRANSACTIONS_SCHEMA, "transactions.json")
            if txn_err:
                st.error(txn_err)
                return

            # Write to data_path
            try:
                data_path.mkdir(parents=True, exist_ok=True)
                (data_path / "accounts.json").write_text(
                    json.dumps(acc_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                (data_path / "transactions.json").write_text(
                    json.dumps(txn_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError as exc:
                st.error(f"Failed to write portfolio files: {exc}")
                return

            st.success(
                f"Portfolio updated: {len(acc_data)} account(s), "
                f"{len(txn_data)} transaction(s). "
                "Switch to the Portfolio tab to see the changes."
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


def _agents_involved(
    plan_payload: Dict, agent_payloads: List[Dict]
) -> List[str]:
    """Compute the list of agents that actually contributed to the answer.

    The single source of truth is the ``agent_responses`` list the
    backend returned: each entry is produced by exactly one agent that
    ran for this turn, so if the Portfolio agent alone answered, only
    ``"portfolio"`` appears here. When no specialists ran (the planner
    set ``handled_by_central: true``) we fall back to reporting the
    central agent. Order is preserved, duplicates are removed.
    """
    seen: set[str] = set()
    agents: List[str] = []
    for ar in agent_payloads:
        name = str(ar.get("agent", "")).strip()
        if name and name not in seen:
            seen.add(name)
            agents.append(name)
    if agents:
        return agents
    if bool(plan_payload.get("handled_by_central")):
        return ["central"]
    return []


def _tools_called(agent_payloads: List[Dict]) -> List[str]:
    """Collect actually invoked MCP tools from each agent response.

    The Portfolio agent sets ``metadata.tools_invoked`` (names the ReAct
    loop executed, in order). Legacy payloads only had ``tool_names``
    (the full bound tool list), which is not the same as calls; we
    ignore ``tool_names`` here so the expander label "Tools called"
    stays truthful.
    """
    tools: List[str] = []
    for ar in agent_payloads:
        meta = ar.get("metadata") or {}
        raw = meta.get("tools_invoked")
        if not isinstance(raw, list):
            continue
        for name in raw:
            if isinstance(name, str) and name:
                tools.append(name)
    return tools


def _render_plan_expander(plan_payload: Dict, agent_payloads: List[Dict]) -> None:
    """Show routing details for the most recent assistant turn.

    The expander is ordered for quick diagnosis:
      1. Which agent(s) actually produced the user-visible answer.
      2. Which tools those agents called (currently the Portfolio
         agent's MCP tool list).
      3. The raw routing plan the central planner emitted.
      4. Each specialist's individual reply, for full traceability.
    """
    agents = _agents_involved(plan_payload, agent_payloads)
    tools = _tools_called(agent_payloads)

    with st.expander("Under the hood", expanded=False):
        agents_line = ", ".join(agents) if agents else "(none)"
        tools_line = ", ".join(tools) if tools else "(none)"
        st.markdown(f"**Agents involved:** {agents_line}")
        st.markdown(f"**Tools called:** {tools_line}")

        # RAG status — shown only when the QnA agent ran.
        for ar in agent_payloads:
            if ar.get("agent") == "qna":
                meta = ar.get("metadata") or {}
                if "rag_used" in meta:
                    rag_used: bool = bool(meta["rag_used"])
                    chunk_count: int = int(meta.get("rag_chunk_count", 0))
                    rag_label = (
                        f"yes ({chunk_count} chunk{'s' if chunk_count != 1 else ''} retrieved)"
                        if rag_used
                        else "no"
                    )
                    st.markdown(f"**RAG invoked:** {rag_label}")
                break

        st.markdown("**Plan**")
        st.json(plan_payload)

        st.markdown("**Agent responses**")
        for ar in agent_payloads:
            body = sanitize_streamlit_markdown(str(ar.get("content", "") or ""))
            st.markdown(f"- **{ar.get('agent', '?')}** — {body}")


def _render_history(history: List[Dict]) -> None:
    """Replay past turns in chronological order (oldest first)."""
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(sanitize_streamlit_markdown(turn["content"] or ""))
            if turn["role"] == "assistant" and "plan" in turn:
                _render_plan_expander(turn["plan"], turn.get("agent_responses", []))


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
        st.markdown(sanitize_streamlit_markdown(user_input))

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

        placeholder.markdown(
            sanitize_streamlit_markdown(response.answer or "(no answer produced)")
        )

    st.session_state[history_key].append({"role": "user", "content": user_input})
    st.session_state[history_key].append(
        {
            "role": "assistant",
            "content": response.answer or "",
            "plan": response.plan.model_dump(mode="json"),
            "agent_responses": [ar.model_dump(mode="json") for ar in response.agent_responses],
        }
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
    suggestions: Optional[List[str]] = None,
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

    # Pop any pending preset before rendering suggestions so the buttons
    # disappear immediately on the same rerun that processes the click.
    preset_key = f"preset_{suffix}"
    preset: Optional[str] = st.session_state.pop(preset_key, None)

    if suggestions and preset is None:
        st.caption("Ask any general finance question. Try any of these to get started.")
        for i in range(0, len(suggestions), 2):
            pair = suggestions[i : i + 2]
            cols = st.columns(len(pair))
            for j, (col, question) in enumerate(zip(cols, pair)):
                if col.button(question, key=f"suggest_{suffix}_{i + j}", use_container_width=True):
                    st.session_state[preset_key] = question
                    st.rerun()

    _render_history(history)

    user_input = preset or st.chat_input(placeholder, key=input_key)
    if not user_input:
        return
    _run_chat_turn(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=suffix,
        user_input=user_input,
        intent_hint=intent_hint,
    )
    st.rerun()


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
        suggestions=_QNA_SUGGESTIONS,
    )


def _render_portfolio_tab(api_base_url: str, session_id: str) -> None:
    """Portfolio tab: graphics on top, chat below (same flow as QnA).

    Laying the graphics above the chat lets ``st.chat_input`` live at
    the tab's top level, which means Streamlit pins it to the bottom
    of the page -- exactly the same flow as the single-pane QnA tab.
    No columns, no fixed-height containers, no ``st.rerun()`` dance.

    The tab does NOT pin an ``intent_hint`` any more: the central
    planner decides which specialist should run so generic financial
    questions asked here still reach the right agent, while any
    mention of the user's own holdings routes to the Portfolio agent
    as expected. The user-visible intro intentionally does NOT expose
    this routing detail -- it just describes what the tab is for.
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
        intent_hint=None,
        intro=(
            "Ask about your accounts, balances, asset allocation, or "
            "recent transactions. The graphics above summarize your "
            "current holdings."
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

    st.markdown(_TAB_LABEL_CSS, unsafe_allow_html=True)

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
