"""Streamlit chat UI for Fincent.

Run as a separate process:

    streamlit run src/web_app/streamlit_app.py

The UI talks to the FastAPI server over HTTP (URL configured via
``ui.api_base_url`` in ``config.yaml`` or the
``FINCENT__UI__API_BASE_URL`` env var).

The conversation is identified by a ``session_id`` query parameter
(``?session_id=<id>``). If none is present, the UI falls back to
``DEFAULT_SESSION_ID``.

The UI is organised as three tabs:

* **QnA**       -- routes questions through the central planner.
                   App identity/features, chit-chat, and out-of-scope
                   requests are answered by the central agent; generic
                   financial questions route to the Q&A agent; anything
                   that touches the user's own portfolio routes to the
                   Portfolio agent.
* **Portfolio** -- same central-planner routing as QnA, plus a
                   right-hand graphics panel rendering plotly
                   charts / tables for the user's static portfolio
                   snapshot. This tab is intended for portfolio
                   deep-dives but will still correctly fall back to
                   the Q&A agent if the user asks a purely generic
                   financial question here.
* **Market Research** -- pinned to the Market Research agent for
                   company, security, filing, risk, and investment
                   theme analysis.
* **Goal Planning** -- pinned to the Goal Planning agent for retirement,
                   college, home purchase, vacation, and stress-test
                   scenarios grounded in the user's portfolio.

Each tab uses its own LangGraph thread (``<session>-qna`` /
``<session>-portfolio`` / ``<session>-market-research`` /
``<session>-goal-planning``) so transcripts are independent and survive
reloads.
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
    get_model,
    health,
    query_fincent,
    rag_status,
    refresh_portfolio,
    reset_thread,
    set_model,
)
from src.web_app.markdownutil import sanitize_streamlit_markdown
from src.web_app.portfolio_view import render_portfolio_panel

# Hardcoded fallback when the visitor does not supply ``?session_id=``.
DEFAULT_SESSION_ID: str = "default-session"


# Per-tab thread-id suffixes so transcripts stay independent even though
# they share a base ``?session_id=``.
_QNA_SUFFIX: str = "qna"
_PORTFOLIO_SUFFIX: str = "portfolio"
_MARKET_RESEARCH_SUFFIX: str = "market-research"
_GOAL_PLANNING_SUFFIX: str = "goal-planning"

# Available chat models offered in the sidebar selector.
_AVAILABLE_MODELS: List[str] = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-4o-mini"]
_DEFAULT_MODEL: str = "gpt-5.4-mini"
_MODEL_STATE_KEY: str = "selected_model"

_QNA_SUGGESTIONS: List[str] = [
    "What can you do for me?",
    "Compare ETFs with mutual funds",
    "How does the New York Stock Exchange (NYSE) work?",
    "What exactly is a tariff and how does it affect prices?",
    "Explain options trading",
    "What is Adjusted Gross Income (AGI) in tax forms?",
]

_MARKET_RESEARCH_SUGGESTIONS: List[str] = [
    "Is Nvidia a good investment?",
    "Compare Procter and Gamble with Unilever",
    "What are the risks of investing in Tesla?",
    "What is the best AI investment today?",
]

_GOAL_PLANNING_SUGGESTIONS: List[str] = [
    "I want to retire at 60 with $8,000/month in today's dollars. Am I on track?",
    "I want to buy a home in 2 years. Is my down payment too exposed to stocks?",
    "My kid starts college in 10 years. Is my 529 on target?",
    "Can I afford a $12,000 vacation next summer?",
    "If my portfolio drops 25%, how many extra years might I need to work?",
]


# CSS overrides applied once per page load. Streamlit's ``st.tabs``
# renders labels inside a BaseWeb tab list; bumping the ``p`` font
# size and colour inside ``data-testid="stMarkdownContainer"`` is the
# documented, version-stable way to restyle the labels without
# reaching into internal class names. Royal blue (#4169E1) keeps the
# tab headers visually prominent at the top of the page.
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

        if st.button("Clear Market Research conversation", key="reset_market_research_btn"):
            try:
                reset_thread(
                    api_base_url,
                    _thread_id_for(session_id, _MARKET_RESEARCH_SUFFIX),
                )
            except FincentApiError as exc:
                st.error(f"Reset failed: {exc}")
            else:
                st.session_state[_history_state_key(_MARKET_RESEARCH_SUFFIX)] = []
                st.rerun()

        if st.button("Clear Goal Planning conversation", key="reset_goal_planning_btn"):
            try:
                reset_thread(
                    api_base_url,
                    _thread_id_for(session_id, _GOAL_PLANNING_SUFFIX),
                )
            except FincentApiError as exc:
                st.error(f"Reset failed: {exc}")
            else:
                st.session_state[_history_state_key(_GOAL_PLANNING_SUFFIX)] = []
                st.rerun()

        st.markdown("---")
        _render_portfolio_upload(cfg, api_base_url)

        st.markdown("---")
        _render_model_selector(api_base_url)

        st.markdown("---")
        qna_tid = html.escape(_thread_id_for(session_id, _QNA_SUFFIX), quote=True)
        port_tid = html.escape(
            _thread_id_for(session_id, _PORTFOLIO_SUFFIX), quote=True
        )
        market_tid = html.escape(
            _thread_id_for(session_id, _MARKET_RESEARCH_SUFFIX), quote=True
        )
        goal_tid = html.escape(
            _thread_id_for(session_id, _GOAL_PLANNING_SUFFIX), quote=True
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
            f"<span style='word-break:break-all;'>{port_tid}</span><br/>"
            f"<span style='color:#374151;font-weight:600;'>Market Research thread</span> · "
            f"<span style='word-break:break-all;'>{market_tid}</span><br/>"
            f"<span style='color:#374151;font-weight:600;'>Goal Planning thread</span> · "
            f"<span style='word-break:break-all;'>{goal_tid}</span></p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Tip: open `?session_id=my-id` in the URL for a separate conversation set."
        )


def _render_model_selector(api_base_url: str) -> None:
    """Sidebar widget: choose the OpenAI chat model used by all agents."""
    # Seed session state from the backend on first render.
    if _MODEL_STATE_KEY not in st.session_state:
        backend_model = get_model(api_base_url)
        st.session_state[_MODEL_STATE_KEY] = (
            backend_model if backend_model in _AVAILABLE_MODELS else _DEFAULT_MODEL
        )

    def _on_model_change() -> None:
        chosen = st.session_state[_MODEL_STATE_KEY]
        try:
            set_model(api_base_url, chosen)
        except FincentApiError as exc:
            st.warning(f"Model switch failed: {exc}")

    st.selectbox(
        "OpenAI model",
        options=_AVAILABLE_MODELS,
        key=_MODEL_STATE_KEY,
        on_change=_on_model_change,
        help="Applies immediately to all subsequent queries — no restart needed.",
    )


def _render_portfolio_upload(cfg: Any, api_base_url: str) -> None:  # noqa: ANN001
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
                    width="stretch",
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
                    width="stretch",
                )
            else:
                st.caption("_(sample-transactions.json not yet available)_")

        st.markdown("---")
        acc_file = st.file_uploader(
            "accounts.json",
            type=["json", "application/json", "text/plain", "text/json"],
            key="upload_accounts",
            help="Array of account objects; see data/default_portfolio/accounts.schema.json",
        )
        txn_file = st.file_uploader(
            "transactions.json",
            type=["json", "application/json", "text/plain", "text/json"],
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

            # Bust the Streamlit-process LRU cache (portfolio graphics).
            load_portfolio(force_refresh=True)

            # Bust the FastAPI-process LRU cache (Portfolio agent).
            try:
                refresh_portfolio(api_base_url)
            except FincentApiError as exc:
                st.warning(f"Portfolio updated on disk but backend cache refresh failed: {exc}")

            st.success(
                f"Portfolio updated: {len(acc_data)} account(s), "
                f"{len(txn_data)} transaction(s)."
            )
            st.rerun()


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
    if not isinstance(plan_payload, dict):
        return []
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


def _tool_limit_notes(agent_payloads: List[Dict]) -> List[str]:
    """Describe any provider-limit tool pruning recorded by specialists."""

    notes: List[str] = []
    for ar in agent_payloads:
        meta = ar.get("metadata") or {}
        dropped = int(meta.get("dropped_tool_count") or 0)
        available = int(meta.get("available_tool_count") or meta.get("tool_count") or 0)
        bound = int(meta.get("tool_count") or 0)
        if dropped <= 0:
            continue
        agent = str(ar.get("agent", "agent") or "agent")
        notes.append(
            f"{agent}: bound {bound} of {available} available tools; "
            f"dropped {dropped} lower-priority tools to stay within provider limits."
        )
    return notes


def _data_sources_footprint_notes(agent_payloads: List[Dict]) -> List[str]:
    """Collect unique FMP / free-data disclaimers from specialist metadata."""

    seen: set[str] = set()
    out: List[str] = []
    for ar in agent_payloads:
        meta = ar.get("metadata") or {}
        raw = meta.get("data_sources_note")
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _agent_error_details(agent_payloads: List[Dict]) -> List[Dict[str, str]]:
    """Collect non-user-facing error diagnostics from specialist metadata."""

    out: List[Dict[str, str]] = []
    for ar in agent_payloads:
        meta = ar.get("metadata") or {}
        if not bool(meta.get("error")):
            continue
        item = {
            "agent": str(ar.get("agent", "?") or "?"),
            "phase": str(meta.get("error_phase", "") or ""),
            "type": str(meta.get("error_type", "Error") or "Error"),
            "message": str(meta.get("error_message", "") or ""),
            "traceback": str(meta.get("error_traceback", "") or ""),
        }
        out.append(item)
    return out


def _render_plan_expander(plan_payload: Dict, agent_payloads: List[Dict]) -> None:
    """Show routing details for the most recent assistant turn.

    The expander is ordered for quick diagnosis:
      1. Which agent(s) actually produced the user-visible answer.
      2. Which tools those agents called (specialists record
         ``metadata.tools_invoked``).
      3. Optional ``st.info`` line(s) when ``metadata.data_sources_note`` explains
         free-tier / non-FMP data (FMP paywall disclaimers).
      4. Non-user-facing error details from ``metadata.error_*``.
      5. The raw routing plan the central planner emitted.
      6. Each specialist's individual reply, for full traceability.
    """
    agents = _agents_involved(plan_payload, agent_payloads)
    tools = _tools_called(agent_payloads)

    with st.expander("Under the hood", expanded=False):
        agents_line = ", ".join(agents) if agents else "(none)"
        tools_line = ", ".join(tools) if tools else "(none)"
        st.markdown(f"**Agents involved:** {agents_line}")
        st.markdown(f"**Tools called:** {tools_line}")

        for note in _tool_limit_notes(agent_payloads):
            st.caption(note)

        for note in _data_sources_footprint_notes(agent_payloads):
            st.info(note)

        for detail in _agent_error_details(agent_payloads):
            title = (
                f"**{detail['agent']} error:** {detail['type']}"
                + (f" during `{detail['phase']}`" if detail["phase"] else "")
            )
            st.error(title)
            if detail["message"]:
                st.code(detail["message"], language="text")
            if detail["traceback"]:
                with st.expander(f"{detail['agent']} traceback", expanded=False):
                    st.code(detail["traceback"], language="text")

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
            if turn["role"] == "assistant" and isinstance(turn.get("plan"), dict):
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
            if getattr(exc, "categories", None):
                answer = f"{exc.detail}\n\nCategories: {', '.join(exc.categories)}"
            else:
                answer = str(exc)
            placeholder.markdown(sanitize_streamlit_markdown(answer))
            st.session_state[history_key].append({"role": "user", "content": user_input})
            st.session_state[history_key].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "agent_responses": [],
                }
            )
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
    suggestions_caption: str = "Ask any general finance question. Try any of these to get started.",
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
        st.caption(suggestions_caption)
        for i in range(0, len(suggestions), 2):
            pair = suggestions[i : i + 2]
            cols = st.columns(len(pair))
            for j, (col, question) in enumerate(zip(cols, pair)):
                if col.button(
                    question,
                    key=f"suggest_{suffix}_{i + j}",
                    width="stretch",
                ):
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


def _render_market_research_tab(api_base_url: str, session_id: str) -> None:
    """Market Research tab: pinned to the Market Research specialist."""

    _render_chat_tab(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=_MARKET_RESEARCH_SUFFIX,
        placeholder="Ask for company, security, or market research...",
        input_key="market_research_chat_input",
        intent_hint="market_research",
        intro=(
            "Ask for company research, investment comparisons, bond or ETF "
            "risk analysis, AI investment themes, or 10-K risk summaries."
        ),
        suggestions=_MARKET_RESEARCH_SUGGESTIONS,
        suggestions_caption=(
            "Ask a market research question. Try any of these to get started."
        ),
    )


def _render_goal_planning_tab(api_base_url: str, session_id: str) -> None:
    """Goal Planning tab: pinned to the Goal Planning specialist."""

    _render_chat_tab(
        api_base_url=api_base_url,
        session_id=session_id,
        suffix=_GOAL_PLANNING_SUFFIX,
        placeholder="Ask about retirement, college, home purchase, or another goal...",
        input_key="goal_planning_chat_input",
        intent_hint="goal_planning",
        intro=(
            "Plan goals against your current portfolio, time horizon, savings "
            "rate, and risk exposure. Results are educational planning scenarios, "
            "not professional financial advice."
        ),
        suggestions=_GOAL_PLANNING_SUGGESTIONS,
        suggestions_caption=(
            "Ask a goal-planning question. Try any of these to get started."
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

    qna_tab, portfolio_tab, market_research_tab, goal_planning_tab = st.tabs(
        ["QnA", "Portfolio", "Market Research", "Goal Planning"]
    )
    with qna_tab:
        _render_qna_tab(api_base_url, session_id)
    with portfolio_tab:
        _render_portfolio_tab(api_base_url, session_id)
    with market_research_tab:
        _render_market_research_tab(api_base_url, session_id)
    with goal_planning_tab:
        _render_goal_planning_tab(api_base_url, session_id)


if __name__ == "__main__":
    main()
