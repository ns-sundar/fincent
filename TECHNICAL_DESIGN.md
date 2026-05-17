# Fincent Technical Design

This document describes the design features adopted to make the deployment robust and resilient. The main one is the decision to separate the UI (StreamLit) and the backend (FastAPI and everything else) into two separate processes, which caused several important consequences. The other features relate to robustness against common disruptive actions for StreamLit, such as HuggingFace Space restart, browser reloads, etc.

## Decoupled Architecture

Fincent uses a decoupled application architecture:

- **Streamlit frontend** for the interactive browser UI.
- **FastAPI + LangServe backend** for typed HTTP endpoints, health checks,
  guardrails, session history, portfolio refresh, and graph invocation.
- **LangGraph core** for the central planner, conditional specialist routing,
  agent fan-out, and final answer aggregation.

This separation has several benefits:

- **Isolation**: UI concerns are separated from backend orchestration,
  guardrails, persistence, and agent execution.
- **Independent deployment and scaling**: the frontend and backend can be run,
  monitored, restarted, and scaled independently.
- **Robustness**: backend initialization, health checks, persistent session
  state, and tool startup can be managed without coupling them directly to
  Streamlit's process lifecycle.

The tradeoff is additional architectural complexity. The main impacts are
session memory management, portfolio upload propagation, health checks, and
developer support.

### Session Memory Management

The backend checkpointer is the source of truth for conversation state. It is backed by a sqlite database.
Streamlit does not own durable memory; it retrieves conversation history from
the backend after reloads or restarts.

The Streamlit UI passes a `session_id` to the backend through FastAPI
endpoints. The backend uses that value as the LangGraph checkpointer
`thread_id`, allowing multiple browser tabs or shared links to maintain
separate conversations.

Each session can be cleared independently through the backend reset endpoint.
Clearing one session does not affect other active sessions.

### Portfolio Uploads

Portfolio edits and uploads originate in the UI but must be propagated to the
backend because the Portfolio agent reads portfolio state from backend-managed
runtime files.

FastAPI exposes `POST /portfolio/refresh` so the UI can notify the backend
when portfolio data changes. The backend can then refresh the in-process
portfolio snapshot used by the Portfolio agent and related UI views.

### Health Check

FastAPI exposes `GET /health` as the backend liveness endpoint.

The Streamlit UI displays the backend URL and API reachability so users and
developers can quickly tell whether the backend is available. This is
especially useful during local development, first startup with RAG ingestion,
and HuggingFace Space restarts.

### Developer Support

The Streamlit UI surfaces backend connection details, including the backend
URL, so developers can directly inspect or call the API with tools such as
`curl`.

This helps debug whether an issue belongs to the frontend, API boundary,
LangGraph workflow, or a downstream agent/tool.

The UI also adds an **Under the Hood** section to each response, showing
developers and curious users which agents and tools were invoked.

## Initialization

Backend initialization is coordinated by FastAPI's lifespan context manager.
The same lifecycle handles:

- RAG ingestion for the FAISS-backed curated knowledge base.
- Portfolio seeding from default data into the runtime portfolio directory.
- FastMCP server startup for Portfolio agent tools.
- Cleanup of FastMCP sessions on termination.

FastAPI does not start serving requests until initialization is complete.
That prevents the app from answering questions against a half-built FAISS
index or partially initialized portfolio/tool state.

The Streamlit UI may become visible before the backend is ready, especially
on first startup or after a HuggingFace Space restart. In that case, the UI
shows backend reachability/status information and communicates that the
backend is not yet available.

## Robustness

The architecture is designed and validated for common lifecycle disruptions:

- Browser reloads.
- Multiple tabs with different `session_id` values.
- HuggingFace Space restarts.
- Backend initialization delays during RAG ingestion.
- Portfolio data refreshes after UI-side edits.
- Optional tool startup failures.

Persistent session state lives in the backend checkpointer, portfolio runtime
state lives under the backend-managed data path, and the UI reconstructs its
view from backend endpoints. This keeps the user experience stable across
frontend reloads and backend restarts.
