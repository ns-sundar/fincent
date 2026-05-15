#!/usr/bin/env bash
# Start FastAPI in the background and Streamlit immediately in the foreground.
# This is important for HuggingFace Spaces: the public $PORT should bind quickly
# even while FastAPI runs startup work such as RAG ingestion.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
PORT="${PORT:-7860}"
API_READY_LOG_WAIT="${FINCENT__SERVER__STARTUP_HEALTH_WAIT_SECONDS:-900}"

export FINCENT__UI__API_BASE_URL="http://127.0.0.1:${API_PORT}"

ensure_dir() {
  local path="$1"
  if [[ -n "${path}" && "${path}" != ":memory:" ]]; then
    mkdir -p "${path}"
  fi
}

ensure_parent_dir() {
  local path="$1"
  if [[ -n "${path}" && "${path}" != ":memory:" ]]; then
    mkdir -p "$(dirname "${path}")"
  fi
}

# Hugging Face persistent storage is mounted at /data when enabled. On Spaces
# without persistent storage, creating /data inside the container still gives the
# app a writable ephemeral runtime directory instead of failing at startup.
ensure_parent_dir "${FINCENT__CHECKPOINTER__PATH:-/data/checkpoints.sqlite}"
ensure_dir "${FINCENT__RAG__VECTOR_DB_PATH:-/data/vector_db}"
ensure_dir "${FINCENT__PORTFOLIO__DATA_PATH:-/data/portfolio}"

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[entrypoint] Starting FastAPI on 0.0.0.0:${API_PORT}"
python -m uvicorn src.workflow.server:app --host 0.0.0.0 --port "${API_PORT}" &
UVICORN_PID=$!

( 
  echo "[entrypoint] FastAPI warming up; checking /health in background for up to ${API_READY_LOG_WAIT}s ..."
  for _ in $(seq 1 "${API_READY_LOG_WAIT}"); do
    if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
      echo "[entrypoint] WARNING: uvicorn exited before becoming healthy" >&2
      exit 0
    fi
    if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
      echo "[entrypoint] FastAPI is healthy"
      exit 0
    fi
    sleep 1
  done
  echo "[entrypoint] WARNING: /health did not respond within ${API_READY_LOG_WAIT}s" >&2
) &

echo "[entrypoint] Starting Streamlit on 0.0.0.0:${PORT}"
# Foreground Streamlit (no exec) so SIGTERM/INT reaches this shell and we kill
# uvicorn — triggers FastAPI lifespan shutdown (MCP sessions close cleanly).
streamlit run src/web_app/streamlit_app.py \
  --server.port "${PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
st_exit=$?
cleanup
exit "${st_exit}"
