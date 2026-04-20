#!/usr/bin/env bash
# Convenience launcher for local Ubuntu development.
# Starts the FastAPI server and the Streamlit UI as two processes.
#
# LangGraph checkpoints default to /data/checkpoints.sqlite (outside this repo).
# /data must exist on the host (e.g. HF mount) or set FINCENT__CHECKPOINTER__PATH.
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f .env ]]; then
    set -a; source .env; set +a
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it or put it in .env" >&2
  exit 1
fi

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

echo "Starting FastAPI on :${API_PORT}"
python -m uvicorn src.workflow.server:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

cleanup() {
  echo "Shutting down (api pid=${API_PID})..."
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export FINCENT__UI__API_BASE_URL="http://localhost:${API_PORT}"

echo "Starting Streamlit on :${UI_PORT}"
streamlit run src/web_app/streamlit_app.py \
  --server.port "${UI_PORT}" \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
