#!/usr/bin/env bash
# Start FastAPI in the background, wait until /health succeeds, then exec
# Streamlit in the foreground. Suitable for HuggingFace Spaces (single
# public port on $PORT, internal API on $API_PORT).
set -euo pipefail

API_PORT="${API_PORT:-8000}"
PORT="${PORT:-7860}"

export FINCENT__UI__API_BASE_URL="http://127.0.0.1:${API_PORT}"

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[entrypoint] Starting FastAPI on 0.0.0.0:${API_PORT}"
python -m uvicorn src.workflow.server:app --host 0.0.0.0 --port "${API_PORT}" &
UVICORN_PID=$!

echo "[entrypoint] Waiting for GET /health ..."
ready=0
for _ in $(seq 1 90); do
  if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[entrypoint] ERROR: uvicorn exited before becoming healthy" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[entrypoint] ERROR: /health did not respond within 90s" >&2
  exit 1
fi

echo "[entrypoint] API ready; starting Streamlit on 0.0.0.0:${PORT}"
trap - EXIT
exec streamlit run src/web_app/streamlit_app.py \
  --server.port "${PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
