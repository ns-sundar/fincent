# =====================================================================
# HuggingFace Spaces (Docker SDK) + generic container runs
#
# - Streamlit binds the public port ($PORT, default 7860 on Spaces).
# - FastAPI + LangGraph run on $API_PORT (default 8000) inside the pod.
# - scripts/docker-entrypoint.sh starts Streamlit immediately so Spaces sees
#   the public port while FastAPI/RAG startup continues in the background.
#
# Required secret at runtime: OPENAI_API_KEY (set in Space Settings).
# =====================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app \
    PORT=7860 \
    API_PORT=8000

WORKDIR /app

# System packages:
#   curl, ca-certificates      -- entrypoint health check + HTTPS to OpenAI
# Persistent state uses /data/checkpoints.sqlite, /data/vector_db, and
# /data/portfolio; the entrypoint creates those paths if the Space does not
# mount persistent storage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Application source + entrypoint
COPY . .
RUN chmod +x scripts/docker-entrypoint.sh

# HuggingFace Spaces maps the public listener to $PORT (7860 default).
EXPOSE 7860

# Default UI→API URL (entrypoint overrides with 127.0.0.1 + $API_PORT)
ENV FINCENT__UI__API_BASE_URL=http://127.0.0.1:8000

# Match config.yaml server.startup_health_wait_seconds / healthcheck_interval_seconds.
ENV FINCENT__SERVER__STARTUP_HEALTH_WAIT_SECONDS=900

# Internal API port (must match API_PORT default).
# start-period: grace while RAG ingestion runs in FastAPI lifespan (no /health yet).
# interval: steady-state probe spacing after start-period (90s).
HEALTHCHECK --interval=90s --timeout=5s --start-period=900s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health >/dev/null || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
