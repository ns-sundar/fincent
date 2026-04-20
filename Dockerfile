# =====================================================================
# HuggingFace Spaces (Docker SDK) + generic container runs
#
# - Streamlit binds the public port ($PORT, default 7860 on Spaces).
# - FastAPI + LangGraph run on $API_PORT (default 8000) inside the pod.
# - scripts/docker-entrypoint.sh waits for GET /health before starting
#   Streamlit so the UI never races the API.
#
# Required secret at runtime: OPENAI_API_KEY (set in Space Settings).
# =====================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    API_PORT=8000

WORKDIR /app

# System packages:
#   curl, ca-certificates  -- entrypoint health check + HTTPS to OpenAI
#   poppler-utils, tesseract-ocr, libmagic1  -- unstructured.io PDF parsing
#                                              (for RAG ingestion at startup)
# Persistent state uses /data/checkpoints.sqlite and /data/vector_db; /data
# is provided by the host (e.g. HuggingFace Spaces mount) -- do not mkdir here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        poppler-utils \
        tesseract-ocr \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application source + entrypoint
COPY . .
RUN chmod +x scripts/docker-entrypoint.sh

# HuggingFace Spaces maps the public listener to $PORT (7860 default).
EXPOSE 7860

# Default UI→API URL (entrypoint overrides with 127.0.0.1 + $API_PORT)
ENV FINCENT__UI__API_BASE_URL=http://127.0.0.1:8000

# Internal API port (must match API_PORT default).
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health >/dev/null || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
