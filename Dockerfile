# =====================================================================
# Multi-process container suitable for HuggingFace Spaces (Docker SDK).
#
# Runs the FastAPI/LangServe server on $API_PORT (default 8000) and
# the Streamlit UI on $PORT (default 7860 -- HuggingFace's expected
# public port).
# =====================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    API_PORT=8000

WORKDIR /app

# System deps (kept minimal).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source.
COPY . .

# HuggingFace Spaces only exposes one public port; we bind Streamlit
# there and keep the API internal on $API_PORT.
EXPOSE 7860

# Tell Streamlit where the API is reachable inside the container.
ENV FINCENT__UI__API_BASE_URL=http://localhost:8000

# Launch both processes; the container exits when either dies.
CMD ["bash", "-lc", "\
    python -m uvicorn src.workflow.server:app --host 0.0.0.0 --port ${API_PORT} & \
    streamlit run src/web_app/streamlit_app.py \
        --server.port ${PORT} \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
"]
