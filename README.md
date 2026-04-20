---
title: Fincent
emoji: 💹
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Fincent

Multi-agent financial assistant scaffold built on **LangGraph**, served
via **LangServe + FastAPI**, with a **Streamlit** UI running as a
separate process.

> Status: initial scaffold. The central orchestrator is fully wired;
> the Q&A agent is RAG-enabled (FAISS + LangChain ingestion of the
> curated catalog at `rag/fincent_rag_articles.json`); three additional
> specialist agents are stubbed for future iterations.

---

## Architecture

```
                +-------------------+
                |  Streamlit (UI)   |   <-- separate process
                +---------+---------+
                          | HTTP
                          v
                +---------+---------+
                |  FastAPI (ingress)|
                |   + LangServe     |
                +---------+---------+
                          |
                          v
                +---------+---------+
                |   LangGraph       |
                |   workflow        |
                +-------------------+
                          |
   +----------+-----------+-----------+----------+
   |          |           |           |          |
 central    qna        agent_two   agent_three  agent_four
(router/    (skeleton)  (stub)      (stub)       (stub)
 direct/
 aggregator)
```

The central agent has three responsibilities:

1. **Plan**: Use the LLM to classify the user's intent and produce a
   `RoutingPlan`.
2. **Direct answer**: For app-info / generic-user questions, answer
   without invoking specialists.
3. **Aggregate**: When one or more specialists run (possibly in
   parallel), merge their replies into a single final answer.

LangGraph's `Send` API is used to fan out to multiple specialist
nodes when the planner picks more than one.

### Session persistence

**Checkpoint location:** **`/data/`** means the directory at the **root of
the filesystem** (absolute path: leading `/`). It is **not** a relative
path `./data` beside the clone, and **not** the Python package folder
`src/data/` in this repo.

Conversation state is persisted with a **LangGraph SQLite
`SqliteSaver` checkpointer** at **`/data/checkpoints.sqlite`** by default
(a host-level directory **outside the git repository** — not under
`fincent/`). **HuggingFace Spaces** mounts **`/data`** for the Space; your
**local** environment should expose the same path if you use the default
(check with your setup). The **Dockerfile does not create `/data`**. Override
the path with `checkpointer.path` in `config.yaml` or
`FINCENT__CHECKPOINTER__PATH` when needed.

Each Streamlit
visitor maps to a LangGraph **`thread_id`**, sourced from the URL
query parameter `?session_id=<id>` (falling back to
`default-session`). Every call to `POST /query` and every LangServe
invocation must include the thread id via:

```json
{"configurable": {"thread_id": "<session_id>"}}
```

On a reload or HuggingFace tab switch, the Streamlit UI calls
`GET /history/{thread_id}` to rebuild the chat from the checkpoint
(mapping LangGraph roles `human` / `ai` to Streamlit's
`user` / `assistant`). The **Clear conversation** button first POSTs
to `/reset/{thread_id}` — the backend uses
`graph.update_state(..., {"messages": [RemoveMessage(id=...), ...]})`,
so prior versions remain in the SQLite checkpoint log.

### RAG (Q&A agent)

The Q&A agent is backed by a **FAISS** vector index built from the
curated catalog at [`rag/fincent_rag_articles.json`](rag/fincent_rag_articles.json)
(IRS / SEC / FINRA / Federal Reserve / CBP / USITC … 50+ URLs to
authoritative PDFs and HTML pages). The ingestion pipeline is a
standard LangChain RAG stack:

1. **Load** each URL:
   - PDFs are downloaded and parsed by **`UnstructuredPDFLoader`**
     (unstructured.io); if unstructured's system dependencies are
     missing, `PyPDFLoader` is used as a transparent fallback.
   - HTML pages are parsed by LangChain's **`BSHTMLLoader`**
     (BeautifulSoup + lxml).
2. **Stamp metadata**: every `Document` gets `url`, `title`, `tags`,
   and `source` attached — these travel through chunking and land on
   every vector in FAISS, so retrieved hits can be cited.
3. **Chunk** with `RecursiveCharacterTextSplitter.from_tiktoken_encoder`,
   **chunk size ≈ 1000 tokens, overlap 200**. The overlap keeps
   definitions next to their examples.
4. **Embed** with `OpenAIEmbeddings` (default model:
   `text-embedding-3-small`; override via
   `FINCENT__RAG__EMBEDDING_MODEL`).
5. **Persist** the FAISS index at **`/data/vector_db`** (same absolute
   `/data` mount as the SQLite checkpoint — provided by the host /
   the HF Space).

**Idempotent / skip-if-present.** The corpus is treated as **static**:
if `index.faiss` + `index.pkl` already exist at
`/data/vector_db`, ingestion is skipped on the next startup and the
retriever loads the existing index.

**Startup integration.** Ingestion runs inside the FastAPI
**`lifespan`** context manager. FastAPI does **not** begin serving
HTTP until ingestion finishes, so the Q&A agent never starts
answering with a half-built index. If ingestion **fails**, the app
still starts; `GET /rag/status` reports `{"state": "failed", ...}`
and the Streamlit UI renders an error banner above the chat while
still accepting queries (the agent degrades to plain LLM answers).

**Status endpoint.** Clients (including the Streamlit UI) poll
`GET /rag/status`; the response shape is:

```json
{
  "state": "ready | pending | ingesting | skipped | disabled | failed",
  "detail": "42 chunk(s) from 58/59 article(s) in 83.4s",
  "error": null,
  "chunk_count": 42,
  "ingested_articles": 58,
  "meta": {"vector_db_path": "/data/vector_db", "failures": [...]}
}
```

**Retrieval at query time.** When the Q&A agent receives a query it
calls the cached retriever (FAISS similarity search with `top_k=4`
by default) and prepends a `<context>` block containing the top
chunks to the system prompt. Source URLs are included so the LLM can
cite inline, and the returned `AgentResponse.metadata.sources` lists
the hits for the UI's routing-details expander.

---

## Repository layout

```
fincent/
  src/
    agents/
      central/      # Orchestrator (planner + direct + aggregator)
      qna/          # Generic financial Q&A (skeleton)
      agent_two/    # Reserved
      agent_three/  # Reserved
      agent_four/   # Reserved
    core/           # Config, LLM factory, shared schemas
    data/           # Python pkg ``src.data`` — NOT the same as ``/data`` on disk
    rag/            # RAG: loaders, ingestion, FAISS retriever, status singleton
    web_app/        # Streamlit UI + API client
    utils/          # Logging, helpers
    workflow/       # LangGraph state, nodes, graph, FastAPI server
  tests/            # pytest unit tests
  rag/              # fincent_rag_articles.json (source catalog; NOT the index)
  config.yaml       # Application configuration
  requirements.txt
  Dockerfile        # HuggingFace Spaces (Docker SDK) compatible
  scripts/docker-entrypoint.sh   # API health-wait + Streamlit (used by Dockerfile)
  .dockerignore     # Smaller / faster image builds
  run_local.sh      # Local Ubuntu launcher (api + UI)
  README.md
```

---

## Configuration

Defaults live in `config.yaml`. Any value can be overridden by an
environment variable using the `FINCENT__SECTION__KEY` convention,
for example:

```bash
export FINCENT__LLM__MODEL=gpt-4o-mini
export FINCENT__SERVER__PORT=8000
export FINCENT__CHECKPOINTER__PATH=/data/checkpoints.sqlite

# RAG (Q&A agent)
export FINCENT__RAG__ENABLED=true
export FINCENT__RAG__VECTOR_DB_PATH=/data/vector_db
export FINCENT__RAG__EMBEDDING_MODEL=text-embedding-3-small
export FINCENT__RAG__CHUNK_SIZE=1000
export FINCENT__RAG__CHUNK_OVERLAP=200
export FINCENT__RAG__TOP_K=4
```

The only **required** environment variable is:

```bash
export OPENAI_API_KEY=sk-...
```

A template lives in `.env.example`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| `OPENAI_API_KEY` | Required at runtime (chat model **and** embeddings). |
| Host directory `/data` | Defaults: `/data/checkpoints.sqlite` (session checkpointer) and `/data/vector_db/` (FAISS index). HF Spaces mounts `/data`; locally it should exist per your setup, or set `FINCENT__CHECKPOINTER__PATH` / `FINCENT__RAG__VECTOR_DB_PATH` to writable locations. |
| System libs for PDF parsing | Needed only if you want `UnstructuredPDFLoader` during ingestion. On Ubuntu: `sudo apt-get install poppler-utils tesseract-ocr libmagic1`. Without them, ingestion transparently falls back to `PyPDFLoader`. |
| `jq` *(optional)* | Pretty-prints API responses in the terminal: `sudo apt-get install jq` |

---

## Running locally (Ubuntu)

```bash
cd fincent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # fill in OPENAI_API_KEY
./run_local.sh
```

This starts:

- FastAPI / LangServe on `http://localhost:8000`
  - `GET  /health`
  - `GET  /rag/status`            (RAG ingestion pipeline state — see below)
  - `POST /query`                 (typed JSON I/O for the UI; carries `session_id`)
  - `GET  /history/{thread_id}`   (rehydrates the UI on reload — see below)
  - `POST /reset/{thread_id}`     (clears the current conversation)
  - `POST /fincent/invoke`        (raw LangServe endpoint for the graph)
  - `POST /fincent/stream`
- Streamlit UI on `http://localhost:8501`
  - Open `http://localhost:8501/?session_id=alice` to pin a named thread.
  - With no query param, the UI falls back to `session_id=default-session`.

Send a quick test:

```bash
# Ask a question (thread = "demo")
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is an ETF?", "session_id": "demo"}' | jq

# Fetch the full transcript back
curl -s http://localhost:8000/history/demo | jq

# Clear the current conversation for that thread
curl -s -X POST http://localhost:8000/reset/demo | jq

# Check the RAG ingestion status (state, counts, any failures)
curl -s http://localhost:8000/rag/status | jq
```

> **First startup note.** With RAG enabled and no existing index at
> `/data/vector_db`, the first run will download and embed the full
> catalog (50+ URLs). Expect a few minutes and some OpenAI embedding
> cost. Subsequent runs find the index on disk and skip ingestion.

---

## Running on HuggingFace Spaces (Docker SDK)

1. **Create a Space** at [huggingface.co/new-space](https://huggingface.co/new-space):
   - **SDK:** Docker
   - **Hardware:** CPU basic is enough; upgrade if you hit rate limits or latency issues.
   - Push this repository (or connect a GitHub repo).

2. **Secrets (required)** — Space **Settings → Repository secrets**:
   - `OPENAI_API_KEY` — your OpenAI API key. The app does not read `.env` in the container; secrets must be set here.

3. **Ports** — Spaces exposes only **`$PORT`** (default **7860**). The image runs:
   - **Streamlit** on `0.0.0.0:$PORT` (what users open in the browser).
   - **FastAPI** on **`API_PORT`** (default **8000**) inside the container only.
   - `scripts/docker-entrypoint.sh` starts FastAPI first, waits until `GET /health` succeeds, then starts Streamlit so the UI never races the API.
   - `PYTHONPATH=/app` is set so `src.*` imports work without extra flags.

   **Why `/health` can look “stuck” during first startup.** RAG ingestion runs inside the FastAPI **lifespan** hook *before* the ASGI app finishes startup. Until that completes, the server does not serve HTTP (including `GET /health`). So probes that expect a quick 200 from `/health` will fail or time out for as long as ingestion takes—not because `/health` is slow, but because **nothing is listening yet**. After ingestion finishes (or fails and the app still starts), `/health` responds normally.

   **Startup vs steady-state timeouts** (see `config.yaml` → `server`):

   | Setting | Default | Role |
   |---------|---------|------|
   | `startup_health_wait_seconds` | **300** (5 min) | Max time `docker-entrypoint.sh` polls `GET /health` before giving up. Must cover worst-case RAG ingestion. |
   | `healthcheck_interval_seconds` | **90** | Docker `HEALTHCHECK` **interval** after the start period—how often the container probes `/health` in steady state. |

   The Dockerfile `HEALTHCHECK` uses **`--start-period=300s`** (same 5 minutes as startup) so Docker does not mark the container unhealthy while ingestion is still running, then **`--interval=90s`** for ongoing checks. Override the entrypoint wait with **`FINCENT__SERVER__STARTUP_HEALTH_WAIT_SECONDS`** (Space variable or `docker run -e`).

4. **Persistent state** — Spaces **mounts `/data`**; the image does **not** run `mkdir` for it. The app writes two things under `/data`:
   - **`/data/checkpoints.sqlite`** — LangGraph session checkpoints (`FINCENT__CHECKPOINTER__PATH` overrides).
   - **`/data/vector_db/`** — FAISS index built by the RAG ingestion pipeline at startup (`FINCENT__RAG__VECTOR_DB_PATH` overrides). The index is treated as static: once present, subsequent container restarts skip ingestion.

5. **Optional overrides** (Space **Settings → Variables**):
   - `API_PORT` — change internal API port (default `8000`). If you change it, you must keep it in sync with `FINCENT__UI__API_BASE_URL` or use `FINCENT__UI__API_BASE_URL=http://127.0.0.1:<API_PORT>`.

6. **Build** — First build may take several minutes (`pip install`). If the Space shows a build error, open the **Logs** tab for the full trace.

7. **Local test with Docker** (optional):

   ```bash
   docker build -t fincent .
   docker run --rm -p 7860:7860 -e OPENAI_API_KEY=sk-... fincent
   ```

   Open `http://localhost:7860`.

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The test suite uses fake LLMs from `langchain_core.language_models.fake_chat_models`
so it does **not** require an `OPENAI_API_KEY` and does not perform
any network I/O.

---

## Adding a new specialist agent

1. Create `src/agents/<name>/{__init__.py,agent.py[,prompts.py]}`.
   The agent must expose `answer(query: str) -> AgentResponse`.
2. Add a new value to `Intent` and `AgentName` in
   `src/core/schemas.py`.
3. Wire it into `SPECIALIST_DISPATCH` and `SPECIALIST_NODE_FOR` in
   `src/workflow/nodes.py`, and add the node + edge in
   `src/workflow/graph.py`.
4. Add a `<name>:` block under `agents:` in `config.yaml` with
   `enabled: true` and a clear `description:` so the central planner
   can route to it.
5. Update the central router system prompt
   (`src/agents/central/prompts.py`) so the LLM knows the new intent
   exists.
