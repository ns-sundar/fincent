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
> curated catalog at `rag/fincent_rag_articles.json`); the Portfolio
> agent is a ReAct tool-caller wired to two MCP servers (OpenBB for
> live market data, Fincent's RAG sidecar for curated documents);
> two additional specialist agents are stubbed for future iterations.

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
 central    qna        portfolio  agent_three  agent_four
(router/    (skeleton)  (grounded)  (stub)       (stub)
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

### Routing rules (both tabs)

Routing is **orchestrated by the central planner for every turn, in
both the Q&A and Portfolio tabs**. The Portfolio tab is _not_ pinned
to the Portfolio agent — it uses the same router as Q&A, so generic
financial questions asked there are still handled by the Q&A agent,
and personal-portfolio questions asked in the Q&A tab correctly reach
the Portfolio agent.

| Query class | Example | Handled by |
|---|---|---|
| Non-financial chit-chat | “Hi, what's your name?” | **Central** (direct answer) |
| App-info | “What can this app do?” | **Central** (direct answer) |
| Generic financial, no personal data | “What is an ETF?”, “How are dividends taxed?” | **Q&A agent** (RAG over curated corpus) |
| Touches the user's own portfolio (even if it also asks a general concept) | “Am I over-concentrated in AAPL?”, “Explain dividend taxation for my holdings.” | **Portfolio agent** (ReAct; may call OpenBB or RAG tools itself) |

Personal-finance questions never fan out to both Q&A and Portfolio:
the Portfolio agent has its own `rag_search` tool for generic context,
so routing to it alone avoids duplicate work.

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

**Retrieval at query time (agentic RAG).** The Q&A agent routes every
query through the canonical search function
[`src.rag.tool.rag_search`](src/rag/tool.py) — the same function the MCP
server (below) exposes. It:

1. Retrieves **top-k** (default **`5`**, `cfg.rag.top_k`) chunks from
   the FAISS index.
2. Re-ranks with **MMR** (Maximal Marginal Relevance) by default
   (`cfg.rag.use_mmr`) so the top-k is not three near-duplicate
   chunks. Tunable via `cfg.rag.mmr_fetch_k` (candidate pool) and
   `cfg.rag.mmr_lambda` in `[0, 1]` (1.0 = pure similarity, 0.0 =
   maximum diversity).
3. Honours **source-filter narrowing** expressed in natural language.
   Phrases such as *“Cite only IRS documents”*, *“Based on SEC
   documents, …”*, *“Per FINRA rules, …”*, *“According to the Federal
   Reserve, …”* are detected by
   [`detect_source_filter`](src/rag/tool.py) and restrict retrieval to
   chunks whose `metadata.tags.source` matches the requested family.
   Supported sources match the values shipped in
   `rag/fincent_rag_articles.json`:
   `irs`, `sec`, `finra`, `fdic`, `fed`, `occ`, `treasury`, `cbp`,
   `nyse`, `investopedia`, `bogleheads`, `fidelity`, `tax-foundation`.

**Citations.** RAG-grounded replies use inline `[n]` markers in the
prose (`n` matches the numbered `<context>` entries). The model is
instructed to end with a single **`## Sources`** section when it cited
at least one chunk, and to list **only** those context entries whose
`[n]` appears in the answer—never every retrieved passage. Titles and
URLs must be copied from the context block (no invented links).

The API still parses `[n]` from the full reply and fills
**`AgentResponse.metadata.sources`** with that **cited subset** (`url`,
`title`, `tags`, `score`), plus **`cited_chunk_indices`**, so UIs and
tests get a reliable structured list even if the model’s `## Sources`
wording differs slightly.

### Portfolio agent (personal finance, ReAct + MCP tools)

The Portfolio agent answers questions grounded in the user's own
account / holding / transaction snapshot (seeded on first boot from
`data/default_portfolio/` into `/data/portfolio/`). Under the hood it
runs a **ReAct tool-calling loop** and can reach for two families of
**MCP tools**, both launched as `stdio` subprocesses so there are no
extra ports to manage:

1. **OpenBB MCP tools** — real-world market data (live quotes,
   historical prices, company news, ETF holdings, economic
   indicators, crypto, FX, …) via the official
   [`openbb-mcp-server`](https://pypi.org/project/openbb-mcp-server/).
   The OpenBB Platform ships free providers out of the box (yfinance,
   SEC, FRED, …); other providers (e.g. Financial Modeling Prep) can
   be installed as OpenBB extensions and configured with API keys in
   `~/.openbb_platform/user_settings.json`. The advertised tool set is
   restricted to a handful of relevant categories
   (`equity,news,etf,currency,economy,crypto`) so the LLM isn't
   overwhelmed by hundreds of tools.
2. **Fincent RAG MCP tool** — the same `rag_search` tool the Q&A
   agent uses internally, exposed here via the existing
   `src/rag/mcp_server.py` so the Portfolio agent can pull in curated
   regulatory / educational context (IRS rules, ETF mechanics,
   Bogleheads guides, …) when a question combines portfolio data with
   a general financial concept.

Both servers are launched via
[`langchain-mcp-adapters`](https://pypi.org/project/langchain-mcp-adapters/)
`MultiServerMCPClient`. Tools are loaded **once per process** and
cached; if any of `langchain-mcp-adapters`, `openbb-mcp-server`, or
`openbb` is missing (or the server is disabled), the agent degrades
to a single LLM call grounded in the snapshot — i.e. identical to the
pre-tooling behaviour — so the app always starts.

The tool registry is controlled by `portfolio.tools.*` in
`config.yaml`; see below for the relevant env overrides.

### MCP server (optional sidecar)

The FAISS vector_db is also wrapped in a **Model Context Protocol**
server at [`src/rag/mcp_server.py`](src/rag/mcp_server.py) which
publishes a single tool — **`rag_search`** (name configurable) — that
any MCP-capable client (Claude Desktop, Cursor, LangChain MCP adapter,
…) can call.

Tool contract:

| Arg      | Type              | Notes |
|----------|-------------------|-------|
| `query`  | `str`             | Natural language query. |
| `source` | `str \| null`     | Canonical source tag (or alias, e.g. `"Federal Reserve"`). Limits hits to that `tags.source` family. |
| `top_k`  | `int \| null`     | Defaults to `cfg.rag.top_k` (5). |

Each result is a dict with `text`, `url`, `title`, `tags`
(`source`, `category[]`), and `score`.

Run it standalone:

```bash
python -m src.rag.mcp_server
```

Transports are configurable via `cfg.rag.mcp_server.transport`:

- **`stdio`** (default) — suitable when the MCP client spawns the
  server (e.g. Claude Desktop / Cursor's MCP config).
- **`streamable-http`** — long-lived HTTP server with bidirectional
  streaming on the **`/mcp`** endpoint at
  `cfg.rag.mcp_server.host:port` (default `127.0.0.1:8765`). This is
  the MCP spec's replacement for the **deprecated SSE** transport; if
  you set `transport: "sse"` the server will refuse to start with a
  pointer to `"streamable-http"`.

The MCP sidecar is **independent** of the in-process Q&A agent: both
call the same `rag_search` function, so disabling MCP (the default)
does not change Q&A behaviour.

---

## Repository layout

```
fincent/
  src/
    agents/
      central/      # Orchestrator (planner + direct + aggregator)
      qna/          # Generic financial Q&A (agentic RAG: top-k + MMR + source-filter citations)
      portfolio/    # Personal portfolio agent (reads /data/portfolio; seeded on first boot from data/default_portfolio)
      agent_three/  # Reserved
      agent_four/   # Reserved
    core/           # Config, LLM factory, shared schemas
    data/           # Python pkg ``src.data`` — NOT the same as ``/data`` on disk
    rag/            # RAG: loaders, ingestion, FAISS retriever, canonical search tool, MCP server, status
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
export FINCENT__RAG__TOP_K=5
# MMR (diversified re-ranking at query time)
export FINCENT__RAG__USE_MMR=true
export FINCENT__RAG__MMR_FETCH_K=20
export FINCENT__RAG__MMR_LAMBDA=0.5
# Optional MCP sidecar (exposes vector_db as a tool)
export FINCENT__RAG__MCP_SERVER__ENABLED=false
export FINCENT__RAG__MCP_SERVER__TRANSPORT=stdio   # or "streamable-http"
export FINCENT__RAG__MCP_SERVER__TOOL_NAME=rag_search

# Portfolio agent MCP tools (stdio subprocesses)
# OpenBB Platform -- real-world market data via yfinance + SEC + FRED + ...
export FINCENT__PORTFOLIO__TOOLS__OPENBB__ENABLED=true
export FINCENT__PORTFOLIO__TOOLS__OPENBB__COMMAND=openbb-mcp
# Fincent RAG MCP server -- the curated vector_db from above, exposed as a tool
export FINCENT__PORTFOLIO__TOOLS__RAG__ENABLED=true
export FINCENT__PORTFOLIO__TOOLS__RAG__COMMAND=python
# Provider API keys for OpenBB live in ~/.openbb_platform/user_settings.json;
# the free providers (yfinance, SEC, FRED, ...) work with no key.
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
| Host directory `/data` | Defaults: `/data/checkpoints.sqlite` (session checkpointer), `/data/vector_db/` (FAISS index), and `/data/portfolio/` (runtime portfolio JSONs, seeded on first boot from `data/default_portfolio/` in the repo). HF Spaces mounts `/data`; locally it should exist per your setup, or set `FINCENT__CHECKPOINTER__PATH` / `FINCENT__RAG__VECTOR_DB_PATH` / `FINCENT__PORTFOLIO__DATA_PATH` to writable locations. |
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

4. **Persistent state** — Spaces **mounts `/data`**; the image does **not** run `mkdir` for it. The app writes three things under `/data`:
   - **`/data/checkpoints.sqlite`** — LangGraph session checkpoints (`FINCENT__CHECKPOINTER__PATH` overrides).
   - **`/data/vector_db/`** — FAISS index built by the RAG ingestion pipeline at startup (`FINCENT__RAG__VECTOR_DB_PATH` overrides). The index is treated as static: once present, subsequent container restarts skip ingestion.
   - **`/data/portfolio/`** — Runtime portfolio JSONs (`accounts.json`, `transactions.json`). On the first boot the FastAPI lifespan seeds this directory from `data/default_portfolio/` in the repo (read-only). Subsequent boots leave existing files alone so future in-app edits survive restarts. Override the location with `FINCENT__PORTFOLIO__DATA_PATH`, and the seed source with `FINCENT__PORTFOLIO__SEED_PATH`.

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
