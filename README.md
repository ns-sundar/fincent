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
> the Q&A agent is a working skeleton; three additional specialist
> agents are stubbed for future iterations.

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
    data/           # Python pkg ``src.data`` — NOT the same as ``/data`` for SQLite
    rag/            # Retrieval utilities (placeholder)
    web_app/        # Streamlit UI + API client
    utils/          # Logging, helpers
    workflow/       # LangGraph state, nodes, graph, FastAPI server
  tests/            # pytest unit tests
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
| `OPENAI_API_KEY` | Required at runtime |
| Host directory `/data` | Default checkpoint DB: `/data/checkpoints.sqlite`. HF Spaces mounts `/data`; locally it should exist per your setup, or set `FINCENT__CHECKPOINTER__PATH` to a writable file. |
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
```

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

4. **Checkpoints** — Spaces **mounts `/data`**; the image does **not** run `mkdir` for it. LangGraph writes **`/data/checkpoints.sqlite`** by default. Override with `FINCENT__CHECKPOINTER__PATH` if your deployment uses a different path.

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
