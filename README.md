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

## Overview

Fincent is a personal financial assistant based on a multi-agent AI framework. It helps users ask general finance questions, learn about stocks,
bonds, ETFs, taxes, market mechanics, and portfolio theory, and analyze
their own accounts, holdings, transactions, allocation, concentration,
exposure, and portfolio risk. It presents two tabs in its UI, one for general financial Q&A, and one meant for analyzing your portfolio that you upload. You can start with a default sample portfolio for convenience. The UI also adds an **Under the Hood** section to each response, showing
developers and curious users which agents and tools were invoked.

Fincent combines several knowledge sources and safety layers:

- **Streamlit** for the browser UI.
- **FastAPI** for typed HTTP endpoints, session history, guardrails, and
  LangGraph invocation.
- **LangGraph** for the central planner/router, specialist fan-out, and
  final response aggregation.
- **OpenBB APIs** for real-time financial and market knowledge, with
  yFinance configured as the default backend for key equity quote and price
  tools.
- **FAISS-backed curated knowledge base** for grounded general finance Q&A.
- **FastMCP Server** for exposing tools such as OpenBB using MCP.
- **OpenAI Moderation API** as an input guardrail on the `/query` endpoint.
- **Markdown validation** as an output guardrail for assistant responses.
- **DeepEval** for end-to-end LLM evaluation against intent-based golden
  datasets.

Fincent is educational software, not a professional financial adviser. It has no access to your personal information except for the portfolio that you upload.

---

## Installation and Usage

### Local

Prerequisites:

- Python 3.11+
- `OPENAI_API_KEY`
- A writable `/data` directory, or overrides for the checkpoint, vector DB,
  and portfolio data paths
- Optional system packages for richer PDF ingestion:
  `poppler-utils`, `tesseract-ocr`, and `libmagic1`

Run locally:

```bash
cd fincent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill OPENAI_API_KEY in .env

./run_local.sh
```

This starts:

- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`

Useful API endpoints:

- `GET /health`
- `GET /rag/status`
- `POST /query`
- `GET /history/{thread_id}`
- `POST /reset/{thread_id}`
- `POST /fincent/invoke`
- `POST /fincent/stream`

Example request:

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is an ETF?", "session_id": "demo"}'
```

Session state is persisted with LangGraph's SQLite checkpointer. By
default, Fincent writes to absolute paths under `/data`:

- `/data/checkpoints.sqlite` for conversation checkpoints
- `/data/vector_db/` for the FAISS index
- `/data/portfolio/` for runtime portfolio JSON files

The first run may take a few minutes if the FAISS index does not already
exist. FastAPI performs RAG ingestion during startup before serving HTTP.
Once the index exists, later runs reuse it.

### HuggingFace Spaces

Fincent is configured for HuggingFace Spaces with the Docker SDK.

1. Create a new Space.
2. Select **Docker** as the SDK.
3. Add `OPENAI_API_KEY` as a Space secret.
4. Push this repository or connect it to GitHub.

The container exposes Streamlit on `$PORT` (default `7860`) and runs
FastAPI internally on `API_PORT` (default `8000`). The Docker entrypoint
starts FastAPI first, waits for `GET /health`, then starts Streamlit so the
UI does not race the API.

HuggingFace Spaces mounts `/data`, which Fincent uses for checkpoints, the
FAISS index, and portfolio runtime data. The image intentionally does not
create `/data`; the platform provides it.

---

## Arch

```text
                         +--------------------+
                         |   Streamlit UI     |
                         +---------+----------+
                                   |
                                   | HTTP
                                   v
                         +---------+----------+
                         | FastAPI ingress    |
                         | guardrails + API   |
                         +---------+----------+
                                   |
                                   v
                         +---------+----------+
                         | LangGraph workflow |
                         +---------+----------+
                                   |
                                   v
                         +---------+----------+
                         | Central planner    |
                         | router/direct      |
                         +----+----------+----+
                              |          |
              central direct  |          | specialist fan-out
                              |          |
                              v          v
                       +------+--+   +---+----------------+
                       | Central |   | Q&A and/or          |
                       | answer  |   | Portfolio agents    |
                       +------+--+   +---+----------------+
                              |          |
                              +-----+----+
                                    |
                                    v
                         +----------+---------+
                         | Central aggregator |
                         +----------+---------+
                                    |
                                    v
                         +----------+---------+
                         | Final answer       |
                         +--------------------+
```

The central agent has three responsibilities:

1. **Plan**: classify the user's request and produce a `RoutingPlan`.
2. **Direct answer**: answer app identity, app features, chit-chat, and
   out-of-scope requests without invoking specialists.
3. **Aggregate**: merge one or more specialist responses into the final
   answer.

The planner can conditionally route to multiple specialist agents. LangGraph
uses `Send` to fan out when the route contains more than one specialist, and
the central aggregator combines the specialist outputs.

Routing is shared by both UI tabs. The Portfolio tab is not hard-pinned to
the Portfolio agent; generic financial questions still route to Q&A, and
personal portfolio questions asked in the Q&A tab still route to Portfolio.

Query classes:

| Query class | Example | Handler |
|---|---|---|
| App identity | "Who are you?" | Central direct answer |
| App features | "What tools do you use?" | Central direct answer |
| Chit-chat | "Thanks for the help." | Central direct answer |
| Out of scope | "What's the weather today?" | Central refusal/redirect |
| Generic finance | "What is an ETF?" | Q&A agent |
| Personal portfolio | "Am I over-concentrated in AAPL?" | Portfolio agent |

For deeper implementation detail, see the
[technical design](TECHNICAL_DESIGN.md), the source under `src/workflow/`
and `src/agents/`, and the eval design in `eval/README.md`.

---

## Guardrails

Guardrails are implemented in FastAPI middleware around the typed `/query`
endpoint.

**Input moderation**: incoming user queries are sent to the OpenAI
Moderation API using `omni-moderation-latest`. If moderation flags the
query, FastAPI returns `400` with the flagged categories and does not invoke
the graph. If the moderation service fails, FastAPI returns `503` rather
than running the unsafe or unchecked request.

**Output validation**: successful JSON responses are inspected for an
`answer` field. FastAPI validates the generated markdown structure,
including unclosed fenced code blocks and malformed GitHub-style markdown
tables. Validation warnings are logged with a capped excerpt for debugging;
the response is still returned to the user.

The Streamlit UI also sanitizes assistant/user text before rendering with
`st.markdown`.

---

## Evals

Fincent uses DeepEval for end-to-end evaluation. Eval dependencies live in
`requirements-eval.txt` so runtime installs can stay lean:

```bash
pip install -r requirements.txt -r requirements-eval.txt
```

Golden datasets are organized by intent under `eval/eval.*.json`:

- `eval.app-identity.json`
- `eval.app-features.json`
- `eval.chit-chat.json`
- `eval.out-of-scope.json`
- `eval.qna.json`
- `eval.portfolio.json`

Each row contains the user input, expected output, optional retrieval
context, expected intent, whether the central agent should handle it
directly, and the expected agent sequence.

Run the e2e evals:
```bash
 deepeval test run eval/test_e2e_intents.py
```

```bash
python -m pytest eval/test_e2e_intents.py
```

The pytest runner loads all golden cases, starts Portfolio FastMCP servers
once for the test session, compiles the LangGraph once, and then runs each
case through the full workflow with a fresh `session_id`. Before invoking
DeepEval, it asserts deterministic routing expectations: central-handled
flag, primary intent, expected agents, and absence of agent errors.

DeepEval metrics include:

- **G-Eval** as the primary intent-aware task success metric. It compares
  actual output with expected output while allowing harmless wording
  differences.
- **AnswerRelevancyMetric** for most intents.
- **FaithfulnessMetric** when retrieval context is present.
- **ContextualRecallMetric** when retrieval context is present.
- **ContextualRelevancyMetric** when retrieval context is present.

`chit_chat` and `out_of_scope` skip generic answer relevancy because polite
redirects can be correct product behavior even when a generic relevancy
judge would penalize them. For those intents, the custom G-Eval rubric is
the source of truth.

See `eval/README.md` for the eval-specific design and operating notes.

---

## Validation

Unit and integration tests live under `tests/` and run with pytest:

```bash
pip install -r requirements.txt
pytest -q
```

The normal unit test suite uses fake LLMs from
`langchain_core.language_models.fake_chat_models`, so it does not require
`OPENAI_API_KEY` and does not perform network I/O.

The validation suite covers configuration loading, central routing behavior,
workflow execution, server endpoints, Streamlit routing helpers, RAG
utilities, and portfolio data handling.

---

## Agents

### Central Agent

The central agent is the planner, direct responder, and aggregator. It
classifies requests into intents such as `app_identity`, `app_features`,
`chit_chat`, `out_of_scope`, `qna`, and `portfolio`.

For central-handled intents, it answers directly using app metadata from
`config.yaml`. For specialist intents, it routes to Q&A, Portfolio, or both
when the routing plan calls for fan-out, then aggregates the specialist
responses.

### Q&A Agent

The Q&A agent handles generic financial and economics questions that do not
depend on the user's personal portfolio. Examples include ETFs, dividends,
brokerage concepts, IRS/tax basics, investment risk, market mechanics, and
general portfolio theory.

It uses RAG over a curated knowledge base stored in
`rag/fincent_rag_articles.json`. At startup, Fincent ingests the corpus into
a FAISS vector index. At query time, the agent calls the canonical
`rag_search` function and retrieves `top_k=5` chunks by default.

Retrieval uses MMR (Maximal Marginal Relevance) by default to improve
diversity and avoid returning near-duplicate passages. The RAG tool also
supports natural-language source narrowing such as "Cite only IRS
documents" or "According to FINRA rules."

RAG-grounded replies are instructed to cite retrieved chunks with inline
`[n]` markers and a `## Sources` section. The API also parses cited sources
into structured response metadata for UI and tests.

### Portfolio Agent

The Portfolio agent handles questions about the user's own accounts,
holdings, transactions, allocation, concentration, exposure, and portfolio
risk. It grounds responses in the runtime portfolio data under
`/data/portfolio/`, seeded on first boot from `data/default_portfolio/`.

The agent runs a ReAct tool-calling loop and can use MCP tools for:

- **OpenBB**: real-time market data such as quotes, historical prices,
  company news, ETF holdings, economic indicators, crypto, and FX.
- **Fincent RAG**: the same curated FAISS-backed `rag_search` capability
  used by the Q&A agent, exposed as an MCP tool for mixed portfolio +
  educational questions.

For OpenBB equity quote and historical price tools, Fincent pins the
provider to **yFinance** by default because it works without an API key for
many US listings. Operators can configure additional OpenBB providers and
API keys outside the app.

Portfolio MCP sessions are started during FastAPI lifespan and reused for
the process. If OpenBB, the MCP adapter, or the RAG sidecar is unavailable,
the Portfolio agent degrades to an LLM response grounded in the portfolio
snapshot so the app can still start.

---

## Repository Layout

```text
fincent/
  src/
    agents/
      central/      # Planner, direct answers, aggregator
      qna/          # Generic financial Q&A with FAISS RAG
      portfolio/    # Personal portfolio agent with OpenBB/RAG tools
    core/           # Config, LLM factory, shared schemas
    data/           # Python package, not the same as absolute /data
    rag/            # Ingestion, retriever, search tool, MCP server, status
    web_app/        # Streamlit UI and API client
    workflow/       # LangGraph state, nodes, graph, FastAPI server
  eval/             # DeepEval datasets, metrics, pytest runner
  tests/            # Unit and integration tests
  rag/              # Curated source catalog
  config.yaml
  requirements.txt
  requirements-eval.txt
  Dockerfile
  run_local.sh
```

---

## Configuration

Defaults live in `config.yaml`. Any value can be overridden with the
`FINCENT__SECTION__KEY` environment variable convention:

```bash
export OPENAI_API_KEY=sk-...
export FINCENT__LLM__MODEL=gpt-5.4-mini
export FINCENT__SERVER__PORT=8000
export FINCENT__CHECKPOINTER__PATH=/data/checkpoints.sqlite

export FINCENT__RAG__ENABLED=true
export FINCENT__RAG__VECTOR_DB_PATH=/data/vector_db
export FINCENT__RAG__TOP_K=5
export FINCENT__RAG__USE_MMR=true

export FINCENT__PORTFOLIO__TOOLS__OPENBB__ENABLED=true
export FINCENT__PORTFOLIO__TOOLS__RAG__ENABLED=true
```

A template lives in `.env.example`.
