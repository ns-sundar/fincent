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

Fincent is a multi-agent personal finance assistant. Aimed at the financially curious,
it offers general Q&A on financial topics, analytics on the user's own uploaded
portfolio, market/company research with real-time data and help with long-run goal
or stress-test modelling.

Four specialist agents coordinate to provide this functionality. The Q&A agent answers
broad finance questions from a FAISS-backed article catalog. The Portfolio agent
interprets  the user's holdings, transactions, allocation, and risk, with optional live quotes when
tools are enabled. The Market Research agent draws on public data about companies,
securities, sectors, filings, and themes without giving personalized buy-or-sell advice.
The Goal Planning agent uses the same portfolio snapshot for retirement, education,
housing, spending, drawdown, time-value-of-money, and Monte Carlo scenarios.

You can run it locally or deploy your own Hugging Face Space ([Installation and Usage](#installation-and-usage)); a preinstalled deployment lives at [huggingface.co/spaces/nssundar/fincent](https://huggingface.co/spaces/nssundar/fincent).

Please note that Fincent is educational software, not a professional adviser. In terms of privacy, it only sees portfolio
data that you upload.

Fincent leverages several open source and proprietary frameworks and APIs to deliver this functionality:

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Framework or API</th>
      <th>Description and Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="vertical-align: middle;">App and orchestration</td>
      <td>Streamlit</td>
      <td>Provides the browser UI for local use and Hugging Face Spaces.</td>
    </tr>
    <tr>
      <td>FastAPI + LangServe</td>
      <td>Expose Fincent as a web service with typed routes, session checkpoints, guardrails, and LangGraph <code>invoke</code> and <code>stream</code> endpoints.</td>
    </tr>
    <tr>
      <td>LangGraph</td>
      <td>Routes intent, fans out work to specialist agents, and aggregates final responses.</td>
    </tr>
    <tr>
      <td rowspan="6" style="vertical-align: middle;">Tools and data</td>
      <td>FastMCP</td>
      <td>Connects tool servers for OpenBB, Financial Modeling Prep, Fincent RAG, Goal Planning math, and optional providers.</td>
    </tr>
    <tr>
      <td>OpenBB + yFinance</td>
      <td>Provide default market data and related public equity context.</td>
    </tr>
    <tr>
      <td>Financial Modeling Prep</td>
      <td>Adds company fundamentals and filings when configured.</td>
    </tr>
    <tr>
      <td>Tavily</td>
      <td>Extends web search and extraction when enabled.</td>
    </tr>
    <tr>
      <td>Alpha Vantage</td>
      <td>Adds technical and sentiment signals when enabled.</td>
    </tr>
    <tr>
      <td>FAISS</td>
      <td>Powers retrieval over the curated article catalog for RAG.</td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle;">Safety and quality</td>
      <td>OpenAI Moderation API</td>
      <td>Screens incoming <code>/query</code> requests before they reach the agents.</td>
    </tr>
    <tr>
      <td>Markdown validation + Streamlit sanitization</td>
      <td>Validate assistant markdown server-side and sanitize rendered content in the UI.</td>
    </tr>
    <tr>
      <td>DeepEval</td>
      <td>Runs golden intent datasets with custom G-Eval rubrics to catch routing and answer-quality regressions.</td>
    </tr>
  </tbody>
</table>

Fincent is architected for robustness across local and hosted deployments. See
[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) for specifics.

---

## Installation and Usage

Install from source on your machine or deploy your own Docker Space on Hugging Face.
To use Fincent without installing anything, open the preinstalled Space at
[huggingface.co/spaces/nssundar/fincent](https://huggingface.co/spaces/nssundar/fincent).

### Prerequisites
The API keys used by Fincent are documented in `.env.example`: look at that for the specific names. **Locally**, set them in `.env` or export them in your shell. **On Hugging Face**, create the same
names under the Space **Settings → Secrets and variables**; store tokens as Secrets, not
public variables.

**API keys**:

- **`OPENAI_API_KEY`** — **Required.** Chat models and `/query` moderation.
- **`FMP_ACCESS_TOKEN`** — **Optional** (alias: **`FMP_API_KEY`**). Same value unlocks the FMP MCP server and OpenBB routes that use the FMP provider.
- **`TAVILY_API_KEY`** — **Optional.** Native Tavily search/extract when enabled in config.
- **`ALPHA_VANTAGE_API_KEY`** — **Optional.** Alpha Vantage MCP tools when that server is turned on (off by default on the stock Python 3.11 Docker image).

**Data storage.** By default the app writes the LangGraph checkpoint database, FAISS
vector store, and runtime portfolio JSON under **`/data`** (`checkpoints.sqlite`,
`vector_db/`, `portfolio/`). **Locally**, create a writable `/data` or point
`FINCENT__CHECKPOINTER__PATH`, `FINCENT__RAG__VECTOR_DB_PATH`, and related settings
elsewhere. **On Hugging Face**, the container uses the same paths; enable Persistent
Storage on the Space if you want `/data` to survive rebuilds.

**Local-only.
** **Python 3.11+** is required for a from-source install (the published
Space runs the bundled Docker image instead).

### Local Install

```bash
cd fincent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# At minimum set OPENAI_API_KEY; add optional keys from Prerequisites.

./run_local.sh
```

`./run_local.sh` starts FastAPI and Streamlit on `API_PORT` and `UI_PORT` (defaults
**8000** and **8501**). Open **http://localhost:8501** for Streamlit; the API lives at
**http://localhost:8000** by default. Common endpoints include `GET /health`,
`GET /rag/status`, `POST /query`, `GET /history/{thread_id}`, `POST /reset/{thread_id}`,
`POST /fincent/invoke`, and `POST /fincent/stream`.

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is an ETF?", "session_id": "demo"}'
```

The first boot can take a while while RAG ingestion builds the FAISS index; later runs
reuse it.

### Hugging Face Spaces Install
 
Create a **Docker** Space, connect this repository, and add the API keys from
**Prerequisites** as Secrets. Turn on **Persistent Storage** if `/data` should
outlast image rebuilds. Streamlit binds **`$PORT`** (often **7860**); FastAPI listens
on **`API_PORT`** (default **8000**) inside the container. The entrypoint brings Streamlit up
quickly so Hugging Face sees a listener while RAG ingestion finishes—the UI may show
the backend as warming until `GET /health` succeeds. The image ships `openbb-mcp`,
`python -m fmp.server`, lazy creation of `/data`, and `pypdf` for PDFs; Docker defaults
**`FINCENT__SERVER__STARTUP_HEALTH_WAIT_SECONDS`** to **900** so first-boot ingestion can
complete.

---

## Architecture

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
                       +------+--+   +---+-----------------------+
                       | Central |   | Q&A / Portfolio /        |
                       | answer  |   | Market Research /        |
                       |         |   | Goal Planning agents     |
                       +------+--+   +---+-----------------------+
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

Routing is shared by the Q&A and Portfolio tabs. The Portfolio tab is not
hard-pinned to the Portfolio agent; generic financial questions still route
to Q&A, and personal portfolio questions asked in the Q&A tab still route to
Portfolio. The Market Research tab pins `intent_hint="market_research"` so
company, security, filing, and investment-theme questions go directly to the
Market Research agent. The Goal Planning tab pins `intent_hint="goal_planning"`
so retirement, college, housing, vacation, and stress-test questions use the
Goal Planning specialist with a separate conversation thread from the other
tabs.

Query classes:

| Query class | Example | Handler |
|---|---|---|
| App identity | "Who are you?" | Central direct answer |
| App features | "What tools do you use?" | Central direct answer |
| Chit-chat | "Thanks for the help." | Central direct answer |
| Out of scope | "What's the weather today?" | Central refusal/redirect |
| Generic finance | "What is an ETF?" | Q&A agent |
| Personal portfolio | "Am I over-concentrated in AAPL?" | Portfolio agent |
| Market research | "Is Nvidia a good investment?" | Market Research agent |
| Market/security risk | "Compare the risks of bond X vs ETF Y" | Market Research agent |
| Goal planning | "If my portfolio drops 25% in a recession, how many extra years might I need to work?" | Goal Planning agent |

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
- `eval.market_research.json`
- `eval.goal_planning.json`

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

The pytest runner loads all golden cases, starts Portfolio, Market
Research, and Goal Planning FastMCP sessions once for the test session,
compiles the LangGraph once, and then runs each case through the full workflow
with a fresh `session_id`. Before invoking DeepEval, it asserts deterministic routing
expectations: central-handled flag, primary intent, expected agents, and
absence of agent errors.

DeepEval metrics include:

- **G-Eval** as the primary intent-aware task success metric. It compares
  actual output with expected output while allowing harmless wording
  differences.
- **Fincent Helpful Relevancy** as a custom G-Eval relevancy rubric that
  allows useful examples, calculations, citations, and brief related offers.
- **FaithfulnessMetric** when retrieval context is actual model-visible
  retrieval context.
- **ContextualRecallMetric** when retrieval context is actual model-visible
  retrieval context.
- **ContextualRelevancyMetric** when retrieval context is actual
  model-visible retrieval context.

Fincent uses the custom helpful relevancy rubric instead of DeepEval's generic
answer relevancy metric because examples, source sections, small tables, and
brief reminders of Fincent's capabilities are often desirable. The contextual
metrics are opt-in because golden snippets are often hints, not the exact
context seen by the model during the live workflow.

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
utilities, portfolio data handling, and goal-planning math/MCP helpers.

---

## Agents

### Central Agent

The central agent is the planner, direct responder, and aggregator. It
classifies requests into intents such as `app_identity`, `app_features`,
`chit_chat`, `out_of_scope`, `qna`, `portfolio`, `market_research`, and
`goal_planning`.

For central-handled intents, it answers directly using app metadata from
`config.yaml`. For specialist intents, it routes to Q&A, Portfolio, Market
Research, Goal Planning, or a fan-out combination when the routing plan calls for it, then
aggregates the specialist responses.

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

### Market Research Agent

The Market Research agent handles non-personal investment research about
public companies, securities, sectors, business risks, financial statements,
and investment themes such as AI. Example questions include "Is Nvidia a
good investment?", "Compare Procter & Gamble with Unilever", "What are the
risks of investing in Tesla?", and "What is the best AI investment today?"

The agent runs a ReAct tool-calling loop and can use MCP tools for:

- **OpenBB**: general financial data, quotes, historical prices, ETF data,
  company news, and economic context when a more specialized tool does not
  fit.
- **Alpha Vantage**: technical indicators such as RSI and MACD, plus Alpha
  Intelligence sentiment and market news signals.
- **Tavily**: current web/news search for recent company developments,
  AI investment announcements, competitive context, and market commentary.
- **Financial Modeling Prep (FMP)**: company fundamentals, statements,
  ratios, filings, and 10-K risk analysis. The Market Research prompt uses
  FMP filings to summarize the top three company risks when available.
  With an FMP Starter (or higher) key configured, Fincent keeps FMP MCP tools
  enabled and exposes OpenBB tools that route to paid FMP fundamentals,
  estimates, and supported historical data. To force free-tier behavior, set
  `FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS=false`; to hide a noisy direct
  FMP MCP tool family, add substrings to
  `market_research.fmp_exclude_tool_name_substrings` in `config.yaml`.

Market Research MCP sessions are started during FastAPI lifespan and reused
for the process. API keys are read from environment variables rather than
stored in `config.yaml`. **`FMP_ACCESS_TOKEN` (and aliases) is copied
into the isolated OpenBB MCP `user_settings.json` as `fmp_api_key`**, because
OpenBB fundamental routes use provider `fmp` and do not read `FMP_ACCESS_TOKEN`
from the environment by default. Intrinio is paid-only and is disabled by
default; the OpenBB tool **`equity_fundamental_reported_financials`** is omitted
unless `FINCENT_OPENBB_ALLOW_INTRINIO=true` and an Intrinio key are both set.
If one of the optional MCP servers is unavailable
or its key is missing, that server is skipped and the agent continues with
the tools that did load.

### Goal Planning Agent

The Goal Planning agent helps users connect their **portfolio snapshot** and
savings assumptions to long-horizon goals: retirement income targets, college
funding, home purchases, discretionary spending (e.g. vacations), and **stress
tests** (e.g. large portfolio drawdowns). Answers are **educational** only,
not personalized financial, tax, or legal advice.

The agent runs a ReAct tool-calling loop and can use:

- **Internal MCP tools** (`python -m src.agents.goal_planning.mcp_server`):
  deterministic time-value-of-money helpers, bounded **Monte Carlo**
  simulation (NumPy), and a compact **portfolio summary** resource aligned
  with the Portfolio agent's snapshot.
- **OpenBB MCP** (optional, `economy` category in `config.yaml`): CPI,
  Treasury yields, and related macro context when enabled.

Goal Planning MCP sessions start during FastAPI lifespan alongside the other
specialists. Monte Carlo size and projection horizons are capped in
`config.yaml` under `goal_planning` so interactive use and Spaces stay
responsive.

---

## Repository Layout

```text
fincent/
  src/
    agents/
      central/      # Planner, direct answers, aggregator
      market_research/ # Market/company research with OpenBB/Alpha/Tavily/FMP tools
      goal_planning/   # Goal planning specialist: TVM, Monte Carlo, OpenBB economy
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
`FINCENT__SECTION__KEY` environment variable convention. When the primary
chat model hits an OpenAI **rate limit** (`RateLimitError`), Fincent
retries with **`llm.rate_limit_fallback_model`** (default **`gpt-5.4`** in
`config.yaml`). Clear `FINCENT__LLM__RATE_LIMIT_FALLBACK_MODEL` to disable.

The supported environment variables are documented in `.env.example`. You can copy and edit it for your install.
