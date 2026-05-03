# Fincent Evals

This directory contains intent-based end-to-end eval datasets and a
DeepEval runner for the Fincent workflow.

## Install

Runtime installs should continue to use only production dependencies:

```bash
pip install -r requirements.txt
```

Eval installs add the eval-only dependency file:

```bash
pip install -r requirements.txt -r requirements-eval.txt
```

`requirements-eval.txt` is intentionally not used by the HuggingFace
Dockerfile.

## Datasets

Each `eval.*.json` file is a JSON array of cases with:

- `input`: user query
- `expected_output`: reference answer for judge metrics
- `retrieval_context`: grounding snippets, if available
- `additional_metadata.intent`: expected router intent
- `additional_metadata.handled_by_central`: expected routing mode
- `additional_metadata.expected_agents`: expected responding agents
- `additional_metadata.context_type`: optional context source label. It
  defaults to `golden_hint`. Set it to `actual_retrieval` only when
  `retrieval_context` contains the actual context the model saw.

Current intent datasets:

- `eval.app-identity.json`
- `eval.app-features.json`
- `eval.chit-chat.json`
- `eval.out-of-scope.json`
- `eval.qna.json`
- `eval.portfolio.json`

## How The Runner Works

`test_e2e_intents.py` runs the live Fincent workflow with:

```python
graph = build_graph()
run_query(QueryRequest(query=case.input, session_id=...), graph=graph)
```

The graph is compiled once per pytest session and reused for all cases.
Each case gets a unique `session_id` so chat state does not leak between
eval rows.

The portfolio FastMCP stdio servers are also started once at eval session
startup and stopped once at teardown. This mirrors the FastAPI lifespan
behavior and avoids respawning OpenBB / Fincent RAG MCP servers for each
test case.

For every case, the runner first checks deterministic routing:

- `plan.handled_by_central`
- first routed intent
- responding agent list
- absence of agent error metadata

Then it creates a DeepEval `LLMTestCase` from the final answer and runs
LLM-judge metrics. **Before** those metrics, `actual_output` and
`expected_output` are passed through ``eval.text_normalize.normalize_answer_for_eval``:
Markdown emphasis (e.g. ``**bold**``) and common typography (Unicode dashes
and curly quotes), citation markers, and trailing ``## Sources`` sections
are stripped so judges compare substance, not formatting.

Each `LLMTestCase` is named as `<intent>: <input>` and tagged with the
intent, dataset file, and source. Pytest parametrization also uses a
slug derived from the input so `pytest -v` output is readable. When a
DeepEval assertion fails, the raised error appends the input, expected
output, and actual output for quick diagnosis.

## Metrics

The primary metric is a custom `GEval` rubric named
`Fincent Intent Task Success`. It checks task completion, expected-answer
agreement, entity/number/date preservation, intent behavior, and whether
the answer avoids unsupported claims. It receives retrieval context only
when `additional_metadata.context_type` is `actual_retrieval`.

All cases also use a custom `GEval` rubric named
`Fincent Helpful Relevancy`. This replaces DeepEval's generic
`AnswerRelevancyMetric` because Fincent's desired behavior allows helpful
examples, calculations, citations, small tables, and brief related offers
when they support the user's question. It also allows chit-chat responses to
include a short reminder of what Fincent can help with.

Cases with non-empty `retrieval_context` additionally use contextual
metrics only when `additional_metadata.context_type` is
`actual_retrieval`:

- `FaithfulnessMetric`
- `ContextualRecallMetric`
- `ContextualRelevancyMetric`

This avoids judging faithfulness or contextual relevance against golden
hint snippets that the model did not actually see during the live
end-to-end workflow.

## Run

Set a real OpenAI key for both Fincent model calls and DeepEval judge
calls:

```bash
export OPENAI_API_KEY=...
```

Run with DeepEval:

```bash
deepeval test run eval/test_e2e_intents.py
```

You can also run the same file with pytest:

```bash
python -m pytest eval/test_e2e_intents.py
```

If `OPENAI_API_KEY` is missing, the eval tests are skipped.
