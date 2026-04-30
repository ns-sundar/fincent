"""End-to-end DeepEval tests for Fincent intent datasets."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Coroutine

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("deepeval")
from deepeval import assert_test  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from eval.datasets import EvalCase, case_ids, load_eval_cases  # noqa: E402
from eval.metrics import metrics_for_case  # noqa: E402
from eval.text_normalize import normalize_answer_for_eval  # noqa: E402
from src.agents.portfolio.mcp_tools import (  # noqa: E402
    start_portfolio_mcp_sessions,
    stop_portfolio_mcp_sessions,
)
from src.core.schemas import QueryRequest  # noqa: E402
from src.workflow.graph import build_graph, run_query  # noqa: E402


CASES = load_eval_cases()


class _BackgroundEventLoop:
    """Run async MCP lifespan hooks on a loop that stays alive during evals."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="fincent-eval-mcp-loop",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()


def _has_real_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key != "test-key-not-real")


pytestmark = pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="DeepEval e2e tests require a real OPENAI_API_KEY.",
)


@pytest.fixture(scope="session")
def portfolio_mcp_sessions():
    """Keep FastMCP stdio servers alive for the full eval session."""

    background_loop = _BackgroundEventLoop()
    background_loop.start()
    background_loop.run(start_portfolio_mcp_sessions())
    try:
        yield
    finally:
        background_loop.run(stop_portfolio_mcp_sessions())
        background_loop.stop()


@pytest.fixture(scope="session")
def eval_graph(portfolio_mcp_sessions):
    """Compile the LangGraph workflow once and reuse it for all eval cases."""

    return build_graph()


def _assert_routing(case: EvalCase, response) -> None:
    """Assert deterministic routing expectations before LLM-judge metrics."""

    assert response.plan.handled_by_central is case.handled_by_central
    assert response.plan.intents, "router returned no intents"
    assert response.plan.intents[0] == case.intent

    observed_agents = [
        agent_response.agent.value
        for agent_response in response.agent_responses
    ]
    assert observed_agents == case.expected_agents
    assert not any(
        agent_response.metadata.get("error")
        for agent_response in response.agent_responses
    )


def _case_report(case: EvalCase, actual_output: str) -> str:
    """Format the high-signal case fields for failure output."""

    return (
        f"\n\nDeepEval case: {case.display_name}\n"
        f"Dataset: {case.dataset}\n"
        f"Input:\n{case.input}\n\n"
        f"Expected output:\n{case.expected_output}\n\n"
        f"Actual output:\n{actual_output}\n"
    )


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_fincent_intent_e2e(case: EvalCase, eval_graph) -> None:
    """Run one dataset row through the full workflow and evaluate the answer."""

    session_id = f"eval-{case.case_id}-{uuid.uuid4()}"
    response = run_query(
        QueryRequest(query=case.input, session_id=session_id),
        graph=eval_graph,
    )

    _assert_routing(case, response)
    assert response.answer.strip()

    test_case = LLMTestCase(
        name=case.display_name,
        tags=[case.intent.value, case.dataset, case.source],
        input=case.input,
        # Normalize so Markdown/typography in model output does not
        # penalize otherwise-correct answers vs plain-text expectations.
        actual_output=normalize_answer_for_eval(response.answer),
        expected_output=normalize_answer_for_eval(case.expected_output),
        retrieval_context=case.retrieval_context,
    )
    try:
        assert_test(test_case, metrics_for_case(case))
    except AssertionError as exc:
        raise AssertionError(str(exc) + _case_report(case, response.answer)) from exc
