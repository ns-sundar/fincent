"""DeepEval metric selection for Fincent intent evals."""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

from eval.datasets import EvalCase
from src.core.schemas import Intent


_CENTRAL_POLICY_INTENTS = {Intent.CHIT_CHAT, Intent.OUT_OF_SCOPE}


TASK_SUCCESS_CRITERIA = """\
Evaluate whether the actual output satisfies the Fincent eval case.

The answer should:
- Address the user's input directly.
- Match the expected output in meaning, allowing harmless wording differences.
- Treat Markdown formatting (e.g. **bold**), list bullets, and minor typography
  (hyphen vs en-dash, straight vs curly quotes) as irrelevant; compare substance.
- Treat equivalent references to the same point in time as matching: e.g.
  "December 31, 2023" vs "2023-12-31" vs "at the end of 2023" when the context
  is clearly the same event (such as a year-end fee or a single dated transaction);
  likewise "January 10, 2022" vs "2022-01-10". Do not penalize for calendar
  expression style if amount and event type align with the expected answer.
- Preserve important numbers, dates, tickers, entities, and account/transaction details.
- For portfolio answers, omitting a parenthetical account id (e.g. ACC-001) when the
  account name and holding details are otherwise correct is acceptable.
- Follow the expected intent behavior: central refusals should refuse politely, chit-chat
  should be brief and warm, app answers should not invent features, and specialist answers
  should answer the finance or portfolio question.
- For chit-chat, a concise offer to help with finance or portfolio questions is acceptable
  and should not be penalized as irrelevant.
- For out-of-scope questions, a refusal plus a concise redirect/offer to help with finance,
  economics, or portfolio-related questions is the desired behavior.
- Avoid unsupported claims or personal facts not present in the expected output or context.
"""


def task_success_metric(case: EvalCase) -> GEval:
    """Primary intent-aware answer quality metric."""

    params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
    if case.retrieval_context:
        params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)

    return GEval(
        name="Fincent Intent Task Success",
        criteria=TASK_SUCCESS_CRITERIA,
        evaluation_params=params,
        threshold=0.75,
    )


def metrics_for_case(case: EvalCase) -> list[object]:
    """Return DeepEval metrics appropriate for one dataset row."""

    metrics: list[object] = [task_success_metric(case)]

    # Generic answer relevancy often penalizes policy-compliant central responses
    # for adding a short redirect/offer (e.g. "I can help with finance...").
    # For these intents, the custom G-Eval rubric is the source of truth.
    if case.intent not in _CENTRAL_POLICY_INTENTS:
        metrics.append(AnswerRelevancyMetric(threshold=0.7))

    if case.retrieval_context:
        metrics.extend(
            [
                FaithfulnessMetric(threshold=0.7),
                ContextualRecallMetric(threshold=0.7),
                ContextualRelevancyMetric(threshold=0.7),
            ]
        )

    return metrics
