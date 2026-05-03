"""DeepEval metric selection for Fincent intent evals."""

from __future__ import annotations

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

from eval.datasets import EvalCase


_CONTEXT_METRIC_TYPES = {"actual_retrieval"}


TASK_SUCCESS_CRITERIA = """\
Evaluate whether the actual output satisfies the Fincent eval case.

The answer should:
- Address the user's input directly.
- Match the expected output in meaning, allowing harmless wording differences.
- Treat optional qualifiers in expected outputs (for example "may", "can", or
  illustrative examples) as acceptable context, not mandatory facts that must
  appear verbatim in every answer.
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


HELPFUL_RELEVANCY_CRITERIA = """\
Evaluate whether the actual output is relevant and helpful for this Fincent
interaction.

The expected intent for this case is: {intent}

A good answer should:
- Directly answer or appropriately respond to the user's main input.
- Stay within Fincent's allowed scope for the expected intent.
- For qna, allow concise examples, calculations, analogies, citations, source
  sections, or step-by-step explanation when they help clarify the answer.
- For qna, allow a brief offer to answer related finance or economics questions
  after the main answer.
- For chit_chat, allow a warm, brief response plus a short reminder of what
  Fincent can help with, such as finance, economics, or portfolio analysis.
- For app_identity, allow concise expanded examples of Fincent's supported
  domain, including stocks, bonds, cash, ETFs, mutual funds, portfolio theory,
  investment risk, market trading, brokers, and general tax topics.
- For app_features, accept and expect answers that name specific configured
  tool integrations (such as OpenBB, fincent-rag), data-access behaviors,
  version, authorship, or other capabilities drawn from the app metadata.
  Tool integrations are first-class app features; do not penalize an answer
  merely because it names backend or infrastructure tools. Penalize only
  long unrelated feature dumps or capabilities that were not asked about.
- For portfolio, allow account identifiers, ticker symbols, transaction ids,
  small tables, and examples from the user's data when they help answer the
  question.
- For out_of_scope, treat a polite refusal plus a brief redirect to finance,
  economics, or portfolio topics as relevant and helpful.

Do not penalize helpful examples or related offers merely because they go beyond
a one-sentence definition. Penalize responses that mostly change the topic, omit
the main answer, introduce unsupported facts, provide personalized financial or
tax advice, or answer an out-of-scope factual question instead of refusing it.
"""


def task_success_metric(case: EvalCase) -> GEval:
    """Primary intent-aware answer quality metric."""

    params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
    if case.retrieval_context and case.context_type in _CONTEXT_METRIC_TYPES:
        params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)

    return GEval(
        name="Fincent Intent Task Success",
        criteria=TASK_SUCCESS_CRITERIA,
        evaluation_params=params,
        threshold=0.75,
    )


def helpful_relevancy_metric(case: EvalCase) -> GEval:
    """Product-specific relevancy metric that allows helpful Fincent behavior."""

    return GEval(
        name="Fincent Helpful Relevancy",
        criteria=HELPFUL_RELEVANCY_CRITERIA.format(intent=case.intent.value),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )


def metrics_for_case(case: EvalCase) -> list[object]:
    """Return DeepEval metrics appropriate for one dataset row."""

    metrics: list[object] = [task_success_metric(case), helpful_relevancy_metric(case)]

    # Only use context metrics when retrieval_context is the actual context the
    # model saw. Golden hints and synthetic portfolio snippets are useful for
    # G-Eval, but DeepEval's contextual metrics are noisy against those stubs.
    if case.retrieval_context and case.context_type in _CONTEXT_METRIC_TYPES:
        metrics.extend(
            [
                FaithfulnessMetric(threshold=0.7),
                ContextualRecallMetric(threshold=0.7),
                ContextualRelevancyMetric(threshold=0.7),
            ]
        )

    return metrics
