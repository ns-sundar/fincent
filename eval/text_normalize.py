"""Normalize model answers for eval comparison.

LLMs often wrap phrases in Markdown (**bold**, *italic*) while dataset
``expected_output`` strings are plain text. Q&A answers may also include
mandatory citations and a ``## Sources`` section. DeepEval metrics can treat
those surface differences as mismatches even when the underlying claims match.
This module strips presentation-only differences so judges compare content,
not markup or citation boilerplate."""

from __future__ import annotations

import re
import unicodedata


def normalize_answer_for_eval(text: str) -> str:
    """Strip markdown-ish formatting and normalize typography.

    - Removes common Markdown emphasis (``**bold**``, `*italic*`, ``__bold__``).
    - Removes Q&A citation markers and trailing ``## Sources`` sections.
    - Normalizes Unicode dashes and quotes to ASCII equivalents.
    - Collapses redundant whitespace.

    Does not rewrite words (e.g. "your" vs "the"); those remain for the
    LLM judge, which already allows harmless wording differences.
    """
    if not text:
        return ""

    # Unicode normalization (compat so NFKC folds some compat chars).
    s = unicodedata.normalize("NFKC", text.strip())

    # Typography → ASCII (en/em dash, minus, curly quotes).
    s = (
        s.replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .replace("\u2212", "-")  # minus sign
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )

    # Markdown bold / italic (non-greedy, single-line segments).
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)
    s = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", s)

    # Q&A answers include citation markers and a trailing Sources section by
    # product design. Eval goldens focus on answer substance, not citations.
    s = re.sub(r"(?im)^\s*##\s+Sources\s*$.*", "", s, flags=re.DOTALL)
    s = re.sub(r"\[(?:\d+)(?:\]\s*\[\d+)*\]", "", s)

    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
