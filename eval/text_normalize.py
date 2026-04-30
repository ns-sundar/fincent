"""Normalize model answers for eval comparison.

LLMs often wrap phrases in Markdown (**bold**, *italic*) while dataset
``expected_output`` strings are plain text. DeepEval metrics can treat
those surface differences as mismatches even when the underlying
claims match. This module strips presentation-only differences so
judges compare content, not markup."""

from __future__ import annotations

import re
import unicodedata


def normalize_answer_for_eval(text: str) -> str:
    """Strip markdown-ish formatting and normalize typography.

    - Removes common Markdown emphasis (``**bold**``, `*italic*`, ``__bold__``).
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

    s = re.sub(r"\s+", " ", s).strip()
    return s
