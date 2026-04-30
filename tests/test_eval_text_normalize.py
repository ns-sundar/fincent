"""Tests for eval answer normalization."""

from eval.text_normalize import normalize_answer_for_eval


def test_strips_markdown_bold():
    raw = "You own **200 shares of NVDA** in your **US Equities – Growth** account."
    out = normalize_answer_for_eval(raw)
    assert "**" not in out
    assert "200 shares of NVDA" in out
    assert "US Equities - Growth" in out


def test_normalizes_en_dash():
    assert normalize_answer_for_eval("a–b") == "a-b"


def test_plain_text_unchanged_up_to_whitespace():
    plain = "You own 200 shares of NVDA in the 'US Equities - Growth' account (ACC-001)."
    assert normalize_answer_for_eval(plain) == plain
