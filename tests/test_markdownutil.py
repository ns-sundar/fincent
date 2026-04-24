"""Tests for Streamlit markdown sanitization."""

from src.web_app.markdownutil import sanitize_streamlit_markdown


def test_sanitize_escapes_currency_dollar_before_digit():
    s = sanitize_streamlit_markdown("About $270.08 per share.")
    assert s == "About \\$270.08 per share."


def test_sanitize_normalizes_unicode_asterisk_operator():
    s = sanitize_streamlit_markdown("a\u2217b")
    assert s == "a*b"
