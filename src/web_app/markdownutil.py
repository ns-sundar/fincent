"""Streamlit markdown helpers.

Streamlit treats ``$...$`` as TeX; currency like ``$270.08`` breaks layout.
Some models emit Unicode asterisk operators (U+2217) that break ``**bold**``.
"""

from __future__ import annotations

import re


def sanitize_streamlit_markdown(text: str) -> str:
    """Make assistant/user text safe for ``st.markdown`` (currency + asterisks)."""
    if not text:
        return text
    s = text.replace("\u2217", "*").replace("\u204e", "*")
    # TeX inline math: escape $ when it starts a number (currency).
    s = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", s)
    return s
