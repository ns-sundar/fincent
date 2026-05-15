"""Tests for narrow third-party log suppression."""

from __future__ import annotations

import logging

from src.utils.logging import _SuppressNoisyThirdPartyWarnings


def _record(name: str, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_suppresses_transformers_image_processing_path_alias_warning() -> None:
    filt = _SuppressNoisyThirdPartyWarnings()
    rec = _record(
        "transformers",
        "Accessing `__path__` from `.models.pi0.image_processing_pi0`. "
        "Returning `__path__` instead. Behavior may be different and this alias "
        "will be removed in future versions.",
    )
    assert not filt.filter(rec)


def test_keeps_other_transformers_warnings() -> None:
    filt = _SuppressNoisyThirdPartyWarnings()
    rec = _record("transformers", "Some other warning")
    assert filt.filter(rec)
