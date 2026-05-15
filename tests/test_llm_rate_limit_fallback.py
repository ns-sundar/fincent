"""Tests for LLM rate-limit fallback wiring."""

from __future__ import annotations

import pytest
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_openai import ChatOpenAI

from src.core.config import reset_config_cache
from src.core.llm import build_chat_model


@pytest.fixture(autouse=True)
def _reset_cfg():
    reset_config_cache()
    yield
    reset_config_cache()


def test_build_chat_model_wraps_primary_with_rate_limit_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    reset_config_cache()
    m = build_chat_model()
    assert isinstance(m, RunnableWithFallbacks)


def test_build_chat_model_plain_chat_when_primary_equals_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    reset_config_cache()
    m = build_chat_model(model="gpt-5.4")
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "gpt-5.4"


def test_build_chat_model_no_fallback_when_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("FINCENT__LLM__RATE_LIMIT_FALLBACK_MODEL", "")
    reset_config_cache()
    m = build_chat_model()
    assert isinstance(m, ChatOpenAI)
