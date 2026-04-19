"""LLM provider factory.

We isolate model construction here so every agent can ask for a chat
model without knowing about provider-specific arguments. Today only
OpenAI is wired up; other providers can be added by branching on
``cfg.llm.provider``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.core.config import LLMConfig, get_config


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM cannot be constructed (e.g. missing API key)."""


def _require_openai_key() -> str:
    """Return the OPENAI_API_KEY env var or raise a helpful error."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it or add it to a local .env file before starting."
        )
    return key


def build_chat_model(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[int] = None,
) -> BaseChatModel:
    """Construct a chat model using config defaults plus optional overrides.

    Args:
        model: Override the configured model name.
        temperature: Override the configured sampling temperature.
        max_tokens: Override the per-call max output tokens.
        request_timeout: Override the per-call timeout in seconds.

    Returns:
        A LangChain ``BaseChatModel`` ready for ``.invoke``/``.ainvoke``.
    """
    cfg: LLMConfig = get_config().llm

    if cfg.provider.lower() != "openai":
        raise LLMConfigurationError(
            f"Unsupported LLM provider '{cfg.provider}'. Only 'openai' is wired."
        )

    return ChatOpenAI(
        api_key=_require_openai_key(),
        model=model or cfg.model,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=max_tokens or cfg.max_tokens,
        timeout=request_timeout or cfg.request_timeout,
    )


@lru_cache(maxsize=8)
def get_default_chat_model() -> BaseChatModel:
    """Cached convenience accessor for the default chat model."""
    return build_chat_model()
