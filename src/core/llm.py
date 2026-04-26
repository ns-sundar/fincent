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


# Runtime model override — None means "use whatever config.yaml says".
# Set via set_current_model(); read by get_default_chat_model().
_runtime_model: Optional[str] = None


def set_current_model(name: str) -> None:
    """Switch the default chat model to *name* without restarting.

    Clears the LRU cache so the next call to
    :func:`get_default_chat_model` builds a fresh instance with the new
    model.  All subsequent agent invocations in this process pick up the
    change immediately.
    """
    global _runtime_model  # noqa: PLW0603
    _runtime_model = name
    get_default_chat_model.cache_clear()


def get_current_model() -> str:
    """Return the model name that :func:`get_default_chat_model` will use.

    This is the runtime override if one has been set, otherwise the value
    from ``config.yaml``.
    """
    return _runtime_model or get_config().llm.model


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
    """Cached convenience accessor for the default chat model.

    Uses the runtime override set by :func:`set_current_model` when
    present, otherwise falls back to ``config.yaml``.  The cache is
    keyed by the Python process lifetime; call
    :func:`set_current_model` to invalidate it.
    """
    return build_chat_model(model=_runtime_model or None)
