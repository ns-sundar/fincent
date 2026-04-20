"""YAML-backed application configuration with env-var overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------


class AppInfo(BaseModel):
    """Static metadata about the application itself."""

    name: str = "Fincent"
    version: str = "0.1.0"
    description: str = ""
    about: str = ""


class LLMConfig(BaseModel):
    """Settings for the chat-LLM backend."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: int = 60


class ServerConfig(BaseModel):
    """FastAPI / LangServe server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    graph_path: str = "/fincent"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    # Container / HF Spaces: scripts/docker-entrypoint.sh and Dockerfile HEALTHCHECK
    # should stay aligned with these values (see README).
    startup_health_wait_seconds: int = 300
    healthcheck_interval_seconds: int = 90


class UIConfig(BaseModel):
    """Streamlit UI settings."""

    api_base_url: str = "http://localhost:8000"
    page_title: str = "Fincent"
    page_icon: str = "FC"


class AgentToggle(BaseModel):
    """Per-agent enable flag and human-readable description."""

    enabled: bool = False
    description: str = ""
    max_fanout: int = 3


class AgentsConfig(BaseModel):
    """Container for every agent toggle entry."""

    central: AgentToggle = AgentToggle(enabled=True, max_fanout=3)
    qna: AgentToggle = AgentToggle(enabled=True)
    agent_two: AgentToggle = AgentToggle(enabled=False)
    agent_three: AgentToggle = AgentToggle(enabled=False)
    agent_four: AgentToggle = AgentToggle(enabled=False)


class LoggingConfig(BaseModel):
    """Logging knobs."""

    level: str = "INFO"
    file: Optional[str] = None


class CheckpointerConfig(BaseModel):
    """Settings for the LangGraph checkpointer."""

    backend: str = "sqlite"
    # Absolute path (e.g. /data/... on disk root), never ./data in the repo.
    # Or ":memory:" for tests.
    path: str = "/data/checkpoints.sqlite"


class RagConfig(BaseModel):
    """Settings for Retrieval-Augmented Generation (Q&A agent).

    See ``config.yaml`` for the narrative description of each field.
    """

    enabled: bool = True
    articles_path: str = "rag/fincent_rag_articles.json"
    # Absolute path: host provides /data (HF Spaces mount / local sudo mkdir).
    vector_db_path: str = "/data/vector_db"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    request_timeout: int = 60
    # Realistic Chrome UA -- many gov / finance sites 403 non-browser UAs.
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    top_k: int = 4


class AppConfig(BaseModel):
    """Top-level configuration object."""

    app: AppInfo = AppInfo()
    llm: LLMConfig = LLMConfig()
    server: ServerConfig = ServerConfig()
    ui: UIConfig = UIConfig()
    agents: AgentsConfig = AgentsConfig()
    checkpointer: CheckpointerConfig = CheckpointerConfig()
    rag: RagConfig = RagConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

_ENV_PREFIX: str = "FINCENT__"


def _default_config_path() -> Path:
    """Resolve the default location of `config.yaml` (project root)."""
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `overrides` into `base` (overrides win)."""
    out = dict(base)
    for key, value in overrides.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: str) -> Any:
    """Best-effort string-to-scalar coercion for env-var overrides."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _env_overrides() -> Dict[str, Any]:
    """Build a nested-dict overlay from `FINCENT__SECTION__KEY` env vars."""
    overlay: Dict[str, Any] = {}
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        path = env_key[len(_ENV_PREFIX) :].lower().split("__")
        cursor = overlay
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw_value)
    return overlay


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load YAML config from disk and overlay environment overrides.

    Args:
        config_path: Optional explicit path; defaults to the repo's
            top-level `config.yaml`.

    Returns:
        A validated `AppConfig` instance.
    """
    path = Path(config_path) if config_path else _default_config_path()
    raw: Dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file {path} must contain a mapping.")
            raw = loaded
    merged = _deep_merge(raw, _env_overrides())
    return AppConfig(**merged)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached accessor for the application config."""
    return load_config()


def reset_config_cache() -> None:
    """Clear the cached config (useful for tests)."""
    get_config.cache_clear()
