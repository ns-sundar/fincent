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
    tools: List["AppTool"] = Field(default_factory=list)


class AppTool(BaseModel):
    """Tool/integration metadata exposed to the central app-features path."""

    name: str
    description: str = ""


class LLMConfig(BaseModel):
    """Settings for the chat-LLM backend."""

    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: int = 60
    # If the primary ``model`` raises an OpenAI rate limit, retry the request
    # with this model (LangChain ``with_fallbacks``). Disable with null or
    # ``FINCENT__LLM__RATE_LIMIT_FALLBACK_MODEL=`` (empty).
    rate_limit_fallback_model: Optional[str] = "gpt-5.4"


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
    portfolio: AgentToggle = AgentToggle(enabled=True)
    market_research: AgentToggle = AgentToggle(enabled=True)


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


class McpServerConfig(BaseModel):
    """Settings for the MCP server that wraps the FAISS vector_db as a tool.

    The server exposes a single tool (``rag_search``) over the MCP
    protocol. It is opt-in; the in-process Q&A agent uses the same
    underlying search function directly regardless.
    """

    enabled: bool = False
    # "stdio" or "streamable-http".
    #   stdio            - client-spawned local servers (Claude
    #                      Desktop, Cursor).
    #   streamable-http  - long-lived HTTP server with bidirectional
    #                      streaming on the /mcp endpoint. This is the
    #                      successor to the deprecated SSE transport
    #                      in the MCP spec / Python SDK.
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765
    # Name the tool advertises to MCP clients.
    tool_name: str = "rag_search"


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
    # Retrieval fan-out at query time.
    top_k: int = 5
    # MMR (Maximal Marginal Relevance) re-ranking trades some similarity
    # for diversity so the top-k is not three near-duplicate chunks.
    use_mmr: bool = True
    # Candidate pool size for MMR before re-ranking down to top_k.
    mmr_fetch_k: int = 20
    # Diversity / relevance knob in [0, 1]. 1.0 == pure similarity,
    # 0.0 == maximise diversity only.
    mmr_lambda: float = 0.5
    # MCP sidecar that publishes the vector store as a tool.
    mcp_server: McpServerConfig = McpServerConfig()


class PortfolioMcpServerSpec(BaseModel):
    """Spawn spec for a single MCP tool server consumed by the Portfolio agent.

    Only the stdio transport is wired today: the Portfolio agent's
    MCP client (see :mod:`src.agents.portfolio.mcp_tools`) launches a
    subprocess per tool invocation, so the server advertises its
    tool list on stdin/stdout. Keep ``command`` resolvable on
    ``$PATH`` inside whichever environment the FastAPI process runs
    in (HF Spaces, local venv, etc.).
    """

    enabled: bool = False
    command: str = ""
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class PortfolioToolsConfig(BaseModel):
    """Container for the Portfolio agent's MCP tool server specs.

    Each entry describes one stdio MCP server. If every entry is
    ``enabled: false`` the agent falls back to the legacy direct-LLM
    answer path (no tool calls). Missing entries are treated as
    disabled so operators can drop whole sections in ``config.yaml``
    without breaking validation.
    """

    # OpenBB Platform MCP server -- real-world financial data tools.
    openbb: PortfolioMcpServerSpec = PortfolioMcpServerSpec()
    # Fincent's own RAG MCP server (FAISS vector_db as an MCP tool).
    rag: PortfolioMcpServerSpec = PortfolioMcpServerSpec()


class PortfolioConfig(BaseModel):
    """Settings for the Portfolio agent.

    The agent reads the user's portfolio from JSON files under
    ``data_path`` at runtime. On first startup the application seeds
    that directory by copying every JSON file from ``seed_path`` (a
    repo-relative read-only snapshot shipped with the code). Once
    seeded, subsequent restarts leave ``data_path`` untouched so any
    future in-app edits to the user's portfolio are preserved.
    """

    # Writable runtime location -- HF Spaces mounts /data; local dev
    # either mounts it too or overrides this path.
    data_path: str = "/data/portfolio"

    # Read-only seed shipped in the repo. Relative paths resolve
    # against the project root.
    seed_path: str = "data/default_portfolio"

    # MCP tool servers advertised to the Portfolio agent at answer
    # time. Loaded lazily on first use and cached for the rest of the
    # process lifetime.
    tools: PortfolioToolsConfig = PortfolioToolsConfig()


class MarketResearchToolsConfig(BaseModel):
    """Container for Market Research MCP tool server specs."""

    # OpenBB Platform MCP server -- general financial data fallback.
    openbb: PortfolioMcpServerSpec = PortfolioMcpServerSpec()
    # Alpha Vantage MCP server -- technical indicators and sentiment.
    alpha_vantage: PortfolioMcpServerSpec = PortfolioMcpServerSpec()
    # Tavily MCP server -- current web/news search.
    tavily: PortfolioMcpServerSpec = PortfolioMcpServerSpec()
    # Financial Modeling Prep MCP server -- filings and company fundamentals.
    fmp: PortfolioMcpServerSpec = PortfolioMcpServerSpec()


class MarketResearchConfig(BaseModel):
    """Settings for the Market Research agent."""

    tools: MarketResearchToolsConfig = MarketResearchToolsConfig()
    # FMP MCP tool names containing any of these substrings (case-insensitive)
    # are dropped from the FMP MCP tool list. The default keeps all tools so
    # configured Starter-plan keys can use paid FMP endpoints.
    fmp_exclude_tool_name_substrings: List[str] = Field(
        default_factory=list,
    )


class AppConfig(BaseModel):
    """Top-level configuration object."""

    app: AppInfo = AppInfo()
    llm: LLMConfig = LLMConfig()
    server: ServerConfig = ServerConfig()
    ui: UIConfig = UIConfig()
    agents: AgentsConfig = AgentsConfig()
    checkpointer: CheckpointerConfig = CheckpointerConfig()
    rag: RagConfig = RagConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    market_research: MarketResearchConfig = MarketResearchConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

_ENV_PREFIX: str = "FINCENT__"
_DOTENV_LOADED: bool = False


def _maybe_load_dotenv() -> None:
    """Load repo-root ``.env`` into the process environment once.

    This makes keys like ``FMP_ACCESS_TOKEN`` available when operators use
    ``python -m uvicorn ...`` without ``run_local.sh``, and complements
    ``run_local.sh`` which also sources ``.env``. Values already present in
    ``os.environ`` are not overwritten (``override=False``).
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _default_config_path().parent.resolve() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


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
    _maybe_load_dotenv()
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


def reset_dotenv_loaded_flag() -> None:
    """Reset the one-shot dotenv guard (tests only)."""

    global _DOTENV_LOADED
    _DOTENV_LOADED = False
