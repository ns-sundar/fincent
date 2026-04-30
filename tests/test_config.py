"""Tests for the configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

from src.core.config import AppConfig, load_config


def test_load_default_config(monkeypatch):
    """The bundled config.yaml should parse cleanly into AppConfig."""
    monkeypatch.delenv("FINCENT__CHECKPOINTER__PATH", raising=False)
    monkeypatch.delenv("FINCENT__RAG__ENABLED", raising=False)
    from src.core.config import reset_config_cache

    reset_config_cache()
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.app.name == "Fincent"
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-5.4-mini"
    assert cfg.agents.qna.enabled is True
    assert cfg.agents.portfolio.enabled is True
    assert cfg.checkpointer.path == "/data/checkpoints.sqlite"
    assert cfg.rag.enabled is True
    assert cfg.rag.vector_db_path == "/data/vector_db"
    assert cfg.rag.chunk_size == 1000
    assert cfg.rag.chunk_overlap == 200
    assert cfg.rag.top_k == 5
    assert cfg.rag.use_mmr is True
    assert cfg.rag.mmr_fetch_k == 20
    assert cfg.rag.mmr_lambda == 0.5
    assert cfg.rag.mcp_server.enabled is False
    assert cfg.rag.mcp_server.transport == "stdio"
    assert cfg.rag.mcp_server.tool_name == "rag_search"
    assert cfg.server.startup_health_wait_seconds == 300
    assert cfg.server.healthcheck_interval_seconds == 90


def test_env_var_overrides(monkeypatch, tmp_path: Path):
    """FINCENT__SECTION__KEY env vars should overlay the YAML defaults."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "llm:\n  model: gpt-4o-mini\n  temperature: 0.1\n"
        "server:\n  port: 8000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINCENT__LLM__MODEL", "gpt-test")
    monkeypatch.setenv("FINCENT__SERVER__PORT", "9999")
    monkeypatch.setenv("FINCENT__LLM__TEMPERATURE", "0.7")

    cfg = load_config(yaml_path)
    assert cfg.llm.model == "gpt-test"
    assert cfg.server.port == 9999
    assert cfg.llm.temperature == 0.7


def test_missing_yaml_uses_defaults(tmp_path: Path):
    """An absent config file is fine; defaults apply."""
    cfg = load_config(tmp_path / "does-not-exist.yaml")
    assert cfg.app.name == "Fincent"
    assert cfg.server.port == 8000
