"""OpenBB MCP subprocess HOME + user_settings wiring."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.agents.portfolio.mcp_tools import (
    _PROJECT_ROOT,
    _ensure_fincent_openbb_mcp_home,
    _server_config,
)
from src.core.config import PortfolioMcpServerSpec


def test_project_root_is_fincent_not_src():
    assert ( _PROJECT_ROOT / "config.yaml").is_file()
    assert (_PROJECT_ROOT / "data" / "openbb_default_user_settings.json").is_file()
    assert not str(_PROJECT_ROOT).endswith(f"{os.sep}src")


def test_openbb_default_user_settings_prefers_yfinance():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "openbb_default_user_settings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    quote = data["defaults"]["commands"]["/equity/price/quote"]
    assert quote["provider"] == "yfinance"


def test_ensure_openbb_home_writes_fmp_api_key_from_env(
    tmp_path: Path, monkeypatch
) -> None:
    """``.env`` uses ``FMP_ACCESS_TOKEN``; OpenBB expects ``fmp_api_key`` in user_settings."""
    import src.agents.portfolio.mcp_tools as mcp_mod

    monkeypatch.setattr(mcp_mod, "_FINCENT_OPENBB_HOME", tmp_path)
    monkeypatch.setenv("FMP_ACCESS_TOKEN", "test-fmp-key-from-env")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    home = _ensure_fincent_openbb_mcp_home()
    assert Path(home) == tmp_path.resolve()
    settings_path = tmp_path / ".openbb_platform" / "user_settings.json"
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data["credentials"]["fmp_api_key"] == "test-fmp-key-from-env"


def test_server_config_openbb_injects_fincent_home_by_default():
    spec = PortfolioMcpServerSpec(
        enabled=True,
        command="openbb-mcp",
        args=["--transport", "stdio"],
        env={},
    )
    cfg = _server_config("openbb", spec)
    assert cfg is not None
    home = cfg["env"].get("HOME", "")
    assert ".fincent_openbb_home" in home


def test_server_config_openbb_respects_explicit_home_in_spec():
    spec = PortfolioMcpServerSpec(
        enabled=True,
        command="openbb-mcp",
        args=[],
        env={"HOME": "/custom/openbb/home"},
    )
    cfg = _server_config("openbb", spec)
    assert cfg is not None
    assert cfg["env"]["HOME"] == "/custom/openbb/home"


def test_server_config_rag_does_not_touch_home():
    spec = PortfolioMcpServerSpec(
        enabled=True,
        command="python",
        args=["-m", "src.rag.mcp_server"],
        env={},
    )
    cfg = _server_config("fincent_rag", spec)
    assert cfg is not None
    # Inherits process HOME; must not point at .fincent_openbb_home
    assert ".fincent_openbb_home" not in (cfg["env"].get("HOME") or "")
