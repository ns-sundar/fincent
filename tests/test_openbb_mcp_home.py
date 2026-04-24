"""OpenBB MCP subprocess HOME + user_settings wiring."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.agents.portfolio.mcp_tools import _PROJECT_ROOT, _server_config
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
