"""Unit tests for the Market Research specialist agent."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool

from src.agents.market_research import answer as market_research_answer
from src.agents.market_research.mcp_tools import reset_market_research_tools_cache
from src.agents.openbb_mcp_invoke import FMP_FOOTPRINT_FREE_DATA_DISCLAIMER

from src.core.config import AppConfig
from src.core.schemas import AgentName


def _fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _force_fmp_free_tier(monkeypatch) -> None:
    monkeypatch.delenv("FMP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)


class _ScriptedToolCallingLLM(FakeListChatModel):
    """Fake chat model that emits one tool call before the final answer."""

    first_turn_tool: str = ""
    first_turn_args: Dict[str, Any] = {}
    _emitted: bool = False

    def bind_tools(self, tools: List[Any], **_kwargs: Any) -> "_ScriptedToolCallingLLM":  # type: ignore[override]
        return self

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        if not self._emitted and self.first_turn_tool:
            self._emitted = True
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.first_turn_tool,
                        "args": dict(self.first_turn_args),
                        "id": "call-1",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _ContextOverflowLLM(FakeListChatModel):
    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise RuntimeError(
            "OpenAIContextOverflowError: Input tokens exceed the configured limit"
        )


class _InternalErrorLLM(FakeListChatModel):
    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise RuntimeError("upstream exploded api_key=super-secret-value")


def test_market_research_answer_attribution_and_metadata(monkeypatch):
    _force_fmp_free_tier(monkeypatch)
    response = market_research_answer(
        "Is Nvidia a good investment?",
        llm=_fake_llm("Nvidia has strong AI demand, but valuation risk matters."),
        tools=[],
    )

    assert response.agent == AgentName.MARKET_RESEARCH
    assert "Nvidia" in response.content
    assert response.metadata["tool_count"] == 0
    assert response.metadata["tool_names"] == []
    assert response.metadata["tools_invoked"] == []
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER


def test_market_research_answer_no_data_sources_note_when_paid_fmp_enabled(
    monkeypatch,
):
    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", "true")
    response = market_research_answer(
        "Is Nvidia a good investment?",
        llm=_fake_llm("Nvidia has strong AI demand, but valuation risk matters."),
        tools=[],
    )
    assert "data_sources_note" not in response.metadata


def test_market_research_answer_no_data_sources_note_when_fmp_key_present(
    monkeypatch,
):
    monkeypatch.delenv("FINCENT_OPENBB_ALLOW_PAID_FMP_FUNDAMENTALS", raising=False)
    monkeypatch.setenv("FMP_ACCESS_TOKEN", "starter-plan-key")
    response = market_research_answer(
        "Is Nvidia a good investment?",
        llm=_fake_llm("Nvidia has strong AI demand, but valuation risk matters."),
        tools=[],
    )
    assert "data_sources_note" not in response.metadata


def test_market_research_context_overflow_returns_clean_message():
    response = market_research_answer(
        "Compare a lot of tickers",
        llm=_ContextOverflowLLM(responses=[]),
        tools=[],
    )
    assert "ran out of model context" in response.content
    assert "OpenAIContextOverflowError" not in response.content
    assert "Input tokens exceed" not in response.content
    assert response.metadata["error"] is True
    assert response.metadata["error_type"] == "RuntimeError"
    assert "OpenAIContextOverflowError" in response.metadata["error_message"]


def test_market_research_internal_error_keeps_details_in_metadata_only():
    response = market_research_answer(
        "Compare Procter Gamble vs Unilever",
        llm=_InternalErrorLLM(responses=[]),
        tools=[],
    )
    assert "internal error" in response.content
    assert "upstream exploded" not in response.content
    assert response.metadata["error"] is True
    assert response.metadata["error_phase"] == "market_research_tool_loop"
    assert response.metadata["error_type"] == "RuntimeError"
    assert "upstream exploded" in response.metadata["error_message"]
    assert "super-secret-value" not in response.metadata["error_message"]


def test_market_research_answer_invokes_mcp_tools_and_returns_final_text(monkeypatch):
    _force_fmp_free_tier(monkeypatch)
    call_log: List[Dict[str, Any]] = []

    @tool("NEWS_SENTIMENT")
    def news_sentiment(symbol: str) -> str:
        """Return stubbed market sentiment."""

        call_log.append({"symbol": symbol})
        return "[stub] sentiment is positive but crowded."

    llm = _ScriptedToolCallingLLM(
        responses=["Sentiment is positive, but it should not be the whole thesis."],
        first_turn_tool="NEWS_SENTIMENT",
        first_turn_args={"symbol": "NVDA"},
    )

    response = market_research_answer(
        "What does sentiment say about Nvidia?",
        llm=llm,
        tools=[news_sentiment],
    )

    assert response.agent == AgentName.MARKET_RESEARCH
    assert "Sentiment" in response.content
    assert response.metadata["tool_count"] == 1
    assert response.metadata["tool_names"] == ["NEWS_SENTIMENT"]
    assert response.metadata["tools_invoked"] == ["NEWS_SENTIMENT"]
    assert response.metadata["data_sources_note"] == FMP_FOOTPRINT_FREE_DATA_DISCLAIMER
    assert call_log == [{"symbol": "NVDA"}]


def test_market_research_caps_bound_tools_for_openai_limit(monkeypatch):
    _force_fmp_free_tier(monkeypatch)
    monkeypatch.delenv("FINCENT_MARKET_RESEARCH_MAX_BOUND_TOOLS", raising=False)
    tools = [
        SimpleNamespace(name=f"crypto_low_value_{idx}")
        for idx in range(142)
    ] + [SimpleNamespace(name="tavily_search")]

    response = market_research_answer(
        "Compare Procter Gamble vs Unilever",
        llm=_fake_llm("The comparison depends on growth, margins, and valuation."),
        tools=tools,
    )

    assert response.metadata["available_tool_count"] == 143
    assert response.metadata["tool_count"] == 128
    assert response.metadata["dropped_tool_count"] == 15
    assert "tavily_search" in response.metadata["tool_names"]


def test_market_research_bound_tool_cap_env_cannot_exceed_provider_limit(monkeypatch):
    from src.agents.market_research.agent import _select_tools_for_binding

    monkeypatch.setenv("FINCENT_MARKET_RESEARCH_MAX_BOUND_TOOLS", "999")
    tools = [SimpleNamespace(name=f"tool_{idx}") for idx in range(130)]

    selected, dropped = _select_tools_for_binding(tools)

    assert len(selected) == 128
    assert dropped == 2


def test_market_research_prompt_requires_company_comparison_framework():
    from src.agents.market_research.prompts import MARKET_RESEARCH_SYSTEM_PROMPT

    required_terms = [
        "return metrics",
        "EBITDA",
        "free cash flow",
        "technical indicators",
        "sentiment",
        "key filing",
        "Do not trail off",
        "parallel structure",
    ]
    for term in required_terms:
        assert term in MARKET_RESEARCH_SYSTEM_PROMPT


def test_market_research_prompt_bans_bullish_bearish_recommendation_framing():
    from src.agents.market_research.prompts import MARKET_RESEARCH_SYSTEM_PROMPT

    assert "bullish case" in MARKET_RESEARCH_SYSTEM_PROMPT
    assert "Never stop mid-thought" in MARKET_RESEARCH_SYSTEM_PROMPT


def test_merge_openbb_coerces_yfinance_to_fmp_for_equity_fundamental_ratios():
    from src.agents.market_research.agent import _merge_openbb_tool_args

    assert _merge_openbb_tool_args(
        "equity_fundamental_ratios",
        {"symbol": "NVDA", "provider": "yfinance"},
    ) == {"symbol": "NVDA", "provider": "fmp"}


def test_merge_openbb_preserves_intrinio_for_equity_fundamental_ratios():
    from src.agents.market_research.agent import _merge_openbb_tool_args

    assert _merge_openbb_tool_args(
        "equity_fundamental_ratios",
        {"symbol": "NVDA", "provider": "intrinio"},
    ) == {"symbol": "NVDA", "provider": "intrinio"}


def test_market_research_openbb_provider_default_can_be_overridden(monkeypatch):
    from src.agents.market_research.agent import _merge_openbb_tool_args

    assert _merge_openbb_tool_args("equity_price_quote", {"symbol": "NVDA"}) == {
        "symbol": "NVDA",
        "provider": "yfinance",
    }

    monkeypatch.setenv("FINCENT_OPENBB_ALLOW_KEYED_PROVIDERS", "true")
    assert _merge_openbb_tool_args(
        "equity_price_quote",
        {"symbol": "NVDA", "provider": "fmp"},
    ) == {"symbol": "NVDA", "provider": "fmp"}


def test_market_research_tools_disabled_in_tests_returns_empty_list(monkeypatch):
    from src.agents.market_research.mcp_tools import get_market_research_tools

    monkeypatch.setenv("TAVILY_API_KEY", "")
    reset_market_research_tools_cache()
    assert get_market_research_tools() == []


def test_market_research_loads_native_tavily_tools_without_mcp(monkeypatch):
    import src.agents.market_research.mcp_tools as mcp_mod
    from src.agents.market_research.mcp_tools import get_market_research_tools

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> Dict[str, Any]:
            return {"answer": "Nvidia announced AI chip updates.", "results": []}

    calls: List[Dict[str, Any]] = []

    def _post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _Response()

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(mcp_mod.requests, "post", _post)
    reset_market_research_tools_cache()

    tools = get_market_research_tools()
    by_name = {str(getattr(t, "name", "")): t for t in tools}

    assert {"tavily_search", "tavily_extract"}.issubset(by_name)
    out = by_name["tavily_search"].invoke(
        {"query": "Nvidia AI investments", "max_results": 1}
    )
    assert "Nvidia announced" in out
    assert calls[0]["url"].endswith("/search")
    assert calls[0]["headers"]["Authorization"] == "Bearer tvly-test"


def test_market_research_loader_expands_env_placeholders(monkeypatch):
    import src.agents.market_research.mcp_tools as mcp_mod
    from src.core.config import load_config, reset_config_cache

    monkeypatch.setenv("FINCENT__MARKET_RESEARCH__TOOLS__TAVILY__ENABLED", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    reset_config_cache()

    cfg = load_config()
    specs = mcp_mod._build_client_specs(cfg)

    assert "tavily" in specs
    assert specs["tavily"]["env"]["TAVILY_API_KEY"] == "test-tavily-key"


def test_market_research_loader_fmp_accepts_fmp_api_key_alias(monkeypatch):
    """FMP docs often use FMP_API_KEY; our YAML placeholder is FMP_ACCESS_TOKEN."""

    import src.agents.market_research.mcp_tools as mcp_mod
    from src.core.config import load_config, reset_config_cache

    monkeypatch.setenv("FINCENT__MARKET_RESEARCH__TOOLS__FMP__ENABLED", "true")
    monkeypatch.delenv("FMP_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("FMP_API_KEY", "fmp-secret-from-alias")
    reset_config_cache()

    cfg = load_config()
    specs = mcp_mod._build_client_specs(cfg)

    assert "fmp" in specs
    assert specs["fmp"]["env"]["FMP_ACCESS_TOKEN"] == "fmp-secret-from-alias"
    assert specs["fmp"]["env"]["FMP_API_KEY"] == "fmp-secret-from-alias"


@pytest.mark.asyncio
async def test_market_research_persistent_startup_keeps_successful_servers(
    monkeypatch,
):
    """A single MCP connection failure must not clear every loaded tool."""

    import src.agents.market_research.mcp_tools as mcp_mod

    monkeypatch.setenv("TAVILY_API_KEY", "")

    class _Session:
        def __init__(self, name: str) -> None:
            self.name = name

        async def initialize(self) -> None:
            if self.name == "bad":
                raise RuntimeError("Connection closed")

    class _SessionContext:
        def __init__(self, name: str) -> None:
            self.session = _Session(name)
            self.closed = False

        async def __aenter__(self) -> _Session:
            return self.session

        async def __aexit__(self, *_exc: object) -> None:
            self.closed = True

    def _create_session(conn: Dict[str, Any]) -> _SessionContext:
        return _SessionContext(str(conn["command"]))

    async def _load_mcp_tools(
        session: _Session,
        *,
        connection: Dict[str, Any],
        server_name: str,
    ) -> List[Any]:
        return [SimpleNamespace(name=f"{session.name}_tool")]

    monkeypatch.setitem(
        sys.modules,
        "langchain_mcp_adapters.sessions",
        SimpleNamespace(create_session=_create_session),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_mcp_adapters.tools",
        SimpleNamespace(load_mcp_tools=_load_mcp_tools),
    )
    monkeypatch.setattr(
        mcp_mod,
        "_build_client_specs",
        lambda _cfg: {
            "bad": {"transport": "stdio", "command": "bad", "args": [], "env": {}},
            "good": {"transport": "stdio", "command": "good", "args": [], "env": {}},
        },
    )
    monkeypatch.setattr(mcp_mod.shutil, "which", lambda cmd: f"/fake/{cmd}")

    await mcp_mod.start_market_research_mcp_sessions(AppConfig())
    try:
        tools = mcp_mod.get_market_research_tools(AppConfig())
        assert [t.name for t in tools] == ["good_tool"]
    finally:
        await mcp_mod.stop_market_research_mcp_sessions()
