"""Unit tests for the Goal Planning specialist agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool

from src.agents.goal_planning import answer as goal_planning_answer
from src.agents.goal_planning.prompts import GOAL_PLANNING_SYSTEM_PROMPT
from src.core.schemas import AgentName


def _fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


class _ScriptedToolCallingLLM(FakeListChatModel):
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


class _InternalErrorLLM(FakeListChatModel):
    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise RuntimeError("planning model failed")


def test_goal_planning_answer_attribution_and_metadata():
    response = goal_planning_answer(
        "Can I afford a $12,000 vacation next summer?",
        llm=_fake_llm("A $12,000 vacation is a near-term cash-flow goal."),
        tools=[],
    )
    assert response.agent == AgentName.GOAL_PLANNING
    assert "vacation" in response.content
    assert response.metadata["goal_type"] == "vacation"
    assert response.metadata["target_amount"] == 12000
    assert response.metadata["time_horizon_years"] == 1.0
    assert response.metadata["tools_invoked"] == []


def test_goal_planning_invokes_calculation_tool():
    calls: List[Dict[str, Any]] = []

    @tool("calculate_fv")
    def calculate_fv_tool(pv: float, rate: float, nper: int, pmt: float = 0.0) -> str:
        """Return a stub future value."""

        calls.append({"pv": pv, "rate": rate, "nper": nper, "pmt": pmt})
        return '{"future_value": 12345}'

    llm = _ScriptedToolCallingLLM(
        responses=["Using the calculation, the future value is about $12,345."],
        first_turn_tool="calculate_fv",
        first_turn_args={"pv": 1000, "rate": 0.01, "nper": 12, "pmt": 100},
    )

    response = goal_planning_answer(
        "Will I make my goal?",
        llm=llm,
        tools=[calculate_fv_tool],
    )
    assert "future value" in response.content
    assert response.metadata["tools_invoked"] == ["calculate_fv"]
    assert calls == [{"pv": 1000, "rate": 0.01, "nper": 12, "pmt": 100}]


def test_goal_planning_native_monte_carlo_tool_accepts_missing_portfolio_mix():
    from src.agents.goal_planning.mcp_tools import get_goal_planning_tools

    tools = {str(getattr(t, "name", "")): t for t in get_goal_planning_tools()}
    out = tools["run_monte_carlo"].invoke(
        {
            "years": 1,
            "monthly_contribution": 1000,
            "target": 12000,
            "simulations": 100,
            "seed": 42,
        }
    )
    assert "probability_of_success" in out
    assert "portfolio_mix" in out


def test_goal_planning_prompt_requires_stress_test_extra_years_estimate():
    assert "25%" in GOAL_PLANNING_SYSTEM_PROMPT or "drawdown" in GOAL_PLANNING_SYSTEM_PROMPT
    assert "extra years" in GOAL_PLANNING_SYSTEM_PROMPT
    assert "illustrative defaults" in GOAL_PLANNING_SYSTEM_PROMPT


def test_goal_planning_internal_error_keeps_details_in_metadata_only():
    response = goal_planning_answer(
        "I want to retire at 60.",
        llm=_InternalErrorLLM(responses=[]),
        tools=[],
    )
    assert "internal error" in response.content
    assert "planning model failed" not in response.content
    assert response.metadata["error"] is True
    assert response.metadata["error_phase"] == "goal_planning_tool_loop"
    assert "planning model failed" in response.metadata["error_message"]

