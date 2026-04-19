"""Unit tests for the Q&A agent skeleton."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.qna import answer
from src.core.schemas import AgentName


def test_qna_returns_agent_response():
    """The skeleton must produce an AgentResponse attributed to qna."""
    fake = FakeListChatModel(responses=["An ETF is an exchange-traded fund."])
    response = answer("What is an ETF?", llm=fake)
    assert response.agent == AgentName.QNA
    assert "ETF" in response.content
    assert response.metadata.get("skeleton") is True
