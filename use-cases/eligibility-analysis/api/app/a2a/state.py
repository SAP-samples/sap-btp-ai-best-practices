"""Agent state definition for the A2A LangGraph agent."""
from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Base agent state extending MessagesState."""

    pass
