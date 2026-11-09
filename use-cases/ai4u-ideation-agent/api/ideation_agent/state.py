"""Agent state definition for the A2A LangGraph agent.

This module defines the state schema that flows through the agent graph.
Extend AgentState to add custom fields for your specific use case.

LangGraph state works as follows:
- State is a TypedDict that defines the shape of data flowing through nodes
- Each node receives the current state and returns updates to it
- MessagesState provides the base 'messages' field for conversation history

Example of extending state for a RAG agent:

    class RAGState(AgentState):
        # Document chunks extracted from files
        chunks: Optional[List[str]]

        # Retrieved context from vector search
        retrieved_context: Optional[List[Dict[str, Any]]]

        # Currently active vector table
        active_table: Optional[str]

Usage in agent.py:
    from .state import AgentState  # or your custom state class

    graph = StateGraph(AgentState)

    async def assistant(state: AgentState):
        # Access state fields
        messages = state["messages"]
        # custom_field = state.get("custom_field")
        ...
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Base agent state extending MessagesState.

    Inherits:
        messages: list[BaseMessage] - The conversation history, automatically
            managed by LangGraph. New messages are appended, not replaced.

    Add custom fields below for your specific agent needs. Use Optional types
    for fields that may not be set in every state update.

    Example fields you might add:
        - user_context: Optional[str] - Additional context about the user
        - tool_results: Optional[Dict[str, Any]] - Cached tool outputs
        - current_task: Optional[str] - Description of current task

    Custom fields used by this template:
        - skill_summaries: lightweight skill metadata injected by the skill node
    """

    skill_summaries: Optional[List[Dict[str, str]]]
    clarification_count: Optional[int]
    last_turn_kind: Optional[str]
    ui_payload: Optional[Dict[str, Any]]
    context_id: Optional[str]
    company_name: Optional[str]
    cse_enrichment: Optional[Dict[str, Any]]
    cse_status: Optional[str]
    perplexity_enrichment: Optional[Dict[str, Any]]
    perplexity_status: Optional[str]
    perplexity_problem_enrichment: Optional[Dict[str, Any]]
    perplexity_problem_status: Optional[str]
    linkedin_enrichment: Optional[Dict[str, Any]]
    linkedin_status: Optional[str]
    refinement_mode: Optional[bool]
    refinement_clarification_count: Optional[int]
    llm_usage_meta: Optional[Dict[str, Any]]
    # Provocateur/Reimagine variant fields
    reimagine_context: Optional[str]
    provocateur_interventions: Optional[int]
    conversation_variant: Optional[str]  # "standard" | "reimagine" | "provocateur"
    mini_reimagine_result: Optional[Dict[str, Any]]
