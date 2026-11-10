"""Handover Agent — LangGraph-powered assistant for filling handover forms.

Uses a simple tool-calling graph: assistant → tools → assistant (loop).
Tools: search_context, fill_fields, get_section_status.

The agent searches linked context, proposes field values, and asks
targeted questions about gaps — all in a single conversational turn.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Annotated, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .agent import AGENT_LLM_MODEL
from .common import make_llm
from .handover_system_prompt import REQUIRED_FIELDS, SECTIONS, get_handover_system_prompt

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────


class HandoverAgentState(TypedDict):
    """State for the handover agent graph."""
    messages: Annotated[list, add_messages]
    form_state: dict
    context_sources: list
    target_section: Optional[str]


# ── Tools ────────────────────────────────────────────────────────────────────

# These are closures that capture the state via a factory pattern.
# The actual tool functions are created per-invocation with the current
# form_state and context_sources bound.


def _make_tools(form_state: dict, context_sources: list):
    """Create tool instances bound to the current handover state."""

    @tool
    def search_context(query: str) -> str:
        """Search uploaded documents and linked ideation output for relevant information.

        Args:
            query: Search terms or question to look for in the context.

        Returns:
            Relevant excerpts from linked sources, or a message if nothing found.
        """
        results = []

        for source in context_sources:
            extracted = source.get("extracted_fields") or {}
            source_ref = source.get("source_ref", "")
            source_type = source.get("source_type", "")
            content_text = source.get("content_text", "")

            # Search in extracted fields
            if extracted:
                extracted_text = json.dumps(extracted, indent=2)
                query_lower = query.lower()
                # Simple keyword match across extracted content
                if any(
                    kw in extracted_text.lower()
                    for kw in query_lower.split()
                ):
                    results.append(
                        f"[From {source_type} '{source_ref[:40]}' extracted fields]:\n"
                        f"{extracted_text[:2000]}"
                    )

            # For manual notes, search the source_ref text directly
            if source_type == "manual" and source_ref:
                query_lower = query.lower()
                if any(kw in source_ref.lower() for kw in query_lower.split()):
                    results.append(
                        f"[From manual note]:\n{source_ref[:2000]}"
                    )

            # For documents, search the extracted file text
            if source_type == "document" and content_text:
                query_lower = query.lower()
                if any(kw in content_text.lower() for kw in query_lower.split()):
                    filename = source_ref or "document"
                    results.append(
                        f"[From file '{filename}']:\n{content_text[:2000]}"
                    )

        if not results:
            return f"No relevant information found for query: '{query}'. Try different keywords or ask the user directly."

        return "\n\n---\n\n".join(results[:3])  # Max 3 results

    @tool
    def fill_fields(fields: dict) -> str:
        """Propose values for handover form fields. Values are shown to the user as suggestions they can accept or dismiss.

        Args:
            fields: Dictionary of field_id -> proposed value. For table fields (data_sources, business_rules, kpis), pass an array of row objects.

        Returns:
            Confirmation of which fields were proposed.
        """
        if not fields:
            return "No fields provided."

        # This is a signal tool — the actual values are captured from the
        # accumulated tool calls after the graph finishes.
        proposed_keys = list(fields.keys())
        return f"Proposed values for: {', '.join(proposed_keys)}. These will appear as inline suggestions for the user to accept or dismiss."

    @tool
    def get_section_status(section_id: str) -> str:
        """Get the current completion status of a form section.

        Args:
            section_id: One of: customer, usecase, process, data, impact, ai, tobe, rules, tech, checklist

        Returns:
            Which fields are filled and which are empty in the requested section.
        """
        if section_id not in SECTIONS:
            available = ", ".join(SECTIONS.keys())
            return f"Unknown section '{section_id}'. Available: {available}"

        section = SECTIONS[section_id]
        filled = []
        empty = []

        for field in section.get("fields", []):
            value = form_state.get(field)
            if value and str(value).strip():
                filled.append(f"{field}: '{str(value)[:50]}...' " if len(str(value)) > 50 else f"{field}: '{value}'")
            else:
                req_marker = " (REQUIRED)" if field in REQUIRED_FIELDS else ""
                empty.append(f"{field}{req_marker}")

        for table in section.get("tables", []):
            rows = form_state.get(table)
            if rows and isinstance(rows, list) and len(rows) > 0:
                filled.append(f"{table}: {len(rows)} rows")
            else:
                empty.append(f"{table} (table, empty)")

        # Handle special sections (impact table, AI capabilities)
        special = section.get("special")
        if special == "business_impact":
            impact = form_state.get("business_impact")
            if isinstance(impact, dict):
                filled_rows = sum(
                    1 for v in impact.values()
                    if isinstance(v, dict) and any(str(c).strip() for c in v.values())
                )
                if filled_rows:
                    filled.append(f"business_impact: {filled_rows} rows filled")
                else:
                    empty.append("business_impact (table, empty)")
            else:
                empty.append("business_impact (table, empty)")
        elif special == "ai_capabilities":
            caps = form_state.get("ai_capabilities")
            if isinstance(caps, list) and len(caps) > 0:
                filled.append(f"ai_capabilities: {len(caps)} selected")
            else:
                empty.append("ai_capabilities (none selected)")
        elif special == "checklist":
            checklist = form_state.get("checklist")
            if isinstance(checklist, list) and len(checklist) > 0:
                checked = sum(1 for item in checklist if item.get("checked"))
                filled.append(f"checklist: {checked}/{len(checklist)} items checked")
            else:
                empty.append("checklist (not started)")

        lines = [f"## Section: {section['title']}"]
        if filled:
            lines.append(f"**Filled ({len(filled)}):**")
            for f in filled:
                lines.append(f"  ✓ {f}")
        if empty:
            lines.append(f"**Empty ({len(empty)}):**")
            for e in empty:
                lines.append(f"  ○ {e}")

        return "\n".join(lines)

    return [search_context, fill_fields, get_section_status]


# ── Graph Definition ─────────────────────────────────────────────────────────


def _build_graph(tools: list, llm):
    """Build the LangGraph StateGraph for the handover agent."""

    llm_with_tools = llm.bind_tools(tools)

    async def assistant_node(state: HandoverAgentState):
        """Call the LLM with the current message history."""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(HandoverAgentState)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────


async def run_handover_agent(
    user_text: str,
    handover_id: str,
    form_state: Dict[str, Any],
    context_sources: List[Dict[str, Any]],
    prior_messages: List[Dict[str, Any]],
    target_section: Optional[str] = None,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one turn of the handover agent graph.

    Args:
        user_text: The user's message.
        handover_id: ID of the handover workspace.
        form_state: Current form payload (all field values).
        context_sources: List of linked context sources with extracted content.
        prior_messages: Previous conversation messages (for continuity).
        target_section: Optional section ID to focus the agent on.
        llm_usage_meta: Observability metadata for LLM usage logging.

    Returns:
        Dict with keys:
            kind: "question" | "fill" | "summary" | "error"
            message: str — user-facing text (always present)
            fields: dict | None — proposed field values (when kind="fill")
            gaps: list | None — gap list (when kind="summary")
            suggested_replies: list — quick-reply suggestions

    Never raises — returns kind="error" on failure.
    """
    try:
        return await _run_agent_internal(
            user_text=user_text,
            handover_id=handover_id,
            form_state=form_state,
            context_sources=context_sources,
            prior_messages=prior_messages,
            target_section=target_section,
            llm_usage_meta=llm_usage_meta,
        )
    except Exception as exc:
        logger.error(
            "Handover agent failed for %s: %s", handover_id, exc, exc_info=True
        )
        return {
            "kind": "error",
            "message": "I encountered an issue processing your request. Please try again.",
            "fields": None,
            "gaps": None,
            "suggested_replies": ["Try again", "What gaps do I have?"],
        }


async def _run_agent_internal(
    user_text: str,
    handover_id: str,
    form_state: Dict[str, Any],
    context_sources: List[Dict[str, Any]],
    prior_messages: List[Dict[str, Any]],
    target_section: Optional[str] = None,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal implementation of run_handover_agent."""
    import asyncio
    from app.observability.llm_usage_logging import emit_llm_usage_for_response

    # Build tools and LLM
    tools = _make_tools(form_state, context_sources)
    llm = make_llm(
        provider="openai",
        model_name=AGENT_LLM_MODEL,
        temperature=0.3,
        max_tokens=4096,
    )

    # Build graph
    graph = _build_graph(tools, llm)

    # Build messages: system + prior history + new user message
    system_prompt = get_handover_system_prompt(form_state, context_sources, target_section)
    messages = [SystemMessage(content=system_prompt)]

    # Add prior conversation (last 20 messages max to stay in context)
    for msg in prior_messages[-20:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current user message
    messages.append(HumanMessage(content=user_text))

    # Run graph
    start = time.perf_counter()

    initial_state: HandoverAgentState = {
        "messages": messages,
        "form_state": form_state,
        "context_sources": context_sources,
        "target_section": target_section,
    }

    # Run with timeout
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        return {
            "kind": "error",
            "message": "The request took too long. Please try a simpler question.",
            "fields": None,
            "gaps": None,
            "suggested_replies": ["What gaps do I have?"],
        }

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Extract final response from graph result
    final_messages = result.get("messages", [])

    # Find last AI message (the final response)
    final_ai_msg = None
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_ai_msg = msg
            break

    if not final_ai_msg:
        return {
            "kind": "error",
            "message": "I wasn't able to generate a response. Please try again.",
            "fields": None,
            "gaps": None,
            "suggested_replies": ["Try again"],
        }

    # Log usage for all LLM calls (tool-calling loops have multiple)
    for msg in final_messages:
        if isinstance(msg, AIMessage) and (msg.content or msg.tool_calls):
            emit_llm_usage_for_response(
                msg,
                meta=llm_usage_meta,
                provider="sap-ai-core",
                model=AGENT_LLM_MODEL,
                step_name="handover_agent",
                feature="handover",
                latency_ms=elapsed_ms if msg is final_ai_msg else 0,
            )

    # Extract proposed fields from fill_fields tool calls
    proposed_fields = _extract_proposed_fields(final_messages)

    # Determine response kind
    kind = "question"
    if proposed_fields:
        kind = "fill"
    elif _looks_like_summary(final_ai_msg.content):
        kind = "summary"

    # Extract suggested replies
    suggested_replies = _extract_suggestions(final_ai_msg.content, kind)

    # Build gaps list for summary kind
    gaps = None
    if kind == "summary":
        gaps = _extract_gaps(final_ai_msg.content, form_state)

    return {
        "kind": kind,
        "message": final_ai_msg.content,
        "fields": proposed_fields if proposed_fields else None,
        "gaps": gaps,
        "suggested_replies": suggested_replies,
    }


# ── Response parsing helpers ─────────────────────────────────────────────────


def _extract_proposed_fields(messages: list) -> Dict[str, Any]:
    """Extract accumulated fill_fields() calls from the message history."""
    proposed = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.get("name") == "fill_fields":
                    args = call.get("args", {})
                    fields = args.get("fields") if isinstance(args, dict) else {}
                    if isinstance(fields, dict):
                        proposed.update(fields)
    return proposed


def _looks_like_summary(text: str) -> bool:
    """Heuristic: does the response look like a progress summary?"""
    summary_indicators = [
        "progress", "completion", "gaps", "remaining",
        "filled", "empty", "missing", "status",
    ]
    text_lower = text.lower()
    matches = sum(1 for indicator in summary_indicators if indicator in text_lower)
    return matches >= 3


def _extract_suggestions(text: str, kind: str) -> List[str]:
    """Generate contextual quick-reply suggestions based on response."""
    if kind == "fill":
        return ["Accept all", "What else can you fill?", "What gaps remain?"]
    if kind == "summary":
        return ["Fill what you can", "Help me with the biggest gap"]
    # Default for questions
    return ["Skip this", "I don't know"]


def _extract_gaps(text: str, form_state: dict) -> List[Dict[str, str]]:
    """Extract gap information for the summary card."""
    from .handover_system_prompt import REQUIRED_FIELDS, SECTIONS

    gaps = []
    for section_id, section in SECTIONS.items():
        section_has_gap = False
        for field in section.get("fields", []):
            if field in REQUIRED_FIELDS:
                value = form_state.get(field)
                if not value or not str(value).strip():
                    section_has_gap = True
                    break
        if section_has_gap:
            gaps.append({"section": section_id, "label": section["title"]})

    return gaps[:8]  # Cap at 8 for UI readability
