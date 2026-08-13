"""GR/IR LangGraph agent — owns the graph singleton, system prompt, and NDJSON streaming."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from ..utils.langgraph.grir_tools import (
    lookup_po, search_pos, list_issue_summary,
    vendor_discrepancy_analysis, vendor_trend_analysis,
    find_aged_items, close_po, check_po_closed, send_notification,
)
from ..utils.langgraph.common import make_llm

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4.1"

_SYSTEM_PROMPT = (Path(__file__).parent / "grir_system_prompt.md").read_text(encoding="utf-8")

_TOOLS = [
    lookup_po, search_pos, list_issue_summary,
    vendor_discrepancy_analysis, vendor_trend_analysis,
    find_aged_items, close_po, check_po_closed, send_notification,
]


def _build_graph():
    llm = make_llm(provider="openai", model_name=MODEL_NAME, temperature=0.0)
    llm_with_tools = llm.bind_tools(_TOOLS)
    sys_msg = SystemMessage(content=_SYSTEM_PROMPT)

    async def assistant(state: MessagesState):
        response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools=_TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


async def stream(message: str, history: list) -> AsyncIterator[str]:
    """
    Append message to history, run the agent, yield NDJSON events.

    history is mutated in place: HumanMessage prepended before the call,
    AIMessage appended after the full response is accumulated.
    """
    history.append(HumanMessage(content=message))
    graph = _get_graph()
    full_text = ""

    try:
        async for event in graph.astream_events({"messages": history}, version="v2"):
            kind = event.get("event")

            if kind == "on_tool_end" and event.get("name") == "send_notification":
                output = _extract_tool_output(event)
                if output.get("notification_sent"):
                    yield json.dumps({"type": "notification", **output}) + "\n"

            if kind == "on_tool_end" and event.get("name") == "vendor_trend_analysis":
                output = _extract_tool_output(event)
                if output.get("found") and output.get("months"):
                    yield json.dumps({"type": "chart", **output}) + "\n"

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_text += token
                    yield json.dumps({"type": "token", "content": token}) + "\n"

        history.append(AIMessage(content=full_text))
        yield json.dumps({"type": "done", "content": full_text}) + "\n"

    except Exception as e:
        logger.error("GR/IR agent error: %s", e)
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"


def _extract_tool_output(event: dict) -> dict:
    raw = event.get("data", {}).get("output")
    if hasattr(raw, "content"):
        try:
            return json.loads(raw.content)
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}
