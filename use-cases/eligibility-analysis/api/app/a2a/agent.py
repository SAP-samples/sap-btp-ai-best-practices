from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .common import make_llm
from .state import AgentState
from .system_prompt import SYSTEM_PROMPT
from .tools import get_all_tools
from .tool_result_utils import build_tool_result_preview

_TOOL_RESULT_PREVIEW_CHARS = int(os.getenv("A2A_TOOL_RESULT_PREVIEW_CHARS", "2000"))


def _extract_tool_calls(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from LangGraph messages."""
    calls: List[Dict[str, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args")
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None)
            if name:
                calls.append({"name": name, "args": args})
    return calls


def _extract_tool_results(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Extract tool results from LangGraph messages."""
    tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
    tool_calls_by_name: Dict[str, Dict[str, Any]] = {}

    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, dict):
                call_id = call.get("id") or call.get("tool_call_id")
                name = call.get("name")
                args = call.get("args")
            else:
                call_id = getattr(call, "id", None) or getattr(call, "tool_call_id", None)
                name = getattr(call, "name", None)
                args = getattr(call, "args", None)
            if call_id:
                tool_calls_by_id[call_id] = {"name": name, "args": args}
            if name:
                tool_calls_by_name[name] = {"name": name, "args": args}

    results: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = getattr(message, "tool_call_id", None)
        name = getattr(message, "name", None)
        base = None
        if tool_call_id and tool_call_id in tool_calls_by_id:
            base = tool_calls_by_id[tool_call_id]
        elif name and name in tool_calls_by_name:
            base = tool_calls_by_name[name]
        preview = build_tool_result_preview(message.content, _TOOL_RESULT_PREVIEW_CHARS)
        results.append(
            {
                "name": (base or {}).get("name") or name,
                "args": (base or {}).get("args"),
                "tool_call_id": tool_call_id,
                "content": preview["content_preview"],
                "content_truncated": preview["content_truncated"],
                "content_char_length": preview["content_char_length"],
                "payload_bytes_estimate": preview["payload_bytes_estimate"],
                "row_count_estimate": preview["row_count_estimate"],
            }
        )
    return results


def _extract_final_text(result: Any) -> str:
    """Extract the final text response from the graph result."""
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                return message.content or ""
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content:
                return content
    return str(result)


_GRAPH: Optional[Any] = None
_CHECKPOINTER: Optional[AsyncSqliteSaver] = None
_DB_CONNECTION: Optional[aiosqlite.Connection] = None
_GRAPH_LOCK = asyncio.Lock()

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "a2a_conversations.db"


async def _get_graph():
    """Get or create the compiled LangGraph agent with checkpointing."""
    global _GRAPH, _CHECKPOINTER, _DB_CONNECTION
    if _GRAPH is not None:
        return _GRAPH
    async with _GRAPH_LOCK:
        if _GRAPH is not None:
            return _GRAPH

        model_name = os.getenv("AICORE_MODEL", "gpt-4.1")
        temperature = float(os.getenv("AICORE_TEMPERATURE", "0.2"))
        llm = make_llm(model_name=model_name, temperature=temperature)
        tools = await get_all_tools()
        llm_with_tools = llm.bind_tools(tools)

        sys_msg = SystemMessage(content=SYSTEM_PROMPT)

        async def assistant(state: AgentState):
            response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("assistant", assistant)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = Path(os.getenv("A2A_CONVERSATIONS_DB", str(_DEFAULT_DB_PATH)))
        _DB_CONNECTION = await aiosqlite.connect(str(db_path))
        _CHECKPOINTER = AsyncSqliteSaver(_DB_CONNECTION)
        await _CHECKPOINTER.setup()

        _GRAPH = graph.compile(checkpointer=_CHECKPOINTER)
        return _GRAPH


async def run_agent(user_text: str, context_id: str) -> Dict[str, Any]:
    """Run the agent with a user message in a specific conversation context."""
    graph = await _get_graph()
    user_message = HumanMessage(content=user_text)
    config = {"configurable": {"thread_id": context_id}}
    result = await graph.ainvoke({"messages": [user_message]}, config=config)
    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    return {
        "text": _extract_final_text(result),
        "tool_calls": _extract_tool_calls(result_messages),
        "tool_results": _extract_tool_results(result_messages),
    }
