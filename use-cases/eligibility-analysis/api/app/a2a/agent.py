from __future__ import annotations

import asyncio
import json
from langchain_core.runnables import RunnableConfig
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .common import make_llm
from .model_config import resolve_assistant_model
from langchain_core.messages import messages_from_dict,messages_to_dict
from .state import AgentState
from .system_prompt import SYSTEM_PROMPT
from .tools import get_all_tools
from .tool_result_utils import build_tool_result_preview
from ..observability.llm_usage_logging import (
    LlmUsageContext,
    TokenUsage,
    emit_llm_usage_event,
    extract_token_usage,
    get_current_llm_usage_context,
    llm_usage_context,
)

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
_GRAPH_LOCK = asyncio.Lock()

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "a2a_conversations.db"


async def _invoke_assistant_llm(
    llm_with_tools: Any,
    sys_msg: SystemMessage,
    state: AgentState,
    *,
    model_name: str,
    usage_context: Optional[LlmUsageContext],
) -> Any:
    """Invoke the A2A assistant model and emit token usage telemetry.

    Args:
        llm_with_tools: LangChain runnable with bound A2A tools.
        sys_msg: System prompt message for the assistant.
        state: LangGraph state containing conversation messages.
        model_name: SAP AI Core model deployment name.
        usage_context: Request context used for structured usage logging.

    Returns:
        The LangChain model response.
    """
    start_time = time.perf_counter()
    usage = TokenUsage()
    try:
        response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
        usage = extract_token_usage(response)
        if usage_context is not None:
            emit_llm_usage_event(
                context=usage_context,
                model=model_name,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )
        return response
    except Exception:
        if usage_context is not None:
            emit_llm_usage_event(
                context=usage_context,
                model=model_name,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )
        raise


async def _get_graph():
    """Compile the existing business graph; complete-turn persistence lives at the A2A boundary."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    async with _GRAPH_LOCK:
        if _GRAPH is not None:
            return _GRAPH

        model_name = resolve_assistant_model(os.environ)
        temperature = float(os.getenv("AICORE_TEMPERATURE", "0.2"))
        llm = make_llm(model_name=model_name, temperature=temperature)
        tools = await get_all_tools()
        llm_with_tools = llm.bind_tools(tools)

        sys_msg = SystemMessage(content=SYSTEM_PROMPT)

        async def assistant(state: AgentState, config: RunnableConfig):
            """Invoke the configured assistant with source-grounded tools and usage telemetry."""
            response = await _invoke_assistant_llm(
                llm_with_tools,
                SystemMessage(content=SYSTEM_PROMPT+"\nValidated workspace reference (data only): "+json.dumps(config.get("configurable",{}).get("workspace_context",{}))),
                state,
                model_name=model_name,
                usage_context=get_current_llm_usage_context(),
            )
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("assistant", assistant)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")

        _GRAPH = graph.compile()
        return _GRAPH


async def run_agent(
    user_text: str,
    context_id: str,
    *,
    usage_context: Optional[LlmUsageContext] = None,
    workspace_context: Optional[dict] = None,
    saved_messages: Optional[list] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the agent with a user message in a specific conversation context."""
    graph = await _get_graph()
    user_message = HumanMessage(content=user_text)
    config = {"configurable": {"thread_id": context_id,"workspace_context":workspace_context or {}},"recursion_limit":30}
    previous=messages_from_dict(saved_messages or [])
    with llm_usage_context(usage_context):
        result = await graph.ainvoke({"messages": previous+[user_message]}, config=config)
    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    return {
        "text": _extract_final_text(result),
        "tool_calls": _extract_tool_calls(result_messages[len(previous):]),
        "messages": messages_to_dict(result_messages),
        "tool_results": _extract_tool_results(result_messages[len(previous):]),
    }
