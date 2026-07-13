"""LangGraph orchestration for the Pharma Procurement Sales Order Agent agent."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agents.pharma_order.prompts import PHARMA_ORDER_PROMPT_VARIANTS
from app.agents.pharma_order.query_rewrite import rewrite_order_question
from app.observability.llm_usage_logging import emit_llm_usage_event
from app.utils.langgraph.common import make_llm

DEFAULT_PROVIDER = os.getenv("PHARMA_ORDER_LLM_PROVIDER", "openai")
DEFAULT_USAGE_PROVIDER = os.getenv("PHARMA_ORDER_USAGE_PROVIDER", "sap-ai-core")
DEFAULT_MODEL = os.getenv("PHARMA_ORDER_MODEL", "gpt-4.1")
logger = logging.getLogger(__name__)

def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _server_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "mcp" / "pharma_order_server.py")


def _mcp_subprocess_env() -> dict[str, str]:
    """Return an environment that is safe for MCP stdio subprocesses.

    Cloud Foundry Python buildpack starts the web process with the right runtime
    environment, but MCP stdio launches a second Python process. Passing the
    environment explicitly avoids losing credentials and shared-library paths.
    """
    env = os.environ.copy()
    if os.name != "nt":
        executable = Path(sys.executable).resolve()
        candidates = [
            str(executable.parent.parent / "lib"),
            str(executable.parent.parent / "lib64"),
            "/home/vcap/deps/0/lib",
            "/home/vcap/deps/0/lib64",
        ]
        existing = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
        merged: list[str] = []
        for part in [*existing, *candidates]:
            if part and part not in merged:
                merged.append(part)
        if merged:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(merged)
    return env


def _make_model(provider: str | None, model_name: str | None, temperature: float, max_tokens: int):
    provider_value = provider or DEFAULT_PROVIDER
    model_value = model_name or DEFAULT_MODEL
    attempts = [
        {"provider": provider_value, "model_name": model_value, "temperature": temperature, "max_tokens": max_tokens},
        {"provider": provider_value, "model": model_value, "temperature": temperature, "max_tokens": max_tokens},
        {"model_name": model_value, "temperature": temperature, "max_tokens": max_tokens},
        {"model": model_value, "temperature": temperature, "max_tokens": max_tokens},
    ]
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return make_llm(**kwargs)
        except TypeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise RuntimeError("Unable to create Pharma Procurement Sales Order Agent model")


def _usage_from_message(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return dict(usage)
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    return dict(token_usage) if isinstance(token_usage, dict) else {}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _extract_trace(messages: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}

    for message in messages:
        message_usage = _usage_from_message(message)
        for key, value in message_usage.items():
            if isinstance(value, int):
                usage[key] = int(usage.get(key, 0)) + value
            elif key not in usage:
                usage[key] = value

        for call in getattr(message, "tool_calls", []) or []:
            tool_calls.append({"name": call.get("name"), "args": call.get("args"), "id": call.get("id")})

        if isinstance(message, ToolMessage):
            content = _content_to_text(message.content)
            parsed_content = None
            try:
                parsed_content = json.loads(content)
            except Exception:
                parsed_content = None
            tool_results.append(
                {
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "content_preview": content[:4000],
                    "content_json": parsed_content,
                }
            )

    return tool_calls, tool_results, usage


def _final_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = _content_to_text(message.content)
            if text:
                return text
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _content_to_text(message.content)
            if text:
                return text
    return "I could not produce an answer from the available prototype data."


def _build_graph(tools: list[Any], provider: str | None, model_name: str | None, temperature: float, max_tokens: int):
    llm = _make_model(provider, model_name, temperature, max_tokens)
    llm_with_tools = llm.bind_tools(tools)

    async def assistant(state: MessagesState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "assistant")
    graph_builder.add_conditional_edges("assistant", tools_condition)
    graph_builder.add_edge("tools", "assistant")
    return graph_builder.compile()


async def run_pharma_order_agent(
    question: str,
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
    prompt_variant: str = "joule",
    include_trace: bool = False,
    route: str = "internal.pharma_order_agent",
    method: str = "INTERNAL",
    correlation_id: str | None = None,
    client_host: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    agent_trace_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    rewrite_result = rewrite_order_question(question)
    cleaned_question = rewrite_result.rewritten_question
    if rewrite_result.changed:
        logger.info(
            "Pharma Procurement Sales Order Agent query rewrite trace_id=%s rule_id=%s rationale=%s original=%r rewritten=%r",
            agent_trace_id,
            rewrite_result.rule_id,
            rewrite_result.rationale,
            rewrite_result.original_question[:500],
            rewrite_result.rewritten_question[:700],
        )
    logger.info(
        "Pharma Procurement Sales Order Agent agent start trace_id=%s prompt_variant=%s provider=%s model=%s max_tokens=%s include_trace=%s route=%s correlation_id=%s question=%r",
        agent_trace_id,
        prompt_variant,
        provider or DEFAULT_PROVIDER,
        model_name or DEFAULT_MODEL,
        max_tokens,
        include_trace,
        route,
        correlation_id or agent_trace_id,
        cleaned_question[:500],
    )

    server_params = StdioServerParameters(command=sys.executable, args=[_server_path()], env=_mcp_subprocess_env())
    system_prompt = PHARMA_ORDER_PROMPT_VARIANTS.get(prompt_variant, PHARMA_ORDER_PROMPT_VARIANTS["joule"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tool_names = [getattr(tool, "name", repr(tool)) for tool in tools]
            logger.info(
                "Pharma Procurement Sales Order Agent agent tools_loaded trace_id=%s tool_count=%s tools=%s",
                agent_trace_id,
                len(tools),
                tool_names,
            )
            graph = _build_graph(tools, provider, model_name, temperature, max_tokens)
            result = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=cleaned_question),
                    ]
                },
                config={"recursion_limit": 10},
            )
            graph_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Pharma Procurement Sales Order Agent agent graph_complete trace_id=%s elapsed_ms=%s message_count=%s",
                agent_trace_id,
                graph_elapsed_ms,
                len(result.get("messages", [])),
            )

    messages = list(result.get("messages", []))
    answer = _final_answer(messages)
    tool_calls, tool_results, usage = _extract_trace(messages)
    total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "Pharma Procurement Sales Order Agent agent complete trace_id=%s elapsed_ms=%s tool_call_count=%s tool_names=%s answer_chars=%s usage=%s",
        agent_trace_id,
        total_elapsed_ms,
        len(tool_calls),
        [call.get("name") for call in tool_calls],
        len(answer),
        usage,
    )
    emit_llm_usage_event(
        route=route,
        method=method,
        user_id=user_id,
        client_host=client_host,
        provider=DEFAULT_USAGE_PROVIDER,
        model=model_name or DEFAULT_MODEL,
        input_tokens=_usage_int(usage, "input_tokens", "prompt_tokens"),
        output_tokens=_usage_int(usage, "output_tokens", "completion_tokens"),
        latency_ms=total_elapsed_ms,
        correlation_id=correlation_id or agent_trace_id,
    )
    payload = {
        "answer": answer,
        "markdown": answer,
        "model": model_name or DEFAULT_MODEL,
        "provider": DEFAULT_USAGE_PROVIDER,
        "tool_call_count": len(tool_calls),
        "usage": usage,
        "rewritten_question": cleaned_question if rewrite_result.changed else None,
        "rewrite_rule": rewrite_result.rule_id,
    }
    if include_trace:
        payload["tool_calls"] = tool_calls
        payload["tool_results"] = tool_results
    return payload





