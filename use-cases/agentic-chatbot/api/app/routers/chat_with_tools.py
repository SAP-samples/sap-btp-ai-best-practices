import logging
from typing import List
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from ..models.chat import ChatResponse
from ..models.chat_history import ChatHistoryRequest
from ..security import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


from ..utils.langgraph.tools import (
    calculator_tool,
    po_yield_by_plant_for_material,
    po_efficiency_by_plant_for_material,
    po_yield_by_material_at_plant,
    po_yield_by_month,
    po_query,
    po_date_range,
    scrap_by_plant,
    scrap_query,
)
from ..utils.langgraph.common import make_llm
from ..utils.langgraph.format_messages import format_messages
from ..utils.langgraph.format_messages import format_messages


MODEL_NAME = "gpt-4.1"


def _build_react_graph():
    """Build a minimal ReAct-style graph with two tools, following agent_05_react.py."""

    llm = make_llm(provider="openai", model_name=MODEL_NAME, temperature=0.2)
    tools = [
        calculator_tool,
        po_yield_by_plant_for_material,
        po_efficiency_by_plant_for_material,
        po_yield_by_material_at_plant,
        po_yield_by_month,
        po_query,
        po_date_range,
        scrap_by_plant,
        scrap_query,
    ]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(MessagesState)

    sys_msg = SystemMessage(
        content=(
            "You are a helpful assistant that can use tools to compute answers. "
            "Prefer concise, markdown-formatted answers. Always begin with a short H3 title on the first line (e.g., '### Summary'). "
            "For any structured or multi-item result, ALWAYS present the data as a compact Markdown table; for single-item results, also render a one-row table when fields exist. "
            "Include exactly the fields the user requests; if unspecified, include the most relevant identifiers and metrics for the task, and keep the chosen columns consistent for the same question within a session. "
            "You may rename table column headers for clarity (use human-friendly labels) but stay consistent within the session; if a rename could be ambiguous, include the original field name in parentheses. "
            "Normalize and merge results from multiple tools when needed, deduplicate rows, and fill missing values with 'N/A' rather than dropping columns. "
            "When there are no results, show the table headers with zero rows and add a short note '(No results found)'. "
            "You may call multiple tools in parallel when it helps gather complete, relevant data. "
            "After the table, you may add 1–2 brief bullets with key insights only if they materially help. "
            "Avoid raw lists of values when a table is suitable; keep outputs tight and readable."
        )
    )

    async def assistant(state: MessagesState):
        response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])  # type: ignore
        return {"messages": [response]}

    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


def _convert_history_to_messages(history: List) -> List:
    """Map request history to LangChain message objects."""
    converted: List = []
    for m in history:
        if m.role == "user":
            converted.append(HumanMessage(m.content))
        elif m.role == "assistant":
            converted.append(AIMessage(m.content))
        elif m.role == "system":
            converted.append(SystemMessage(m.content))
    return converted


@router.post("/tools", response_model=ChatResponse)
async def chat_with_tools(request: ChatHistoryRequest) -> ChatResponse:
    """Chat endpoint that can call tools via a LangGraph ReAct loop."""
    try:
        app = _build_react_graph()
        messages = _convert_history_to_messages(request.messages or [])
        # Log input/history messages
        format_messages(messages)

        result = await app.ainvoke({"messages": messages})
        all_messages = result.get("messages", [])
        # Log all agent messages (includes tool calls and assistant outputs)
        try:
            format_messages(all_messages)
        except Exception:
            logger.debug("format_messages failed for all_messages")
        ai_messages = [m for m in reversed(all_messages) if isinstance(m, AIMessage)]
        final_text = ""
        for m in ai_messages:
            if not getattr(m, "tool_calls", None):
                final_text = m.content or ""
                break
        if not final_text:
            final_text = ai_messages[0].content if ai_messages else ""

        return ChatResponse(text=final_text, model=MODEL_NAME, success=True, usage=None)
    except Exception as e:
        logger.error(f"Tools chat error: {e}")
        return ChatResponse(text="", model=MODEL_NAME, success=False, error=str(e))


@router.post("/tools/stream")
async def chat_with_tools_stream(request: ChatHistoryRequest):
    """NDJSON streaming endpoint that emits step-wise updates while the agent runs.

    Each line is a JSON object with a minimal schema, for example:
    - {"type":"status","message":"assistant"}
    - {"type":"tool","name":"calculator","args":{...}}
    - {"type":"assistant","content":"final markdown answer"}
    - {"type":"done"}
    """
    try:
        app = _build_react_graph()
        messages = _convert_history_to_messages(request.messages or [])
        # Log input/history messages
        format_messages(messages)

        async def iter_events():
            try:
                # Initial status
                yield json.dumps({"type": "status", "message": "start"}) + "\n"

                # Skip over the initial history we provided, so we only stream new steps
                last_len = len(messages)
                # Track how many tool calls we've already announced per message index
                tool_counts = {}
                # Stream state updates from the graph
                async for update in app.astream(
                    {"messages": messages}, stream_mode="values"
                ):
                    # We expect updates to include the evolving MessagesState under key "messages"
                    state_messages = (
                        update.get("messages") if isinstance(update, dict) else None
                    )
                    if not state_messages:
                        continue

                    # Emit only new tool calls per message (concise dedup)
                    for idx, msg in enumerate(state_messages):
                        tool_calls = getattr(msg, "tool_calls", None)
                        if not tool_calls:
                            continue
                        prev = tool_counts.get(idx, 0)
                        new_calls = tool_calls[prev:]
                        if new_calls:
                            if len(new_calls) > 1:
                                items = []
                                for tc in new_calls:
                                    name = (
                                        tc.get("name")
                                        if isinstance(tc, dict)
                                        else getattr(tc, "name", None)
                                    )
                                    args = (
                                        tc.get("args")
                                        if isinstance(tc, dict)
                                        else getattr(tc, "args", None)
                                    )
                                    items.append({"name": name, "args": args})
                                yield json.dumps(
                                    {"type": "tools", "items": items}
                                ) + "\n"
                            else:
                                tc = new_calls[0]
                                name = (
                                    tc.get("name")
                                    if isinstance(tc, dict)
                                    else getattr(tc, "name", None)
                                )
                                args = (
                                    tc.get("args")
                                    if isinstance(tc, dict)
                                    else getattr(tc, "args", None)
                                )
                                yield json.dumps(
                                    {"type": "tool", "name": name, "args": args}
                                ) + "\n"
                            tool_counts[idx] = len(tool_calls)

                    # Only emit deltas for newly appended messages (assistant final text)
                    try:
                        current_len = len(state_messages)
                    except Exception:
                        current_len = 0

                    if current_len <= last_len:
                        continue

                    new_messages = state_messages[last_len:]
                    last_len = current_len

                    for m in new_messages:
                        # Log only newly appended messages for readability
                        try:
                            format_messages([m])
                        except Exception:
                            logger.debug("format_messages failed for new message")
                        # Assistant plain text messages (only when no tool calls on that message)
                        tool_calls = getattr(m, "tool_calls", None)
                        if isinstance(m, AIMessage) and not tool_calls:
                            yield (
                                json.dumps(
                                    {
                                        "type": "assistant",
                                        "content": m.content or "",
                                    }
                                )
                                + "\n"
                            )

                # Done
                yield json.dumps({"type": "done"}) + "\n"
            except Exception as exc:
                logger.error(f"Tools stream error: {exc}")
                yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

        return StreamingResponse(
            iter_events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        logger.error(f"Failed to start tools stream: {e}")

        # Return a tiny NDJSON error stream to avoid breaking the consumer
        def err():
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(err(), media_type="application/x-ndjson")
