"""
Apex Chat Router - Main chat endpoints for Apex Automotive Services.

Implements:
- POST /apex/chat - Single response chat with tools
- POST /apex/chat/stream - NDJSON streaming with tool execution updates
- POST /apex/reset - Reset session and get initial greeting
"""

import logging
import json
from typing import List

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from ..models.chat import ChatResponse
from ..models.chat_history import ChatMessage
from ..models.session import SessionChatRequest, SessionResetResponse
from ..security import get_api_key
from ..config import INITIAL_PROMPT, LLM_MODEL_NAME
from ..utils.langgraph.common import make_llm
from ..utils.langgraph.apex_tools import APEX_TOOLS
from ..services.session_manager import session_manager

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


def _build_apex_agent(session_id: str):
    """Build LangGraph ReAct agent with Apex tools."""
    llm = make_llm(provider="openai", model_name=LLM_MODEL_NAME, temperature=0.0)

    # Create tools with session_id injected
    # Note: The tools access session_id via their parameter
    tools = APEX_TOOLS

    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(MessagesState)

    # System message with Apex persona and session context
    sys_msg = SystemMessage(
        content=f"{INITIAL_PROMPT}\n\nIMPORTANT: When calling tools, always include session_id=\"{session_id}\" as a parameter."
    )

    async def assistant(state: MessagesState):
        response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


def _convert_history_to_messages(history: List[dict]) -> List:
    """Convert chat history dicts to LangChain message objects."""
    converted = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content))
        elif role == "assistant":
            converted.append(AIMessage(content))
        elif role == "system":
            converted.append(SystemMessage(content))
    return converted


@router.post("/apex/chat", response_model=ChatResponse)
async def apex_chat(
    request: SessionChatRequest,
    response: Response,
    x_session_id: str = Header(None, alias="X-Session-Id")
) -> ChatResponse:
    """Chat endpoint for Apex Assistant with tool support."""
    try:
        session_id, state = session_manager.get_or_create(x_session_id)

        # Set session ID in response header
        response.headers["X-Session-Id"] = session_id

        # Add user message to history
        state.chat_history.append({"role": "user", "content": request.message})

        # Build agent and invoke
        app = _build_apex_agent(session_id)
        messages = _convert_history_to_messages(state.chat_history)

        result = await app.ainvoke({"messages": messages})

        # Extract final response
        all_messages = result.get("messages", [])
        ai_messages = [m for m in reversed(all_messages) if isinstance(m, AIMessage)]
        final_text = ""
        for m in ai_messages:
            if not getattr(m, "tool_calls", None):
                final_text = m.content or ""
                break

        if not final_text and ai_messages:
            final_text = ai_messages[0].content if ai_messages else ""

        # Update session history
        state.chat_history.append({"role": "assistant", "content": final_text})
        session_manager.update(session_id, state)

        return ChatResponse(
            text=final_text,
            model=LLM_MODEL_NAME,
            success=True,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Apex chat error: {e}")
        return ChatResponse(
            text="",
            model=LLM_MODEL_NAME,
            success=False,
            error=str(e),
            session_id=x_session_id
        )


@router.post("/apex/chat/stream")
async def apex_chat_stream(
    request: SessionChatRequest,
    x_session_id: str = Header(None, alias="X-Session-Id")
):
    """NDJSON streaming endpoint that emits step-wise updates while the agent runs.

    Each line is a JSON object with a minimal schema:
    - {"type":"status","message":"start"}
    - {"type":"tool","name":"find_client","args":{...}}
    - {"type":"assistant","content":"final response"}
    - {"type":"done","session_id":"..."}
    """
    try:
        session_id, state = session_manager.get_or_create(x_session_id)

        # Add user message to history
        state.chat_history.append({"role": "user", "content": request.message})

        app = _build_apex_agent(session_id)
        messages = _convert_history_to_messages(state.chat_history)

        async def iter_events():
            final_text = ""
            try:
                # Initial status with session ID
                yield json.dumps({"type": "status", "message": "start", "session_id": session_id}) + "\n"

                last_len = len(messages)
                tool_counts = {}

                async for update in app.astream(
                    {"messages": messages}, stream_mode="values"
                ):
                    state_messages = (
                        update.get("messages") if isinstance(update, dict) else None
                    )
                    if not state_messages:
                        continue

                    # Emit tool calls
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
                                yield json.dumps({"type": "tools", "items": items}) + "\n"
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
                                yield json.dumps({"type": "tool", "name": name, "args": args}) + "\n"
                            tool_counts[idx] = len(tool_calls)

                    # Emit new assistant messages
                    try:
                        current_len = len(state_messages)
                    except Exception:
                        current_len = 0

                    if current_len <= last_len:
                        continue

                    new_messages = state_messages[last_len:]
                    last_len = current_len

                    for m in new_messages:
                        tool_calls = getattr(m, "tool_calls", None)
                        if isinstance(m, AIMessage) and not tool_calls:
                            final_text = m.content or ""
                            yield json.dumps({
                                "type": "assistant",
                                "content": final_text
                            }) + "\n"

                # Update session with final response
                if final_text:
                    state.chat_history.append({"role": "assistant", "content": final_text})
                    session_manager.update(session_id, state)

                # Done
                yield json.dumps({"type": "done", "session_id": session_id}) + "\n"

            except Exception as exc:
                logger.error(f"Apex stream error: {exc}")
                yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

        return StreamingResponse(
            iter_events(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Session-Id": session_id
            }
        )
    except Exception as e:
        logger.error(f"Failed to start apex stream: {e}")

        def err():
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(err(), media_type="application/x-ndjson")


@router.post("/apex/reset", response_model=SessionResetResponse)
async def apex_reset(
    response: Response,
    x_session_id: str = Header(None, alias="X-Session-Id")
) -> SessionResetResponse:
    """Reset conversation and get initial greeting."""
    try:
        session_id, _ = session_manager.get_or_create(x_session_id)
        state = session_manager.reset(session_id)

        # Set session ID in response header
        response.headers["X-Session-Id"] = session_id

        # Generate initial greeting
        llm = make_llm(provider="openai", model_name=LLM_MODEL_NAME, temperature=0.2)

        greeting_prompt = """Start the conversation by greeting the user warmly in English.
        Introduce yourself as Apex Assistant, the customer service assistant for Apex Automotive Services.
        Ask them how you can help them today and mention they can identify themselves with their name, email, or phone number to get started.
        Keep the greeting friendly but concise (2-3 sentences)."""

        response_msg = await llm.ainvoke([
            SystemMessage(content=INITIAL_PROMPT),
            HumanMessage(content=greeting_prompt)
        ])

        initial_greeting = response_msg.content

        # Store greeting in history
        state.chat_history.append({"role": "assistant", "content": initial_greeting})
        session_manager.update(session_id, state)

        return SessionResetResponse(
            message=initial_greeting,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Apex reset error: {e}")
        # Fallback greeting
        fallback_greeting = "Hello! I'm Apex Assistant, your customer service helper for Apex Automotive Services. How can I assist you today? Please share your name, email, or phone number to get started."
        return SessionResetResponse(
            message=fallback_greeting,
            session_id=x_session_id or "error"
        )
