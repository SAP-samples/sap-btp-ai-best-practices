"""Page-session workspace chat with safe NDJSON activity and independent jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Annotated, Any, Literal
from weakref import WeakValueDictionary

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.workspace_tools import workspace_tools
from app.workspace.dependencies import get_service

router = APIRouter(prefix="/api/chat", tags=["chat"])
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "agent/config/agent.yaml"
_CONTEXT_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_ACTIVE_TURNS: set[asyncio.Task] = set()
_TURN_CONTEXTS: dict[asyncio.Task, str] = {}
_LOGGER = logging.getLogger(__name__)


class ChatHistoryMessage(BaseModel):
    """One bounded prior turn supplied by the current browser page only."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100000)


class ChatMessage(BaseModel):
    """One user message, page history, and optional selected workspace IDs."""

    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=100000)
    context_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    plant_profile_id: str | None = None
    dataset_id: str | None = None
    draft_id: str | None = None
    run_id: str | None = None
    point_index: int | None = Field(default=None, ge=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=40)
    context_summary: str | None = Field(default=None, max_length=12_000)


def _context_lock(context_id: str) -> asyncio.Lock:
    """Share one in-process turn lock per conversation without retaining idle IDs."""
    lock = _CONTEXT_LOCKS.get(context_id)
    if lock is None:
        lock = asyncio.Lock()
        _CONTEXT_LOCKS[context_id] = lock
    return lock


async def create_runtime(service: Any, context_id: str, on_event: Callable) -> Any:
    """Create the configured runtime with shared tools and disabled durable memory."""
    from app.agent.runtime import AgentRuntime

    choice = await asyncio.to_thread(service.get_ai_model_settings)
    return await AgentRuntime.create(
        _CONFIG_PATH,
        extra_tools=workspace_tools(service, context_id, on_event),
        model_name=choice["model"],
    )


def get_runtime_factory() -> Callable:
    """Expose injectable runtime construction so tests never initialize providers."""
    return create_runtime


def _public_event(event: dict) -> dict | None:
    """Allow tool activity and mutation events while excluding model reasoning."""
    event_type = event.get("type")
    if event_type == "run_progress":
        return {
            key: event.get(key)
            for key in ("type", "run_id", "status", "stage", "progress")
        }
    if event_type == "tool_call":
        return {
            "type": "tool_call",
            "name": event.get("name"),
            "args": event.get("args") or {},
        }
    if event_type == "tool_result":
        return {
            "type": "tool_result",
            "name": event.get("name"),
            "preview": str(event.get("preview", ""))[:12000],
        }
    if event_type == "draft_changed":
        return {
            key: event[key]
            for key in ("type", "draft_id", "revision", "draft")
            if key in event
        }
    if event_type == "run_created":
        return {key: event[key] for key in ("type", "run_id", "run") if key in event}
    return None


def chat_error_message(error: Exception) -> str:
    """Return actionable public failure guidance without exposing provider response data."""
    independent = " Submitted optimizer runs continue independently; check Run history before launching again."
    if getattr(error, "status_code", None) == 429:
        return (
            "SAP AI Core's configured model reached its rate limit. Retry the chat after the quota recovers."
            + independent
        )
    if type(error).__name__ == "GraphRecursionError":
        return (
            "The agent reached its step limit. Use New conversation to clear its history and retry."
            + independent
        )
    if isinstance(error, TimeoutError) or "Timeout" in type(error).__name__:
        return (
            "The configured AI Core model request timed out. Retry the chat."
            + independent
        )
    return (
        "The chat request could not be completed. Check the selected workspace and retry."
        + independent
    )


@router.post("/{context_id}/reset")
async def reset_chat(context_id: str) -> dict:
    """Stop active/queued chat turns for this page; leave optimizer jobs untouched."""
    tasks = [
        task for task, context in list(_TURN_CONTEXTS.items()) if context == context_id
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return {"reset": True}


@router.post("")
async def chat_message(
    body: ChatMessage,
    service: Annotated[Any, Depends(get_service)],
    runtime_factory: Annotated[Callable, Depends(get_runtime_factory)],
) -> StreamingResponse:
    """Stream safe activity and final text for a serialized page session.

    Disconnecting stops delivery only. The detached turn finishes its current
    agent work; independently submitted optimization jobs have their own lifecycle.
    """
    if not body.message.strip():
        raise HTTPException(422, "Message cannot be blank")
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    lock = _context_lock(body.context_id)

    def on_event(event: dict) -> None:
        """Transfer callbacks from LangChain worker threads to the response loop."""
        public = _public_event(event)
        if public is not None:
            loop.call_soon_threadsafe(queue.put_nowait, public)

    async def run_turn() -> None:
        """Finish one context-serialized turn and always release owned runtime state."""
        runtime = None
        try:
            async with lock:
                patch = body.model_dump(
                    exclude={"message", "context_id", "history", "context_summary"},
                    exclude_unset=True,
                )
                await asyncio.to_thread(service.context, body.context_id, patch or None)
                runtime = await runtime_factory(service, body.context_id, on_event)
                invocation = {
                    "on_event": on_event,
                    "session_history": [item.model_dump() for item in body.history],
                }
                if body.context_summary is not None:
                    invocation["context_summary"] = body.context_summary
                result = await runtime.ainvoke(body.message, body.context_id, **invocation)
                await asyncio.sleep(0)
                assistant_event = {"type": "assistant", "text": result.output_text}
                if getattr(result, "history_compacted", False):
                    assistant_event["context_summary"] = result.context_summary
                    assistant_event["history_compacted"] = True
                queue.put_nowait(assistant_event)
        except Exception as exc:
            _LOGGER.exception("Workspace chat turn failed")
            queue.put_nowait(
                {
                    "type": "error",
                    "message": chat_error_message(exc),
                }
            )
        finally:
            if runtime is not None:
                try:
                    await runtime.aclose()
                except Exception:
                    _LOGGER.exception("Workspace chat runtime cleanup failed")
            # Flush callbacks scheduled by worker threads before the terminal marker.
            await asyncio.sleep(0)
            queue.put_nowait(None)

    task = asyncio.create_task(run_turn())
    _ACTIVE_TURNS.add(task)
    _TURN_CONTEXTS[task] = body.context_id

    def forget_turn(finished: asyncio.Task) -> None:
        """Release per-turn tracking after normal completion or explicit reset."""
        _ACTIVE_TURNS.discard(finished)
        _TURN_CONTEXTS.pop(finished, None)

    task.add_done_callback(forget_turn)

    async def events() -> AsyncIterator[str]:
        """Yield newline-delimited public events until the detached turn completes."""
        while True:
            event = await queue.get()
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
