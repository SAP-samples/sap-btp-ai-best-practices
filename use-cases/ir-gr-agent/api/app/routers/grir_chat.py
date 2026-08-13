"""GR/IR procurement chat endpoint — multi-turn LangGraph agent with streaming."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import grir_agent
from .grir_session_store import GrirSessionStore
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

_store = GrirSessionStore()


class GRIRChatRequest(BaseModel):
    message: str
    session_id: str = ""


@router.post("/chat")
async def grir_chat(request: GRIRChatRequest):
    """Multi-turn GR/IR chat endpoint. Streams NDJSON tokens."""
    session_id = request.session_id or str(uuid.uuid4())
    history = _store.get_or_create(session_id)

    return StreamingResponse(
        grir_agent.stream(request.message, history),
        media_type="application/x-ndjson",
        headers={"X-Session-Id": session_id},
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    _store.clear(session_id)
    return {"cleared": session_id}
