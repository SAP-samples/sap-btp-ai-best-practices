from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.chat.service import (
    ChatInvalidDeclineError,
    ChatThreadNotFoundError,
    chat_workflow_service,
)
from app.models.nbo import ChatMessageRequest, ChatThreadStateResponse, CreateThreadResponse, DeclineProgramRequest
from app.security import get_api_key


router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("/threads", response_model=CreateThreadResponse, status_code=201)
async def create_thread() -> CreateThreadResponse:
    state = chat_workflow_service.create_thread()
    return CreateThreadResponse(thread_id=state.thread_id, state=state)


@router.get("/threads/{thread_id}", response_model=ChatThreadStateResponse)
async def get_thread(thread_id: str) -> ChatThreadStateResponse:
    try:
        return chat_workflow_service.get_thread(thread_id)
    except ChatThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/threads/{thread_id}/messages", response_model=ChatThreadStateResponse)
async def post_message(thread_id: str, request: ChatMessageRequest) -> ChatThreadStateResponse:
    try:
        return chat_workflow_service.post_message(thread_id, request.message)
    except ChatThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/threads/{thread_id}/decline", response_model=ChatThreadStateResponse)
async def decline_program(thread_id: str, request: DeclineProgramRequest) -> ChatThreadStateResponse:
    try:
        return chat_workflow_service.decline_program(thread_id, request.program_id)
    except ChatThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatInvalidDeclineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
