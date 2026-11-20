"""User-triggered assistant endpoints with bounded browser-memory conversation input.

``POST /ecns/{id}/chat`` answers about the open ECN (and other ECNs when the
user asks). ``POST /chat`` is the global chat for pages without an open ECN:
it reads every stored ECN and cannot save reviews.
"""

import asyncio
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.dependencies import ServiceContainer, get_services
from app.ecn.errors import PersistenceStateError
from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.reviews import actor_id

router = APIRouter()


class ChatTurn(BaseModel):
    """Only user and final assistant messages may be supplied as conversational context."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class GlobalChatRequest(BaseModel):
    """Carry one user message and bounded browser-memory turns; reject unbounded context."""

    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def bounded_history(self):
        """Cap total prior content so many individually valid turns cannot bypass the budget."""
        if sum(len(t.content) for t in self.history) > 24000:
            raise ValueError("Conversation context exceeds 24000 characters")
        if not self.message.strip():
            raise ValueError("Enter a message")
        return self


class ChatRequest(GlobalChatRequest):
    """Bind browser-memory turns to the exact selected ECN."""

    assessment_id: UUID


def _timeout_error() -> PersistenceStateError:
    """Return the shared 504 for an investigation that exceeded its time limit."""
    return PersistenceStateError(
        "CHAT_TIMEOUT",
        "The investigation timed out. The snapshot remains available; try a narrower question.",
        status_code=504,
    )


@router.post("/ecns/{ecn_row_id}/chat")
async def chat(
    ecn_row_id: UUID,
    body: ChatRequest,
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Make model calls only for a submitted user message.

    The only write the assistant can make is an engineering review the user
    confirmed in chat; it is attributed to the caller's hashed API key, like
    reviews saved from the review card.
    """
    if body.assessment_id != ecn_row_id:
        raise PersistenceStateError(
            "CHAT_SNAPSHOT_MISMATCH",
            "This conversation belongs to another ECN snapshot.",
            status_code=409,
        )
    await run_blocking(
        services.rdf_projection_service.require_current,
        str(ecn_row_id),
    )
    try:
        return await services.assistant_service.ainvoke(
            body.message,
            str(ecn_row_id),
            [t.model_dump() for t in body.history],
            actor_id=actor_id(request.headers.get("X-API-Key", "")),
        )
    except asyncio.TimeoutError:
        raise _timeout_error() from None


@router.post("/chat")
async def global_chat(
    body: GlobalChatRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Answer one message about all stored ECNs; read-only.

    No ECN is open, so no RDF precheck runs here: graph tools skip ECNs whose
    projection is not current and name them in their result.

    Args:
        body: The message and prior browser turns.
        services: Application service container.

    Returns:
        The assistant response with ``assessment_id: None``.
    """
    try:
        return await services.assistant_service.ainvoke(
            body.message, None, [t.model_dump() for t in body.history]
        )
    except asyncio.TimeoutError:
        raise _timeout_error() from None
