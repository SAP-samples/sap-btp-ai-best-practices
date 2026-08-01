"""
FastAPI router for the AI Chat module.
Mounts at /api/chat — does NOT modify any existing routes.
"""

import logging

from fastapi import APIRouter, HTTPException

from src.chat.models import ChatMessageRequest, ChatMessageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# The ChatService instance is injected at startup from api.py
# to share the FinancialProcessor singleton.
_chat_service = None


def init_chat_service(processor) -> None:
    """
    Called once from api.py after the FinancialProcessor is created.
    Injects the shared processor so the chat layer reuses the cached dataset.
    """
    global _chat_service
    from src.chat.chat_service import ChatService
    _chat_service = ChatService(processor)
    logger.info("chat.routes: ChatService initialized")


@router.post("/message", response_model=ChatMessageResponse)
async def post_chat_message(request: ChatMessageRequest):
    """
    POST /api/chat/message

    Accepts a user message and returns an AI-generated financial analysis response.
    Maintains multi-turn conversation context via conversation_id.
    """
    if _chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="Chat service not initialized. Please try again in a moment.",
        )

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = _chat_service.handle_message(
            message=request.message.strip(),
            conversation_id=request.conversation_id,
            context=request.context,
        )
        return ChatMessageResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            metadata=result.get("metadata", {}),
        )
    except Exception as exc:
        logger.error("chat.routes: unhandled error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(exc)[:200]}",
        )


@router.get("/health")
async def chat_health():
    """Quick health check for the chat module."""
    return {
        "status": "ok",
        "service_ready": _chat_service is not None,
    }