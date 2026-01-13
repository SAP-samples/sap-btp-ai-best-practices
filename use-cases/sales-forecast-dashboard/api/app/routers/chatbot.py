"""
Chatbot API routes for the forecasting agent.

Provides endpoints for:
- Sending messages to the AI agent
- Managing chat sessions
- Retrieving conversation history
"""

import logging
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..models.chatbot import (
    Attachment,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    DeleteResponse,
    ResetResponse,
    SessionInfo,
    ToolCall,
)
from ..security import get_api_key
from ..agent.session_store import get_session_store

# Output directory for agent-generated files
OUTPUT_DIR = Path(__file__).parent.parent / "agent" / "output"

logger = logging.getLogger(__name__)

# Main router with API key authentication
router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
    dependencies=[Depends(get_api_key)],
)

# Public router for file serving (no API key required for browser image requests)
public_router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    """
    Send a message to the forecasting agent and get a response.

    Creates a new session if session_id is not provided.
    Sessions persist across requests for multi-turn conversations.

    Parameters
    ----------
    request : ChatRequest
        The chat request containing the message and optional session_id

    Returns
    -------
    ChatResponse
        The agent's response with session_id for continuation
    """
    session_store = get_session_store()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Chat request for session {session_id}: {request.message[:100]}...")

    # Get or create session
    session = session_store.get_or_create(session_id)

    # Clear previously generated files before this query
    session.session_manager.clear_generated_files()

    try:
        result = session.run_query(request.message)

        # Convert tool calls to response model
        tool_calls = [
            ToolCall(name=tc["name"], args=tc.get("args", {}))
            for tc in result.get("tool_calls", [])
        ]

        # Build attachments from generated files
        attachments = []
        for file_path in session.session_manager.get_generated_files():
            path = Path(file_path)
            if path.exists():
                suffix = path.suffix.lower()
                if suffix in [".png", ".jpg", ".jpeg"]:
                    file_type = "image"
                elif suffix == ".pdf":
                    file_type = "pdf"
                elif suffix == ".csv":
                    file_type = "csv"
                else:
                    file_type = suffix.lstrip(".")
                attachments.append(
                    Attachment(
                        filename=path.name,
                        file_type=file_type,
                        url=f"/api/chatbot/files/{path.name}",
                    )
                )

        return ChatResponse(
            session_id=session_id,
            message=result["final_response"],
            tool_calls=tool_calls,
            attachments=attachments,
        )
    except Exception as e:
        logger.error(f"Error in chat for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> SessionInfo:
    """
    Get information about an existing session.

    Returns session metadata including origin date, scenarios,
    predictions, and model state.

    Parameters
    ----------
    session_id : str
        The session identifier

    Returns
    -------
    SessionInfo
        Session information and state summary
    """
    session_store = get_session_store()
    session = session_store.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    info = session.get_info()
    return SessionInfo(**info)


@router.delete("/session/{session_id}", response_model=DeleteResponse)
async def delete_session(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> DeleteResponse:
    """
    Delete a session and free resources.

    Parameters
    ----------
    session_id : str
        The session identifier

    Returns
    -------
    DeleteResponse
        Confirmation of deletion
    """
    session_store = get_session_store()
    deleted = session_store.delete(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Deleted session {session_id}")
    return DeleteResponse(status="deleted", session_id=session_id)


@router.get("/session/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    api_key: str = Depends(get_api_key),
) -> ConversationHistory:
    """
    Get conversation history for a session.

    Parameters
    ----------
    session_id : str
        The session identifier
    limit : int
        Maximum number of messages to return (default 50)

    Returns
    -------
    ConversationHistory
        The conversation messages
    """
    session_store = get_session_store()
    session = session_store.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session.get_history(limit=limit)
    return ConversationHistory(**history)


@router.post("/session/{session_id}/reset", response_model=ResetResponse)
async def reset_session(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> ResetResponse:
    """
    Reset a session's state while preserving conversation history.

    Clears scenarios, predictions, and cached models.

    Parameters
    ----------
    session_id : str
        The session identifier

    Returns
    -------
    ResetResponse
        Confirmation of reset
    """
    session_store = get_session_store()
    session = session_store.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.reset()
    logger.info(f"Reset session {session_id}")
    return ResetResponse(status="reset", session_id=session_id)


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(
    api_key: str = Depends(get_api_key),
) -> List[SessionInfo]:
    """
    List all active sessions.

    Returns
    -------
    List[SessionInfo]
        List of all active session summaries
    """
    session_store = get_session_store()
    sessions = session_store.list_sessions()
    return [SessionInfo(**s) for s in sessions]


@router.post("/cleanup")
async def cleanup_sessions(
    api_key: str = Depends(get_api_key),
) -> dict:
    """
    Clean up expired sessions.

    Removes sessions that have been inactive longer than the TTL.

    Returns
    -------
    dict
        Number of sessions removed
    """
    session_store = get_session_store()
    removed = session_store.cleanup_expired()
    logger.info(f"Cleaned up {removed} expired sessions")
    return {"removed": removed, "active": session_store.active_count}


@public_router.get("/files/{filename}")
async def get_file(
    filename: str,
) -> FileResponse:
    """
    Serve a file from the agent output directory.

    Used to retrieve images, CSVs, and PDFs generated by agent tools.

    Parameters
    ----------
    filename : str
        Name of the file to retrieve

    Returns
    -------
    FileResponse
        The requested file with appropriate content type
    """
    # Security: Validate filename (prevent path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on file extension
    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )


__all__ = ["router", "public_router"]
