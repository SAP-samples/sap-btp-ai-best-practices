"""Session-related Pydantic models for Apex API."""

from pydantic import BaseModel
from typing import Optional


class SessionChatRequest(BaseModel):
    """Request model for session-based chat.

    Attributes:
        message: The user's input text/prompt.
        temperature: Controls response randomness (0.0-1.0). Defaults to 0.0.
        max_tokens: Maximum number of tokens in the response. Defaults to 2000.
    """

    message: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2000


class SessionResetResponse(BaseModel):
    """Response model for session reset.

    Attributes:
        message: The initial greeting message.
        session_id: The session ID for the conversation.
    """

    message: str
    session_id: str
