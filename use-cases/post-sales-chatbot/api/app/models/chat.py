"""Chat-related Pydantic models for Apex API."""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class ChatRequest(BaseModel):
    """Request model for chat endpoints.

    Attributes:
        message: The user's input text/prompt to send to the LLM.
        temperature: Controls response randomness (0.0-1.0). Defaults to 0.0.
        max_tokens: Maximum number of tokens in the response. Defaults to 2000.
    """

    message: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2000


class ChatResponse(BaseModel):
    """Unified response model for all chat endpoints.

    Attributes:
        text: The generated response text from the LLM. Empty string if request failed.
        model: Identifier of the model used.
        success: Whether the request completed successfully.
        session_id: The session ID for the conversation.
        usage: Token usage statistics. None if not available.
        error: Error message if request failed. None when successful.
    """

    text: str
    model: str
    success: bool
    session_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
