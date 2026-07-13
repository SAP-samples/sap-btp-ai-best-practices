from typing import List, Optional, Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Single chat message for history-based endpoints."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatHistoryRequest(BaseModel):
    """Request model for history-based chat endpoints.

    Attributes:
        messages: Ordered list of conversation messages (system/user/assistant)
        temperature: Generation temperature (optional)
        max_tokens: Max tokens for the response (optional)
    """

    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 1000
