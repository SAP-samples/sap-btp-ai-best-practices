# Models package for organizing domain-specific Pydantic models
# This allows for better organization as the application grows

# Re-export models for convenience
from .chat import ChatRequest, ChatResponse
from .chat_history import ChatMessage, ChatHistoryRequest
from .common import ErrorResponse, HealthResponse

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "ChatMessage",
    "ChatHistoryRequest",
]
