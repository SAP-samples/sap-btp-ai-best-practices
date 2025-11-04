"""Chat-related Pydantic models for LLM integration endpoints."""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ChatRequest(BaseModel):
    """Request model for chat endpoints.

    Attributes:
        message: The user's input text/prompt to send to the LLM.
        temperature: Controls response randomness (0.0-1.0). Lower values (0.0-0.3)
            produce more focused/deterministic responses, higher values (0.7-1.0)
            produce more creative/random responses. Defaults to 0.6.
        max_tokens: Maximum number of tokens in the response. Defaults to 1000.
    """

    message: str
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024


class ChatResponse(BaseModel):
    """Unified response model for all chat endpoints.

    Attributes:
        text: The generated response text from the LLM. Empty string if request failed.
        model: Identifier of the model used (e.g., "anthropic--claude-4-sonnet",
            "gpt-4.1", "gemini-2.5-flash").
        success: Whether the request completed successfully.
        usage: Token usage statistics containing prompt_tokens, completion_tokens,
            and total_tokens. None if not available.
        error: Error message if request failed. None when successful.
    """

    text: str
    model: str
    success: bool
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Optional serialized message log for clients that want to render a trace
    messages: Optional[List[Dict[str, Any]]] = None


class ChatMessage(BaseModel):
    """Represents a single chat message in a conversation history."""

    role: str  # "system" | "user" | "assistant"
    content: str


class ChatHistoryRequest(BaseModel):
    """Request model for chat with history endpoints.

    Attributes:
        messages: The conversation so far (in order). Should include the latest user message.
        temperature: See ChatRequest.temperature.
        max_tokens: See ChatRequest.max_tokens.
    """

    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
