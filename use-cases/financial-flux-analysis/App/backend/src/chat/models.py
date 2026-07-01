"""
Pydantic models for the AI Chat module.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatMessageResponse(BaseModel):
    response: str
    conversation_id: str
    metadata: Dict[str, Any] = {}