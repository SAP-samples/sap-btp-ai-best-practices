from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..security import get_api_key
from ..services.chat_service import ChatService, SupplierRef

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)], tags=["chat"])

chat_service = ChatService()


class SupplierInput(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    kg_json: Optional[Dict[str, Any]] = None


class ChatAskRequest(BaseModel):
    question: str
    supplier1: SupplierInput
    supplier2: SupplierInput
    model: Optional[str] = None


class ChatAskResponse(BaseModel):
    answer_markdown: str
    sources: list[Dict[str, Any]]


@router.post("/v1/chat/ask", response_model=ChatAskResponse)
async def chat_ask(request: ChatAskRequest) -> ChatAskResponse:
    try:
        result = chat_service.ask(
            question=request.question,
            supplier1=SupplierRef(id=request.supplier1.id, name=request.supplier1.name, kg_json=request.supplier1.kg_json),
            supplier2=SupplierRef(id=request.supplier2.id, name=request.supplier2.name, kg_json=request.supplier2.kg_json),
            model=request.model,
        )
        return ChatAskResponse(**result)
    except FileNotFoundError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat ask failed: {e}")
        raise HTTPException(status_code=500, detail="Chat ask failed")
