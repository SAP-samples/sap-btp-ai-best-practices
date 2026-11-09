"""Evaluation router — scores ideation solutions via AI."""

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import UserIdentity, get_current_user
from ..services.evaluation import evaluate_solution

logger = logging.getLogger(__name__)
router = APIRouter()


class EvaluationRequest(BaseModel):
    industry: str
    title: str
    problem: str
    solution: str
    app_type: str = "Agent"
    company_context: Optional[str] = None
    context_id: Optional[str] = None


class EvaluationResponse(BaseModel):
    score: int = Field(..., ge=0, le=1000)
    feedback: str
    detailedAnalysis: str
    suggestions: List[str]
    appropriatenessLevel: Literal["excellent", "good", "fair", "needs-improvement"]
    success: bool
    error: Optional[str] = None


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluationRequest,
    user: UserIdentity = Depends(get_current_user),
) -> EvaluationResponse:
    """Evaluate an ideation solution using AI scoring (1-1000)."""
    import asyncio

    result = await asyncio.to_thread(
        evaluate_solution,
        industry=request.industry,
        title=request.title,
        problem=request.problem,
        solution=request.solution,
        app_type=request.app_type,
        company_context=request.company_context,
        context_id=request.context_id or "",
    )

    return EvaluationResponse(**result)
