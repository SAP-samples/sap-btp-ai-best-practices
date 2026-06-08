from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.models.nbo import EvaluateAccountRequest, RecommendationResponse, serialize_recommendation
from app.security import get_api_key
from app.services.recommendations import RecommendationService
from app.services.saved_answers import saved_answer_service


router = APIRouter(dependencies=[Depends(get_api_key)])
recommendation_service = RecommendationService()


@router.post("/evaluate", response_model=RecommendationResponse)
async def evaluate_account(request: EvaluateAccountRequest) -> RecommendationResponse:
    """Evaluate one account with persisted and newly supplied answer overlays."""
    saved_answers = saved_answer_service.get_answer_values(request.billing_account)
    request_answers = saved_answer_service.valid_answer_values(request.user_answers)
    if request_answers:
        saved_answer_service.save_answers(
            request.billing_account,
            request_answers,
            source_surface="lookup",
        )
    merged_answers = {**saved_answers, **request_answers}
    try:
        result = recommendation_service.evaluate_account(
            request.billing_account,
            user_answers=merged_answers,
            declined_programs=request.declined_programs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return serialize_recommendation(result)
