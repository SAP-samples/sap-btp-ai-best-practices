from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.nbo import ResetAnswersResponse, SavedAnswersResponse
from app.security import get_api_key
from app.services.saved_answers import normalize_billing_account, saved_answer_service


router = APIRouter(dependencies=[Depends(get_api_key)])


@router.get("/{billing_account}", response_model=SavedAnswersResponse)
async def get_saved_answers(billing_account: str) -> SavedAnswersResponse:
    """Return saved customer answer overlays for one billing account."""
    normalized_account = normalize_billing_account(billing_account)
    return SavedAnswersResponse(
        billing_account=normalized_account,
        answers=saved_answer_service.get_answer_values(normalized_account),
    )


@router.delete("/{billing_account}", response_model=ResetAnswersResponse)
async def reset_account_answers(billing_account: str) -> ResetAnswersResponse:
    """Delete saved customer answer overlays for one billing account."""
    normalized_account = normalize_billing_account(billing_account)
    deleted_count = saved_answer_service.reset_account(normalized_account)
    return ResetAnswersResponse(
        billing_account=normalized_account,
        deleted_count=deleted_count,
    )


@router.delete("", response_model=ResetAnswersResponse)
async def reset_all_answers() -> ResetAnswersResponse:
    """Delete all saved customer answer overlays."""
    return ResetAnswersResponse(
        billing_account=None,
        deleted_count=saved_answer_service.reset_all(),
    )
