import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.security import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


class SampleSummaryResponse(BaseModel):
    total_items: int
    active_count: int
    status: str
    last_updated: str


@router.get("/summary", response_model=SampleSummaryResponse)
async def get_sample_summary() -> SampleSummaryResponse:
    """Return a sample summary. Called by Joule capability via BTP destination."""
    return SampleSummaryResponse(
        total_items=42,
        active_count=37,
        status="healthy",
        last_updated="2025-01-15T10:30:00Z",
    )
