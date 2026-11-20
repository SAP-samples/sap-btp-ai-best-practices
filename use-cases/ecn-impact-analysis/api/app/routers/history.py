"""Protected configuration-preserving history reset endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_mapping import history_reset_response_from_counts
from app.ecn.api_models import HistoryResetRequest, HistoryResetResponse
from app.ecn.services.blocking_io import run_blocking

router = APIRouter()


@router.delete("/history", response_model=HistoryResetResponse)
async def reset_history(
    request: HistoryResetRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> HistoryResetResponse:
    """Clear idle analysis history while preserving rules and table definitions.

    Args:
        request: Body carrying the exact destructive confirmation phrase.
        services: Application service container.

    Returns:
        Exact committed deletion counts for all seven history tables.
    """
    counts = await run_blocking(
        services.history_service.reset_history, request.confirmation
    )
    if getattr(services, "snapshot_service", None) is not None:
        services.snapshot_service.clear()
    return history_reset_response_from_counts(counts)
