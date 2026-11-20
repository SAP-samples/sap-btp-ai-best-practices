"""Analysis progress and failed-only retry endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_mapping import retry_response_from_result
from app.ecn.api_models import AnalysisProgressResponse, RetryResponse
from app.ecn.errors import PersistenceNotFoundError, PersistenceStateError
from app.ecn.services.blocking_io import run_blocking
from app.persistence.records import RetryQueueStatus

router = APIRouter()


@router.get("/analysis-progress", response_model=AnalysisProgressResponse)
async def get_analysis_progress(
    services: Annotated[ServiceContainer, Depends(get_services)],
    upload_id: Annotated[UUID | None, Query()] = None,
) -> AnalysisProgressResponse:
    """Return global or upload-scoped mutually exclusive job-state counts.

    Args:
        upload_id: Optional upload batch UUID used to scope the counts.
        services: Application service container.

    Returns:
        Validated aggregate counts including the requested scope identity.
    """
    canonical_id = str(upload_id) if upload_id is not None else None
    progress = await run_blocking(
        services.analysis_repository.get_progress, canonical_id
    )
    return AnalysisProgressResponse(upload_id=upload_id, **progress.model_dump())


@router.post(
    "/ecns/{ecn_row_id}/analysis/retry",
    response_model=RetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_analysis(
    ecn_row_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> RetryResponse:
    """Queue one failed analysis and schedule it only after the commit succeeds.

    Args:
        ecn_row_id: Internal immutable ECN occurrence UUID.
        services: Application service container.

    Returns:
        Committed queued job identity.
    """
    canonical_id = str(ecn_row_id)
    result = await services.retry_service.queue_failed(canonical_id)
    if result.status is RetryQueueStatus.ECN_NOT_FOUND:
        raise PersistenceNotFoundError(
            "ECN_NOT_FOUND",
            "The requested ECN occurrence was not found",
            status_code=404,
        )
    if result.status is RetryQueueStatus.NOT_RETRYABLE:
        raise PersistenceStateError(
            "ANALYSIS_NOT_RETRYABLE",
            "Only a failed ECN analysis without a stored result can be retried",
            status_code=409,
        )
    response = retry_response_from_result(canonical_id, result)
    services.runtime.schedule_retry(str(response.job_id))
    return response
