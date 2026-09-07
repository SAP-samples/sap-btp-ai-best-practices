"""Asynchronous API routes for AI-generated assessment PDF reports."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from app.models.reports import (
    AssessmentReportJobResponse,
    AssessmentReportJobStatusResponse,
    AssessmentReportRequest,
)
from app.routers.ai_review import get_ai_review_repository
from app.security import get_api_key
from app.services.assessment_report_context import build_assessment_report_source

router = APIRouter(dependencies=[Depends(get_api_key)])


def _attachment_file_name(file_name: str) -> str:
    """Return a header-safe basename for a generated report download.

    Inputs:
        file_name: Persisted generated report filename.

    Outputs:
        str: Basename without quote, slash, or newline header characters.
    """

    basename = Path(file_name).name
    return basename.replace('"', "").replace("\\", "").replace("\r", "").replace("\n", "")


@router.post(
    "/calification-reports",
    response_model=AssessmentReportJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_assessment_report_job(
    request: AssessmentReportRequest,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentReportJobResponse:
    """Snapshot current HANA answers and enqueue asynchronous report work.

    Inputs:
        request: Assessment ID/language plus ignored deprecated compatibility fields.
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        AssessmentReportJobResponse: Accepted job identity for browser polling.
    """

    try:
        source = build_assessment_report_source(
            repository=repository,
            assessment_id=request.assessment_id,
            language=request.language,
        )
        return repository.create_assessment_report_job(source)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/calification-reports/{job_id}",
    response_model=AssessmentReportJobStatusResponse,
)
async def get_assessment_report_job_status(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentReportJobStatusResponse:
    """Return current progress for one assessment report job.

    Inputs:
        job_id: Report job identifier returned by the enqueue route.
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        AssessmentReportJobStatusResponse: Polling state and download metadata.
    """

    try:
        return repository.get_assessment_report_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/calification-reports/{job_id}/file")
async def download_assessment_report(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> Response:
    """Download the PDF for a completed, unexpired report job.

    Inputs:
        job_id: Report job identifier returned by the enqueue route.
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        Response: PDF bytes with an attachment disposition header.
    """

    try:
        download = repository.get_assessment_report_download(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    file_name = _attachment_file_name(download.file_name)
    return Response(
        content=download.content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )
