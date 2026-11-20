"""Authenticated multipart upload endpoint and bounded transport conversion."""

from __future__ import annotations

from typing import Annotated
from datetime import date
from dataclasses import replace
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile, status

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_mapping import upload_response_from_result
from app.ecn.api_models import UploadResponse
from app.ecn.errors import SourceValidationError
from app.ecn.intake.files import read_upload_bytes, validate_upload_metadata
from app.ecn.models import ArtifactKind
from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.ingestion import UploadBytesRequest, UploadFileBytes
from app.ecn.settings import EcnSettings

router = APIRouter()


async def _bounded_file(
    upload: UploadFile,
    kind: ArtifactKind,
    maximum_bytes: int,
) -> UploadFileBytes:
    """Validate safe metadata before reading one multipart file within its bound.

    Args:
        upload: FastAPI multipart file stream.
        kind: Expected domain artifact kind and its extension/media allowlist.
        maximum_bytes: Maximum accepted streamed byte count.

    Returns:
        Immutable sanitized metadata and complete bounded bytes.
    """
    metadata = validate_upload_metadata(upload.filename, upload.content_type, kind)
    content = await read_upload_bytes(upload, maximum_bytes)
    return UploadFileBytes(metadata.file_name, metadata.media_type, content)


async def build_bounded_upload_request(
    request_id: UUID,
    ecn_file: UploadFile,
    settings: EcnSettings,
) -> UploadBytesRequest:
    """Validate metadata and read multipart files within configured byte limits.

    Args:
        request_id: Parsed idempotency UUID validated before this function runs.
        ecn_file: Required ECN JSON multipart file.
        settings: Runtime upload byte bounds.

    Returns:
        Immutable ingestion request containing only bounded source bytes.
    """
    ecn = await _bounded_file(
        ecn_file, ArtifactKind.ECN_JSON, settings.max_json_upload_bytes
    )
    return UploadBytesRequest(request_id, ecn, None, actor_id=None)


async def _schedule_automatic_analyses(
    services: ServiceContainer,
    ecn_row_ids: tuple[str, ...],
    job_ids: tuple[str, ...],
) -> None:
    """Start committed summary jobs and exact local drawing fixtures after upload success."""
    if job_ids:
        services.runtime.schedule_jobs(job_ids)
    runtime = getattr(services, "manufacturing_feasibility_runtime", None)
    if runtime is None or services.snapshot_service is None:
        return
    material_numbers: dict[str, tuple[str, ...]] = {}
    for ecn_row_id in ecn_row_ids:
        snapshot = await run_blocking(services.snapshot_service.get, ecn_row_id)
        material_numbers[ecn_row_id] = tuple(
            sorted(
                {
                    str(item["number"])
                    for item in snapshot.get("objects", [])
                    if item.get("technical")
                    and "ecn" in item.get("origins", [])
                    and item.get("number")
                }
            )
        )
    runtime.schedule(ecn_row_ids, material_numbers)


@router.post(
    "/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_sources(
    request: Request,
    request_id: Annotated[UUID, Form()],
    ecn_file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    services: Annotated[ServiceContainer, Depends(get_services)],
    assessment_date: Annotated[date | None, Form()] = None,
    data_group: Annotated[str, Form(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")] = "customer",
) -> UploadResponse:
    """Publish source facts, then automatically start committed AI assessments.

    The response remains a deterministic upload acknowledgment; independently
    tracked summary and drawing jobs run only after publication succeeds.
    """
    submitted_fields = set((await request.form()).keys())
    extras = submitted_fields - {"request_id", "ecn_file", "assessment_date", "data_group"}
    if extras:
        raise SourceValidationError(
            "UPLOAD_FIELDS_INVALID",
            "ECN upload accepts source, request identity, assessment date and data group; manage reusable sources in Settings.",
            status_code=400,
            details={"fields": sorted(extras)},
        )
    upload_request = await build_bounded_upload_request(
        request_id, ecn_file, services.settings
    )
    upload_request = replace(upload_request, assessment_date=assessment_date, data_group=data_group)
    result = await run_blocking(
        services.ingestion_service.ingest_upload, upload_request
    )
    if result.projection_ecn_ids:
        background_tasks.add_task(
            run_blocking,
            services.rdf_projection_service.project_many,
            result.projection_ecn_ids,
        )
    background_tasks.add_task(
        _schedule_automatic_analyses,
        services,
        result.projection_ecn_ids,
        result.queued_job_ids,
    )
    progress = await run_blocking(
        services.analysis_repository.get_progress, result.upload.upload_id
    )
    # Response remains deterministic; background task owns automatic scheduling.
    return upload_response_from_result(result, progress)
