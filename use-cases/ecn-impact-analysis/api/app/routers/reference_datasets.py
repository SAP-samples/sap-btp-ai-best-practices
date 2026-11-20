"""Authenticated Settings endpoints for immutable reusable source datasets."""

from __future__ import annotations

import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import ValidationError

from app.dependencies import ServiceContainer, get_services
from app.ecn.errors import (
    PersistenceNotFoundError,
    PersistenceStateError,
    SourceValidationError,
)
from app.ecn.intake.files import read_upload_bytes, validate_upload_metadata
from app.ecn.reference_data.api_models import (
    ReferenceDatasetDetailResponse,
    ReferenceDatasetListResponse,
    ReferenceDatasetView,
    dataset_list_response,
    dataset_view,
)
from app.ecn.reference_data.models import ReferenceDatasetType
from app.ecn.reference_data.scopes import DatasetContext
from app.ecn.reference_data.service import (
    ReferenceUploadRequest,
    artifact_kind_for_dataset,
)
from app.ecn.services.blocking_io import run_blocking

router = APIRouter()


def _configured(services: ServiceContainer) -> None:
    """Reject reference routes when a test container omits their dependencies."""
    if (
        services.reference_dataset_service is None
        or services.reference_dataset_repository is None
    ):
        raise PersistenceStateError(
            "REFERENCE_DATA_UNAVAILABLE",
            "Reference dataset management is unavailable.",
            status_code=503,
        )


@router.get("/reference-datasets", response_model=ReferenceDatasetListResponse)
async def list_reference_datasets(
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> ReferenceDatasetListResponse:
    """List immutable publications and identify each active type/scope source."""
    _configured(services)
    assert services.reference_dataset_repository is not None
    records = await run_blocking(services.reference_dataset_repository.list)
    return dataset_list_response(records)


@router.post(
    "/reference-datasets",
    response_model=ReferenceDatasetView,
    status_code=status.HTTP_201_CREATED,
)
async def publish_reference_dataset(
    dataset_type: Annotated[ReferenceDatasetType, Form()],
    source_file: Annotated[UploadFile, File()],
    services: Annotated[ServiceContainer, Depends(get_services)],
    context_json: Annotated[str, Form(max_length=4000)] = "{}",
) -> ReferenceDatasetView:
    """Validate and atomically activate one Settings workbook publication.

    No plant is declared: plants are derived from conditional rules and the
    Plant Code Mapping when ECNs are assessed.
    """
    _configured(services)
    assert services.reference_dataset_service is not None
    try:
        context = DatasetContext.model_validate(json.loads(context_json)).model_dump(mode="json")
    except (ValueError, ValidationError) as exc:
        raise SourceValidationError("DATASET_CONTEXT_INVALID", "Provide valid source scope and snapshot metadata.", status_code=400) from exc
    metadata = validate_upload_metadata(
        source_file.filename,
        source_file.content_type,
        artifact_kind_for_dataset(dataset_type),
    )
    content = await read_upload_bytes(
        source_file, services.settings.max_workbook_upload_bytes
    )
    record = await run_blocking(
        services.reference_dataset_service.publish,
        ReferenceUploadRequest(
            dataset_type=dataset_type,
            file_name=metadata.file_name,
            media_type=metadata.media_type,
            content=content,
            context=context,
        ),
    )
    return dataset_view(record)


@router.get(
    "/reference-datasets/{dataset_id}",
    response_model=ReferenceDatasetDetailResponse,
)
async def get_reference_dataset(
    dataset_id: str,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> ReferenceDatasetDetailResponse:
    """Return one dataset's metadata and bounded normalized row sample."""
    _configured(services)
    assert services.reference_dataset_repository is not None
    detail = await run_blocking(services.reference_dataset_repository.get, dataset_id)
    if detail is None:
        raise PersistenceNotFoundError(
            "REFERENCE_DATASET_NOT_FOUND",
            "The requested reference dataset does not exist.",
            status_code=404,
        )
    samples = [
        {
            "sheet_name": row.sheet_name,
            "source_row_number": row.source_row_number,
            **row.values,
        }
        for row in detail.sample_rows
    ]
    return ReferenceDatasetDetailResponse(
        dataset=dataset_view(detail.dataset), sample_rows=samples
    )


@router.delete("/reference-datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_reference_dataset(
    dataset_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> None:
    """Remove an unpinned Settings publication and its stored workbook.

    Args:
        dataset_id: UUID of the publication selected in Settings.
        services: Request-scoped service container.

    Returns:
        No content on success; 404 or 409 errors when removal cannot proceed.
    """
    _configured(services)
    assert services.reference_dataset_service is not None
    deleted = await run_blocking(services.reference_dataset_service.delete, str(dataset_id))
    if not deleted:
        raise PersistenceNotFoundError(
            "REFERENCE_DATASET_NOT_FOUND",
            "The requested reference dataset does not exist.",
            status_code=404,
        )
