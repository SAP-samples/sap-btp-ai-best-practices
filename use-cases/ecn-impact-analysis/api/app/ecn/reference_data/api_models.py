"""Business-safe HTTP response models for Settings reference datasets."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ecn.reference_data.models import ReferenceDatasetRecord, ReferenceDatasetType
from app.ecn.reference_data.service import (
    PROCUREMENT_STATUS_LEGEND,
    PROCUREMENT_STATUS_LEGEND_VERSION,
)


class ReferenceDatasetView(BaseModel):
    """Expose one immutable dataset publication without artifact payloads."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_type: ReferenceDatasetType
    file_name: str
    sha256: str
    scope_code: str | None
    scope_name: str | None
    row_count: int
    warning_count: int
    warning_counts: dict[str, int]
    parser_version: str
    uploaded_at: datetime
    active: bool
    context: dict[str, Any] = Field(default_factory=dict)


class ReferenceDatasetListResponse(BaseModel):
    """Return all source publications and the versioned source-code legend."""

    model_config = ConfigDict(extra="forbid")

    items: list[ReferenceDatasetView]
    status_legend_version: str
    procurement_status_legend: dict[str, str]


class ReferenceDatasetDetailResponse(BaseModel):
    """Return metadata and a small normalized row preview for operator checks."""

    model_config = ConfigDict(extra="forbid")

    dataset: ReferenceDatasetView
    sample_rows: list[dict[str, Any]] = Field(max_length=100)


def dataset_view(record: ReferenceDatasetRecord) -> ReferenceDatasetView:
    """Map an immutable repository record to its safe HTTP projection."""
    return ReferenceDatasetView(
        dataset_id=record.dataset_id,
        dataset_type=record.dataset_type,
        file_name=record.file_name,
        sha256=record.sha256,
        scope_code=record.scope_code,
        scope_name=record.scope_name,
        row_count=record.row_count,
        warning_count=sum(record.warning_counts.values()),
        warning_counts=record.warning_counts,
        parser_version=record.parser_version,
        uploaded_at=record.created_at,
        active=record.active,
        context=record.context,
    )


def dataset_list_response(
    records: tuple[ReferenceDatasetRecord, ...],
) -> ReferenceDatasetListResponse:
    """Map repository records and the code legend to the list response."""
    return ReferenceDatasetListResponse(
        items=[dataset_view(record) for record in records],
        status_legend_version=PROCUREMENT_STATUS_LEGEND_VERSION,
        procurement_status_legend=dict(PROCUREMENT_STATUS_LEGEND),
    )
