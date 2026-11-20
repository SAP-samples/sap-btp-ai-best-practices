"""Immutable domain records for Settings-managed reference datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.ecn.models import BomRow, JsonValue, SourceArtifact, canonical_uuid


REFERENCE_DATA_PARSER_VERSION = "ecn-reference-v1"
"""Version of normalized Settings reference-workbook parsing."""


class ReferenceDatasetType(StrEnum):
    """Enumerate reusable source snapshots managed through Settings."""

    STOCK = "STOCK"
    BOM = "BOM"
    CONDITIONAL_RULES = "CONDITIONAL_RULES"
    MATERIAL_MASTER = "MATERIAL_MASTER"
    PURCHASE_INFO_RECORDS = "PURCHASE_INFO_RECORDS"
    PURCHASE_INFO_PRICES = "PURCHASE_INFO_PRICES"


@dataclass(frozen=True)
class NormalizedReferenceRow:
    """Hold one source row with normalized fields and exact workbook location.

    Args:
        row_id: Immutable row UUID.
        artifact_id: Owning source-artifact UUID.
        sheet_name: Exact source worksheet name.
        source_row_number: One-based worksheet row number.
        values: Dataset-specific normalized business values.
    """

    row_id: str
    artifact_id: str
    sheet_name: str
    source_row_number: int
    values: dict[str, JsonValue]

    def __post_init__(self) -> None:
        """Validate row/artifact identity and source coordinate bounds."""
        canonical_uuid(self.row_id)
        canonical_uuid(self.artifact_id)
        if not self.sheet_name or len(self.sheet_name) > 500:
            raise ValueError("sheet_name is blank or exceeds 500 characters")
        if self.source_row_number < 1:
            raise ValueError("source_row_number must be positive")


@dataclass(frozen=True)
class ParsedReferenceWorkbook:
    """Return a validated normalized reference workbook before persistence.

    Args:
        dataset_type: Kind of reusable dataset parsed.
        rows: Normalized source rows in workbook order.
        warning_counts: Aggregated safe warning codes and occurrence counts.
    """

    dataset_type: ReferenceDatasetType
    rows: tuple[NormalizedReferenceRow, ...]
    warning_counts: dict[str, int]

    @property
    def row_count(self) -> int:
        """Return the number of normalized nonblank source rows."""
        return len(self.rows)


@dataclass(frozen=True)
class ReferenceDatasetPublication:
    """Carry one validated immutable Settings publication into persistence.

    Args:
        dataset_id: Canonical dataset publication UUID.
        dataset_type: Reusable source type.
        artifact: Valid source artifact containing the original upload.
        rows: Normalized source-specific rows; BOM rows are stored separately.
        row_count: Number of logical normalized rows in the publication.
        warning_counts: Aggregated safe validation warning counts.
        scope_code: Optional plant/site scope code.
        scope_name: Optional human-readable scope name.
        created_by: Optional opaque actor identifier.
        created_at: Timezone-aware publication time.
    """

    dataset_id: str
    dataset_type: ReferenceDatasetType
    artifact: SourceArtifact
    rows: tuple[NormalizedReferenceRow, ...]
    row_count: int
    warning_counts: dict[str, int]
    scope_code: str | None
    scope_name: str | None
    created_by: str | None
    created_at: datetime
    bom_rows: tuple[BomRow, ...] = ()
    context: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate immutable identity, counts, and site metadata bounds."""
        canonical_uuid(self.dataset_id)
        if self.row_count < 0:
            raise ValueError("row_count must be nonnegative")
        if self.scope_code is not None and not self.scope_code.strip():
            raise ValueError("scope_code cannot be blank")
        if self.scope_code is not None and len(self.scope_code) > 40:
            raise ValueError("scope_code exceeds 40 characters")
        if self.scope_name is not None and len(self.scope_name) > 200:
            raise ValueError("scope_name exceeds 200 characters")


@dataclass(frozen=True)
class ReferenceDatasetRecord:
    """Describe one immutable Settings dataset without returning its BLOB.

    Args:
        dataset_id: Canonical dataset publication UUID.
        dataset_type: Reusable source type.
        artifact_id: Canonical source-artifact UUID.
        publication_order: Monotonic global publication sequence.
        file_name: Sanitized original file name.
        sha256: Original file hash.
        scope_code: Optional plant/site scope.
        scope_name: Optional display name for the scope.
        row_count: Logical normalized row count.
        warning_counts: Safe grouped validation warnings.
        parser_version: Parser contract pinned to this publication.
        created_at: Timezone-aware publication time.
        active: Whether this is the latest publication for its type/scope.
    """

    dataset_id: str
    dataset_type: ReferenceDatasetType
    artifact_id: str
    publication_order: int
    file_name: str
    sha256: str
    scope_code: str | None
    scope_name: str | None
    row_count: int
    warning_counts: dict[str, int]
    parser_version: str
    created_at: datetime
    active: bool
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceDatasetDetail:
    """Return dataset metadata plus bounded representative normalized rows."""

    dataset: ReferenceDatasetRecord
    sample_rows: tuple[NormalizedReferenceRow, ...]


@dataclass(frozen=True)
class ReferenceDatasetPin:
    """Pin one active reference publication to an ECN upload batch."""

    dataset_type: ReferenceDatasetType
    dataset_id: str

    def __post_init__(self) -> None:
        """Validate the immutable dataset identity."""
        canonical_uuid(self.dataset_id)
