"""Tests for validation-first reference dataset publication and activation."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import pytest
from openpyxl import Workbook

from app.ecn.errors import SourceValidationError
from app.ecn.evidence.snapshot import procurement_type_label
from app.ecn.models import ArtifactKind
from app.ecn.reference_data.models import ReferenceDatasetRecord, ReferenceDatasetType
from app.ecn.reference_data.service import (
    PROCUREMENT_STATUS_LEGEND,
    ReferenceDatasetService,
    ReferenceUploadRequest,
    display_source_code,
)
from app.ecn.settings import EcnSettings


NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _workbook(rows: list[list[object]], title: str = "Sheet1") -> bytes:
    """Build a compact XLSX fixture entirely in memory."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = title
    for row in rows:
        sheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


class _Repository:
    def list(self):
        """Return no preceding publications for first-upload service tests."""
        return ()

    """Record fully validated publications and return a metadata projection."""

    def __init__(self) -> None:
        """Initialize with no publication calls."""
        self.publications = []

    def publish(self, publication):
        """Record one publication and return a matching active record."""
        self.publications.append(publication)
        return ReferenceDatasetRecord(
            dataset_id=publication.dataset_id,
            dataset_type=publication.dataset_type,
            artifact_id=publication.artifact.artifact_id,
            publication_order=len(self.publications),
            file_name=publication.artifact.file_name,
            sha256=publication.artifact.sha256,
            scope_code=publication.scope_code,
            scope_name=publication.scope_name,
            row_count=publication.row_count,
            warning_counts=publication.warning_counts,
            parser_version="ecn-reference-v1",
            created_at=publication.created_at,
            active=True,
        )


def _service(repository: _Repository) -> ReferenceDatasetService:
    """Build a deterministic service with canonical test UUIDs."""
    identifiers = iter(
        [
            "00000000-0000-4000-8000-000000000201",
            "00000000-0000-4000-8000-000000000202",
        ]
    )
    return ReferenceDatasetService(
        repository,  # type: ignore[arg-type]
        EcnSettings(),
        clock=lambda: NOW,
        uuid_factory=lambda: next(identifiers),
    )


def test_valid_material_master_activates_with_source_metadata() -> None:
    """Publish a parsed material master and retain exact metadata and row count."""
    repository = _Repository()
    service = _service(repository)
    content = _workbook(
        [
            [
                "Material", "Material Description", "Plnt", "ProcType",
                "Cross plant status", "Plant specific status", "Matl Group",
                "Commodity Code number",
            ],
            ["M-1", "Bracket", "PL10", "E", "", "", "42", "8708"],
        ]
    )
    record = service.publish(
        ReferenceUploadRequest(
            ReferenceDatasetType.MATERIAL_MASTER,
            "materials.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            content,
        )
    )
    assert record.active is True
    assert record.row_count == 1
    publication = repository.publications[0]
    assert publication.artifact.artifact_kind is ArtifactKind.MATERIAL_MASTER_WORKBOOK
    assert publication.rows[0].values["material_number"] == "M-1"


def test_purchase_info_price_csv_activates_as_an_independent_overlay() -> None:
    """Publish a validated CSV price source without modifying purchase setup rows."""
    repository = _Repository()
    service = _service(repository)
    record = service.publish(
        ReferenceUploadRequest(
            ReferenceDatasetType.PURCHASE_INFO_PRICES,
            "prices.csv",
            "text/csv",
            (
                "Info Rec.,Vendor,Name 1,Material,Plnt,PGr,Net Price,Crcy,Per,OPU\n"
                '500,900,"Supplier A",MAT-1,PL10,F01,12.50,EUR,1,EA\n'
            ).encode(),
        )
    )

    assert record.active is True
    assert record.row_count == 1
    publication = repository.publications[0]
    assert publication.artifact.artifact_kind is ArtifactKind.PURCHASE_INFO_PRICE_CSV
    assert publication.rows[0].values["net_price"] == "12.50"



    """Keep the active source unchanged by rejecting invalid input before HANA."""
    repository = _Repository()
    service = _service(repository)
    with pytest.raises(SourceValidationError):
        service.publish(
            ReferenceUploadRequest(
                ReferenceDatasetType.MATERIAL_MASTER,
                "invalid.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                _workbook([["Wrong"], ["value"]]),
            )
        )
    assert repository.publications == []


def test_procurement_status_legend_agrees_with_snapshot_labels() -> None:
    """Guard the single procurement-label source of truth across both surfaces.

    The Settings reference-data legend (``PROCUREMENT_STATUS_LEGEND`` and
    ``display_source_code``) and the evidence snapshot (``procurement_type_label``)
    must return the same maintained label for every known ProcType code, and code
    ``X`` must read exactly the binding global-constraint wording. This test fails
    if either surface ever diverges again.
    """
    # Code X must carry the exact maintained wording on the legend surface.
    assert (
        PROCUREMENT_STATUS_LEGEND["X"] == "In-house and external procurement allowed"
    )
    assert display_source_code("X") == "In-house and external procurement allowed"
    # Both surfaces must agree with the snapshot label for every known code.
    for code in ("E", "F", "X"):
        assert PROCUREMENT_STATUS_LEGEND[code] == procurement_type_label(code)
        assert display_source_code(code) == procurement_type_label(code)
    # The published legend exposes only the maintained codes.
    assert set(PROCUREMENT_STATUS_LEGEND) == {"E", "F", "X"}


def test_display_source_code_preserves_unknown_codes() -> None:
    """Keep the legend's raw-passthrough fallback distinct from the snapshot.

    Unlike ``procurement_type_label`` (which returns ``"Unknown"``), the Settings
    legend must preserve an unrecognised raw code and return ``None`` for ``None``.
    This guards against accidentally swapping in the snapshot's "Unknown" fallback.
    """
    assert display_source_code("Z") == "Z"
    assert display_source_code(None) is None
