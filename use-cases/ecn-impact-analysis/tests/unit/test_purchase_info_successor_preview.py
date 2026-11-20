"""Tests for local, non-published Purchase Info successor preview workbooks."""

from __future__ import annotations

from io import BytesIO

import pytest
from openpyxl import load_workbook

from app.ecn.reference_data.purchase_info_reconciliation import (
    reconcile_targeted_purchase_info,
)
from app.ecn.reference_data.purchase_info_successor_preview import (
    PURCHASE_INFO_SUCCESSOR_HEADERS,
    build_purchase_info_successor_workbook,
)


def test_successor_preview_preserves_rows_and_embeds_lineage_manifest() -> None:
    """Create the review workbook without embedding source prices in setup rows."""
    reconciliation = reconcile_targeted_purchase_info(
        baseline_rows=[
            {
                "info_record": "100001",
                "vendor": "200001",
                "supplier_name": "Historical Supplier",
                "material_number": "MAT-1",
                "plant_code": "PK01",
                "purchasing_group": "P01",
                "sheet_name": "Legacy PIR",
                "source_row_number": 8,
            }
        ],
        price_rows=[
            {
                "info_record": "100002",
                "vendor": "200002",
                "supplier_name": "New Supplier",
                "material_number": "MAT-2",
                "plant_code": "PL10",
                "purchasing_group": "P02",
                "net_price": "12.50",
                "currency": "EUR",
                "price_per": "1",
                "order_price_unit": "EA",
                "sheet_name": "Purchase Info Prices",
                "source_row_number": 2,
            }
        ],
        relevant_pairs=[("MAT-1", "PK01"), ("MAT-2", "PL10")],
    )

    content = build_purchase_info_successor_workbook(
        reconciliation,
        baseline_dataset_id="baseline-123",
        price_source_sha256="a" * 64,
    )

    workbook = load_workbook(BytesIO(content), data_only=True)
    records = workbook["Purchase Info Records"]
    assert tuple(cell.value for cell in records[1]) == PURCHASE_INFO_SUCCESSOR_HEADERS
    assert [tuple(cell.value for cell in row) for row in records.iter_rows(min_row=2)] == [
        ("100001", "200001", "Historical Supplier", "MAT-1", "PK01", "P01"),
        ("100002", "200002", "New Supplier", "MAT-2", "PL10", "P02"),
    ]

    manifest = workbook["Reconciliation Manifest"]
    manifest_values = [tuple(cell.value for cell in row[:2]) for row in manifest.iter_rows()]
    manifest_rows = [tuple(cell.value for cell in row[:4]) for row in manifest.iter_rows()]
    assert ("Preview status", "NOT PUBLISHED") in manifest_values
    assert ("Baseline dataset ID", "baseline-123") in manifest_values
    assert (2, "HANA_CARRIED_FORWARD", "Legacy PIR", 8) in manifest_rows
    assert (3, "PRICE_CSV_TARGETED", "Purchase Info Prices", 2) in manifest_rows


def test_successor_preview_refuses_to_drop_legacy_purchase_info_fields() -> None:
    """Prevent an export layout from silently discarding baseline business data."""
    reconciliation = reconcile_targeted_purchase_info(
        baseline_rows=[
            {
                "info_record": "100001",
                "vendor": "200001",
                "material_number": "MAT-1",
                "plant_code": "PK01",
                "category": "Z1",
            }
        ],
        price_rows=[],
        relevant_pairs=[],
    )

    with pytest.raises(ValueError, match="category"):
        build_purchase_info_successor_workbook(
            reconciliation,
            baseline_dataset_id="baseline-123",
            price_source_sha256="a" * 64,
        )
