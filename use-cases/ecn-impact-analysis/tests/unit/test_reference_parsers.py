"""Tests for normalized Settings reference-data workbook parsers."""

from __future__ import annotations

from io import BytesIO

import pytest
from app.ecn.errors import SourceValidationError
from app.ecn.reference_data.models import ReferenceDatasetType
from app.ecn.reference_data.parsers import (
    parse_purchase_info_price_csv,
    parse_reference_workbook,
)
from openpyxl import Workbook


def _workbook_bytes(rows: list[list[object]], title: str = "Sheet1") -> bytes:
    """Create an in-memory XLSX workbook for parser contract tests."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = title
    for row in rows:
        sheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def test_material_master_parser_normalizes_business_fields() -> None:
    """Normalize exact material, plant, procurement, status, and grouping fields."""
    content = _workbook_bytes(
        [
            [
                "Material",
                "Material Description",
                "Plnt",
                "ProcType",
                "Cross plant status",
                "Plant specific status",
                "Matl Group",
                "Commodity Code number",
            ],
            ["000123", "Bracket", "PL10", "E", "30", "98", "42", "8708"],
        ]
    )
    parsed = parse_reference_workbook(
        ReferenceDatasetType.MATERIAL_MASTER,
        content,
        "00000000-0000-4000-8000-000000000111",
    )
    assert parsed.row_count == 1
    assert parsed.rows[0].values == {
        "material_number": "000123",
        "description": "Bracket",
        "plant_code": "PL10",
        "procurement_type": "E",
        "cross_plant_status": "30",
        "plant_status": "98",
        "material_group": "42",
        "commodity_code": "8708",
    }


def test_purchase_info_parser_keeps_vendor_as_non_predictive_source_data() -> None:
    """Persist purchase record provenance without making it a model feature."""
    content = _workbook_bytes(
        [
            ["Info Rec.", "Vendor", "Plnt", "Cat", "PGr", "Tx", "FFD", "Material"],
            ["500", "900", "PL10", "0", "F01", "", "", "MAT-1"],
        ]
    )
    parsed = parse_reference_workbook(
        ReferenceDatasetType.PURCHASE_INFO_RECORDS,
        content,
        "00000000-0000-4000-8000-000000000112",
    )
    assert parsed.rows[0].values["material_number"] == "MAT-1"
    assert parsed.rows[0].values["vendor"] == "900"


def test_purchase_info_price_csv_preserves_legacy_fields_and_normalizes_grouped_numbers() -> None:
    """Accept the approved price CSV without losing its purchasing setup keys."""
    parsed = parse_purchase_info_price_csv(
        (
            "Info Rec.,Vendor,Name 1,Material,Plnt,PGr,Net Price,Crcy,Per,OPU\n"
            '500,900,"Supplier A",MAT-1,PL10,F01,"1,234.50",EUR,"1,000",EA\n'
            '501,901,"Supplier B",MAT-2,PK01,F02,0,USD,1,PC\n'
        ).encode(),
        "00000000-0000-4000-8000-000000000113",
    )

    assert parsed.dataset_type is ReferenceDatasetType.PURCHASE_INFO_PRICES
    assert parsed.warning_counts == {"PURCHASE_INFO_PRICE_ZERO": 1}
    assert parsed.rows[0].values == {
        "info_record": "500",
        "vendor": "900",
        "supplier_name": "Supplier A",
        "material_number": "MAT-1",
        "plant_code": "PL10",
        "purchasing_group": "F01",
        "net_price": "1234.50",
        "currency": "EUR",
        "price_per": "1000",
        "order_price_unit": "EA",
    }


def test_purchase_info_price_csv_rejects_ambiguous_comma_decimal_format() -> None:
    """Reject locale guesses rather than silently changing a source price."""
    with pytest.raises(SourceValidationError) as caught:
        parse_purchase_info_price_csv(
            (
                "Info Rec.,Vendor,Name 1,Material,Plnt,PGr,Net Price,Crcy,Per,OPU\n"
                '500,900,"Supplier A",MAT-1,PL10,F01,"1,23",EUR,1,EA\n'
            ).encode(),
            "00000000-0000-4000-8000-000000000114",
        )
    assert caught.value.code == "PURCHASE_INFO_PRICE_NUMBER_INVALID"


def test_exactly_six_reference_dataset_types_remain() -> None:
    """Exactly five source types remain; historical/rule ingestion types are fully retired."""
    maintained = {item.value for item in ReferenceDatasetType}
    assert maintained == {
        "STOCK",
        "BOM",
        "CONDITIONAL_RULES",
        "MATERIAL_MASTER",
        "PURCHASE_INFO_RECORDS",
        "PURCHASE_INFO_PRICES",
    }


def test_invalid_headers_are_rejected_before_publication() -> None:
    """Reject a workbook with missing required columns and no partial rows."""
    content = _workbook_bytes([["Wrong", "Headers"], ["x", "y"]])
    with pytest.raises(SourceValidationError) as caught:
        parse_reference_workbook(
            ReferenceDatasetType.MATERIAL_MASTER,
            content,
            "00000000-0000-4000-8000-000000000115",
        )
    assert caught.value.code == "REFERENCE_HEADERS_INVALID"
