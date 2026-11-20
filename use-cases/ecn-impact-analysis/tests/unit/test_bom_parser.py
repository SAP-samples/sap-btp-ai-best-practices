"""Unit tests for workbook parsing: role detection, limit enforcement, and field mapping.

All fixtures are synthetic — no files under data/ are read.  Three TDD waves
mirror the brief's numbered steps:

  Wave 1 (roles/limits/corruption)    – Steps 1-5
  Wave 2 (field mapping / ROW_JSON)   – Steps 6-9
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.ecn.errors import SourceValidationError
from app.ecn.intake.workbook_parser import parse_bom_workbook
from app.ecn.models import BomKind, BomRowClassification

from tests.fixtures.builders import (
    bom_workbook_bytes,
    enriched_headers,
    enriched_row,
    mbom_headers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACT_ID = "00000000-0000-4000-8000-000000000001"
_LIMITS = dict(maximum_rows=100, maximum_cells=10_000)


# ---------------------------------------------------------------------------
# Wave 1: Role detection, limits, corruption
# ---------------------------------------------------------------------------


def test_roles_are_detected_by_headers_not_sheet_names_or_order() -> None:
    """Role detection uses normalized header signatures, not sheet titles or order."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    assert set(parsed.sheet_roles) == {
        BomKind.ENRICHED_EBOM, BomKind.RAW_EBOM_130, BomKind.MBOM_130
    }
    assert parsed.sheet_roles[BomKind.RAW_EBOM_130] == "Renamed raw evidence"
    assert parsed.sheet_roles[BomKind.ENRICHED_EBOM] == "Renamed enriched evidence"
    assert parsed.sheet_roles[BomKind.MBOM_130] == "Renamed manufacturing evidence"


def test_extra_unmatched_sheet_is_ignored() -> None:
    """A sheet with unrecognized headers does not prevent detection of the three roles."""
    parsed = parse_bom_workbook(
        bom_workbook_bytes(add_unmatched_sheet=True), ARTIFACT_ID, **_LIMITS
    )
    assert set(parsed.sheet_roles) == {
        BomKind.ENRICHED_EBOM, BomKind.RAW_EBOM_130, BomKind.MBOM_130
    }


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"not-a-zip", "BOM_WORKBOOK_CORRUPT"),
        (bom_workbook_bytes(remove_role="MBOM_130"), "BOM_SHEET_ROLE_MISSING"),
        (bom_workbook_bytes(duplicate_role="RAW_EBOM_130"), "BOM_SHEET_ROLE_DUPLICATE"),
        (bom_workbook_bytes(duplicate_required_header="Typ"), "BOM_REQUIRED_HEADER_DUPLICATE"),
    ],
)
def test_workbook_contract_failures_are_safe(content: bytes, expected_code: str) -> None:
    """Contract violations are safe HTTP 400 errors with no source cell payloads."""
    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(content, ARTIFACT_ID, **_LIMITS)
    err = caught.value
    assert err.code == expected_code
    assert err.status_code == 400
    # Error details must not include raw cell content.
    details_str = str(err.details or {})
    # Specifically, no ECN identifiers or row data should leak.
    assert "ROOT001" not in details_str
    assert "0000123" not in details_str


def test_malformed_lazy_worksheet_xml_is_reported_as_corrupt_workbook() -> None:
    """Map XML errors raised during lazy row iteration to the safe workbook error."""
    source = ZipFile(BytesIO(bom_workbook_bytes()))
    output = BytesIO()
    with source, ZipFile(output, "w", ZIP_DEFLATED) as rewritten:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename == "xl/worksheets/sheet1.xml":
                payload = b"<worksheet><broken>"
            rewritten.writestr(member, payload)

    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(output.getvalue(), ARTIFACT_ID, **_LIMITS)
    assert caught.value.code == "BOM_WORKBOOK_CORRUPT"
    assert caught.value.status_code == 400


def test_row_limit_crossed_by_one_raises_error() -> None:
    """BOM_ROW_LIMIT_EXCEEDED is raised when visited row count exceeds the limit."""
    # Default workbook has 4 raw rows + 2 enriched + 1 MBOM = 7 data rows.
    # Set limit to 6 (one less than total) to force the error.
    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, maximum_rows=6, maximum_cells=100_000)
    assert caught.value.code == "BOM_ROW_LIMIT_EXCEEDED"
    assert caught.value.status_code == 400


def test_row_limit_exactly_at_threshold_does_not_raise() -> None:
    """Row limit is inclusive — exactly at the threshold is accepted."""
    # 7 total data rows; limit=7 should succeed.
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, maximum_rows=7, maximum_cells=100_000)
    assert parsed.row_count > 0


def test_cell_limit_crossed_raises_error() -> None:
    """BOM_CELL_LIMIT_EXCEEDED is raised when visited cell count exceeds the limit."""
    # Default workbook header rows: 15 + 18 + 11 = 44 cells.
    # Set limit to 40 to force failure before headers are done.
    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, maximum_rows=100, maximum_cells=40)
    assert caught.value.code == "BOM_CELL_LIMIT_EXCEEDED"
    assert caught.value.status_code == 400


def test_zero_row_limit_raises_value_error() -> None:
    """A zero row limit is rejected as invalid input (not a source error)."""
    with pytest.raises(ValueError):
        parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, maximum_rows=0, maximum_cells=10_000)


def test_negative_cell_limit_raises_value_error() -> None:
    """A negative cell limit is rejected as invalid input."""
    with pytest.raises(ValueError):
        parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, maximum_rows=100, maximum_cells=-1)


def test_invalid_artifact_id_raises_value_error() -> None:
    """A non-canonical artifact UUID string is rejected before any I/O."""
    with pytest.raises(ValueError):
        parse_bom_workbook(bom_workbook_bytes(), "not-a-uuid", **_LIMITS)


def test_formula_is_preserved_not_evaluated() -> None:
    """data_only=False is in effect: formula cells are kept as strings, not computed."""
    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    # Build minimal raw sheet with a formula in the Quantity column.
    raw_headers = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    raw_sheet = wb.create_sheet("Formula raw")
    raw_sheet.append(raw_headers)
    raw_sheet.append([None, ".1", "0", "FORM001", "Formula part", "=1+1", "EA", "01.01.2020", "31.12.9999", "", "", "", "ND", "ECN1", ""])

    enriched_sheet = wb.create_sheet("Enriched sheet")
    enriched_sheet.append(enriched_headers())

    mbom_sheet = wb.create_sheet("MBOM sheet")
    mbom_sheet.append(mbom_headers())

    stream = BytesIO()
    wb.save(stream)
    content = stream.getvalue()

    parsed = parse_bom_workbook(content, ARTIFACT_ID, maximum_rows=10, maximum_cells=1_000)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    assert len(raw_rows) == 1
    # The row_json quantity entry should contain the formula string, not a computed value.
    qty_entry = next(e for e in raw_rows[0].row_json if e["normalized_header"] == "Quantity")
    assert qty_entry["value"] == "=1+1"


# ---------------------------------------------------------------------------
# Wave 2: Field mapping, ROW_JSON, quality
# ---------------------------------------------------------------------------


def test_all_three_roles_are_present_in_parsed_rows() -> None:
    """All three BomKind variants appear in parsed.rows after a successful parse."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    kinds = {row.bom_kind for row in parsed.rows}
    assert kinds == {BomKind.ENRICHED_EBOM, BomKind.RAW_EBOM_130, BomKind.MBOM_130}


def test_raw_row_fields_and_duplicate_ms_headers() -> None:
    """Raw EBOM rows map all required fields and retain two MS columns positionally."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [row for row in parsed.rows if row.bom_kind is BomKind.RAW_EBOM_130]
    # row at index 1 (source row 3) is Level "..2", Item "150"
    material = raw_rows[1]
    assert material.item_value == "150"
    assert material.position_value == "150"
    assert material.component_number == "0000123"
    assert material.material_number == "0000123"
    assert material.change_number == "ECN2"
    assert material.valid_from_date == date(1900, 1, 2)
    assert material.is_migration_date is True
    ms_entries = [entry for entry in material.row_json if entry["normalized_header"] == "MS"]
    assert [entry["column_index"] for entry in ms_entries] == [11, 12]


def test_raw_row_json_contains_every_source_column() -> None:
    """ROW_JSON contains one entry per source column, including blank headers and cells."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [row for row in parsed.rows if row.bom_kind is BomKind.RAW_EBOM_130]
    root = raw_rows[0]
    # Fixture has 15 columns (including leading None).
    assert len(root.row_json) == 15
    # First column has blank normalized header (None source).
    assert root.row_json[0]["column_index"] == 1
    assert root.row_json[0]["normalized_header"] == ""


def test_raw_row_occurrence_uuids_are_distinct() -> None:
    """Each BomRow receives a unique UUID regardless of component repetition."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [row for row in parsed.rows if row.bom_kind is BomKind.RAW_EBOM_130]
    ids = [r.bom_row_id for r in raw_rows]
    assert len(ids) == len(set(ids)), "All bom_row_id values must be distinct"


def test_enriched_row_fields_and_not_applicable_classification() -> None:
    """Enriched EBOM rows map all required fields and carry NOT_APPLICABLE classification."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    enriched_rows = [r for r in parsed.rows if r.bom_kind is BomKind.ENRICHED_EBOM]
    main_row = enriched_rows[0]
    assert main_row.row_classification is BomRowClassification.NOT_APPLICABLE
    assert main_row.component_number == "MAIN"
    assert main_row.material_number == "MAIN"
    assert main_row.item_value is None  # Enriched has no Item column.
    assert main_row.main_value == "MAIN"
    assert main_row.parent_material_number == "ROOT"
    # Hierarchy is reconstructed: level-1 row has no parent row ID.
    assert main_row.parent_bom_row_id is None


def test_mbom_row_fields_hierarchy_path_and_not_applicable_classification() -> None:
    """MBOM rows map Item→parent, Component→component, and produce Item/Component path."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    mbom = next(row for row in parsed.rows if row.bom_kind is BomKind.MBOM_130)
    assert mbom.item_value == "SUB"
    assert mbom.position_value is None
    assert mbom.parent_material_number == "SUB"
    assert mbom.component_number == "0000123"
    assert mbom.hierarchy_path == "SUB/0000123"
    assert mbom.row_classification is BomRowClassification.NOT_APPLICABLE


def test_migration_date_is_preserved_not_substituted() -> None:
    """Year-1900 valid_from date is stored as-is; no substitute date is injected."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    migration_rows = [r for r in raw_rows if r.is_migration_date]
    assert len(migration_rows) >= 1
    for row in migration_rows:
        assert row.valid_from_date is not None
        assert row.valid_from_date.year == 1900
        # The raw text must be preserved exactly.
        assert row.valid_from_raw is not None


def test_d_prefix_item_is_confirmed_document_in_raw_sheet() -> None:
    """Raw rows with a D-prefixed Item column value are classified as CONFIRMED_DOCUMENT."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    doc_rows = [r for r in raw_rows if r.item_value and r.item_value.startswith("D")]
    assert len(doc_rows) >= 1
    for row in doc_rows:
        assert row.row_classification is BomRowClassification.CONFIRMED_DOCUMENT
        assert row.is_documentation is True


def test_material_like_rows_carry_nd_not_confirmed_qualifier() -> None:
    """MATERIAL_LIKE_OBSERVED rows carry TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    material_rows = [r for r in raw_rows if r.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED]
    assert len(material_rows) >= 1
    for row in material_rows:
        assert "TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED" in row.quality_qualifiers


def test_parsed_bom_workbook_row_count_matches_rows_tuple() -> None:
    """ParsedBomWorkbook.row_count equals len(rows) (nonblank retained rows only)."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    assert parsed.row_count == len(parsed.rows)
    assert parsed.row_count > 0


def test_cell_count_is_positive_and_includes_headers() -> None:
    """cell_count is positive and at least the sum of all role header widths."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    # Minimum cell count = 15 raw headers + 18 enriched headers + 11 MBOM headers = 44.
    assert parsed.cell_count >= 44


def test_sheet_roles_maps_bom_kind_to_exact_sheet_names() -> None:
    """sheet_roles contains exactly the three required BomKind keys."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    assert len(parsed.sheet_roles) == 3
    for kind in (BomKind.ENRICHED_EBOM, BomKind.RAW_EBOM_130, BomKind.MBOM_130):
        assert kind in parsed.sheet_roles
        assert isinstance(parsed.sheet_roles[kind], str)


def test_enriched_hierarchy_is_reconstructed_in_parsed_workbook() -> None:
    """Enriched EBOM rows in the parsed workbook have hierarchy_path set."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    enriched_rows = [r for r in parsed.rows if r.bom_kind is BomKind.ENRICHED_EBOM]
    # Level-1 row: path = "MAIN"; level-2 row: path = "MAIN/SUB".
    paths = [r.hierarchy_path for r in enriched_rows if r.hierarchy_path]
    assert "MAIN" in paths
    assert "MAIN/SUB" in paths


def test_raw_hierarchy_is_reconstructed_in_parsed_workbook() -> None:
    """Raw EBOM rows in the parsed workbook have hierarchy_path set."""
    parsed = parse_bom_workbook(bom_workbook_bytes(), ARTIFACT_ID, **_LIMITS)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    root_row = raw_rows[0]
    # Level ".1" is depth 1; path segment = component_number.
    assert root_row.hierarchy_path is not None
    assert root_row.parent_bom_row_id is None


def test_over_limit_component_number_in_raw_sheet_raises_source_value_too_long() -> None:
    """A Component number exceeding 200 chars in the raw sheet raises SOURCE_VALUE_TOO_LONG.

    The error details must carry field_name, sheet_name, source_row_number, and
    maximum_length — but NEVER the raw over-limit value itself.
    """
    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    over_limit_value = "C" * 201  # 201 chars, exceeds COMPONENT_NUMBER NVARCHAR(200)

    raw_headers = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    raw_sheet = wb.create_sheet("raw sheet")
    raw_sheet.append(raw_headers)
    raw_sheet.append([None, ".1", "0", over_limit_value, "desc", 1, "EA", "01.01.2020", "31.12.9999", "", "", "", "ND", "ECN1", ""])

    enriched_sheet = wb.create_sheet("enriched sheet")
    enriched_sheet.append(enriched_headers())

    mbom_sheet = wb.create_sheet("mbom sheet")
    mbom_sheet.append(mbom_headers())

    stream = BytesIO()
    wb.save(stream)

    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(stream.getvalue(), ARTIFACT_ID, maximum_rows=10, maximum_cells=1_000)

    err = caught.value
    assert err.code == "SOURCE_VALUE_TOO_LONG"
    assert err.status_code == 400
    # Safe coordinates must be present.
    assert err.details is not None
    assert err.details["field_name"] == "component_number"
    assert err.details["maximum_length"] == 200
    assert "source_row_number" in err.details
    assert "sheet_name" in err.details
    # The over-limit value itself must NOT appear in the error.
    full_error_text = str(err) + str(err.details)
    assert over_limit_value not in full_error_text


def test_over_limit_component_in_enriched_sheet_raises_source_value_too_long() -> None:
    """A Component value exceeding 200 chars in the enriched sheet raises SOURCE_VALUE_TOO_LONG."""
    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    over_limit_value = "E" * 201  # 201 chars

    enriched_sheet = wb.create_sheet("enriched sheet")
    enriched_sheet.append(enriched_headers())
    # Build an enriched row with an over-limit Component value.
    row = enriched_row(level=1, component=over_limit_value, parent="ROOT", main=over_limit_value)
    enriched_sheet.append(row)

    raw_sheet = wb.create_sheet("raw sheet")
    raw_headers = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    raw_sheet.append(raw_headers)

    mbom_sheet = wb.create_sheet("mbom sheet")
    mbom_sheet.append(mbom_headers())

    stream = BytesIO()
    wb.save(stream)

    with pytest.raises(SourceValidationError) as caught:
        parse_bom_workbook(stream.getvalue(), ARTIFACT_ID, maximum_rows=10, maximum_cells=1_000)

    err = caught.value
    assert err.code == "SOURCE_VALUE_TOO_LONG"
    assert err.status_code == 400
    assert err.details is not None
    assert err.details["field_name"] == "component_number"
    assert err.details["maximum_length"] == 200
    # The raw value must NOT appear anywhere in the error output.
    assert over_limit_value not in str(err.details)


def test_within_limit_values_parse_successfully() -> None:
    """Values at exactly the column limit (200 chars) do not raise an error."""
    at_limit_value = "X" * 200  # exactly 200 chars — must be accepted

    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    raw_headers = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    raw_sheet = wb.create_sheet("raw sheet")
    raw_sheet.append(raw_headers)
    raw_sheet.append([None, ".1", "0", at_limit_value, "desc", 1, "EA", "01.01.2020", "31.12.9999", "", "", "", "ND", "ECN1", ""])

    enriched_sheet = wb.create_sheet("enriched sheet")
    enriched_sheet.append(enriched_headers())

    mbom_sheet = wb.create_sheet("mbom sheet")
    mbom_sheet.append(mbom_headers())

    stream = BytesIO()
    wb.save(stream)

    # Must parse without error.
    parsed = parse_bom_workbook(stream.getvalue(), ARTIFACT_ID, maximum_rows=10, maximum_cells=1_000)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    assert raw_rows[0].component_number == at_limit_value


def test_quantity_failure_qualifier_added_for_non_representable_quantity() -> None:
    """A cell that can't be parsed as a valid DECIMAL(31,14) gets a quality qualifier."""
    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    raw_headers = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    raw_sheet = wb.create_sheet("raw")
    raw_sheet.append(raw_headers)
    # Use a string that fails Decimal parsing.
    raw_sheet.append([None, ".1", "0", "QTY001", "Bad qty", "abc", "EA", "01.01.2020", "31.12.9999", "", "", "", "ND", "ECN1", ""])

    enriched_sheet = wb.create_sheet("enriched")
    enriched_sheet.append(enriched_headers())

    mbom_sheet = wb.create_sheet("mbom")
    mbom_sheet.append(mbom_headers())

    stream = BytesIO()
    wb.save(stream)

    parsed = parse_bom_workbook(stream.getvalue(), ARTIFACT_ID, maximum_rows=10, maximum_cells=1_000)
    raw_rows = [r for r in parsed.rows if r.bom_kind is BomKind.RAW_EBOM_130]
    assert len(raw_rows) == 1
    assert raw_rows[0].quantity is None
    assert "QUANTITY_NOT_EXACTLY_REPRESENTABLE" in raw_rows[0].quality_qualifiers
