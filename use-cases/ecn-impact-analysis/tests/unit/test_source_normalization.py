"""Unit tests for workbook role contracts, source value normalization,
effectivity parsing, and the four-way raw-EBOM classification heuristic.

All fixtures are synthetic — no files under data/ are read.  The three
TDD waves mirror the brief's numbered steps:

  Wave 1 (header/contract)    – Steps 1-4
  Wave 2 (identifier/cell/qty) – Steps 5-8
  Wave 3 (effectivity/class)  – Steps 9-12
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.ecn.errors import SourceValidationError
from app.ecn.models import BomKind, BomRowClassification

# ---------------------------------------------------------------------------
# Wave 1: Workbook role contract and header normalization
# ---------------------------------------------------------------------------

from app.ecn.intake.workbook_contracts import (
    WORKBOOK_SHEET_CONTRACTS,
    WorkbookSheetContract,
    normalize_header,
)


def test_header_normalization_changes_whitespace_only() -> None:
    """normalize_header collapses internal whitespace but does NOT case-fold."""
    assert normalize_header("  Component   number  ") == "Component number"
    # Case preserved — different capitalizations are NOT the same header.
    assert normalize_header("n° of modification") != "N° of modification"


def test_normalize_header_handles_none_and_blank() -> None:
    """None and blank/empty strings normalize to the empty string."""
    assert normalize_header(None) == ""
    assert normalize_header("   ") == ""
    assert normalize_header("") == ""


def test_normalize_header_preserves_accents_and_punctuation() -> None:
    """Accents and punctuation characters survive normalization unchanged."""
    assert normalize_header("N° of modification") == "N° of modification"
    assert normalize_header("  N°   of   modification  ") == "N° of modification"


def test_three_roles_have_exact_required_signatures() -> None:
    """WORKBOOK_SHEET_CONTRACTS has exactly the three required BomKind roles."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    assert set(contracts) == {
        BomKind.ENRICHED_EBOM, BomKind.RAW_EBOM_130, BomKind.MBOM_130
    }
    assert "Typ" in contracts[BomKind.RAW_EBOM_130].required_headers
    assert contracts[BomKind.ENRICHED_EBOM].field_headers["component_number"] == "Component"
    assert contracts[BomKind.MBOM_130].field_headers["parent_material_number"] == "Item"


def test_enriched_ebom_all_required_headers_present() -> None:
    """ENRICHED_EBOM contract has exactly the 18 required headers from the brief."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    ebom = contracts[BomKind.ENRICHED_EBOM]
    expected = (
        "Level", "Post number", "Component", "Item designation",
        "Component quantity", "Quantity unit", "N° of modification",
        "Start of validity", "Up to modif. Number", "Expiration date",
        "Parent", "Main", "Sub", "VC", "Level 4", "Level 5",
        "Level 6", "Alternative",
    )
    assert ebom.required_headers == expected


def test_raw_ebom_130_all_required_headers_present() -> None:
    """RAW_EBOM_130 contract has exactly the 12 required headers from the brief."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    raw = contracts[BomKind.RAW_EBOM_130]
    expected = (
        "Level no.", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "Change No.",
        "Chg No. To", "Typ", "AltItm",
    )
    assert raw.required_headers == expected


def test_mbom_130_all_required_headers_present() -> None:
    """MBOM_130 contract has exactly the 11 required headers from the brief."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    mbom = contracts[BomKind.MBOM_130]
    expected = (
        "Item", "Component", "Component name", "Quantity",
        "N° of modification", "Start of validity", "at the number of modif.",
        "Expiration date", "alternative postal group", "Sequence",
        "Employment probability",
    )
    assert mbom.required_headers == expected


def test_enriched_ebom_field_headers_complete() -> None:
    """ENRICHED_EBOM field_headers maps all BomRow fields to their source columns."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    fh = contracts[BomKind.ENRICHED_EBOM].field_headers
    assert fh["dotted_level"] == "Level"
    assert fh["position_value"] == "Post number"
    assert fh["component_number"] == "Component"
    assert fh["description_text"] == "Item designation"
    assert fh["quantity"] == "Component quantity"
    assert fh["unit"] == "Quantity unit"
    assert fh["change_number"] == "N° of modification"
    assert fh["valid_from"] == "Start of validity"
    assert fh["change_number_to"] == "Up to modif. Number"
    assert fh["valid_to"] == "Expiration date"
    assert fh["parent_material_number"] == "Parent"
    assert fh["main_value"] == "Main"
    assert fh["sub_value"] == "Sub"
    assert fh["vc_value"] == "VC"
    assert fh["level_4_value"] == "Level 4"
    assert fh["level_5_value"] == "Level 5"
    assert fh["level_6_value"] == "Level 6"
    assert fh["alternative"] == "Alternative"


def test_raw_ebom_field_headers_complete() -> None:
    """RAW_EBOM_130 field_headers maps all BomRow fields to their source columns."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    fh = contracts[BomKind.RAW_EBOM_130].field_headers
    assert fh["dotted_level"] == "Level no."
    assert fh["item_value"] == "Item"
    assert fh["component_number"] == "Component number"
    assert fh["description_text"] == "Object description"
    assert fh["quantity"] == "Quantity"
    assert fh["unit"] == "Un"
    assert fh["valid_from"] == "Valid from"
    assert fh["valid_to"] == "Valid to"
    assert fh["change_number"] == "Change No."
    assert fh["change_number_to"] == "Chg No. To"
    assert fh["type_value"] == "Typ"
    assert fh["alternative"] == "AltItm"


def test_mbom_field_headers_complete() -> None:
    """MBOM_130 field_headers maps all BomRow fields to their source columns."""
    contracts = {c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS}
    fh = contracts[BomKind.MBOM_130].field_headers
    assert fh["parent_material_number"] == "Item"
    assert fh["component_number"] == "Component"
    assert fh["description_text"] == "Component name"
    assert fh["quantity"] == "Quantity"
    assert fh["change_number"] == "N° of modification"
    assert fh["valid_from"] == "Start of validity"
    assert fh["change_number_to"] == "at the number of modif."
    assert fh["valid_to"] == "Expiration date"
    assert fh["alternative"] == "alternative postal group"


def test_duplicate_ms_not_a_required_or_mapped_field() -> None:
    """MS (a duplicate header in the raw sheet) is not in any contract's header sets.

    It is retained positionally as ordered ROW_JSON entries and must not
    overwrite a neighbor or appear in required_headers / field_headers.
    """
    for contract in WORKBOOK_SHEET_CONTRACTS:
        assert "MS" not in contract.required_headers
        assert "MS" not in contract.field_headers


def test_workbook_sheet_contract_is_frozen() -> None:
    """WorkbookSheetContract is an immutable frozen dataclass."""
    contract = next(c for c in WORKBOOK_SHEET_CONTRACTS if c.bom_kind is BomKind.ENRICHED_EBOM)
    with pytest.raises((TypeError, AttributeError)):
        contract.bom_kind = BomKind.RAW_EBOM_130  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Wave 2: Source text, JSON-safe cells, fit checks, and quantity normalization
# ---------------------------------------------------------------------------

from app.ecn.intake.normalization import (
    QuantityNormalization,
    fit_bom_identifier,
    json_safe_cell,
    normalize_quantity,
    ordered_unique,
    source_text,
)


def test_source_text_trims_only_external_whitespace() -> None:
    """source_text trims outer whitespace but preserves case and leading zeros."""
    assert source_text("  00150  ") == "00150"
    assert source_text(" AbC ") == "AbC"
    assert source_text("  ") is None
    assert source_text("") is None
    assert source_text(None) is None


def test_identifier_text_keeps_case_and_leading_zeros() -> None:
    """String values: case and leading zeros survive source_text unchanged."""
    assert source_text("  00150  ") == "00150"
    assert source_text(" AbC ") == "AbC"


def test_source_text_converts_numeric_to_string() -> None:
    """Non-string non-None values are converted via str(); leading zeros are lost for ints."""
    assert source_text(150) == "150"
    assert source_text(1.5) == "1.5"


def test_ordered_unique_preserves_first_seen_order() -> None:
    """ordered_unique removes duplicates while preserving insertion order."""
    result = ordered_unique(["c", "a", "b", "a", "c"])
    assert result == ("c", "a", "b")


def test_ordered_unique_drops_blank_strings() -> None:
    """ordered_unique silently drops blank and whitespace-only strings."""
    result = ordered_unique(["a", "  ", "", "b"])
    assert result == ("a", "b")


def test_ordered_unique_empty_input() -> None:
    """ordered_unique of an empty iterable returns an empty tuple."""
    assert ordered_unique([]) == ()


def test_quantity_integer_digit_boundary_17() -> None:
    """17 integer digits are accepted; 18 integer digits are rejected."""
    accepted = normalize_quantity("1" * 17)
    rejected = normalize_quantity("1" * 18)
    assert accepted.value == Decimal("1" * 17)
    assert accepted.qualifiers == ()
    assert rejected.value is None
    assert rejected.qualifiers == ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)


def test_quantity_fractional_digit_boundary_14() -> None:
    """14 fractional digits are accepted; 15 fractional digits are rejected."""
    accepted = normalize_quantity("0." + "1" * 14)
    rejected = normalize_quantity("0." + "1" * 15)
    assert accepted.value == Decimal("0." + "1" * 14)
    assert accepted.qualifiers == ()
    assert rejected.value is None
    assert rejected.qualifiers == ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)


def test_quantity_never_rounds_to_decimal_contract() -> None:
    """normalize_quantity accepts 123.00000000000001 (14 frac) but rejects 1.000000000000001 (15 frac)."""
    accepted = normalize_quantity("123.00000000000001")
    rejected = normalize_quantity("1.000000000000001")
    assert accepted.value == Decimal("123.00000000000001")
    assert accepted.qualifiers == ()
    assert rejected.value is None
    assert rejected.qualifiers == ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)


def test_quantity_negative_and_zero() -> None:
    """Negative quantities and zero are accepted."""
    assert normalize_quantity("0").value == Decimal("0")
    assert normalize_quantity("-1.5").value == Decimal("-1.5")


def test_quantity_scientific_notation_accepted() -> None:
    """Scientific notation is parsed; the DECIMAL(31,14) check applies to the expanded form."""
    result = normalize_quantity("1e3")
    assert result.value == Decimal("1E+3")
    assert result.qualifiers == ()


def test_quantity_infinity_and_nan_rejected() -> None:
    """Infinity and NaN are not finite and therefore not exactly representable."""
    for s in ("Infinity", "nan", "-Infinity", "Inf"):
        r = normalize_quantity(s)
        assert r.value is None, f"expected None for {s!r}"
        assert r.qualifiers == ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)


def test_quantity_blank_and_none_produce_empty_result() -> None:
    """Blank and None quantity values produce (None, ()) — no qualifier."""
    result_none = normalize_quantity(None)
    result_blank = normalize_quantity("  ")
    assert result_none.value is None
    assert result_none.qualifiers == ()
    assert result_blank.value is None
    assert result_blank.qualifiers == ()


def test_quantity_non_numeric_string_rejected() -> None:
    """Non-numeric strings produce (None, QUANTITY_NOT_EXACTLY_REPRESENTABLE)."""
    result = normalize_quantity("not a number")
    assert result.value is None
    assert result.qualifiers == ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)


def test_json_safe_cell_passthrough_types() -> None:
    """None, bool, int, finite float, and str pass through with empty qualifiers."""
    assert json_safe_cell(None) == (None, ())
    assert json_safe_cell(True) == (True, ())
    assert json_safe_cell(False) == (False, ())
    assert json_safe_cell(42) == (42, ())
    assert json_safe_cell(3.14) == (3.14, ())
    assert json_safe_cell("hello") == ("hello", ())


def test_json_safe_cell_date_datetime_to_iso() -> None:
    """date and datetime objects are converted to their ISO-format strings."""
    d = date(2024, 3, 15)
    dt = datetime(2024, 3, 15, 10, 30, 0)
    assert json_safe_cell(d) == ("2024-03-15", ())
    assert json_safe_cell(dt) == ("2024-03-15T10:30:00", ())


def test_json_safe_cell_decimal_to_exact_text() -> None:
    """Decimal values are serialized to their exact string form, no qualifier."""
    d = Decimal("1.23456789012345")
    result, qualifiers = json_safe_cell(d)
    assert result == "1.23456789012345"
    assert qualifiers == ()


def test_json_safe_cell_opaque_object_gets_stringified_qualifier() -> None:
    """Opaque (unknown-type) objects are str()-converted and flagged."""

    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    result, qualifiers = json_safe_cell(Opaque())
    assert result == "opaque-value"
    assert "SOURCE_CELL_TYPE_STRINGIFIED" in qualifiers


def test_json_safe_cell_non_finite_float_stringified() -> None:
    """Non-finite floats (inf, nan) are stringified with SOURCE_CELL_TYPE_STRINGIFIED."""
    _, q_inf = json_safe_cell(float("inf"))
    _, q_nan = json_safe_cell(float("nan"))
    assert "SOURCE_CELL_TYPE_STRINGIFIED" in q_inf
    assert "SOURCE_CELL_TYPE_STRINGIFIED" in q_nan


def test_fit_bom_identifier_accepts_value_at_limit() -> None:
    """fit_bom_identifier accepts a value whose trimmed length equals maximum_length."""
    value = "A" * 200
    result = fit_bom_identifier(
        value, field_name="component_number", maximum_length=200,
        sheet_name="Raw", source_row_number=2,
    )
    assert result == value


def test_fit_bom_identifier_returns_none_for_none_and_blank() -> None:
    """fit_bom_identifier returns None for None or whitespace-only values."""
    assert fit_bom_identifier(
        None, field_name="component_number", maximum_length=200,
        sheet_name="Raw", source_row_number=2,
    ) is None
    assert fit_bom_identifier(
        "   ", field_name="component_number", maximum_length=200,
        sheet_name="Raw", source_row_number=2,
    ) is None


def test_overlong_join_key_error_does_not_echo_value() -> None:
    """SOURCE_VALUE_TOO_LONG error carries coordinates but NOT the overlong value."""
    secret_value = "X" * 201
    with pytest.raises(SourceValidationError) as caught:
        fit_bom_identifier(
            secret_value, field_name="component_number", maximum_length=200,
            sheet_name="Raw", source_row_number=4,
        )
    assert secret_value not in str(caught.value)
    err = caught.value
    assert err.code == "SOURCE_VALUE_TOO_LONG"
    assert err.details is not None
    assert err.details["maximum_length"] == 200
    assert err.details["field_name"] == "component_number"
    assert err.details["sheet_name"] == "Raw"
    assert err.details["source_row_number"] == 4


def test_fit_bom_identifier_sheet_name_width_500() -> None:
    """Sheet name slots have maximum width 500."""
    assert fit_bom_identifier(
        "A" * 500, field_name="sheet_name", maximum_length=500,
        sheet_name="S", source_row_number=1,
    ) == "A" * 500
    with pytest.raises(SourceValidationError):
        fit_bom_identifier(
            "A" * 501, field_name="sheet_name", maximum_length=500,
            sheet_name="S", source_row_number=1,
        )


def test_fit_bom_identifier_unit_width_40() -> None:
    """Unit field has maximum width 40."""
    assert fit_bom_identifier(
        "A" * 40, field_name="unit", maximum_length=40,
        sheet_name="S", source_row_number=1,
    ) == "A" * 40
    with pytest.raises(SourceValidationError):
        fit_bom_identifier(
            "A" * 41, field_name="unit", maximum_length=40,
            sheet_name="S", source_row_number=1,
        )


def test_fit_bom_identifier_plant_width_80() -> None:
    """Plant field has maximum width 80."""
    assert fit_bom_identifier(
        "A" * 80, field_name="plant", maximum_length=80,
        sheet_name="S", source_row_number=1,
    ) == "A" * 80
    with pytest.raises(SourceValidationError):
        fit_bom_identifier(
            "A" * 81, field_name="plant", maximum_length=80,
            sheet_name="S", source_row_number=1,
        )


def test_fit_bom_identifier_component_width_200() -> None:
    """Component/material/change/alternative fields have maximum width 200."""
    for width_200_field in ("component_number", "material_number", "change_number"):
        assert fit_bom_identifier(
            "A" * 200, field_name=width_200_field, maximum_length=200,
            sheet_name="S", source_row_number=1,
        ) == "A" * 200
        with pytest.raises(SourceValidationError):
            fit_bom_identifier(
                "A" * 201, field_name=width_200_field, maximum_length=200,
                sheet_name="S", source_row_number=1,
            )


# ---------------------------------------------------------------------------
# Wave 3: Effectivity normalization and four-way raw EBOM classification
# ---------------------------------------------------------------------------

from app.ecn.intake.normalization import (
    EffectivityField,
    EffectivityFormat,
    classify_raw_row,
    normalize_effectivity_value,
)


def test_raw_date_is_day_month_year_and_1900_only_marks_valid_from() -> None:
    """Raw dd.mm.yyyy dates parse correctly; year-1900 marks VALID_FROM only."""
    valid_from = normalize_effectivity_value(
        "02.01.1900", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    valid_to = normalize_effectivity_value(
        "02.01.1900", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_TO
    )
    assert valid_from.parsed_date == date(1900, 1, 2)
    assert valid_from.is_migration_placeholder is True
    assert valid_to.is_migration_placeholder is False


def test_raw_date_02_01_1900_is_january_2_not_february_1() -> None:
    """02.01.1900 means 2 January (dd first), not 1 February (mm first)."""
    result = normalize_effectivity_value(
        "02.01.1900", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    assert result.parsed_date is not None
    assert result.parsed_date.month == 1
    assert result.parsed_date.day == 2


def test_raw_date_normal_year_not_migration_placeholder() -> None:
    """A valid parsed date with year != 1900 is never a migration placeholder."""
    result = normalize_effectivity_value(
        "15.03.2024", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    assert result.parsed_date == date(2024, 3, 15)
    assert result.is_migration_placeholder is False


def test_raw_date_1900_valid_to_is_not_migration_placeholder() -> None:
    """VALID_TO with year 1900 is never a migration placeholder (VALID_FROM only)."""
    result = normalize_effectivity_value(
        "01.01.1900", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_TO
    )
    assert result.parsed_date == date(1900, 1, 1)
    assert result.is_migration_placeholder is False


def test_enriched_date_accepts_python_date_object() -> None:
    """EXCEL_TYPED_OR_ISO accepts a Python date object directly."""
    d = date(2024, 6, 1)
    result = normalize_effectivity_value(
        d, EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    assert result.parsed_date == d
    assert result.is_migration_placeholder is False
    assert result.parse_warning is None


def test_enriched_date_accepts_python_datetime_object() -> None:
    """EXCEL_TYPED_OR_ISO extracts the date component from a Python datetime."""
    dt = datetime(2024, 6, 1, 12, 0, 0)
    result = normalize_effectivity_value(
        dt, EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    assert result.parsed_date == date(2024, 6, 1)


def test_enriched_date_accepts_iso_string() -> None:
    """EXCEL_TYPED_OR_ISO parses an ISO-format YYYY-MM-DD string."""
    result = normalize_effectivity_value(
        "2024-06-01", EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    assert result.parsed_date == date(2024, 6, 1)
    assert result.parse_warning is None


def test_effectivity_blank_and_none_produce_null_result() -> None:
    """Blank (None, empty string, whitespace) values produce all-null output."""
    for fmt in (EffectivityFormat.RAW_DD_MM_YYYY, EffectivityFormat.EXCEL_TYPED_OR_ISO):
        for field in (EffectivityField.VALID_FROM, EffectivityField.VALID_TO):
            r_none = normalize_effectivity_value(None, fmt, field)
            assert r_none.raw_text is None
            assert r_none.parsed_date is None
            assert r_none.is_migration_placeholder is False
            assert r_none.parse_warning is None
    r_blank = normalize_effectivity_value(
        "   ", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    assert r_blank.raw_text is None
    assert r_blank.parsed_date is None


def test_effectivity_invalid_nonblank_raw_sets_parse_warning() -> None:
    """Invalid nonblank raw dates preserve raw_text and set EFFECTIVITY_PARSE_FAILED."""
    result = normalize_effectivity_value(
        "not-a-date", EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    assert result.raw_text == "not-a-date"
    assert result.parsed_date is None
    assert result.parse_warning == "EFFECTIVITY_PARSE_FAILED"
    assert result.is_migration_placeholder is False


def test_effectivity_invalid_iso_string_sets_parse_warning() -> None:
    """Invalid non-ISO date string for EXCEL_TYPED_OR_ISO sets EFFECTIVITY_PARSE_FAILED."""
    result = normalize_effectivity_value(
        "2024/06/01", EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    assert result.raw_text == "2024/06/01"
    assert result.parsed_date is None
    assert result.parse_warning == "EFFECTIVITY_PARSE_FAILED"


def test_effectivity_no_active_current_boolean_on_model() -> None:
    """NormalizedEffectivity carries no active/current attribute — it models intervals only."""
    from app.ecn.models import NormalizedEffectivity

    field_names = {f.name for f in dataclasses.fields(NormalizedEffectivity)}
    for name in field_names:
        assert "active" not in name.lower(), f"unexpected 'active' in field {name!r}"
        assert "current" not in name.lower(), f"unexpected 'current' in field {name!r}"


def test_raw_classification_precedence_and_position_semantics() -> None:
    """D-prefix > ZVD pattern > ND type > UNRESOLVED; 150/160 are plain positions."""
    assert classify_raw_row("D001", "ND", "PART") is BomRowClassification.CONFIRMED_DOCUMENT
    assert classify_raw_row("150", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED
    assert classify_raw_row("160", "", "PART") is BomRowClassification.UNRESOLVED
    assert classify_raw_row("10", "ND", "PART ZVD 000 01") is BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE


def test_classification_confirmed_document_d_prefix_various() -> None:
    """Any item beginning with uppercase 'D' (after trim) is CONFIRMED_DOCUMENT."""
    assert classify_raw_row("D001", "ND", "anything") is BomRowClassification.CONFIRMED_DOCUMENT
    assert classify_raw_row("D-PREFIX", "", "anything") is BomRowClassification.CONFIRMED_DOCUMENT
    # Trimmed whitespace — still a D-prefix.
    assert classify_raw_row("  D001  ", "ND", "PART") is BomRowClassification.CONFIRMED_DOCUMENT


def test_classification_lowercase_d_is_not_document() -> None:
    """Lowercase 'd' does NOT trigger CONFIRMED_DOCUMENT — heuristic is case-sensitive."""
    assert classify_raw_row("d001", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED


def test_classification_document_reference_candidate_zvd_zvs_zvg() -> None:
    """Component matching ^.+?\\s+ZV[DSG]\\s+\\S+\\s+\\S+$ is DOCUMENT_REFERENCE_CANDIDATE."""
    assert classify_raw_row("10", "ND", "PART ZVD 000 01") is BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE
    assert classify_raw_row("10", "X", "PART ZVS 001 02") is BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE
    assert classify_raw_row("10", "", "PART ZVG 001 02") is BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE


def test_classification_partial_zvd_pattern_not_candidate() -> None:
    """Incomplete ZVD patterns (missing segments, wrong variant) do not match."""
    # Missing last segment.
    assert classify_raw_row("10", "", "PART ZVD 000") is BomRowClassification.UNRESOLVED
    # ZVX is not ZV[DSG].
    assert classify_raw_row("10", "", "PART ZVX 000 01") is BomRowClassification.UNRESOLVED
    # Leading-only match (fullmatch required).
    assert classify_raw_row("10", "", "PART ZVD 000 01 EXTRA") is BomRowClassification.UNRESOLVED


def test_classification_material_like_observed_exact_nd() -> None:
    """Exact 'ND' Typ (no D-prefix, no ZVD component) yields MATERIAL_LIKE_OBSERVED."""
    assert classify_raw_row("150", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED
    assert classify_raw_row("10", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED


def test_classification_lowercase_nd_is_unresolved() -> None:
    """'nd' (lowercase) does NOT trigger MATERIAL_LIKE_OBSERVED — case-sensitive."""
    assert classify_raw_row("10", "nd", "PART") is BomRowClassification.UNRESOLVED


def test_classification_blank_type_with_no_document_pattern() -> None:
    """Empty string Typ with no D-prefix and no ZVD component → UNRESOLVED."""
    assert classify_raw_row("150", "", "PART") is BomRowClassification.UNRESOLVED
    assert classify_raw_row("160", "", "PART") is BomRowClassification.UNRESOLVED


def test_classification_none_type_is_unresolved() -> None:
    """None Typ (blank cell) → UNRESOLVED when no D-prefix or ZVD component."""
    assert classify_raw_row("150", None, "PART") is BomRowClassification.UNRESOLVED


def test_classification_d_prefix_beats_zvd_component() -> None:
    """D-prefix item takes precedence over a matching ZVD component (precedence rule 1 > 2)."""
    assert (
        classify_raw_row("D001", "ND", "PART ZVD 000 01")
        is BomRowClassification.CONFIRMED_DOCUMENT
    )


def test_classification_zvd_component_beats_nd_type() -> None:
    """ZVD component pattern takes precedence over ND type (precedence rule 2 > 3)."""
    assert (
        classify_raw_row("10", "ND", "PART ZVD 000 01")
        is BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE
    )


def test_classification_positions_150_and_160_no_special_treatment() -> None:
    """150 and 160 are observed position values; they receive standard classification."""
    # With ND type: MATERIAL_LIKE_OBSERVED (not a special-cased type).
    assert classify_raw_row("150", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED
    assert classify_raw_row("160", "ND", "PART") is BomRowClassification.MATERIAL_LIKE_OBSERVED
    # No ND, no ZVD → UNRESOLVED.
    assert classify_raw_row("150", "XX", "PART") is BomRowClassification.UNRESOLVED
    assert classify_raw_row("160", "", "PART") is BomRowClassification.UNRESOLVED
