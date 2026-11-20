"""Exercise delivered layout changes and snapshot semantics without relying on customer files."""

from io import BytesIO
from uuid import uuid4

import pytest
from openpyxl import Workbook

from app.ecn.errors import SourceValidationError
from app.ecn.evidence.snapshot import BOM_SCOPE, build_snapshot, procurement_type_label
from app.ecn.models import (
    BomKind,
    BomRow,
    BomRowClassification,
    NormalizedEcn,
    NormalizedSourceObject,
    SourceObjectRole,
)
from app.ecn.reference_data.models import ReferenceDatasetType as Kind
from app.ecn.reference_data.parsers import parse_reference_workbook


# ---------------------------------------------------------------------------
# Minimal fixture builders for snapshot tests
# ---------------------------------------------------------------------------


def _minimal_ecn(**overrides) -> NormalizedEcn:
    """Return a NormalizedEcn with no affected or resulting objects.

    Args:
        **overrides: Optional field overrides applied to the default values.

    Returns:
        A minimal NormalizedEcn suitable for snapshot construction tests.
    """
    defaults = dict(
        ecn_row_id="11111111-1111-4111-8111-111111111111",
        artifact_id="22222222-2222-4222-8222-222222222222",
        source_ordinal=0,
        ecn_number="ECN-001",
        title="Test ECN",
        description_summary=None,
        description_text=None,
        affected_objects=(),
        resulting_objects=(),
        source_json={},
        normalization_qualifiers=(),
    )
    defaults.update(overrides)
    return NormalizedEcn(**defaults)


def _source_object(number: str, obj_id: str, is_wtpart: bool = True) -> NormalizedSourceObject:
    """Return a minimal NormalizedSourceObject for snapshot matching tests.

    Args:
        number: The part number exposed in the snapshot object entry.
        obj_id: Canonical UUID for this object (must be unique per test).
        is_wtpart: Whether the object is a technical WTPart; defaults True.

    Returns:
        A minimal NormalizedSourceObject with the given number and WTPart flag.
    """
    return NormalizedSourceObject(
        source_object_id=obj_id,
        source_path="$[0].affectedObjects[0]",
        link_path="$[0].affectedObjects[0]",
        role=SourceObjectRole.AFFECTED,
        source_ordinal=0,
        technical_id=None,
        version_id=None,
        number=number,
        name=None,
        description=None,
        object_type=None,
        revision=None,
        version=None,
        state=None,
        created_on=None,
        last_modified=None,
        source_subtype=None,
        relationship_values={},
        raw_object={},
        is_technical_wtpart=is_wtpart,
        quality_qualifiers=(),
    )


def _bom_row(component_number: str, row_id: str) -> BomRow:
    """Return a minimal ENRICHED_EBOM BomRow for the given component number.

    Args:
        component_number: The part number the row represents.
        row_id: Canonical UUID for this row (must be unique per test).

    Returns:
        A minimal BomRow with ENRICHED_EBOM kind and the given component number.
    """
    return BomRow(
        bom_row_id=row_id,
        artifact_id="55555555-5555-4555-8555-555555555555",
        sheet_name="EBOM Q900V",
        bom_kind=BomKind.ENRICHED_EBOM,
        source_row_number=1,
        parent_bom_row_id=None,
        dotted_level="1",
        hierarchy_path=None,
        main_value=None,
        sub_value=None,
        vc_value=None,
        level_4_value=None,
        level_5_value=None,
        level_6_value=None,
        item_value=None,
        position_value=None,
        material_number=None,
        parent_material_number=None,
        component_number=component_number,
        change_number=None,
        change_number_to=None,
        description_text=None,
        unit=None,
        plant=None,
        alternative=None,
        valid_from_raw=None,
        valid_to_raw=None,
        quantity=None,
        valid_from_date=None,
        valid_to_date=None,
        row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
        is_documentation=False,
        is_migration_date=False,
        row_json=(),
        quality_qualifiers=(),
        quality_details={},
    )


# ---------------------------------------------------------------------------
# Source-semantics tests (Step 2 of the brief)
# ---------------------------------------------------------------------------


def test_procurement_type_label_maps_all_four_cases() -> None:
    """Exact labels for E/F/X; None, empty string, and unknown codes all return Unknown."""
    assert procurement_type_label("E") == "In-house production"
    assert procurement_type_label("F") == "External procurement"
    assert procurement_type_label("X") == "In-house and external procurement allowed"
    assert procurement_type_label(None) == "Unknown"
    assert procurement_type_label("") == "Unknown"
    assert procurement_type_label("Z") == "Unknown"
    assert procurement_type_label("e") == "Unknown"  # case-sensitive: raw code preserved


def test_build_snapshot_adds_bom_scope_to_overview() -> None:
    """Snapshot overview must carry bom_scope so tools and prompts always see Q900V-only scope."""
    snapshot = build_snapshot(_minimal_ecn(), [], [])
    assert snapshot["bom_scope"] == BOM_SCOPE
    assert snapshot["bom_scope"] == "Q900V only"


def test_build_snapshot_exposes_existing_owner_category_and_resolution_context() -> None:
    """The decision brief uses concise context already present in the ECN source."""
    ecn = _minimal_ecn(source_json={
        "REQUESTER_NAME": "Dana Whitfield",
        "CATEGORY2": {"Value": "CORRECTION", "Display": "Correction"},
        "ResolutionDate": "2026-05-19",
    })

    context = build_snapshot(ecn, [], [])["ecn_context"]

    assert context == {
        "owner_name": "Dana Whitfield",
        "category_label": "Correction",
        "business_driver": None,
        "cib_target_date": "2026-05-19",
        "financial_impact": "UNKNOWN",
        "financial_notes": None,
        "risk_level": None,
        "source": "SOURCE",
    }


def test_build_snapshot_adds_bom_scope_to_every_bom_location() -> None:
    """Every BOM location emitted by the snapshot must carry the same bom_scope marker."""
    obj = _source_object("MAT-001", "33333333-3333-4333-8333-333333333333")
    ecn = _minimal_ecn(affected_objects=(obj,))
    row = _bom_row("MAT-001", "44444444-4444-4444-8444-444444444444")
    snapshot = build_snapshot(ecn, [row], [])
    entry = next(o for o in snapshot["objects"] if o["number"] == "MAT-001")
    assert entry["locations"], "expected at least one BOM location for the test to be meaningful"
    for loc in entry["locations"]:
        assert loc["bom_scope"] == "Q900V only"


def test_build_snapshot_applies_procurement_type_label_to_material_master_reference_records() -> None:
    """Material-master reference records in the snapshot carry procurement_type_label; others do not."""
    obj_e = _source_object("MAT-E", "33333333-3333-4333-8333-000000000001")
    obj_f = _source_object("MAT-F", "33333333-3333-4333-8333-000000000002")
    obj_x = _source_object("MAT-X", "33333333-3333-4333-8333-000000000003")
    obj_n = _source_object("MAT-N", "33333333-3333-4333-8333-000000000004")
    obj_u = _source_object("MAT-U", "33333333-3333-4333-8333-000000000005")
    ecn = _minimal_ecn(
        affected_objects=(obj_e, obj_f, obj_x, obj_n, obj_u),
        resulting_objects=(),
    )
    references = [
        {"dataset_type": "MATERIAL_MASTER", "material_number": "MAT-E", "procurement_type": "E"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "MAT-F", "procurement_type": "F"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "MAT-X", "procurement_type": "X"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "MAT-N", "procurement_type": None},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "MAT-U", "procurement_type": "UNKNOWN_CODE"},
        # Non-material-master row: must not receive a procurement_type_label.
        {"dataset_type": "STOCK", "material_number": "MAT-E"},
    ]
    snapshot = build_snapshot(ecn, [], references)
    by_number = {o["number"]: o for o in snapshot["objects"]}
    # Verify each expected label.
    assert by_number["MAT-E"]["reference_records"][0]["procurement_type_label"] == "In-house production"
    assert by_number["MAT-F"]["reference_records"][0]["procurement_type_label"] == "External procurement"
    assert by_number["MAT-X"]["reference_records"][0]["procurement_type_label"] == "In-house and external procurement allowed"
    assert by_number["MAT-N"]["reference_records"][0]["procurement_type_label"] == "Unknown"
    assert by_number["MAT-U"]["reference_records"][0]["procurement_type_label"] == "Unknown"
    # STOCK reference must not have the label field added.
    stock_refs = [r for r in by_number["MAT-E"]["reference_records"] if r.get("dataset_type") == "STOCK"]
    for ref in stock_refs:
        assert "procurement_type_label" not in ref


def workbook_bytes(headers, values):
    """Return a minimal uploaded workbook for the given exact header and data row."""
    workbook = Workbook()
    workbook.active.append(headers)
    workbook.active.append(values)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_new_material_layout_preserves_status_order_without_old_commodity():
    """Duplicate MS headers have meanings only within the recognized delivered layout."""
    result = parse_reference_workbook(Kind.MATERIAL_MASTER, workbook_bytes(
        ["Material", "Material Description", "Plnt", "ProcType", "MS", "MS", "Matl Group", "Material Group Desc."],
        ["0007", "Bracket", "PL10", "F", "30", "40", "001", "Hydraulics"],
    ), str(uuid4()))
    values = result.rows[0].values
    assert values["material_number"] == "0007"
    assert (values["cross_plant_status"], values["plant_status"]) == ("30", "40")
    assert values["material_group_description"] == "Hydraulics"
    assert "commodity_code" not in values


def test_supplier_name_is_preserved_without_inventing_legacy_columns():
    """The second purchase export carries names but omits category/tax/date."""
    result = parse_reference_workbook(Kind.PURCHASE_INFO_RECORDS, workbook_bytes(
        ["Info Rec.", "Vendor", "Name 1", "Material", "Plnt", "PGr"],
        ["12", "003", "Supplier name", "0007", "PL10", "A"],
    ), str(uuid4()))
    assert result.rows[0].values["supplier_name"] == "Supplier name"
    assert "category" not in result.rows[0].values


@pytest.mark.parametrize("quantity", [0, -2, 3.25])
def test_stock_keeps_reported_quantity_and_explicit_source_limits(quantity):
    """Zero and negative reported quantities remain facts, not synthesized absence."""
    result = parse_reference_workbook(Kind.STOCK, workbook_bytes(
        ["Material", "Plant", "Storage Location", "Unrestricted"],
        ["0007", "PL10", "0001", quantity],
    ), str(uuid4()))
    assert float(result.rows[0].values["unrestricted_quantity"]) == quantity
    assert result.rows[0].values["storage_location"] == "0001"
    assert result.warning_counts == {"STOCK_UNIT_UNAVAILABLE": 1, "STOCK_DATE_UNAVAILABLE": 1}


def test_non_numeric_stock_is_rejected():
    """Unusable quantities fail intake instead of appearing as zero stock."""
    with pytest.raises(SourceValidationError, match="finite number"):
        parse_reference_workbook(Kind.STOCK, workbook_bytes(
            ["Material", "Plant", "Storage Location", "Unrestricted"],
            ["0007", "PL10", "0001", "not a quantity"],
        ), str(uuid4()))
