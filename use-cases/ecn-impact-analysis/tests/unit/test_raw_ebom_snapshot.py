"""A part started only in the 130% EBOM reaches BOM changes, object locations and programs.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_raw_ebom_snapshot.py
"""

from datetime import date
from decimal import Decimal

from app.ecn.evidence.snapshot import build_snapshot
from app.ecn.models import BomKind, BomRowClassification
from tests.fixtures.builders import make_bom_row, make_normalized_ecn, synthetic_uuid

ECN_NUMBER = "ECN193101"


def raw_row():
    """A 130% material row at level 4 (group / sub-assembly / part) started by the ECN."""
    return make_bom_row(
        bom_row_id=synthetic_uuid("raw"), bom_kind=BomKind.RAW_EBOM_130, dotted_level="....4",
        hierarchy_path="Q900VV0170801/Q921PD3001908/X596516039000", parent_material_number="Q921PD3001908",
        component_number="X596516039000", item_value="10", position_value="10", change_number=ECN_NUMBER,
        valid_from_date=date(2017, 3, 8), valid_to_date=date(9999, 12, 31), quantity=Decimal("2"),
        row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED)


def twin_row():
    """The 150% row for the same position (position written with zero padding)."""
    return make_bom_row(
        bom_row_id=synthetic_uuid("enriched"), bom_kind=BomKind.ENRICHED_EBOM, dotted_level="5",
        vc_value="Q900VV0170801", parent_material_number="Q921PD3001908", component_number="X596516039000",
        position_value="0010", change_number=ECN_NUMBER, valid_from_date=date(2017, 3, 8),
        valid_to_date=date(9999, 12, 31), quantity=Decimal("2"))


def snapshot_of(*rows):
    """Build the v2 snapshot of ECN193101 over the given rows."""
    return build_snapshot(make_normalized_ecn(ecn_number=ECN_NUMBER), rows, assessment_date=date(2026, 9, 23))


def test_raw_only_row_gives_a_bom_change_a_location_and_a_program_entry():
    """Without a 150% twin the 130% row is the evidence in every part of the snapshot."""
    snapshot = snapshot_of(raw_row())
    [line] = snapshot["bom_changes"]["lines"]
    assert line["change"] == "ADDED" and line["incoming"]["number"] == "X596516039000"
    assert line["position"] == "0010" and "EBOM_130_SOURCE" in line["qualifiers"]
    assert line["vc_groups"] == ["Q900VV0170801"]
    part = next(o for o in snapshot["objects"] if o["number"] == "X596516039000")
    assert [(l["kind"], l["raw_only"]) for l in part["locations"]] == [("RAW_EBOM_130", True)]
    [program] = [p for p in snapshot["programs"] if p["ebom"]]
    assert [row["number"] for row in program["ebom"]] == ["X596516039000"]


def test_150_twin_wins_and_the_130_row_is_not_counted_again():
    """With both rows present exactly one line exists and it comes from the 150% row."""
    snapshot = snapshot_of(raw_row(), twin_row())
    [line] = snapshot["bom_changes"]["lines"]
    assert "EBOM_130_SOURCE" not in line["qualifiers"] and line["sources"][0]["kind"] == "ENRICHED_EBOM"
    part = next(o for o in snapshot["objects"] if o["number"] == "X596516039000")
    assert [(l["kind"], l["raw_only"]) for l in part["locations"]] == [("ENRICHED_EBOM", False)]
