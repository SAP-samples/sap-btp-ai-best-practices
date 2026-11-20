"""Opt-in acceptance of BOM changes and plant rows on the real Q900V extraction.

Run:
    DEVELOPMENT_FIXTURE_DIR=<repo-root>/data \
      PYTHONPATH=api .venv/bin/python -m pytest -q -m development_fixture \
      tests/integration/test_real_bom_changes_fixture.py

Reads the ignored files under ``data/`` (EBOM/MBOM workbook, conditional-rule
workbook and ``new_16-09/ECN_full.json``) and checks ECN345274 and ECN346863
against the facts read by hand from the MBOM sheet and the rules. Plants come
from the rules with JB/IS -> PL10 Plant A and JD -> PK01 Plant B, excluding
"To be Obsolete" rules. Skips when the files are absent.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from app.ecn.evidence.raw_ebom import raw_only_row_ids
from app.ecn.evidence.snapshot import build_snapshot
from app.ecn.intake.json_parser import parse_ecn_json
from app.ecn.intake.workbook_parser import parse_bom_workbook
from app.ecn.rules.importer import normalize_rule_workbook
from app.ecn.rules.models import RuleArtifactMetadata
from app.ecn.rules.plant_scope import build_vc_index

pytestmark = pytest.mark.development_fixture
DATA = Path(os.environ.get("DEVELOPMENT_FIXTURE_DIR", str(Path(__file__).resolve().parents[2] / "data"))).resolve()
BOM_PATH = DATA / "EBOM MBOM_Q900V_04.09.2026.xlsx"
ECN_PATH = DATA / "new_16-09" / "ECN_full.json"
RULES_PATH = next(DATA.glob("R*gles*conditionnelles*.xlsx"), DATA / "missing-rules.xlsx")
BOM_ARTIFACT_ID = "00000000-0000-0000-0000-000000000202"
RULES_ARTIFACT_ID = "00000000-0000-4000-8000-000000000203"
MAPPINGS = [{"suffix": "IS", "plant_code": "PL10", "plant_name": "Plant A"},
            {"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"},
            {"suffix": "JD", "plant_code": "PK01", "plant_name": "Plant B"}]
ECN_ARTIFACT_ID = "00000000-0000-0000-0000-000000000201"


@pytest.fixture(scope="module")
def sources():
    """Parse the real workbook and ECN export once; skip when not available."""
    if not BOM_PATH.exists() or not ECN_PATH.exists() or not RULES_PATH.exists():
        pytest.skip("real Q900V development files are not available")
    rows = parse_bom_workbook(BOM_PATH.read_bytes(), BOM_ARTIFACT_ID, maximum_rows=100_000, maximum_cells=3_000_000).rows
    ecns = {e.ecn_number: e for e in parse_ecn_json(ECN_PATH.read_bytes(), ECN_ARTIFACT_ID).ecns}
    dataset = {"dataset_type": "BOM", "artifact_id": BOM_ARTIFACT_ID, "file_name": BOM_PATH.name,
               "model_range": "Q900V", "data_group": "real-q900v",
               "environment": "PRODUCTION", "snapshot_date": "2026-09-04", "coverage_complete": True, "comparison": False}
    metadata = RuleArtifactMetadata(RULES_ARTIFACT_ID, RULES_PATH.name, "application/vnd.ms-excel", datetime.now(timezone.utc))
    rules = normalize_rule_workbook(RULES_PATH.read_bytes(), metadata).rules
    plant_context = {"index": build_vc_index(rules, ("To be Obsolete",)), "mappings": MAPPINGS, "rules_pinned": True}
    return rows, ecns, dataset, plant_context


def _snapshot(sources, number, references=()):
    """Build the snapshot of one ECN against the whole pinned BOM with rule-derived plants."""
    rows, ecns, dataset, plant_context = sources
    return build_snapshot(ecns[number], rows, references, datasets=[dataset], assessment_date=date(2026, 9, 24),
                          plant_context=plant_context)


def test_ecn345274_replacements_with_positions_dates_and_flags(sources):
    """Two old parts replaced by their A revisions in both VC groups, fully phased out."""
    changes = _snapshot(sources, "ECN345274")["bom_changes"]
    replaced = {(l["assembly"], l["outgoing"]["number"]): l for l in changes["lines"] if l["change"] == "REPLACED"}
    assert set(replaced) == {(g, p) for g in ("Q923VV2526401", "Q923VV2526404") for p in ("QKX4638810", "QKX8914160")}
    line = replaced[("Q923VV2526401", "QKX4638810")]
    assert line["incoming"]["number"] == "QKX463881A" and line["position"] == "0150"
    assert (line["end_date"], line["manufacturing_date"], line["engineering_date"]) == ("2026-06-29", "2026-06-30", "2026-01-16")
    assert replaced[("Q923VV2526401", "QKX8914160")]["position"] == "0160"
    flags = {m["number"]: m["status"] for m in changes["materials"]}
    assert flags == {"QKX4638810": "FULLY_PHASED_OUT", "QKX8914160": "FULLY_PHASED_OUT",
                     "QKX463881A": "NEW_PART", "QKX891416A": "NEW_PART"}


def test_ecn345274_rules_place_it_in_plant_a_only(sources):
    """Q923VV2526401/404 are selected only by JB rules, so only Plant A is affected."""
    snapshot = _snapshot(sources, "ECN345274")
    assert {p["plant_code"] for l in snapshot["bom_changes"]["lines"] for p in l["plants"]} == {"PL10"}
    [plant] = snapshot["plant_impact"]["plants"]
    assert (plant["plant_code"], plant["plant_name"], plant["status"]) == ("PL10", "Plant A", "CONFIRMED")
    assert {"QKX4638810", "QKX463881A", "Q923VV2526401"} <= {m["number"] for m in plant["materials"]}

def test_ecn346863_removals_quantities_additions_and_plant(sources):
    """3 removals and 2 quantity changes in Q900VV0080402, additions elsewhere, nothing paired."""
    snapshot = _snapshot(sources, "ECN346863")
    lines = snapshot["bom_changes"]["lines"]
    in_402 = {(l["change"], (l["outgoing"] or l["incoming"])["number"]) for l in lines if l["assembly"] == "Q900VV0080402"}
    assert in_402 == {("REMOVED", "QKX2833570"), ("REMOVED", "6646743T1"), ("REMOVED", "3388708T1"),
                      ("QUANTITY_CHANGED", "8241814T1"), ("QUANTITY_CHANGED", "33380541"), ("ADDED", "QKX7702260")}
    assert not any(l["change"] == "REPLACED" for l in lines)
    added = {(l["assembly"], l["incoming"]["number"]) for l in lines if l["change"] == "ADDED"}
    assert {("Q900VV0080301", "QKX0187890"), ("Q900VV0080460", "QKX3105210"), ("Q900VV0080461", "QKX5949130"),
            ("Q900VV0080462", "QKW8860810"), ("Q900VV0080463", "QKW8860810")} <= added
    flags = {m["number"]: m for m in snapshot["bom_changes"]["materials"]}
    assert flags["QKX2833570"]["status"] == "FULLY_PHASED_OUT" and flags["QKX2833570"]["date"] == "2026-05-18"
    assert flags["3388708T1"]["status"] == "STILL_USED"
    # New VC groups are in the MBOM through their children, so no MBOM question is raised.
    assert "MBOM_INTRODUCTION_UNCONFIRMED" not in {q["code"] for q in snapshot["questions"]}
    plant_b, plant_a = snapshot["plant_impact"]["plants"]
    assert (plant_a["plant_code"], plant_a["status"], plant_a["cutover_from"]) == ("PL10", "CONFIRMED", "2026-05-19")
    # Only Q900VV0080301 has a JD rule: its new part is the one Plant B change.
    assert (plant_b["plant_code"], plant_b["changes"]) == ("PK01", {"ADDED": 1})
    assert {m["number"] for m in plant_b["materials"]} >= {"QKX0187890", "Q900VV0080301"}
    assert "QKX7702260" not in {m["number"] for m in plant_b["materials"]}
    assert [p["model_range"] for p in snapshot["programs"]][:1] == ["Q900V"]


def test_bom_only_removed_parts_are_inspectable_and_counted(sources):
    """A removed material absent from the ECN JSON still has a working part view."""
    snapshot = _snapshot(sources, 'ECN346863')
    objects = {obj['number']: obj for obj in snapshot['objects']}
    assert 'QKX2833570' in objects
    assert 'bom' in objects['QKX2833570']['origins']
    assert objects['QKX2833570']['locations']
    program = next(p for p in snapshot['programs'] if p['model_range'] == 'Q900V')
    assert any(row['number'] == 'QKX2833570' for row in program['mbom'])


def test_real_stock_and_purchasing_follow_rule_plants(sources):
    """PK01 purchasing counts only for the Plant B VC group; five old parts are stocked at PL10."""
    from app.ecn.reference_data.models import ReferenceDatasetType
    from app.ecn.reference_data.parsers import parse_reference_workbook
    references = []
    for kind, name in [('STOCK', 'SAP AI4U - Stock Information.XLSX'),
                       ('PURCHASE_INFO_RECORDS', 'SAP AI4U - Purchase Info Record2.XLSX'),
                       ('MATERIAL_MASTER', 'SAP AI4U - Material Master2.XLSX')]:
        parsed = parse_reference_workbook(ReferenceDatasetType(kind), (DATA / 'new_16-09' / name).read_bytes(), BOM_ARTIFACT_ID)
        references.extend(dict(row.values, dataset_type=kind) for row in parsed.rows)
    snapshot = _snapshot(sources, 'ECN346863', references)
    plants = {p['plant_code']: p for p in snapshot['plant_impact']['plants']}
    assert plants['PL10']['status'] == 'CONFIRMED'
    # SAP rows count only where the rules place the material.
    assert plants['PK01']['new_parts_with_purchasing'] == ['QKX0187890']
    assert {'plant_code': 'PK01', 'materials': ['QKX7702260']} in snapshot['plant_impact']['records_outside_rule_plants']
    assert {'number': 'QKX2833570', 'quantity': '5'} in plants['PL10']['old_part_stock']


def _lines(sources, number):
    """BOM change lines of one ECN on the real workbook."""
    return _snapshot(sources, number)["bom_changes"]["lines"]


def test_raw_only_rows_are_the_measured_489_with_11_on_published_json_ecns(sources):
    """The twin logic finds exactly the measured 130%-only material rows on the workbook.

    The published ECN export (data) keeps 11 of the 557 source ECNs, so
    11 of the 489 rows belong to an ECN present in that JSON (60 on the full export).
    """
    rows, ecns, _dataset, _context = sources
    ids = raw_only_row_ids(rows)
    assert len(ids) == 489
    assert sum(1 for row in rows if row.bom_row_id in ids and row.change_number in ecns) == 11


def test_ecns_that_exist_only_in_the_130_sheet_now_have_bom_changes(sources):
    """ECN193101/301489/304579/307099 have rows only in the 130% sheet; each row is now an added line."""
    lines = _lines(sources, "ECN193101")
    assert len(lines) == 6 and {l["change"] for l in lines} == {"ADDED"}
    # VC groups come from the rules via the 130% path root; ``assembly`` is the direct parent.
    assert {g for l in lines for g in l["vc_groups"]} == {"Q900VV0170801", "Q957VV0195800"}
    assert {l["position"] for l in lines} == {"0010", "0020"}
    assert all("EBOM_130_SOURCE" in l["qualifiers"] and l["engineering_date"] == "2017-03-08" for l in lines)
    (single,) = _lines(sources, "ECN301489")
    assert (single["incoming"]["number"], single["position"], single["vc_groups"]) == ("916100410830", "0022", ["Q900VV0140800"])
    assert sorted(g for l in _lines(sources, "ECN304579") for g in l["vc_groups"]) == ["Q900VV0145150", "Q900VV0145151"]
    assert sorted(l["position"] for l in _lines(sources, "ECN307099")) == ["0400", "0410"]


def test_150_rows_still_win_and_document_only_ecns_stay_empty(sources):
    """Existing ECNs keep their 150%/MBOM lines untouched; a documents-only ECN still has no BOM change."""
    lines = _lines(sources, "ECN345274")
    assert len(lines) == 4 and {l["change"] for l in lines} == {"REPLACED"}
    assert not any("EBOM_130_SOURCE" in l["qualifiers"] for l in lines)
    assert _lines(sources, "ECN142964") == []
