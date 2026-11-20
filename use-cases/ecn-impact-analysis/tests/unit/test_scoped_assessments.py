"""Protect scope isolation, historical selection, and dated material usage.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_scoped_assessments.py
"""
from datetime import date
from types import SimpleNamespace

import pytest

from app.ecn.reference_data.scopes import DatasetContext, select_assessment_datasets
from app.ecn.evidence.scoped_usage import usage_status


def dataset(identity, plant, model, *, order=1, comparison=None, group="demo"):
    """Build an immutable metadata observation without emulating persistence."""
    return SimpleNamespace(dataset_id=identity, artifact_id=identity, dataset_type="BOM",
        scope_code=plant, scope_name=None, file_name=identity, sha256="hash", publication_order=order,
        context={"model_range":model,"data_group":group,"environment":"SYNTHETIC",
                 "snapshot_date":"2026-09-01","coverage_complete":True,"comparison_dataset_id":comparison})


def test_scope_selection_pins_current_and_explicit_history_without_customer_mix():
    """Two models at one plant coexist; baseline history does not become current use."""
    records = [dataset("base", "P1", "R1"), dataset("new", "P1", "R1", order=2, comparison="base"),
               dataset("other", "P1", "R2"), dataset("customer", "P1", "R1", group="customer")]
    result = select_assessment_datasets(records, "demo")
    assert {(r["dataset_id"], r["comparison"]) for r in result} == {("base",True),("new",False),("other",False)}
    assert DatasetContext().coverage_complete is False
    with pytest.raises(ValueError):
        DatasetContext(environment="invented")


def test_scope_selection_rejects_cross_scope_comparison():
    """A foreign plant cannot be selected as a material's prior BOM snapshot."""
    with pytest.raises(ValueError):
        select_assessment_datasets([dataset("old","P2","R1"), dataset("new","P1","R1",comparison="old")],"demo")


def test_usage_dates_do_not_treat_unknowns_as_expired_or_historical_as_current():
    """Date boundaries are inclusive; missing dates and migration placeholders are unknown."""
    day = date(2026,9,22)
    base = {"valid_from_date":"2026-09-01","valid_to_date":"2026-09-22"}
    assert usage_status(base,day) == "EFFECTIVE"
    assert usage_status({**base,"valid_to_date":"2026-09-21"},day) == "EXPIRED"
    assert usage_status({**base,"valid_from_date":"2026-09-23","valid_to_date":"9999-12-31"},day) == "FUTURE"
    assert usage_status({**base,"valid_from_date":None},day) == "UNRESOLVED"
    assert usage_status({**base,"migration_date":True},day) == "UNRESOLVED"
    assert usage_status({**base,"comparison":True},day) == "HISTORICAL"


def test_snapshot_uses_historical_ebom_pairing_without_counting_history_as_use():
    """Complete scoped follow-up excludes expired history and retains an unrelated-number pair."""
    from tests.unit.test_material_changes import ecn, source
    from tests.fixtures.builders import make_bom_row, synthetic_uuid
    from app.ecn.evidence.snapshot import build_snapshot
    data = ecn([source("SIM2100","1","APPROVEDFORSERIES")],
        [(source("SIM2100","2","TOBEOBSOLETE"),"OBSOLESCENCE"),
         (source("SIM9800","1","APPROVEDFORSERIES"),"PRODUCTION_RELEASED")])
    old_id, new_id = synthetic_uuid("before"), synthetic_uuid("after")
    before = make_bom_row(artifact_id=old_id, component_number="SIM2100", parent_material_number="VC",
        position_value="0260", valid_from_date=date(2026,1,1), valid_to_date=date(9999,12,31))
    after = make_bom_row(artifact_id=new_id, component_number="SIM9800", parent_material_number="VC",
        position_value="260", valid_from_date=date(2026,9,15), valid_to_date=date(9999,12,31))
    scope = {"dataset_type":"BOM","plant_code":"P1","model_range":"R1","environment":"SYNTHETIC",
             "data_group":"demo","coverage_complete":True,"snapshot_date":"2026-09-22"}
    snapshot = build_snapshot(data,(after,), comparison_rows=(before,), assessment_date=date(2026,9,22),
        datasets=({**scope,"artifact_id":old_id,"comparison":True},{**scope,"artifact_id":new_id,"comparison":False}))
    changes = {c["number"]: c for c in snapshot["changes"]}
    assert changes["SIM2100"]["counterparts"] == ["SIM9800"]
    assert changes["SIM2100"]["pairing_basis"] == "EBOM_POSITION"
    assert changes["SIM2100"]["usage"]["status"] == "NO_REMAINING_USE_IN_SCOPE"
    assert next(o for o in snapshot["objects"] if o["number"] == "SIM2100")["locations"][0]["usage_status"] == "HISTORICAL"


def test_followups_name_missing_materials_and_distinguish_future_usage():
    """Missing placement and a scheduled occurrence require different evidence and actions."""
    from app.ecn.evidence.snapshot import build_snapshot
    from tests.fixtures.builders import make_bom_row, make_normalized_ecn, make_source_object

    data = make_normalized_ecn(affected_objects=tuple(make_source_object(number=n)
        for n in ("MISSING", "SCHEDULED")), resulting_objects=())
    row = make_bom_row(component_number="SCHEDULED", parent_material_number="ASSEMBLY",
        valid_from_date=date(2026,10,1), valid_to_date=date(9999,12,31))
    scope = {"dataset_type":"BOM", "artifact_id":row.artifact_id, "file_name":"range.xlsx",
        "plant_code":"P1", "model_range":"R1", "environment":"SYNTHETIC",
        "data_group":"demo", "coverage_complete":True, "snapshot_date":"2026-09-22"}
    snapshot = build_snapshot(data, (row,), datasets=(scope,), assessment_date=date(2026,9,23))
    questions = {q["code"]: q for q in snapshot["questions"]}
    missing = questions["BOM_LOCATION_UNAVAILABLE"]
    assert [s["number"] for s in missing["subjects"]] == ["MISSING"]
    assert "R1" in " ".join(missing["scopes"]) and "P1" not in " ".join(missing["scopes"])
    assert "range.xlsx" in " ".join(missing["scopes"])
    assert missing["question"] == "Material not found in supplied EBOMs or MBOMs"
    assert "parent assembly and position in the BOM hierarchy" in missing["why"]
    assert "not found as a component" in missing["subjects"][0]["details"][0]
    future = questions["BOM_FUTURE_USAGE"]
    assert [s["number"] for s in future["subjects"]] == ["SCHEDULED"]
    assert "2026-10-01" in " ".join(future["subjects"][0]["details"])
    assert "ASSEMBLY" in " ".join(future["subjects"][0]["details"])


def test_stock_followups_only_request_missing_fields_at_exact_source_locations():
    """A known extraction date must not be requested; complete and historical rows add no gaps."""
    from app.ecn.evidence.snapshot import build_snapshot
    from tests.fixtures.builders import make_normalized_ecn, make_source_object

    data = make_normalized_ecn(affected_objects=(make_source_object(number="PART"),), resulting_objects=())
    row = {"dataset_type":"STOCK", "material_number":"PART", "plant_code":"P1",
        "storage_location":"S1", "file_name":"stock.xlsx", "sheet_name":"Stock",
        "source_row_number":2, "snapshot_date":"2026-09-22", "unrestricted_quantity":"0"}
    snapshot = build_snapshot(data, (), (row,))
    question = next(q for q in snapshot["questions"] if q["code"] == "STOCK_CONTEXT_LIMITED")
    assert [s["number"] for s in question["subjects"]] == ["PART"]
    assert "extraction date" not in question["resolution"].lower()
    details = " ".join(question["subjects"][0]["details"])
    assert all(v in details for v in ("P1", "S1", "stock.xlsx", "row 2", "2026-09-22"))
    complete = build_snapshot(data, (), ({**row, "unit":"EA"}, {**row, "comparison":True}))
    assert not any(q["code"] == "STOCK_CONTEXT_LIMITED" for q in complete["questions"])


def test_phase_in_with_future_mbom_requests_date_confirmation_not_another_mbom():
    """Existing scheduled manufacturing evidence must not become a missing-source request."""
    from app.ecn.evidence.snapshot import build_snapshot
    from tests.unit.test_material_changes import ecn, source
    from tests.fixtures.builders import make_bom_row
    from app.ecn.models import BomKind

    data = ecn([], [(source("INCOMING", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    engineering = make_bom_row(component_number="INCOMING", valid_from_date=date(2026,1,1), valid_to_date=date(9999,12,31))
    manufacturing = make_bom_row(component_number="INCOMING", bom_kind=BomKind.MBOM_130,
        valid_from_date=date(2026,10,1), valid_to_date=date(9999,12,31))
    snapshot = build_snapshot(data, (engineering, manufacturing), assessment_date=date(2026,9,23))
    questions = {q["code"]:q for q in snapshot["questions"]}
    assert "MBOM_INTRODUCTION_UNCONFIRMED" not in questions
    assert questions["MBOM_FUTURE_USAGE"]["subjects"][0]["number"] == "INCOMING"
    assert "2026-10-01" in " ".join(questions["MBOM_FUTURE_USAGE"]["subjects"][0]["details"])


@pytest.mark.parametrize("mbom_end", [None, date(2026, 1, 31)])
def test_manufacturing_followup_distinguishes_ebom_evidence_from_valid_mbom_use(mbom_end):
    """Present EBOM evidence must not imply that a missing or expired MBOM is valid."""
    from app.ecn.evidence.snapshot import build_snapshot
    from tests.unit.test_material_changes import ecn, source
    from tests.fixtures.builders import make_bom_row
    from app.ecn.models import BomKind

    data = ecn([], [(source("INCOMING", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    rows = [make_bom_row(component_number="INCOMING", valid_from_date=date(2026, 1, 1), valid_to_date=date(9999, 12, 31))]
    if mbom_end:
        rows.append(make_bom_row(component_number="INCOMING", bom_kind=BomKind.MBOM_130,
            valid_from_date=date(2026, 1, 1), valid_to_date=mbom_end))
    snapshot = build_snapshot(data, tuple(rows), assessment_date=date(2026, 9, 23))
    question = next(q for q in snapshot["questions"] if q["code"] == "MBOM_INTRODUCTION_UNCONFIRMED")
    assert "engineering BOM (EBOM)" in question["why"]
    assert "no manufacturing BOM (MBOM) row valid on the assessment date" in question["why"]
