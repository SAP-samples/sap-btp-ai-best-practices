"""Per-part acceptance plan and its missing-price block, from hand-made snapshots.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_acceptance_plan.py
"""

from types import SimpleNamespace

import pytest

from app.ecn.evidence.acceptance_plan import build_acceptance_plan


def _material(number, role, status, date=None, still_used_in=()):
    """Return one bom_changes material as _material_flags builds it."""
    return {"number": number, "description": f"{number} description", "role": role, "status": status,
            "date": date, "still_used_in": list(still_used_in)}


def _snapshot(priced=("NEW1",)):
    """PL10 holds two new parts (one already set up there) and three old parts."""
    materials = [
        _material("NEW1", "INCOMING", "NEW_PART", "2026-03-13"),
        _material("NEW2", "INCOMING", "NEW_PART", "9999-12-31"),
        _material("OLD1", "OUTGOING", "FULLY_PHASED_OUT", "2026-05-19"),
        _material("OLD2", "OUTGOING", "STILL_USED", "2026-05-19", ["VC1", "VC2", "VC3"]),
        _material("OLD3", "OUTGOING", "NO_REMAINING_USE_FOUND", "2026-05-19"),
    ]
    pl10 = {"plant_code": "PL10", "new_parts_with_purchasing": [], "new_parts_without_purchasing": ["NEW1"],
            "materials": [{"number": m["number"]} for m in materials]}
    undetermined = {"plant_code": None, "materials": [{"number": "NEW1"}]}
    return {"assessment_id": "a1", "evidence_hash": "h" * 64,
            "bom_changes": {"materials": materials, "lines": [{}, {}]},
            "plant_impact": {"plants": [pl10, undetermined], "priced_new_parts": list(priced)}}


def test_each_part_gets_one_plain_line_and_a_price_state():
    """New parts first; old parts need no price; a part still used elsewhere is left alone."""
    plan = build_acceptance_plan(_snapshot())
    by_number = {p["number"]: p for p in plan["parts"]}
    assert [p["number"] for p in plan["parts"]] == ["NEW1", "NEW2", "OLD1", "OLD2", "OLD3"]
    assert by_number["NEW1"] == {"number": "NEW1", "description": "NEW1 description", "role": "NEW",
                                      "master_data": "Release in PL10 from 2026-03-13", "price": "HAS_PRICE"}
    assert by_number["NEW2"]["master_data"] == "Create and release in PL10 date not known"
    assert by_number["NEW2"]["price"] == "MISSING"
    assert by_number["OLD1"]["master_data"] == "Phase out in all plants from 2026-05-19"
    assert by_number["OLD2"]["master_data"] == "No change: still used in VC1, VC2 +1 more"
    assert by_number["OLD3"]["master_data"] == "Phase out in PL10 from 2026-05-19"
    assert {by_number[n]["price"] for n in ("OLD1", "OLD2", "OLD3")} == {"NOT_NEEDED"}
    assert plan["bom_item_count"] == 2


def test_a_new_part_without_price_blocks_the_ecn():
    """Any new part missing from the purchase info records blocks acceptance, with a one-line reason."""
    plan = build_acceptance_plan(_snapshot())
    assert plan["blocked"] is True and plan["missing_price"] == ["NEW2"]
    assert plan["block_reason"] == "Can't accept: 1 new part has no price."
    plan = build_acceptance_plan(_snapshot(priced=()))
    assert plan["block_reason"] == "Can't accept: 2 new parts have no price."
    assert build_acceptance_plan(_snapshot(priced=("NEW1", "NEW2")))["blocked"] is False


def test_snapshot_without_bom_changes_has_no_parts_and_no_block():
    """Version 1 snapshots have nothing to price, so nothing blocks."""
    plan = build_acceptance_plan({"assessment_id": "a1", "evidence_hash": "h"})
    assert plan["available"] is False and plan["parts"] == [] and plan["blocked"] is False


def test_record_review_refuses_to_accept_a_blocked_ecn_but_can_send_it_back():
    """The shared save path enforces the block for the card, the API and the assistant."""
    from app.ecn.errors import PersistenceStateError
    from app.ecn.services.reviews import record_review

    snapshot = _snapshot() | {"version": "ecn-snapshot-v2"}
    saved = []
    workspace = SimpleNamespace(save_review=saved.append, reviews=lambda identity: saved)
    snapshots = SimpleNamespace(overview=lambda identity: snapshot)
    common = {"request_id": "r1", "reviewer": "Ana", "actor": "hashed",
              "evidence_hash": snapshot["evidence_hash"], "snapshot_version": "ecn-snapshot-v2"}
    with pytest.raises(PersistenceStateError) as error:
        record_review(snapshots, workspace, "a1", outcome="ACCEPTED", comments="", **common)
    assert error.value.code == "ECN_ACCEPT_BLOCKED" and error.value.status_code == 409 and not saved
    record_review(snapshots, workspace, "a1", outcome="CHANGES_REQUESTED", comments="Add prices", **common)
    assert [row["OUTCOME"] for row in saved] == ["CHANGES_REQUESTED"]
