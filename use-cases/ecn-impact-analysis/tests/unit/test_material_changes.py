"""Check lifecycle and pairing decisions using explicit, independent expectations.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_material_changes.py
"""

from uuid import uuid4

from app.ecn.intake.json_parser import _normalize_ecn
from app.ecn.evidence.material_changes import interpret_changes


def source(number, revision, state, kind="wt.part.WTPart"):
    """Return a source object with stable technical class and explicit revision."""
    return {"Number": number, "Revision": revision, "State": {"Value": state},
            "ID": f"OR:{kind}:{number}-{revision}", "ObjectType": "Part"}


def ecn(affected, resulting):
    """Normalize source objects and their target transitions through the real parser."""
    return _normalize_ecn({"Number": "ECN-DEMO", "CNAffectLinks": [{"AffectedObjects": affected}],
        "ImplementationPlan": [{"ResultingLinks": [
            {"TargetTransition": {"Value": target}, "ResultingObjects": [obj]}
            for obj, target in resulting]}]}, str(uuid4()), 0)


def test_retirement_revision_and_unique_number_pattern():
    """Retirement wins over revision and unique suffix pairing remains labelled."""
    data = ecn([source("SIM1000", "3", "APPROVEDFORSERIES"), source("VC1", "5", "APPROVEDFORSERIES")],
        [(source("SIM1000", "4", "TOBEOBSOLETE"), "OBSOLESCENCE"),
         (source("SIM100A", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
         (source("VC1", "6", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    changes = interpret_changes(data)
    old = next(c for c in changes if c["number"] == "SIM1000")
    assert (old["action"], old["revision_from"], old["revision_to"]) == ("PHASE_OUT", "3", "4")
    assert old["counterparts"] == ["SIM100A"]
    assert old["pairing_basis"] == "NUMBER_PATTERN"
    assert next(c for c in changes if c["number"] == "VC1")["action"] == "REVISION"


def test_ebom_position_pairs_unrelated_numbers_but_not_other_parents_or_mbom():
    """Only an exact scoped EBOM position across snapshots establishes a pair."""
    data = ecn([source("SIM2100", "1", "APPROVEDFORSERIES")],
        [(source("SIM2100", "2", "TOBEOBSOLETE"), "OBSOLESCENCE"),
         (source("SIM9800", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    common = {"kind": "ENRICHED_EBOM", "parent": "VC", "scope_id": "R1/P1", "alternative": None}
    rows = [{**common, "number": "SIM2100", "item_or_position": "0260", "comparison": True},
            {**common, "number": "SIM9800", "item_or_position": "260", "comparison": False}]
    assert interpret_changes(data, rows)[0]["pairing_basis"] == "EBOM_POSITION"
    rows[1]["parent"] = "OTHER"
    assert interpret_changes(data, rows)[0]["counterparts"] == []
    rows[1]["parent"] = "VC"
    for row in rows:
        row["kind"] = "MBOM_130"
    assert interpret_changes(data, rows)[0]["counterparts"] == []


def test_conflicts_ambiguous_suffixes_and_missing_revisions_remain_unresolved():
    """Do not choose among competing suffixes or infer a missing previous revision."""
    data = ecn([], [(source("SIM1000", "4", "TOBEOBSOLETE"), "OBSOLESCENCE"),
                    (source("SIM100A", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
                    (source("SIM100B", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
                    (source("BAD", "2", "TOBEOBSOLETE"), "PRODUCTION_RELEASED")])
    changes = {c["number"]: c for c in interpret_changes(data)}
    assert changes["SIM1000"]["counterparts"] == []
    assert "AMBIGUOUS_PAIRING" in changes["SIM1000"]["qualifiers"]
    assert changes["SIM100A"]["revision_from"] is None
    assert changes["BAD"]["action"] == "REVIEW"


def test_explicit_source_pair_has_priority_and_unchanged_revision_is_not_revised():
    """Explicit endpoints beat naming conventions; equal revisions stay unchanged."""
    data = ecn([source("SIM1000", "1", "APPROVEDFORSERIES"), source("VC", "7", "APPROVEDFORSERIES")],
        [(source("SIM1000", "2", "TOBEOBSOLETE"), "OBSOLESCENCE"),
         (source("SIM100A", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
         (source("OTHER", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
         (source("VC", "7", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    data.source_json["ImplementationPlan"][0]["ResultingLinks"][2]["OemReplacesAffected"] = "SIM1000"
    changes = {c["number"]: c for c in interpret_changes(data)}
    assert changes["SIM1000"]["counterparts"] == ["OTHER"]
    assert changes["SIM1000"]["pairing_basis"] == "SOURCE_RELATION"
    assert changes["SIM100A"]["counterparts"] == []
    assert changes["VC"]["action"] == "UNCHANGED"


def test_missing_revision_and_conflicting_historical_position_do_not_invent_comparisons():
    """Missing revision is not unchanged; observed contrary positions block name fallback."""
    data = ecn([source("SIM1000", "1", "APPROVEDFORSERIES"), source("VC", None, "APPROVEDFORSERIES")],
        [(source("SIM1000", "2", "TOBEOBSOLETE"), "OBSOLESCENCE"),
         (source("SIM100A", "1", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED"),
         (source("VC", "6", "APPROVEDFORSERIES"), "PRODUCTION_RELEASED")])
    common = {"kind": "ENRICHED_EBOM", "parent": "VC", "scope_id": "R/P", "item_or_position": "150"}
    rows = [{**common, "number": "SIM1000", "comparison": True, "alternative": "1"},
            {**common, "number": "SIM100A", "comparison": False, "alternative": "2"}]
    changes = {c["number"]: c for c in interpret_changes(data, rows)}
    assert changes["SIM1000"]["counterparts"] == []
    assert changes["VC"]["action"] == "RELEASE"
    assert "REVISION_COMPARISON_UNAVAILABLE" in changes["VC"]["qualifiers"]
