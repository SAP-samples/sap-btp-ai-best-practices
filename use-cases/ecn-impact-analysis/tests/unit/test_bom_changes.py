"""BOM change lines and phase flags built from dated BOM rows carrying the ECN code.

Rows mirror the real Q900V MBOM pattern: the outgoing row ends with a derived
MBOM code (MPXA + ECN digits + suffix), the incoming row starts with it, both
under the same VC group, the old row ending the day before the new one starts.
"""

from app.ecn.evidence.bom_changes import assembly_key, build_bom_changes, change_code_matches

SCOPE = "real/PRODUCTION/Q900V"
PLANT_A = {"plant_code": "PL10", "plant_name": "Plant A"}
PLANT_B = {"plant_code": "PK01", "plant_name": "Plant B"}


def loc(number, parent, *, kind="MBOM_130", start="2021-03-19", end="9999-12-31", change=None, change_to=None,
        qty="2", position=None, comparison=False, scope=SCOPE, plants=(PLANT_A,), description=None):
    """Build one snapshot BOM location dict with only the fields the engine reads.

    ``plant_scope`` mimics ``resolve_row``: the VC group is the parent without
    its model suffix and ``plants`` are the rule-derived plants.
    """
    return {"number": number, "parent": parent, "kind": kind, "valid_from_date": start, "valid_to_date": end,
            "change_number": change, "change_number_to": change_to, "quantity": qty,
            "item_or_position": position, "scope_id": scope,
            "plant_scope": {"vc_groups": [assembly_key(parent)], "plants": list(plants),
                            "reason": None if plants else "BRAND_NOT_MAPPED", "unmapped_brands": [] if plants else ["XX"]},
            "model_range": "Q900V", "comparison": comparison, "description": description or number + " description",
            "file_name": "bom.xlsx", "sheet": "sheet", "row": 1}


def replacement_rows():
    """ECN345274 in one assembly: two old parts replaced by their A revisions."""
    parent = "Q923VV2526401_Q900"
    return [
        loc("QKX4638810", parent, end="2026-06-29", change="MPXA27909600", change_to="MPXA3452740A"),
        loc("QKX8914160", parent, end="2026-06-29", change="MPXA27909600", change_to="MPXA3452740A"),
        loc("QKX463881A", parent, start="2026-06-30", change="MPXA3452740A"),
        loc("QKX891416A", parent, start="2026-06-30", change="MPXA3452740A"),
        loc("QKX463881A", "Q923VV2526401", kind="ENRICHED_EBOM", start="2026-01-16", change="ECN345274", position="0150"),
        loc("QKX891416A", "Q923VV2526401", kind="ENRICHED_EBOM", start="2026-01-16", change="ECN345274", position="0160"),
    ]


def test_change_codes_and_assembly_keys():
    """MBOM codes map to ECN digits; unrelated codes and suffix-free parents behave."""
    assert change_code_matches("ECN345274", "MPXA3452740A")
    assert change_code_matches("ECN345274", "ECN345274")
    assert not change_code_matches("ECN345274", "MPXA34527500")
    assert not change_code_matches("ECN345274", None)
    assert assembly_key("Q923VV2526401_Q900") == assembly_key("Q923VV2526401") == "Q923VV2526401"


def test_replacement_linked_by_number_with_position_and_dates():
    """Old and new part share one line with EBOM position, EBOM date and MBOM cut-over."""
    result = build_bom_changes("ECN345274", replacement_rows(), coverage_complete=True)
    lines = {l["outgoing"]["number"]: l for l in result["lines"]}
    line = lines["QKX4638810"]
    assert line["change"] == "REPLACED" and line["incoming"]["number"] == "QKX463881A"
    assert line["pairing"] == "NUMBER" and line["position"] == "0150"
    assert (line["end_date"], line["manufacturing_date"], line["engineering_date"]) == ("2026-06-29", "2026-06-30", "2026-01-16")
    assert line["levels"] == ["EBOM", "MBOM"] and line["assembly"] == "Q923VV2526401"
    assert lines["QKX8914160"]["incoming"]["number"] == "QKX891416A" and lines["QKX8914160"]["position"] == "0160"
    flags = {m["number"]: m["status"] for m in result["materials"]}
    assert flags == {"QKX4638810": "FULLY_PHASED_OUT", "QKX8914160": "FULLY_PHASED_OUT",
                     "QKX463881A": "NEW_PART", "QKX891416A": "NEW_PART"}
    assert result["counts"]["REPLACED"] == 2 and result["counts"]["new_parts"] == 2
    assert line["plants"] == [PLANT_A] and line["vc_groups"] == ["Q923VV2526401"] and line["plant_reason"] is None


def test_assembly_description_is_taken_from_the_observed_bom_hierarchy():
    """A parent that occurs in the BOM carries its description into the change line."""
    rows = [
        loc("Q900VV0080402", "Q900VS0080", description="M1V decal assembly"),
        loc("NEW1", "Q900VV0080402_Q900", start="2026-05-19", change="MPXA34686300"),
    ]
    [line] = build_bom_changes("ECN346863", rows)["lines"]
    assert line["assembly"] == "Q900VV0080402"
    assert line["assembly_description"] == "M1V decal assembly"


def test_line_plants_union_all_source_rows() -> None:
    """A VC group built in two plants gives one line listing both; no plant keeps its reason."""
    parent = "Q900VV0080301_Q900"
    rows = [loc("OLD1", parent, end="2026-01-01", change_to="MPXA1234560A", plants=(PLANT_A, PLANT_B)),
            loc("OLD1A", parent, start="2026-01-02", change="MPXA1234560A", plants=(PLANT_A, PLANT_B)),
            loc("NEW7", "Q900VV9999999_Q900", start="2026-01-02", change="MPXA1234560A", plants=())]
    lines = {(l["incoming"] or {}).get("number"): l for l in build_bom_changes("ECN123456", rows)["lines"]}
    assert lines["OLD1A"]["plants"] == [PLANT_B, PLANT_A]
    assert lines["NEW7"]["plants"] == [] and lines["NEW7"]["plant_reason"] == "BRAND_NOT_MAPPED"


def test_ecn_pair_beats_number_pattern():
    """A pair declared by the ECN lifecycle is used even when numbers do not match."""
    rows = [loc("OLD1", "G_Q900", end="2026-01-01", change_to="MPXA1234560A"),
            loc("NEW9", "G_Q900", start="2026-01-02", change="MPXA1234560A"),
            loc("OLD2", "G_Q900", end="2026-01-01", change_to="MPXA1234560A"),
            loc("NEW8", "G_Q900", start="2026-01-02", change="MPXA1234560A")]
    lifecycle = [{"number": "OLD1", "counterparts": ["NEW9"]}, {"number": "NEW9", "counterparts": ["OLD1"]}]
    result = build_bom_changes("ECN123456", rows, lifecycle)
    pairs = {(l["outgoing"] or {}).get("number"): (l["incoming"] or {}).get("number") for l in result["lines"]}
    # OLD1/NEW9 from the ECN; OLD2/NEW8 are then the only remaining pair.
    assert pairs == {"OLD1": "NEW9", "OLD2": "NEW8"}


def test_quantity_change_removals_and_unconfirmed_addition():
    """ECN346863 pattern: 3 out, 1 in, 2 quantity changes; nothing is guessed."""
    parent = "Q900VV0080402_Q900"
    rows = [
        loc("QKX2833570", parent, qty="1", end="2026-05-18", change_to="MPXA34686300"),
        loc("6646743T1", parent, qty="1", end="2026-05-18", change_to="MPXA34686300"),
        loc("3388708T1", parent, qty="1", end="2026-05-18", change_to="MPXA34686300"),
        loc("8241814T1", parent, qty="1", end="2026-05-18", change_to="MPXA34686300"),
        loc("8241814T1", parent, qty="2", start="2026-05-19", change="MPXA34686300"),
        loc("QKX7702260", parent, qty="1", start="2026-05-19", change="MPXA34686300"),
        loc("QKX0187890", "Q900VV0080301_Q900", qty="1", start="2026-05-19", change="MPXA34686300"),
        # 3388708T1 keeps an effective row in another assembly.
        loc("3388708T1", "Q900VV0080999_Q900", qty="1"),
    ]
    result = build_bom_changes("ECN346863", rows, coverage_complete=True)
    by = {(l["change"], (l["outgoing"] or l["incoming"])["number"]): l for l in result["lines"]}
    qty = by[("QUANTITY_CHANGED", "8241814T1")]
    assert (qty["outgoing"]["quantity"], qty["incoming"]["quantity"]) == ("1", "2")
    assert {n for c, n in by if c == "REMOVED"} == {"QKX2833570", "6646743T1", "3388708T1"}
    assert by[("ADDED", "QKX7702260")]["qualifiers"] == ["REPLACEMENT_NOT_CONFIRMED"]
    assert by[("ADDED", "QKX0187890")]["qualifiers"] == []
    flags = {m["number"]: m for m in result["materials"]}
    assert flags["QKX2833570"]["status"] == "FULLY_PHASED_OUT" and flags["QKX2833570"]["date"] == "2026-05-18"
    assert flags["3388708T1"]["status"] == "STILL_USED" and flags["3388708T1"]["still_used_in"] == ["Q900VV0080999"]
    assert "8241814T1" not in flags


def test_existing_part_and_incomplete_coverage():
    """An incoming part used before is not new; incomplete coverage never claims full phase-out."""
    rows = [loc("OLD", "G_Q900", end="2026-01-01", change_to="MPXA1234560A"),
            loc("REUSED", "G_Q900", start="2026-01-02", change="MPXA1234560A"),
            loc("REUSED", "OTHER_Q900", start="2020-01-01")]
    result = build_bom_changes("ECN123456", rows, coverage_complete=False)
    flags = {m["number"]: m for m in result["materials"]}
    assert flags["OLD"]["status"] == "NO_REMAINING_USE_FOUND"
    assert flags["REUSED"]["status"] == "ALREADY_USED" and flags["REUSED"]["still_used_in"] == ["OTHER"]
    assert flags["REUSED"]["prior_manufacturing_bom_use"] is True


def test_ebom_release_without_mbom_start_is_flagged():
    """An EBOM row released by the ECN with no MBOM start shows MBOM not updated."""
    rows = [loc("NEWPART", "G", kind="ENRICHED_EBOM", start="2026-03-13", change="ECN346863", position="0010")]
    [line] = build_bom_changes("ECN346863", rows)["lines"]
    assert line["change"] == "ADDED" and line["levels"] == ["EBOM"]
    assert line["qualifiers"] == ["MBOM_NOT_UPDATED"] and line["manufacturing_date"] is None
    assert line["engineering_date"] == "2026-03-13" and line["position"] == "0010"


def test_comparison_rows_are_ignored_for_lines():
    """Historical comparison rows never create change lines."""
    rows = [loc("OLD", "G_Q900", end="2026-01-01", change_to="MPXA1234560A", comparison=True)]
    assert build_bom_changes("ECN123456", rows)["lines"] == []


def test_new_assembly_is_in_mbom_when_its_children_start():
    """A VC group released in EBOM counts as in the MBOM once the ECN starts rows under it."""
    rows = [loc("Q900VV0080460", "Q900VS0080", kind="ENRICHED_EBOM", start="2026-03-13", change="ECN346863", position="0455"),
            loc("QKX3105210", "Q900VV0080460_Q900", start="2026-05-19", change="MPXA34686300")]
    lines = {l["incoming"]["number"]: l for l in build_bom_changes("ECN346863", rows)["lines"]}
    group = lines["Q900VV0080460"]
    assert group["qualifiers"] == [] and group["levels"] == ["EBOM", "MBOM"] and group["manufacturing_date"] == "2026-05-19"


def test_program_matches_lists_declared_programs_without_bom():
    """Declared products without an uploaded BOM stay visible; rows split by BOM level."""
    from app.ecn.evidence.snapshot import program_matches
    snapshot = {
        "declared_products": "Q938V, Q900V, ",
        "datasets": [{"dataset_type": "BOM", "model_range": "Q900V"}],
        "objects": [{"technical": True, "origins": ["ecn"], "number": "QKX463881A", "locations": [
            loc("QKX463881A", "Q923VV2526401", kind="ENRICHED_EBOM", position="0150"),
            loc("QKX463881A", "Q923VV2526401_Q900"),
            loc("QKX463881A", "OLD", comparison=True),
        ]}],
    }
    programs = program_matches(snapshot)
    assert [p["model_range"] for p in programs] == ["Q900V", "Q938V"]
    q900v, q938v = programs
    assert q900v["bom_uploaded"] and len(q900v["ebom"]) == 1 and len(q900v["mbom"]) == 1
    assert q900v["ebom"][0]["position"] == "0150" and q900v["mbom"][0]["position"] is None
    assert not q938v["bom_uploaded"] and q938v["ebom"] == q938v["mbom"] == []


def test_new_vc_group_started_in_mbom_raises_no_manufacturing_question():
    """A phase-in VC group whose MBOM rows the ECN starts is not 'MBOM introduction unconfirmed'."""
    from app.ecn.evidence.review_questions import build_review_questions
    ebom = loc("Q900VV0080460", "Q900VS0080", kind="ENRICHED_EBOM", start="2026-03-13", change="ECN346863")
    ebom["usage_status"] = "EFFECTIVE"
    snapshot = {"version": "ecn-snapshot-v2", "datasets": [], "objects": [{
        "technical": True, "origins": ["ecn"], "number": "Q900VV0080460", "description": None, "object_key": "k",
        "types": ["OEM VC Group"], "locations": [ebom], "reference_records": [], "change": {"action": "PHASE_IN"}}]}
    snapshot["bom_changes"] = {"lines": [{"incoming": {"number": "Q900VV0080460"}, "levels": ["EBOM", "MBOM"],
                                        "manufacturing_usage_statuses": ["EFFECTIVE"]}]}
    assert "MBOM_INTRODUCTION_UNCONFIRMED" not in {q["code"] for q in build_review_questions(snapshot)}
    snapshot["bom_changes"]["lines"][0]["levels"] = ["EBOM"]
    assert "MBOM_INTRODUCTION_UNCONFIRMED" in {q["code"] for q in build_review_questions(snapshot)}


def test_alternatives_are_not_paired_and_duplicate_parts_are_preserved():
    """Changes in different BOM alternatives stay separate without dropping repeats."""
    rows = [loc('OLD', 'G', end='2026-01-01', change_to='SIM-ECN-001') | {'alternative': 'A'},
            loc('NEW', 'G', start='2026-01-02', change='SIM-ECN-001') | {'alternative': 'B'},
            loc('NEW', 'G', start='2026-01-02', change='SIM-ECN-001') | {'alternative': 'C'}]
    lines = build_bom_changes('SIM-ECN-001', rows)['lines']
    assert len(lines) == 3
    assert not any(line['change'] == 'REPLACED' for line in lines)


def test_unknown_dates_cannot_establish_full_phase_out_or_new_part():
    """Missing dates remain unknown, even with declared complete BOM coverage."""
    rows = [loc('OLD', 'G', end=None, change_to='SIM-ECN-001'),
            loc('NEW', 'G', start=None, change='SIM-ECN-001')]
    flags = build_bom_changes('SIM-ECN-001', rows, coverage_complete=True)['materials']
    assert {item['status'] for item in flags} == {'TIMING_UNKNOWN'}


def test_ebom_removal_and_replacement_are_not_reported_as_only_additions():
    """Dated engineering changes work even before a manufacturing BOM is supplied."""
    rows = [loc('OLD', 'G', kind='ENRICHED_EBOM', end='2026-01-01', change_to='SIM-ECN-001', position='0010'),
            loc('NEW', 'G', kind='ENRICHED_EBOM', start='2026-01-02', change='SIM-ECN-001', position='0010')]
    [line] = build_bom_changes('SIM-ECN-001', rows)['lines']
    assert line['change'] == 'REPLACED'
    assert line['outgoing']['number'] == 'OLD'
    assert line['position'] == '0010' and line['levels'] == ['EBOM']
    assert line['manufacturing_date'] is None


def test_mbom_change_pattern_does_not_match_an_engineering_code():
    """Embedded codes are MBOM-specific; callers may supply a configured layout."""
    assert not change_code_matches('ECN123456', 'MPXA12345600', 'ENRICHED_EBOM')
    assert change_code_matches('ECN123456', 'LOCAL-123456', 'MBOM_130', r'^LOCAL-(?P<digits>\d{6})$')


def test_bom_change_settings_are_validated():
    """Operators can supply MBOM change-number layouts without changing code."""
    import pytest
    from app.ecn.settings import load_ecn_settings
    config = load_ecn_settings({'MBOM_CHANGE_NUMBER_PATTERN': r'^LOCAL-(?P<digits>\d{6})$'})
    assert config.mbom_change_number_pattern == r'^LOCAL-(?P<digits>\d{6})$'
    with pytest.raises(ValueError):
        load_ecn_settings({'MBOM_CHANGE_NUMBER_PATTERN': '^NO_CAPTURE$'})


def test_dated_ebom_and_mbom_pairs_merge_without_false_removal():
    """Both engineering sides belong to the combined manufacturing replacement."""
    rows = [loc('OLD', 'G_Q900', end='2026-06-29', change_to='SIM-ECN-001'),
            loc('NEW', 'G_Q900', start='2026-06-30', change='SIM-ECN-001'),
            loc('OLD', 'G', kind='ENRICHED_EBOM', end='2026-01-15', change_to='SIM-ECN-001', position='0010'),
            loc('NEW', 'G', kind='ENRICHED_EBOM', start='2026-01-16', change='SIM-ECN-001', position='0010')]
    [line] = build_bom_changes('SIM-ECN-001', rows)['lines']
    assert line['change'] == 'REPLACED' and line['qualifiers'] == []
    assert len(line['sources']) == 4
    assert line['engineering_end_date'] == '2026-01-15'


def test_mixed_known_and_missing_dates_cannot_establish_material_wide_flags():
    """A dated use in one assembly cannot hide an undated use in another."""
    rows = [loc('OLD', 'G1', end='2026-01-01', change_to='SIM-ECN-001'),
            loc('NEW', 'G1', start='2026-01-02', change='SIM-ECN-001'),
            loc('OLD', 'G2', end=None, change_to='SIM-ECN-001'),
            loc('NEW', 'G2', start=None, change='SIM-ECN-001')]
    flags = build_bom_changes('SIM-ECN-001', rows, coverage_complete=True)['materials']
    assert {item['status'] for item in flags} == {'TIMING_UNKNOWN'}
    assert all(item['date'] is None for item in flags)


def test_future_and_undated_parent_mbom_introductions_keep_readiness_questions():
    """Manufacturing presence alone cannot confirm current assembly usage."""
    from app.ecn.evidence.review_questions import build_review_questions
    for status, expected in [('FUTURE', 'MBOM_FUTURE_USAGE'), ('UNRESOLVED', 'MBOM_INTRODUCTION_UNCONFIRMED')]:
        ebom = loc('VC1', 'TOP', kind='ENRICHED_EBOM', change='SIM-ECN-001') | {'usage_status': 'EFFECTIVE'}
        child = loc('PART1', 'VC1_Q900', start='2026-10-01' if status == 'FUTURE' else None,
                    change='SIM-ECN-001') | {'usage_status': status}
        snapshot = {'version': 'ecn-snapshot-v2', 'coverage_complete': True, 'datasets': [], 'objects': [{
            'technical': True, 'origins': ['ecn'], 'number': 'VC1', 'object_key': 'vc',
            'locations': [ebom], 'reference_records': [], 'change': {'action': 'PHASE_IN'}}],
            'bom_changes': build_bom_changes('SIM-ECN-001', [ebom, child])}
        assert expected in {q['code'] for q in build_review_questions(snapshot)}


def test_ecn_phase_out_with_only_bom_starts_is_a_conflict_not_a_new_part():
    """Synthetic feedback case: an active starting row cannot make a phase-out part new."""
    from app.ecn.evidence.plant_impact import build_plant_impact
    rows = [loc('OUTGOING', 'G', start='2026-01-01', change='SIM-ECN-001')]
    lifecycle = [{'number': 'OUTGOING', 'action': 'PHASE_OUT', 'counterparts': []}]
    changes = build_bom_changes('SIM-ECN-001', rows, lifecycle, coverage_complete=True)
    assert changes['lines'][0]['qualifiers'] == ['DECLARED_PHASE_OUT']
    [material] = changes['materials']
    assert material['role'] == 'OUTGOING' and material['status'] == 'PHASE_OUT_NOT_CONFIRMED'
    assert changes['counts']['new_parts'] == 0
    [plant] = build_plant_impact(changes, [], {})['plants']
    assert plant['status'] == 'CHECK'


def raw_only(*args, **kwargs):
    """Build a 130% EBOM location the snapshot flagged as having no 150% twin."""
    return {**loc(*args, kind="RAW_EBOM_130", **kwargs), "raw_only": True}


def test_raw_only_row_started_by_the_ecn_becomes_an_added_line_from_the_130_sheet():
    """ECN193101 pattern: a part started in the 130% sheet only is an EBOM-side addition, flagged by source."""
    rows = [raw_only("X596516039000", "Q921PD3001908", start="2017-03-08", change="ECN193101", position="10", qty="2")]
    result = build_bom_changes("ECN193101", rows)
    [line] = result["lines"]
    assert line["change"] == "ADDED" and line["incoming"]["number"] == "X596516039000"
    assert line["levels"] == ["EBOM"] and line["position"] == "0010" and line["engineering_date"] == "2017-03-08"
    assert line["qualifiers"] == ["EBOM_130_SOURCE", "MBOM_NOT_UPDATED"]
    assert line["sources"][0]["kind"] == "RAW_EBOM_130"
    assert result["counts"]["ADDED"] == 1


def test_130_row_that_has_a_150_twin_is_ignored():
    """A raw row not flagged raw-only would double-count the 150% row and must produce nothing."""
    rows = [loc("X1", "G", kind="RAW_EBOM_130", start="2017-03-08", change="ECN193101", position="10")]
    assert build_bom_changes("ECN193101", rows)["lines"] == []


def test_raw_only_row_keeps_an_outgoing_part_in_use():
    """A part ended in the MBOM that still has a valid 130%-only row elsewhere is STILL_USED."""
    rows = [loc("OLD", "G_Q900", end="2026-01-01", change_to="MPXA1234560A"),
            raw_only("OLD", "OTHER", start="2020-01-01", position="10")]
    result = build_bom_changes("ECN123456", rows, coverage_complete=True)
    flags = {m["number"]: m for m in result["materials"]}
    assert flags["OLD"]["status"] == "STILL_USED" and flags["OLD"]["still_used_in"] == ["OTHER"]
