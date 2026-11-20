"""Rule-based plant derivation: brands, rule-state filter, VC groups per BOM kind.

Rule texts mirror patterns of the real conditional-rule workbook (typed here,
never read from ``data/``).

Run with::

    PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_plant_scope.py
"""

from dataclasses import replace

from app.ecn.models import BomKind
from app.ecn.rules.parser import parse_rule_definition
from app.ecn.rules.plant_scope import (
    EMPTY_INDEX,
    brand_suffixes,
    build_vc_index,
    merge_resolutions,
    resolve_row,
    resolve_vc_groups,
    row_vc_groups,
)

from tests.fixtures.builders import make_bom_row

MAPPINGS = [
    {"suffix": "IS", "plant_code": "PL10", "plant_name": "Plant A"},
    {"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"},
    {"suffix": "JD", "plant_code": "PK01", "plant_name": "Plant B"},
]


def _rule(text, number, state="App For Series"):
    """Parse one rule and stamp its business number and lifecycle state."""
    rule = parse_rule_definition(text, f"00000000-0000-4000-8000-{int(number[-6:]):012d}")
    return replace(rule, rule_number=number, lifecycle_state=state)


RULES = [
    _rule("(GEN_E_BRAND = E_BRAND_IS / GEN_E_BRAND = E_BRAND_JB) + GEN_E_FRONT_LINKAGE = E_F_LINKAGE_W "
          "--> Inclure (Q900VS1030 = Q900VV1030207)", "RL0000108513"),
    _rule("GEN_E_BRAND = E_BRAND_JD + GEN_E_FRONT_LINKAGE = E_F_LINKAGE_W --> Inclure (Q900VS1030 = Q900VV1030206)",
          "RL0000108512"),
    _rule("GEN_E_BRAND = E_BRAND_JD + GEN_E_COUNTRY = FI --> Inclure (Q900VS1530 = Q932VV1530701)", "RL0000108505",
          "To be Obsolete"),
    _rule("GEN_E_BRAND = E_BRAND_XX --> Inclure (Q900VS1530 = Q900VV1530999)", "RL0000100001"),
    _rule("GEN_E_GEARBOX = E_GEARBOX_1 --> Inclure (Q900VS1530 = Q900VV1530888)", "RL0000100002"),
]


def test_brand_suffixes_ignore_negated_brands() -> None:
    """OR-ed brands are collected; a negated brand never places a VC group in that plant."""
    assert brand_suffixes(RULES[0].condition) == {"IS", "JB"}
    negated = _rule("-(GEN_E_BRAND = E_BRAND_JD) + GEN_E_BRAND = E_BRAND_JB --> Inclure (S = V)", "RL0000100003")
    assert brand_suffixes(negated.condition) == {"JB"}


def test_index_skips_excluded_states_and_records_slots() -> None:
    """Excluded lifecycle states drop their rules; slots list every selectable VC group."""
    index = build_vc_index(RULES, ("To be Obsolete",))
    assert index.brands["Q900VV1030207"] == {"IS", "JB"}
    assert "Q932VV1530701" not in index.brands
    assert index.slots["Q900VS1030"] == {"Q900VV1030206", "Q900VV1030207"}
    assert index.rule_numbers["Q900VV1030206"] == ("RL0000108512",)
    assert "Q932VV1530701" in build_vc_index(RULES, ()).brands


def test_row_vc_groups_per_bom_kind() -> None:
    """150% EBOM uses the VC column or slot members; 130% uses its root; MBOM strips the model suffix."""
    index = build_vc_index(RULES)
    enriched = make_bom_row(bom_kind=BomKind.ENRICHED_EBOM, dotted_level="5", vc_value="Q900VV1030207")
    slot = make_bom_row(bom_kind=BomKind.ENRICHED_EBOM, dotted_level="2", sub_value="Q900VS1030")
    main = make_bom_row(bom_kind=BomKind.ENRICHED_EBOM, dotted_level="1")
    raw = make_bom_row(bom_kind=BomKind.RAW_EBOM_130, hierarchy_path="Q923VV2526401/QKX463881A")
    mbom = make_bom_row(bom_kind=BomKind.MBOM_130, item_value="Q900VV0120800_Q900")
    assert row_vc_groups(enriched, index) == ("Q900VV1030207",)
    assert row_vc_groups(slot, index) == ("Q900VV1030206", "Q900VV1030207")
    assert row_vc_groups(main, index) is None
    assert row_vc_groups(raw, index) == ("Q923VV2526401",)
    assert row_vc_groups(mbom, index) == ("Q900VV0120800",)


def test_resolution_maps_brands_and_explains_missing_plants() -> None:
    """Shared plants merge; each unresolved case carries its own reason."""
    index = build_vc_index(RULES, ("To be Obsolete",))
    slot = make_bom_row(bom_kind=BomKind.ENRICHED_EBOM, dotted_level="2", sub_value="Q900VS1030")
    both = resolve_row(slot, index, MAPPINGS)
    assert [p["plant_code"] for p in both["plants"]] == ["PK01", "PL10"] and both["reason"] is None
    assert resolve_vc_groups(("Q900VV1530999",), index, MAPPINGS) == {
        "vc_groups": ["Q900VV1530999"], "plants": [], "reason": "BRAND_NOT_MAPPED", "unmapped_brands": ["XX"]}
    assert resolve_vc_groups(("Q900VV1530888",), index, MAPPINGS)["reason"] == "NO_BRAND_IN_RULES"
    assert resolve_vc_groups(("Q932VV1530701",), index, MAPPINGS)["reason"] == "NO_RULE_FOR_VC_GROUP"
    assert resolve_vc_groups(None, index, MAPPINGS)["reason"] == "LEVEL_1_COMMON"
    assert resolve_vc_groups((), index, MAPPINGS)["reason"] == "NOT_IN_BOM"
    assert resolve_vc_groups(("Q900VV1030207",), index, [])["reason"] == "MAPPING_EMPTY"
    assert resolve_vc_groups(("Q900VV1030207",), EMPTY_INDEX, MAPPINGS, rules_pinned=False)["reason"] == "NO_RULES_PINNED"


def test_merge_unions_plants_and_keeps_first_reason() -> None:
    """A material under two VC groups gets both plants; with no plant the first reason wins."""
    index = build_vc_index(RULES, ("To be Obsolete",))
    merged = merge_resolutions([resolve_vc_groups(("Q900VV1030207",), index, MAPPINGS),
                                resolve_vc_groups(("Q900VV1030206",), index, MAPPINGS)])
    assert [p["plant_name"] for p in merged["plants"]] == ["Plant B", "Plant A"]
    assert merge_resolutions([])["reason"] == "NOT_IN_BOM"
