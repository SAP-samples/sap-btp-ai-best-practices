"""Raw 130% EBOM rows without a 150% twin are identified exactly.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_raw_ebom.py
"""

from app.ecn.evidence.raw_ebom import promote_raw_only, raw_only_row_ids
from app.ecn.models import BomKind, BomRowClassification
from tests.fixtures.builders import make_bom_row, synthetic_uuid

GROUP = "Q900VV0170302"


def raw(label, *, group=GROUP, parent=GROUP, component="PART1", item="140", **extra):
    """Build a raw 130% material row under a VC group (path root = the group)."""
    fields = dict(
        bom_row_id=synthetic_uuid(label), bom_kind=BomKind.RAW_EBOM_130, dotted_level="..2",
        hierarchy_path=f"{group}/{component}", parent_material_number=parent, component_number=component,
        item_value=item, position_value=item, row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED)
    # Callers override defaults such as row_classification or dotted_level through ``extra``.
    fields.update(extra)
    return make_bom_row(**fields)


def enriched(label, *, group=GROUP, parent=GROUP, component="PART1", position="0140", **extra):
    """Build a 150% row at level 4 (direct child of the VC group)."""
    return make_bom_row(
        bom_row_id=synthetic_uuid(label), bom_kind=BomKind.ENRICHED_EBOM, dotted_level="4",
        vc_value=group, parent_material_number=parent, component_number=component,
        position_value=position, **extra)


def test_raw_row_with_150_twin_is_not_raw_only():
    """Zero padding must not hide the twin: 0140 in the 150% sheet equals 140 in the 130% sheet."""
    assert raw_only_row_ids([raw("r"), enriched("e")]) == frozenset()


def test_raw_row_without_twin_is_raw_only():
    """No 150% row at all: the raw row is the only engineering source."""
    assert raw_only_row_ids([raw("r")]) == {synthetic_uuid("r")}


def test_twin_needs_same_group_parent_component_and_position():
    """The one 150% row is the twin of "match" only; each row differing in a single coordinate stays raw-only."""
    rows = [enriched("e"),
            raw("match"),                           # same group, parent, component and position: twin found
            raw("group", group="Q900VV0000001"),    # only the VC group differs
            raw("parent", parent="OTHERPARENT"),    # only the parent differs
            raw("component", component="PART2"),    # only the component differs
            raw("position", item="150")]            # only the position differs
    assert raw_only_row_ids(rows) == {synthetic_uuid(label) for label in ("group", "parent", "component", "position")}


def test_twin_never_crosses_artifacts():
    """A 150% row of another uploaded BOM does not count as the twin."""
    other = enriched("e", artifact_id=synthetic_uuid("other-artifact"))
    assert raw_only_row_ids([raw("r"), other]) == {synthetic_uuid("r")}


def test_documents_group_headers_and_unresolved_rows_are_never_promoted():
    """Only material-like rows with a parent, a component and an item can be promoted."""
    document = raw("d", row_classification=BomRowClassification.CONFIRMED_DOCUMENT)
    header = raw("h", parent=None, dotted_level=".1")
    unresolved = raw("u", row_classification=BomRowClassification.UNRESOLVED)
    no_item = raw("n", item=None)
    assert raw_only_row_ids([document, header, unresolved, no_item]) == frozenset()


def test_promote_raw_only_presents_150_shape_and_keeps_provenance():
    """Raw-only locations become 150% locations with a padded position; others are untouched."""
    raw_location = {"kind": "RAW_EBOM_130", "raw_only": True, "item_or_position": "22", "number": "P"}
    twin_location = {"kind": "RAW_EBOM_130", "raw_only": False, "item_or_position": "22", "number": "P"}
    mbom_location = {"kind": "MBOM_130", "item_or_position": None, "number": "P"}
    promoted, twin, mbom = promote_raw_only([raw_location, twin_location, mbom_location])
    assert promoted == {"kind": "ENRICHED_EBOM", "source_kind": "RAW_EBOM_130", "raw_only": True,
                        "item_or_position": "0022", "number": "P"}
    assert twin is twin_location and mbom is mbom_location
    assert raw_location["kind"] == "RAW_EBOM_130"  # the input is never mutated
