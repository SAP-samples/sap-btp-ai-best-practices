"""Unit tests for raw and enriched hierarchy reconstruction.

All fixtures are synthetic BomRow objects built in memory — no workbook files
are read.  Three TDD waves mirror the brief's numbered steps:

  Wave 1 (dotted depth parsing)          – Steps 10-13
  Wave 2 (raw hierarchy / doc owner)     – Steps 10-13
  Wave 3 (enriched hierarchy / named paths) – Steps 14-17
"""

from __future__ import annotations

import pytest

from app.ecn.intake.hierarchy import (
    build_enriched_hierarchy_path,
    parse_dotted_depth,
    reconstruct_enriched_hierarchy,
    reconstruct_raw_hierarchy,
)
from app.ecn.models import BomKind, BomRowClassification

from tests.fixtures.builders import enriched_hierarchy_rows, raw_hierarchy_rows


# ---------------------------------------------------------------------------
# Wave 1: parse_dotted_depth
# ---------------------------------------------------------------------------


def test_parse_dotted_depth_single_dot() -> None:
    """A single dot followed by 1 is depth 1 with no qualifiers."""
    depth, qualifiers = parse_dotted_depth(".1")
    assert depth == 1
    assert qualifiers == ()


def test_parse_dotted_depth_two_dots() -> None:
    """Two dots followed by 2 is depth 2 with no qualifiers."""
    depth, qualifiers = parse_dotted_depth("..2")
    assert depth == 2
    assert qualifiers == ()


def test_parse_dotted_depth_three_dots() -> None:
    """Three dots followed by 3 is depth 3 with no qualifiers."""
    depth, qualifiers = parse_dotted_depth("...3")
    assert depth == 3
    assert qualifiers == ()


def test_parse_dotted_depth_label_mismatch_still_returns_depth() -> None:
    """Dots count determines depth; a mismatched label adds DOTTED_LEVEL_LABEL_MISMATCH."""
    depth, qualifiers = parse_dotted_depth("..1")
    assert depth == 2
    assert "DOTTED_LEVEL_LABEL_MISMATCH" in qualifiers


def test_parse_dotted_depth_none_returns_none_with_invalid_qualifier() -> None:
    """None input returns None depth with DOTTED_LEVEL_INVALID qualifier."""
    depth, qualifiers = parse_dotted_depth(None)
    assert depth is None
    assert "DOTTED_LEVEL_INVALID" in qualifiers


def test_parse_dotted_depth_plain_number_is_invalid() -> None:
    """A plain integer string without leading dot(s) is not a valid dotted level."""
    depth, qualifiers = parse_dotted_depth("1")
    assert depth is None
    assert "DOTTED_LEVEL_INVALID" in qualifiers


def test_parse_dotted_depth_zero_label_is_invalid() -> None:
    """A zero label is rejected because the pattern requires a positive integer."""
    depth, qualifiers = parse_dotted_depth(".0")
    assert depth is None
    assert "DOTTED_LEVEL_INVALID" in qualifiers


def test_parse_dotted_depth_whitespace_trimmed() -> None:
    """Leading/trailing whitespace is trimmed before matching."""
    depth, qualifiers = parse_dotted_depth("  .1  ")
    assert depth == 1
    assert qualifiers == ()


# ---------------------------------------------------------------------------
# Wave 2: reconstruct_raw_hierarchy
# ---------------------------------------------------------------------------


def test_raw_hierarchy_keeps_document_and_finds_material_owner() -> None:
    """Document row finds its nearest MATERIAL_LIKE_OBSERVED ancestor as owner."""
    rows = raw_hierarchy_rows([
        (".1",   "0",    "ROOT", "ND"),
        ("..2",  "150",  "PART", "ND"),
        ("...3", "D001", "PART ZVD 000 01", ""),
    ])
    result = reconstruct_raw_hierarchy(rows)
    root, material, document = result
    assert document.parent_bom_row_id == material.bom_row_id
    assert document.parent_material_number == "PART"
    assert document.is_documentation is True
    assert document.quality_details["document_owner_bom_row_id"] == material.bom_row_id
    assert document.quality_details["document_owner_material_number"] == "PART"
    assert len(result) == 3


def test_raw_hierarchy_preserves_all_rows_including_documents() -> None:
    """reconstruct_raw_hierarchy never removes any row, including documentation."""
    rows = raw_hierarchy_rows([
        (".1", "0",    "ROOT", "ND"),
        ("..2", "D002", "ROOT ZVD 000 01", ""),
    ])
    result = reconstruct_raw_hierarchy(rows)
    assert len(result) == 2


def test_impossible_depth_jump_is_not_attached_to_false_parent() -> None:
    """A depth jump greater than one produces RAW_HIERARCHY_ORPHAN and null parent."""
    root, orphan = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        (".1",   "0",  "ROOT",  "ND"),
        ("...3", "10", "CHILD", "ND"),
    ]))
    assert orphan.parent_bom_row_id is None
    assert orphan.parent_material_number is None
    assert "RAW_HIERARCHY_ORPHAN" in orphan.quality_qualifiers


def test_first_row_depth_greater_than_one_is_orphan() -> None:
    """A first row at depth > 1 is an orphan with no false parent attached."""
    (orphan,) = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        ("...3", "30", "PART", "ND"),
    ]))
    assert orphan.parent_bom_row_id is None
    assert "RAW_HIERARCHY_ORPHAN" in orphan.quality_qualifiers


def test_sibling_rows_reset_stack_at_same_depth() -> None:
    """Two sibling rows at the same depth each get the same parent."""
    root, first, second = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        (".1",  "0",  "ROOT",  "ND"),
        ("..2", "10", "SIBA",  "ND"),
        ("..2", "20", "SIBB",  "ND"),
    ]))
    assert first.parent_bom_row_id == root.bom_row_id
    assert second.parent_bom_row_id == root.bom_row_id


def test_hierarchy_path_uses_component_number() -> None:
    """Path segment is the component_number when available."""
    root, child = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        (".1",  "0", "ROOT",  "ND"),
        ("..2", "1", "CHILD", "ND"),
    ]))
    assert root.hierarchy_path == "ROOT"
    assert child.hierarchy_path == "ROOT/CHILD"


def test_hierarchy_path_uses_row_number_when_no_component() -> None:
    """When component_number is empty, the path segment is [row:<source_row_number>]."""
    (no_comp,) = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        (".1", "0", "", "ND"),
    ]))
    assert no_comp.hierarchy_path is not None
    assert "row:" in no_comp.hierarchy_path


def test_structural_parent_material_number_is_direct_parent_even_if_document() -> None:
    """parent_material_number is the direct parent's component, even if that parent is a document."""
    rows = raw_hierarchy_rows([
        (".1",   "0",    "ROOT",              "ND"),
        ("..2",  "D002", "ROOT ZVD 000 01",   ""),   # document parent
        ("...3", "30",   "CHILD",             "ND"),
    ])
    result = reconstruct_raw_hierarchy(rows)
    _, doc_parent, child = result
    # Child's structural parent is the document row.
    assert child.parent_bom_row_id == doc_parent.bom_row_id
    assert child.parent_material_number == "ROOT ZVD 000 01"


def test_document_with_document_parent_finds_material_grandparent() -> None:
    """Document owners are resolved by traversing the full parent chain."""
    rows = raw_hierarchy_rows([
        (".1",    "0",    "ROOT",             "ND"),
        ("..2",   "D002", "ROOT ZVD 000 01",  ""),
        ("...3",  "D003", "ROOT ZVD 001 01",  ""),
    ])
    result = reconstruct_raw_hierarchy(rows)
    root, doc_parent, deep_doc = result
    # deep_doc's parent is doc_parent (which is also documentation).
    # The owner should traverse up to ROOT (the nearest MATERIAL_LIKE).
    assert deep_doc.quality_details.get("document_owner_bom_row_id") == root.bom_row_id
    assert deep_doc.quality_details.get("document_owner_material_number") == "ROOT"


def test_document_with_no_material_ancestor_is_unresolved() -> None:
    """A document with no MATERIAL_LIKE_OBSERVED ancestor gets DOCUMENT_OWNER_UNRESOLVED."""
    (doc,) = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        (".1", "D001", "DOC ZVD 000 01", ""),
    ]))
    assert doc.is_documentation is True
    assert "DOCUMENT_OWNER_UNRESOLVED" in doc.quality_qualifiers


def test_invalid_dotted_level_row_is_orphan() -> None:
    """A row with an invalid dotted level string (no dots) becomes an orphan."""
    (bad,) = reconstruct_raw_hierarchy(raw_hierarchy_rows([
        ("1", "0", "PART", "ND"),  # no leading dot
    ]))
    assert bad.parent_bom_row_id is None
    assert "RAW_HIERARCHY_ORPHAN" in bad.quality_qualifiers


def test_raw_hierarchy_rejects_non_raw_rows() -> None:
    """reconstruct_raw_hierarchy raises ValueError for non-RAW_EBOM_130 rows."""
    from tests.fixtures.builders import make_bom_row

    enriched_row = make_bom_row(bom_kind=BomKind.ENRICHED_EBOM)
    with pytest.raises(ValueError, match="RAW_EBOM_130"):
        reconstruct_raw_hierarchy((enriched_row,))


def test_label_mismatch_does_not_change_depth() -> None:
    """Depth is determined by dot count even when the numeric label differs."""
    rows = raw_hierarchy_rows([
        (".1",  "0", "ROOT",  "ND"),
        ("..1", "1", "CHILD", "ND"),  # two dots, label=1 → depth=2, label mismatch
    ])
    result = reconstruct_raw_hierarchy(rows)
    root, child = result
    # Depth is 2 (two dots), so CHILD attaches to ROOT at depth 1.
    assert child.parent_bom_row_id == root.bom_row_id
    assert "DOTTED_LEVEL_LABEL_MISMATCH" in child.quality_qualifiers


def test_repeated_components_under_different_parents_are_distinct_occurrences() -> None:
    """The same component number under two different parents creates two distinct rows."""
    rows = raw_hierarchy_rows([
        (".1",  "0", "PARENT_A", "ND"),
        ("..2", "1", "SHARED",   "ND"),
        (".1",  "2", "PARENT_B", "ND"),
        ("..2", "3", "SHARED",   "ND"),
    ])
    result = reconstruct_raw_hierarchy(rows)
    pa, shared_a, pb, shared_b = result
    # Both SHARED rows have the same component_number but different parent row IDs.
    assert shared_a.component_number == shared_b.component_number == "SHARED"
    assert shared_a.parent_bom_row_id == pa.bom_row_id
    assert shared_b.parent_bom_row_id == pb.bom_row_id
    assert shared_a.bom_row_id != shared_b.bom_row_id


# ---------------------------------------------------------------------------
# Wave 3: build_enriched_hierarchy_path and reconstruct_enriched_hierarchy
# ---------------------------------------------------------------------------


def test_enriched_named_path_and_parent_row_are_consistent() -> None:
    """Level-1 root has no parent row; level-2 sub is linked to the root by ID."""
    main, sub = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT", "main": "MAIN"},
        {"level": "2", "component": "SUB",  "parent": "MAIN", "main": "MAIN", "sub": "SUB"},
    ]))
    assert main.hierarchy_path == "MAIN"
    assert main.parent_bom_row_id is None
    assert main.parent_material_number == "ROOT"
    assert sub.hierarchy_path == "MAIN/SUB"
    assert sub.parent_bom_row_id == main.bom_row_id
    assert sub.parent_material_number == "MAIN"


def test_enriched_source_parent_contradiction_is_visible_not_repaired() -> None:
    """Parent mismatch adds ENRICHED_PATH_CONTRADICTION without changing parent_material_number."""
    main, sub = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT",  "main": "MAIN"},
        {"level": "2", "component": "SUB",  "parent": "OTHER", "main": "MAIN", "sub": "SUB"},
    ]))
    assert sub.parent_material_number == "OTHER"
    assert sub.parent_bom_row_id is None
    assert "ENRICHED_PATH_CONTRADICTION" in sub.quality_qualifiers
    assert sub.quality_details["expected_parent_material_number"] == "MAIN"


def test_enriched_level_out_of_range_is_contradiction() -> None:
    """A level value outside 1–6 triggers ENRICHED_PATH_CONTRADICTION."""
    (bad,) = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "7", "component": "X", "parent": "Y", "main": "X"},
    ]))
    assert "ENRICHED_PATH_CONTRADICTION" in bad.quality_qualifiers


def test_enriched_slot_mismatch_component_not_in_expected_slot() -> None:
    """Component not equal to its level slot triggers ENRICHED_PATH_CONTRADICTION."""
    (bad,) = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT", "main": "WRONG"},
    ]))
    assert "ENRICHED_PATH_CONTRADICTION" in bad.quality_qualifiers


def test_enriched_unexpected_later_slot_is_contradiction() -> None:
    """A nonblank slot beyond the current level is a contradiction."""
    (bad,) = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT", "main": "MAIN", "sub": "EXTRA"},
    ]))
    assert "ENRICHED_PATH_CONTRADICTION" in bad.quality_qualifiers


def test_enriched_missing_earlier_slot_is_contradiction() -> None:
    """A blank slot before the current level is a contradiction."""
    (bad,) = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        # Level 2 but main is blank — the earlier slot is missing.
        {"level": "2", "component": "SUB", "parent": "ROOT", "main": None, "sub": "SUB"},
    ]))
    assert "ENRICHED_PATH_CONTRADICTION" in bad.quality_qualifiers


def test_enriched_level_one_has_no_parent_bom_row_id() -> None:
    """Level-1 rows represent external root context; parent_bom_row_id is always None."""
    (root,) = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT", "main": "MAIN"},
    ]))
    assert root.parent_bom_row_id is None
    assert root.parent_material_number == "ROOT"


def test_enriched_path_through_level_three() -> None:
    """A three-level enriched path is correctly assembled."""
    main, sub, vc = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "M1",  "parent": "ROOT", "main": "M1"},
        {"level": "2", "component": "S1",  "parent": "M1",  "main": "M1", "sub": "S1"},
        {"level": "3", "component": "V1",  "parent": "S1",  "main": "M1", "sub": "S1", "vc": "V1"},
    ]))
    assert vc.hierarchy_path == "M1/S1/V1"
    assert vc.parent_bom_row_id == sub.bom_row_id


def test_enriched_hierarchy_rejects_non_enriched_rows() -> None:
    """reconstruct_enriched_hierarchy raises ValueError for non-ENRICHED_EBOM rows."""
    from tests.fixtures.builders import make_bom_row

    raw_row = make_bom_row(bom_kind=BomKind.RAW_EBOM_130)
    with pytest.raises(ValueError, match="ENRICHED_EBOM"):
        reconstruct_enriched_hierarchy((raw_row,))


def test_build_enriched_hierarchy_path_valid_level_one() -> None:
    """build_enriched_hierarchy_path returns a path with no qualifiers for a valid level-1 row."""
    (row,) = enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT", "main": "MAIN"},
    ])
    result = build_enriched_hierarchy_path(row)
    assert result.hierarchy_path == "MAIN"
    assert result.quality_qualifiers == ()
    assert result.parent_material_number == "ROOT"


def test_build_enriched_hierarchy_path_invalid_level_zero() -> None:
    """An unparseable dotted_level (e.g. non-integer string) yields a contradiction."""
    from tests.fixtures.builders import make_bom_row
    from app.ecn.models import BomRowClassification

    bad_row = make_bom_row(
        bom_kind=BomKind.ENRICHED_EBOM,
        dotted_level="not-a-number",
        component_number="X",
        main_value="X",
    )
    result = build_enriched_hierarchy_path(bad_row)
    assert "ENRICHED_PATH_CONTRADICTION" in result.quality_qualifiers


def test_enriched_contradiction_preserves_source_parent_material_number() -> None:
    """Source parent_material_number is preserved on contradiction, never overwritten."""
    main, sub = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN", "parent": "ROOT",    "main": "MAIN"},
        {"level": "2", "component": "SUB",  "parent": "DIFFERENT", "main": "MAIN", "sub": "SUB"},
    ]))
    # Source Parent column value must be retained verbatim.
    assert sub.parent_material_number == "DIFFERENT"


def test_enriched_repeated_siblings_under_same_parent() -> None:
    """Two sibling rows at the same level both link to the same parent row."""
    main, sib_a, sib_b = reconstruct_enriched_hierarchy(enriched_hierarchy_rows([
        {"level": "1", "component": "MAIN",  "parent": "ROOT", "main": "MAIN"},
        {"level": "2", "component": "SUB_A", "parent": "MAIN", "main": "MAIN", "sub": "SUB_A"},
        {"level": "2", "component": "SUB_B", "parent": "MAIN", "main": "MAIN", "sub": "SUB_B"},
    ]))
    assert sib_a.parent_bom_row_id == main.bom_row_id
    assert sib_b.parent_bom_row_id == main.bom_row_id
