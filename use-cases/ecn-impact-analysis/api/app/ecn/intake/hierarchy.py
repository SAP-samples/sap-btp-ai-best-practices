"""Hierarchy reconstruction for enriched and raw EBOM workbook rows.

Provides three public functions used by the workbook parser after all BomRow
objects for a given role have been built:

- ``parse_dotted_depth`` – decode one raw leading-dot level string into a depth
  integer and diagnostic qualifiers.
- ``build_enriched_hierarchy_path`` – validate one enriched row's Level/Component/
  named-slot consistency and return a ``HierarchyPathResult``.
- ``reconstruct_enriched_hierarchy`` – apply enriched path validation to a full
  sequence of ENRICHED_EBOM rows and wire parent row IDs via a depth stack.
- ``reconstruct_raw_hierarchy`` – assign parent/path links to RAW_EBOM_130 rows
  using dotted-depth ordering, then resolve document owners through the
  MATERIAL_LIKE_OBSERVED ancestor chain.

Design constraints
------------------
- Documentation rows (CONFIRMED_DOCUMENT) are never filtered or removed.
- ``parent_material_number`` is always the direct structural parent's component,
  even when that parent is a document.
- Document owner resolution traverses parent row IDs; it does not simply pick the
  nearest non-D row.
- Source ``parent_material_number`` is preserved on contradiction even when the
  structural stack disagrees.
- No date is substituted; no active/current conclusion is drawn.

Usage::

    from app.ecn.intake.hierarchy import (
        parse_dotted_depth,
        build_enriched_hierarchy_path,
        reconstruct_enriched_hierarchy,
        reconstruct_raw_hierarchy,
    )
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from app.ecn.models import (
    BomKind,
    BomRow,
    BomRowClassification,
    HierarchyPathResult,
    JsonValue,
)
from app.ecn.intake.normalization import ordered_unique

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

DOTTED_LEVEL = re.compile(r"^(?P<dots>\.+)(?P<label>[1-9][0-9]*)$")
"""Match a valid raw dotted level: one or more leading dots then a positive integer."""

# ---------------------------------------------------------------------------
# Named-slot field names for enriched path validation (in slot order).
# ---------------------------------------------------------------------------

ENRICHED_PATH_FIELDS: tuple[str, ...] = (
    "main_value",
    "sub_value",
    "vc_value",
    "level_4_value",
    "level_5_value",
    "level_6_value",
)
"""BomRow field names for the six named hierarchy slots, in ascending slot order."""


# ---------------------------------------------------------------------------
# Public: parse_dotted_depth
# ---------------------------------------------------------------------------


def parse_dotted_depth(value: str | None) -> tuple[int | None, tuple[str, ...]]:
    """Return relative depth from leading dots and qualify a label mismatch.

    Accepted syntax: one or more leading ASCII dots followed by a positive
    integer (e.g. ``.1``, ``..2``, ``...3``).  Depth equals the dot count.
    A numeric label that differs from the dot count adds
    ``DOTTED_LEVEL_LABEL_MISMATCH`` but does NOT change the returned depth.

    Args:
        value: Raw cell value from the Level no. column; may be ``None`` or
            a string with optional surrounding whitespace.

    Returns:
        A ``(depth, qualifiers)`` tuple where ``depth`` is the dot count (or
        ``None`` on any parse failure) and ``qualifiers`` is a tuple of
        applicable qualifier codes.
    """
    if value is None:
        return None, ("DOTTED_LEVEL_INVALID",)
    stripped = value.strip()
    match = DOTTED_LEVEL.fullmatch(stripped)
    if match is None:
        return None, ("DOTTED_LEVEL_INVALID",)
    depth = len(match.group("dots"))
    label = int(match.group("label"))
    qualifiers: tuple[str, ...] = () if label == depth else ("DOTTED_LEVEL_LABEL_MISMATCH",)
    return depth, qualifiers


# ---------------------------------------------------------------------------
# Public: build_enriched_hierarchy_path
# ---------------------------------------------------------------------------


def build_enriched_hierarchy_path(row: BomRow) -> HierarchyPathResult:
    """Validate one enriched row's Level/Component/named-slot consistency.

    Validates that:
    1. Level is an integer in the range 1–6.
    2. The slot at position ``Level-1`` in ENRICHED_PATH_FIELDS equals the row's
       ``component_number``.
    3. All earlier slots are nonblank.
    4. All later slots are ``None``.
    5. For Level > 1, the row's ``parent_material_number`` equals the slot at
       position ``Level-2`` (the immediately preceding named slot).

    On any failure, ``ENRICHED_PATH_CONTRADICTION`` is added to qualifiers and
    ``quality_details`` records the expected parent and source row number.
    Source ``parent_material_number`` is always preserved, never fabricated.

    Args:
        row: A frozen ``BomRow`` from the ENRICHED_EBOM role.

    Returns:
        A ``HierarchyPathResult`` with the slash-joined named path, the source
        parent material number, optional qualifiers, and diagnostic details.
        ``parent_bom_row_id`` is always ``None`` here — the caller resolves it.
    """
    # Parse level integer from dotted_level (which stores "1", "2", etc.).
    try:
        level = int(row.dotted_level or "")
    except ValueError:
        level = 0

    # Collect the six named slot values in slot order.
    parts: list[str | None] = [getattr(row, field) for field in ENRICHED_PATH_FIELDS]

    # Determine the expected parent for the source-parent cross-check.
    # For level=1, parent is external root context; for level>=2, it's slot level-2.
    expected_parent: str | None
    if 2 <= level <= 6:
        expected_parent = parts[level - 2]
    else:
        expected_parent = row.parent_material_number

    valid = (
        1 <= level <= 6
        # Current slot matches the component number.
        and parts[level - 1] == row.component_number
        # All earlier slots are nonblank.
        and all(parts[i] for i in range(level - 1))
        # All later slots are None or empty.
        and all(not parts[i] for i in range(level, 6))
        # Parent agrees with preceding slot (or is external for level 1).
        and (level == 1 or row.parent_material_number == expected_parent)
    )

    qualifiers: tuple[str, ...] = () if valid else ("ENRICHED_PATH_CONTRADICTION",)
    details: dict[str, JsonValue] = {}
    if not valid:
        details = {
            "expected_level": level,
            "expected_parent_material_number": expected_parent,
            "source_row_number": row.source_row_number,
        }

    # Build the hierarchy path from the nonblank named slots (in slot order).
    path = "/".join(part for part in parts if part)
    return HierarchyPathResult(
        hierarchy_path=path or None,
        parent_bom_row_id=None,
        parent_material_number=row.parent_material_number,
        quality_qualifiers=qualifiers,
        quality_details=details,
    )


# ---------------------------------------------------------------------------
# Public: reconstruct_enriched_hierarchy
# ---------------------------------------------------------------------------


def reconstruct_enriched_hierarchy(
    rows: Sequence[BomRow],
) -> tuple[BomRow, ...]:
    """Assign enriched parent/path links in original row order via a depth stack.

    For each row:
    1. ``build_enriched_hierarchy_path`` validates the named slot consistency.
    2. ``parent_bom_row_id`` is resolved from the depth stack only when the path
       is valid (no contradiction), depth is sequential, and the candidate
       parent's component_number matches the row's source ``Parent`` value.
    3. Source ``parent_material_number`` is always preserved verbatim.
    4. Quality data from the path result is merged onto the row.

    Input must consist exclusively of ``ENRICHED_EBOM`` rows.

    Args:
        rows: Sequence of frozen ``BomRow`` objects in original sheet order.

    Returns:
        A tuple of updated ``BomRow`` objects with hierarchy links assigned.

    Raises:
        ValueError: If any row's ``bom_kind`` is not ``ENRICHED_EBOM``.
    """
    if any(row.bom_kind is not BomKind.ENRICHED_EBOM for row in rows):
        raise ValueError("reconstruct_enriched_hierarchy accepts only ENRICHED_EBOM rows")

    # depth → most-recently-seen BomRow at that level.
    stack: dict[int, BomRow] = {}
    resolved: list[BomRow] = []

    for row in rows:
        path_result = build_enriched_hierarchy_path(row)

        # Parse level integer for stack management.
        try:
            level = int(row.dotted_level or "")
        except ValueError:
            level = 0

        # Resolve parent_bom_row_id only when the path is unambiguous.
        parent_bom_row_id: str | None = None
        has_contradiction = bool(path_result.quality_qualifiers)
        if not has_contradiction and level >= 2:
            parent_candidate = stack.get(level - 1)
            if (
                parent_candidate is not None
                and parent_candidate.component_number == row.parent_material_number
            ):
                parent_bom_row_id = parent_candidate.bom_row_id

        updated = replace(
            row,
            hierarchy_path=path_result.hierarchy_path,
            parent_bom_row_id=parent_bom_row_id,
            # Preserve source parent_material_number regardless of contradiction.
            parent_material_number=row.parent_material_number,
            quality_qualifiers=ordered_unique(
                (*row.quality_qualifiers, *path_result.quality_qualifiers)
            ),
            quality_details={**row.quality_details, **path_result.quality_details},
        )
        resolved.append(updated)

        # Update the depth stack and prune deeper entries.
        if level >= 1:
            stack[level] = updated
            stack = {k: v for k, v in stack.items() if k <= level}

    return tuple(resolved)


# ---------------------------------------------------------------------------
# Public: reconstruct_raw_hierarchy
# ---------------------------------------------------------------------------


def reconstruct_raw_hierarchy(
    rows: Sequence[BomRow],
) -> tuple[BomRow, ...]:
    """Assign raw dotted parents/paths in original order, then resolve document owners.

    Processing steps:
    1. For each row, decode the dotted level via ``parse_dotted_depth``.
    2. Determine the parent from the depth stack (depth-1 entry); a missing
       parent for depth > 1 (or an invalid level) marks the row as
       ``RAW_HIERARCHY_ORPHAN``.
    3. Build the hierarchy path from parent path + component segment.
    4. After all parent links are assigned, call ``_resolve_document_owners``
       to fill ``document_owner_bom_row_id`` / ``document_owner_material_number``
       for every documentation row by traversing the parent chain to the nearest
       ``MATERIAL_LIKE_OBSERVED`` ancestor.

    Documentation rows are never removed or filtered.  ``parent_material_number``
    is the direct parent's component even when that parent is documentation.

    Input must consist exclusively of ``RAW_EBOM_130`` rows.

    Args:
        rows: Sequence of frozen ``BomRow`` objects in original sheet order.

    Returns:
        A tuple of updated ``BomRow`` objects with hierarchy links and document
        ownership assigned.

    Raises:
        ValueError: If any row's ``bom_kind`` is not ``RAW_EBOM_130``.
    """
    if any(row.bom_kind is not BomKind.RAW_EBOM_130 for row in rows):
        raise ValueError("reconstruct_raw_hierarchy accepts only RAW_EBOM_130 rows")

    # depth → most-recently-seen BomRow at that level.
    stack: dict[int, BomRow] = {}
    linked: list[BomRow] = []

    for row in rows:
        depth, level_qualifiers = parse_dotted_depth(row.dotted_level)

        # Identify the structural parent (depth-1 from the stack).
        parent: BomRow | None
        if depth is None or depth == 1:
            parent = None
        else:
            parent = stack.get(depth - 1)

        # Orphan when depth is invalid or when a required parent is absent.
        orphan = depth is None or (depth > 1 and parent is None)

        # Build path segment: prefer component_number, fall back to row marker.
        segment = row.component_number or f"[row:{row.source_row_number}]"
        path = segment if parent is None else f"{parent.hierarchy_path}/{segment}"

        updated = replace(
            row,
            parent_bom_row_id=None if parent is None else parent.bom_row_id,
            parent_material_number=None if parent is None else parent.component_number,
            hierarchy_path=path,
            quality_qualifiers=ordered_unique((
                *row.quality_qualifiers,
                *level_qualifiers,
                *(("RAW_HIERARCHY_ORPHAN",) if orphan else ()),
            )),
        )
        linked.append(updated)

        if depth is not None:
            stack[depth] = updated
            # Prune entries deeper than the current level.
            stack = {k: v for k, v in stack.items() if k <= depth}

    return _resolve_document_owners(tuple(linked))


# ---------------------------------------------------------------------------
# Internal: document owner resolution
# ---------------------------------------------------------------------------


def _resolve_document_owners(rows: tuple[BomRow, ...]) -> tuple[BomRow, ...]:
    """Fill document_owner fields for documentation rows by traversing parent links.

    For each row classified as ``CONFIRMED_DOCUMENT``:
    - Traverse ``parent_bom_row_id`` links through the ID map.
    - Stop at the first ancestor classified as ``MATERIAL_LIKE_OBSERVED``.
    - Set ``document_owner_bom_row_id`` and ``document_owner_material_number``
      in ``quality_details``.
    - If no material-like ancestor exists, add ``DOCUMENT_OWNER_UNRESOLVED`` to
      qualifiers and ``quality_details``.

    Non-documentation rows are returned unchanged.

    Args:
        rows: Tuple of ``BomRow`` objects with parent links already assigned.

    Returns:
        A new tuple where each documentation row has owner details merged into
        its ``quality_details``.
    """
    # Build a lookup map from bom_row_id to the most-current version of the row.
    id_map: dict[str, BomRow] = {row.bom_row_id: row for row in rows}

    result: list[BomRow] = []
    for row in rows:
        if not row.is_documentation:
            result.append(row)
            continue

        # Traverse parent chain to find nearest MATERIAL_LIKE_OBSERVED ancestor.
        owner: BomRow | None = None
        current_id = row.parent_bom_row_id
        while current_id is not None:
            parent = id_map.get(current_id)
            if parent is None:
                break
            if parent.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED:
                owner = parent
                break
            current_id = parent.parent_bom_row_id

        if owner is not None:
            new_details: dict[str, JsonValue] = {
                **row.quality_details,
                "document_owner_bom_row_id": owner.bom_row_id,
                "document_owner_material_number": owner.component_number,
            }
            result.append(replace(row, quality_details=new_details))
        else:
            new_details = {
                **row.quality_details,
                "document_owner_resolution": "DOCUMENT_OWNER_UNRESOLVED",
            }
            result.append(replace(
                row,
                quality_details=new_details,
                quality_qualifiers=ordered_unique(
                    (*row.quality_qualifiers, "DOCUMENT_OWNER_UNRESOLVED")
                ),
            ))

    return tuple(result)
