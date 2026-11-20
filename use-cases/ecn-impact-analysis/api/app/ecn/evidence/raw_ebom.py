"""Raw 130% EBOM rows as engineering evidence when the 150% EBOM has no row.

The workbook carries the engineering BOM twice: the 150% sheet (``EBOM Q900V``,
parsed as ``ENRICHED_EBOM``) and the 130% sheet (``EBOM 130% KM_...``, parsed as
``RAW_EBOM_130``). On the rows they share, both hold the same change number and
start date, so either can say which ECN started a part in an assembly. The 150%
sheet is preferred; a 130% material row is used only when the 150% sheet has no
row at the same position (a "raw-only" row).

Two functions:

* ``raw_only_row_ids`` decides, from parsed ``BomRow`` objects, which raw rows
  are raw-only.
* ``promote_raw_only`` re-presents raw-only snapshot locations as 150%
  locations so the change builder needs no second code path.
"""

from collections.abc import Iterable

from app.ecn.models import BomKind, BomRow, BomRowClassification

RAW_KIND = BomKind.RAW_EBOM_130.value
ENRICHED_KIND = BomKind.ENRICHED_EBOM.value


def _normalized_position(value: object) -> str:
    """Return a BOM position with numeric zero padding removed.

    Args:
        value: Raw ``Post number`` (150%, e.g. ``"0140"``) or ``Item`` (130%,
            e.g. ``"140"``) cell value; may be ``None``.

    Returns:
        ``"140"`` for both examples; non-numeric text is only stripped; ``""``
        for ``None``.
    """
    text = "" if value is None else str(value).strip()
    return str(int(text)) if text.isdigit() else text


def _twin_key(row: BomRow) -> tuple:
    """Return the coordinates that identify one BOM position across both sheets.

    The VC group is the ``VC`` column of a 150% row and the root of the
    hierarchy path of a 130% row (the level-1 node of that sheet). Both sheets
    hang a part under the same parent component, so parent and component are
    compared directly.

    Args:
        row: A 150% (``ENRICHED_EBOM``) or 130% (``RAW_EBOM_130``) row.

    Returns:
        ``(artifact_id, vc_group, parent_number, component_number, position)``.
    """
    if row.bom_kind is BomKind.ENRICHED_EBOM:
        group, position = row.vc_value, row.position_value
    else:
        group, position = (row.hierarchy_path or "").split("/")[0], row.item_value
    return (row.artifact_id, group, row.parent_material_number, row.component_number, _normalized_position(position))


def raw_only_row_ids(rows: Iterable[BomRow]) -> frozenset[str]:
    """Find the raw 130% material rows that have no 150% twin.

    A row qualifies only when it is a real part row: classified
    ``MATERIAL_LIKE_OBSERVED`` (not a document), below level 1 (it has a parent;
    level-1 rows are VC group headers), and carrying a component and an item.
    It is dropped from the result when a 150% row of the same upload has the same
    twin key (see ``_twin_key``).

    Args:
        rows: Parsed BOM rows of the pinned datasets (any mix of kinds).

    Returns:
        ``bom_row_id`` of every raw-only row, as a frozenset.
    """
    rows = tuple(rows)
    # Every position the 150% sheet already covers.
    covered = {_twin_key(row) for row in rows if row.bom_kind is BomKind.ENRICHED_EBOM}
    return frozenset(
        row.bom_row_id
        for row in rows
        if row.bom_kind is BomKind.RAW_EBOM_130
        and row.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED
        and row.parent_material_number and row.component_number and row.item_value
        and _twin_key(row) not in covered
    )


def promote_raw_only(locations: Iterable[dict]) -> list[dict]:
    """Present raw-only snapshot locations as 150% locations for the change builder.

    A location is raw-only when the snapshot set ``raw_only`` on it (see
    ``raw_only_row_ids``). It is copied with ``kind`` changed to
    ``ENRICHED_EBOM`` so the builder pairs it, joins it to the MBOM and derives
    part flags with no extra code path. The true sheet kind stays in
    ``source_kind``. A numeric position is padded to four digits, the way the
    150% sheet writes it (``10`` becomes ``0010``), so positions read alike.

    Args:
        locations: Snapshot location dicts (any kind).

    Returns:
        A new list; raw-only locations are new dicts, every other location is
        the very same object. Inputs are never mutated.
    """
    promoted = []
    for location in locations:
        if not location.get("raw_only"):
            promoted.append(location)
            continue
        position = str(location.get("item_or_position") or "").strip()
        promoted.append({
            **location,
            "kind": ENRICHED_KIND,
            "source_kind": RAW_KIND,
            "item_or_position": position.zfill(4) if position.isdigit() else location.get("item_or_position"),
        })
    return promoted
