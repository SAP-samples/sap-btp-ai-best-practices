"""Workbook role contracts for BOM intake: sheet-role identification by normalized header signature.

Each ``WorkbookSheetContract`` identifies one logical BOM role by the exact set of
normalized header strings that must appear in a worksheet.  Normalization is
whitespace-only — external trim plus internal whitespace collapse — with no case
folding, punctuation stripping, or accent removal.

Three canonical roles are defined:

- ``ENRICHED_EBOM`` — the engineering BOM enriched with hierarchy-path columns
  (Main, Sub, VC, Level 4-6, Parent).
- ``RAW_EBOM_130`` — the raw 130-percent EBOM export from the source system,
  including the ``Typ`` column used by the four-way classification heuristic.
- ``MBOM_130`` — the manufacturing BOM export.

Design constraints
------------------
- Column letters are never referenced; detection is purely header-signature-based.
- Product-specific sheet names are never hard-coded; the real sheet name is
  preserved at parse time (Task 12).
- Duplicate headers (e.g. the two ``MS`` columns present in the raw sheet) are
  not listed in ``required_headers`` or ``field_headers``.  They are retained
  positionally as ordered ``ROW_JSON`` entries during parsing.
- ``field_headers`` values are the **source** header strings as they appear in the
  workbook (after normalization), not Python field names.

Example usage::

    from app.ecn.intake.workbook_contracts import WORKBOOK_SHEET_CONTRACTS, normalize_header

    normalized = normalize_header("  N°   of   modification  ")
    # → "N° of modification"

    for contract in WORKBOOK_SHEET_CONTRACTS:
        if all(h in sheet_headers for h in contract.required_headers):
            recognized_role = contract.bom_kind
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ecn.models import BomKind


# ---------------------------------------------------------------------------
# Header normalization (whitespace-only — no case folding)
# ---------------------------------------------------------------------------


def normalize_header(value: object) -> str:
    """Normalize only surrounding and repeated whitespace in a source header cell.

    Converts ``value`` to a string, trims external whitespace, and collapses
    every internal run of whitespace to a single ASCII space.  Does NOT
    case-fold, remove punctuation/accents, or deduplicate.  Two headers that
    differ only in case or punctuation will NOT match after normalization.

    Args:
        value: The raw cell value from a header row.  ``None`` maps to ``""``;
            any non-string type is converted via ``str()``.

    Returns:
        The whitespace-normalized header string, or ``""`` when the result is
        empty or entirely whitespace.
    """
    if value is None:
        return ""
    return " ".join(str(value).split())


# ---------------------------------------------------------------------------
# Contract dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkbookSheetContract:
    """Identify one logical workbook role by its required normalized header signature.

    A worksheet is associated with this role when every string in
    ``required_headers`` is found (verbatim, after ``normalize_header``) among
    the sheet's actual header cells.  The real sheet name is preserved by the
    caller and is never stored here.

    Args:
        bom_kind: The logical BOM role this contract represents.
        required_headers: Ordered tuple of exact normalized header strings that
            must all be present to identify the sheet role.  Order follows the
            typical source-column sequence but is not significant for matching.
        field_headers: Mapping from ``BomRow`` Python field name to the source
            header string that supplies its value.  Only named structural fields
            are listed; extra columns are retained positionally in
            ``BomRow.row_json``.
    """

    bom_kind: BomKind
    required_headers: tuple[str, ...]
    field_headers: dict[str, str]


# ---------------------------------------------------------------------------
# Canonical workbook role constants
# ---------------------------------------------------------------------------

WORKBOOK_SHEET_CONTRACTS: tuple[WorkbookSheetContract, ...] = (
    # ------------------------------------------------------------------
    # Role 1: Enriched engineering BOM — includes full hierarchy columns.
    #
    # BomRow assignments:
    #   material_number = component_number = "Component"
    #   item_value = None  (no Item column in this layout)
    #   row_classification = NOT_APPLICABLE
    #   parent_bom_row_id / hierarchy_path resolved in Task 12
    # ------------------------------------------------------------------
    WorkbookSheetContract(
        bom_kind=BomKind.ENRICHED_EBOM,
        required_headers=(
            "Level", "Post number", "Component", "Item designation",
            "Component quantity", "Quantity unit", "N° of modification",
            "Start of validity", "Up to modif. Number", "Expiration date",
            "Parent", "Main", "Sub", "VC", "Level 4", "Level 5",
            "Level 6", "Alternative",
        ),
        field_headers={
            "dotted_level": "Level",
            "position_value": "Post number",
            "component_number": "Component",
            "description_text": "Item designation",
            "quantity": "Component quantity",
            "unit": "Quantity unit",
            "change_number": "N° of modification",
            "valid_from": "Start of validity",
            "change_number_to": "Up to modif. Number",
            "valid_to": "Expiration date",
            "parent_material_number": "Parent",
            "main_value": "Main",
            "sub_value": "Sub",
            "vc_value": "VC",
            "level_4_value": "Level 4",
            "level_5_value": "Level 5",
            "level_6_value": "Level 6",
            "alternative": "Alternative",
        },
    ),
    # ------------------------------------------------------------------
    # Role 2: Raw 130-percent engineering BOM.
    #
    # BomRow assignments:
    #   item_value = position_value = "Item"
    #   material_number = component_number = "Component number"
    #   classification from Item/Typ/component_number (Task 11 heuristic)
    #   Main/Sub/VC/L4-L6 stay None
    #   parent_bom_row_id / hierarchy_path resolved in Task 12
    #
    # Note: the raw sheet contains two "MS" headers that are not listed here.
    #       They are preserved positionally as ordered ROW_JSON entries.
    # ------------------------------------------------------------------
    WorkbookSheetContract(
        bom_kind=BomKind.RAW_EBOM_130,
        required_headers=(
            "Level no.", "Item", "Component number", "Object description",
            "Quantity", "Un", "Valid from", "Valid to", "Change No.",
            "Chg No. To", "Typ", "AltItm",
        ),
        field_headers={
            "dotted_level": "Level no.",
            "item_value": "Item",
            "component_number": "Component number",
            "description_text": "Object description",
            "quantity": "Quantity",
            "unit": "Un",
            "valid_from": "Valid from",
            "valid_to": "Valid to",
            "change_number": "Change No.",
            "change_number_to": "Chg No. To",
            "type_value": "Typ",
            "alternative": "AltItm",
        },
    ),
    # ------------------------------------------------------------------
    # Role 3: Manufacturing BOM.
    #
    # BomRow assignments:
    #   item_value = parent_material_number = "Item" (parent context, not a position)
    #   material_number = component_number = "Component"
    #   hierarchy_path = Item/Component
    #   row_classification = NOT_APPLICABLE
    #   Sequence and Employment probability remain only in ROW_JSON
    # ------------------------------------------------------------------
    WorkbookSheetContract(
        bom_kind=BomKind.MBOM_130,
        required_headers=(
            "Item", "Component", "Component name", "Quantity",
            "N° of modification", "Start of validity", "at the number of modif.",
            "Expiration date", "alternative postal group", "Sequence",
            "Employment probability",
        ),
        field_headers={
            "parent_material_number": "Item",
            "component_number": "Component",
            "description_text": "Component name",
            "quantity": "Quantity",
            "change_number": "N° of modification",
            "valid_from": "Start of validity",
            "change_number_to": "at the number of modif.",
            "valid_to": "Expiration date",
            "alternative": "alternative postal group",
        },
    ),
)
