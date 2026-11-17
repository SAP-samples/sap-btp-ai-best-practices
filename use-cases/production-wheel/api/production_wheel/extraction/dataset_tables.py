"""Structural workbook profiles, optional table aliases and string line evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from production_wheel.extraction.common import (
    PLANT_TOKEN,
    TableData,
    as_text,
    table_name_matches,
)

# Alias tuples list Excel table names; ``{plant}`` stands for a plant code
# embedded in the source table name (see ``common.table_name_matches``).
SUPPORTING_TABLES = {
    "planning_calendar_inform": (
        "FINI_Planning_calendar_at_P_site",
        "SB_FINI_Planning_calendar_at_P_site",
    ),
    "planning_calendar_no_calendar": (
        "FINI_Planning_calendar_at_P_site___no_calendar",
    ),
    "fini_cycle_stock_at_dc": (
        "FINI_Cycle_stock_at_DC",
        "SB_FINI_Cycle_stock_at_DC",
        "FINI_Cycle_stock_at_DC_WITH_pal_identified",
    ),
    "fini_cycle_stock_without_pallet_conversion": (
        "FINI_Cycle_stock_at_DC_without_pallet_conversion",
        "SB_FINI_Cycle_stock_without_pallet_conversion",
        "FINI_Cycle_stock_at_DC_WITHOUT_pal_identified",
    ),
    "apo_sefi_lot_size": (
        "APO_SEFI_lot_size_per_recipe_combination",
        "SB_APO_SEFI_lot_size_per_recipe_combination",
    ),
    "fini_new_recipe_combination": (
        "FINI_new_recipe_combination",
        "SB_FINI_new_recipe_combination",
    ),
    "sap_character_catalog": ("Newrecipe",),
    "sap_character_compatibility": ("Compatibility",),
    "line_corrections": ("FixedLine",),
    "legacy_changeover_estimation": ("Change_over_estimation",),
    "legacy_sales_forecast_summary": ("Table11", "Table7"),
    "legacy_baseline_groups": ("Table15", "Groups29june", "Maygroups"),
    "recipe_characters": ("RecipeCharacters",),
    "sefi_zone": ("SEFIzone",),
    "legacy_recipe_alternatives": ("Newrecipe_1",),
    "selling_dcs_per_material": ("{plant}_specific___Selling_DCs_per_material",),
    "sefi_filling_line_evidence": (
        "{plant}_specific___SEFIs_with_several_filling_lines",
    ),
    "legacy_abc_cycle_stock": (
        "ABC_calculation___cycle_stock",
        "{plant}_specific___ABC_in_{plant}",
    ),
    "legacy_solution_summary": ("Solution_summary",),
    "legacy_average_batch_size": ("Avg_FINI_batch_size",),
    "cycle_stock_lot_size_01": ("SB_FINI_Cycle_stock_at_DC___Lot_size_01_in_{plant}",),
    "pck_line_source": ("PCK_per_filling_line___{plant}_specific",),
}

FIELD_ALIASES = {
    "Material": "material",
    "FINI": "material",
    "Plant": "plant",
    "Plnt": "plant",
    "P-site": "plant",
    "SEFI": "sefi",
    "Subgroup": "subgroup",
    "SAP character": "sap_character",
    "Recipe combination": "recipe_combination",
    "Planning calendar": "planning_calendar",
    "Lot size Profile": "lot_size_profile",
    "Lot size": "lot_size",
    "Min lot size at DC": "min_lot_size_at_dc",
    "Rounding value": "rounding_value",
    "Defined in wheel": "defined_in_wheel",
    "Lot size KG": "lot_size_kg",
    "Correct Filling line": "correct_filling_line",
}


def parse_string_lines(value: Any) -> tuple[str, ...] | None:
    """Parse source alternatives/position patterns into exact string identifiers.

    Blank input remains unknown; dash-only input is known empty. Identifiers are
    split only on source separators and are never converted into invented lines.
    """
    text = as_text(value)
    if text is None:
        return None
    return tuple(
        sorted({part for part in re.split(r"[\s/|;,]+", text) if part and part != "-"})
    )


def detect_profile(tables: dict[str, TableData]) -> str:
    """Identify the workbook family from named tables, independently of filename.

    Args:
        tables: Workbook table catalog keyed by Excel table name.

    Returns:
        ``dc_selling_profile`` (plant-specific DC selling/ABC tables),
        ``fixed_line_profile`` (FixedLine/SEFIzone tables), ``pck_line_profile``
        (plant-specific PCK filling-line table) or ``production_wheel_common``.
    """
    names = set(tables)
    if any(re.fullmatch(rf"{PLANT_TOKEN}_specific___.+", name) for name in names):
        return "dc_selling_profile"
    if "FixedLine" in names or "SEFIzone" in names:
        return "fixed_line_profile"
    if any(table_name_matches("PCK_per_filling_line___{plant}_specific", name) for name in names):
        return "pck_line_profile"
    return "production_wheel_common"


def is_supporting_table(name: str) -> bool:
    """Return whether an Excel table name matches any optional supporting alias.

    Args:
        name: Actual Excel table name.

    Returns:
        True when ``name`` satisfies one alias contract in ``SUPPORTING_TABLES``.
    """
    return any(
        table_name_matches(alias, name)
        for aliases in SUPPORTING_TABLES.values()
        for alias in aliases
    )


def supporting_header_map(table: TableData) -> dict[str, str]:
    """Map source headers to distinct, bounded relational identifiers.

    Preserve source-to-canonical provenance, including punctuation-only and
    Unicode headers that have no ASCII word characters.
    """
    mapping = {}
    used = {"source_sheet", "source_row", "source_table", "owner_id", "row_id"}
    for record in table.records:
        for header in record:
            if header == "_source_row" or header in mapping:
                continue
            slug = FIELD_ALIASES.get(header) or re.sub(
                r"[^a-z0-9]+", "_", header.lower()
            ).strip("_")
            if not slug:
                slug = "field"
            elif slug[0].isdigit():
                slug = "field_" + slug
            digest = hashlib.sha256(header.encode("utf-8")).hexdigest()[:16]
            # Keep familiar names where possible. The suffix distinguishes
            # otherwise identical slugs without losing the source symbol.
            candidate = slug if len(slug) <= 127 else f"{slug[:100]}_{digest}"
            if candidate in used or slug == "field":
                candidate = f"{slug[:100]}_{digest}"
            counter = 2
            while candidate in used:
                candidate = f"{slug[:100]}_{digest}_{counter}"
                counter += 1
            mapping[header] = candidate
            used.add(candidate)
    return mapping


def supporting_rows(tables: dict[str, TableData]) -> dict[str, list[dict[str, Any]]]:
    """Map optional family aliases into canonical row collections with lineage."""
    result = {}
    for target, aliases in SUPPORTING_TABLES.items():
        rows = []
        for alias in aliases:
            matched = [tables[name] for name in sorted(tables) if table_name_matches(alias, name)]
            for table in matched:
                mapping = supporting_header_map(table)
                for record in table.records:
                    row = {
                        mapping[key]: value
                        for key, value in record.items()
                        if key != "_source_row"
                    }
                    row.update(
                        source_sheet=table.sheet,
                        source_row=record["_source_row"],
                        source_table=table.name,
                    )
                    rows.append(row)
        result[target] = rows
    return result


def apply_line_evidence(rows: list[dict], corrections: list[dict]) -> list[dict]:
    """Apply unique FixedLine corrections and return conflict issues with evidence."""
    issues = []
    by_material: dict[str, list[dict]] = {}
    for correction in corrections:
        by_material.setdefault(as_text(correction.get("material")) or "", []).append(
            correction
        )
    for row in rows:
        evidence = by_material.get(row["material"], [])
        values = {as_text(item.get("correct_filling_line")) for item in evidence} - {
            None
        }
        row["filling_line_correction_raw"] = (
            "|".join(sorted(values)) if values else None
        )
        row["line_correction_evidence"] = evidence
        effective = row["filling_line_raw"]
        if len(values) == 1:
            effective = next(iter(values))
        elif len(values) > 1:
            issues.append(
                {
                    "rule_id": "CONFLICTING_LINE_CORRECTIONS",
                    "severity": "error",
                    "entity_key": row["fini_plant_key"],
                    "action": "Resolve conflicting FixedLine source rows",
                }
            )
            effective = None
        parsed = parse_string_lines(effective)
        row["eligible_lines"] = "|".join(parsed) if parsed is not None else None
        row["line_data_status"] = "known" if parsed is not None else "missing"
        row["line_resolution_source"] = (
            "FixedLine" if len(values) == 1 else "main_table"
        )
    return issues
