"""Read the SAP-settings lookup tables from the uploaded inputs workbook.

SAP settings generation (cycle stock lot size per DC, APO lot size in KG,
recipe characters) needs reference data that the optimizer itself does not use.
The customer's inputs workbook already carries it as Excel tables, produced
upstream from SAP extractions; that upstream extraction is out of scope here
and documented in ``docs/sap-data-integration-points.md``.

The tables are large (35k-63k rows), so they are streamed in read-only mode
over the ranges reported by ``xlsx_table_catalog`` and filtered to the dataset
plant before they reach HANA. A missing table or header is a warning, never an
error: the optimizer dataset stays valid and only the dependent SAP tab is
blocked at generation time.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import range_boundaries

from production_wheel.extraction.common import as_number, as_text, xlsx_table_catalog

# Excel table -> (HANA input view, source header -> canonical field).
LOOKUP_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "Network___Primary_DC": (
        "dc_network",
        {
            "Material": "material",
            "Production plant": "plant",
            "Primary DC": "dc",
            "Hub": "hub",
            "FINI forecast (litres)": "forecast_litres",
            "Has forecast": "has_forecast",
        },
    ),
    "ABC_calculated": (
        "dc_abc_classes",
        {"Material": "material", "Plant": "dc", "ABC": "abc_class"},
    ),
    "FINI_pallets": (
        "dc_pallet_quantities",
        {
            "Material": "material",
            "Producing plant": "plant",
            "FINI/P-site": "fini_plant_key",
            "Pallet qty": "pallet_qty",
            "Layer qty": "layer_qty",
            "Volume": "package_volume",
        },
    ),
    "Production_versions___List": (
        "production_version_masses",
        {
            "SEFI/P-site": "sefi_plant_key",
            "Clasificador": "production_version",
            "Lot size": "lot_size_kg",
            "Lot size (Litres)": "lot_size_litres",
        },
    ),
    # Optional: some sites keep the character list in the primary workbook instead.
    "RecipeCharacters": (
        "recipe_characters",
        {"List of characters allowed in BW": "list_of_characters_allowed_in_bw"},
    ),
}
LOOKUP_VIEWS = tuple(view for view, _ in LOOKUP_SOURCES.values())
_NUMERIC_FIELDS = {"forecast_litres", "pallet_qty", "layer_qty", "package_volume", "lot_size_kg", "lot_size_litres"}


def _table_rows(workbook, catalog_entry: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield ``(source_row, header -> value)`` for one Excel table in read-only mode."""

    min_col, min_row, max_col, max_row = range_boundaries(catalog_entry["range"])
    rows = workbook[catalog_entry["sheet"]].iter_rows(
        min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, values_only=True
    )
    headers = [as_text(value) for value in next(rows, ())]
    for offset, values in enumerate(rows, start=1):
        yield min_row + offset, dict(zip(headers, values))


def _plant_filters(plant: str) -> dict[str, Callable[[dict[str, Any]], bool]]:
    """Return the per-view row filters that keep only the dataset plant's evidence."""

    suffix = f"/{plant}"
    return {
        "dc_network": lambda row: row["plant"] == plant,
        "dc_pallet_quantities": lambda row: row["plant"] == plant
        or str(row["fini_plant_key"] or "").endswith(suffix),
        "production_version_masses": lambda row: str(row["sefi_plant_key"] or "").endswith(suffix),
    }


def extract_sap_lookups(path: Path, plant: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Read every SAP lookup table available in the inputs workbook for one plant.

    Args:
        path: Uploaded inputs (enrichment) workbook.
        plant: Producing plant of the dataset; rows of other plants are dropped.

    Returns:
        ``(tables, issues)``. ``tables`` maps each found view to its rows, with
        ``source_sheet``/``source_row`` lineage. ABC classes are kept only for
        materials present in this plant's DC network. ``issues`` holds
        ``SAP_LOOKUP_MISSING`` warnings for absent tables or headers.
    """

    catalog = {entry["name"]: entry for entry in xlsx_table_catalog(path)}
    filters = _plant_filters(plant)
    tables: dict[str, list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    try:
        for table_name, (view, mapping) in LOOKUP_SOURCES.items():
            entry = catalog.get(table_name)
            if entry is None:
                issues.append(_missing(table_name, view, "table not found in the inputs workbook"))
                continue
            rows = []
            for source_row, record in _table_rows(workbook, entry):
                missing = [header for header in mapping if header not in record]
                if missing:
                    issues.append(_missing(table_name, view, f"missing headers {missing}"))
                    rows = None
                    break
                row = {
                    field: as_number(record[header]) if field in _NUMERIC_FIELDS else as_text(record[header])
                    for header, field in mapping.items()
                }
                if not any(value is not None for value in row.values()):
                    continue
                if view in filters and not filters[view](row):
                    continue
                rows.append({**row, "source_sheet": entry["sheet"], "source_row": source_row})
            if rows is not None:
                tables[view] = rows
    finally:
        workbook.close()
    for row in tables.get("production_version_masses", []):
        row["sefi"], _, row["plant"] = str(row["sefi_plant_key"]).partition("/")
    if "dc_abc_classes" in tables and "dc_network" in tables:
        materials = {row["material"] for row in tables["dc_network"]}
        tables["dc_abc_classes"] = [row for row in tables["dc_abc_classes"] if row["material"] in materials]
    return tables, issues


def _missing(table_name: str, view: str, reason: str) -> dict[str, Any]:
    """Build the non-blocking quality issue for an unavailable lookup table."""

    return {
        "rule_id": "SAP_LOOKUP_MISSING",
        "severity": "warning",
        "entity_key": table_name,
        "reason": f"{view}: {reason}",
        "action": "SAP settings tabs that need this lookup stay blocked; see docs/sap-data-integration-points.md",
    }


def merge_sap_lookups(tables: dict[str, list[dict[str, Any]]], lookups: dict[str, list[dict[str, Any]]]) -> None:
    """Add lookup views to extracted dataset tables in place.

    Recipe characters may already come from the primary workbook (supporting
    table alias). Both sources are kept, de-duplicated by character, so either
    workbook can provide the list.
    """

    for view, rows in lookups.items():
        if view != "recipe_characters":
            tables[view] = rows
            continue
        merged = list(tables.get(view) or [])
        seen = {as_text(row.get("list_of_characters_allowed_in_bw")) for row in merged}
        merged.extend(row for row in rows if as_text(row.get("list_of_characters_allowed_in_bw")) not in seen)
        tables[view] = merged
