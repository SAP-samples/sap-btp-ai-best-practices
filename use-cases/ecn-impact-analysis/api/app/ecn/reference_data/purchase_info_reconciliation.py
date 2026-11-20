"""Reconcile a targeted Purchase Info successor without losing historical rows.

This module is deliberately pure: it neither connects to HANA nor publishes a
reference dataset.  Callers receive a deterministic preview that can be tested
and approved before any data-changing publication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal


PurchaseInfoValues = Mapping[str, object]
MaterialPlant = tuple[str, str]


@dataclass(frozen=True)
class ReconciledPurchaseInfoRow:
    """One successor Purchase Info row with its reconciliation provenance."""

    values: dict[str, object]
    origin: Literal["HANA_CARRIED_FORWARD", "PRICE_CSV_TARGETED"]
    source_sheet: str | None
    source_row_number: int | None


@dataclass(frozen=True)
class PurchaseInfoReconciliation:
    """Preview the rows and coverage gaps of one targeted successor dataset."""

    rows: tuple[ReconciledPurchaseInfoRow, ...]
    baseline_pair_count: int
    targeted_pair_count: int
    uncovered_target_pairs: tuple[MaterialPlant, ...]

    @property
    def carried_forward_count(self) -> int:
        """Return the number of preserved historical source rows."""
        return sum(row.origin == "HANA_CARRIED_FORWARD" for row in self.rows)

    @property
    def targeted_price_csv_count(self) -> int:
        """Return the number of new source rows selected from the price CSV."""
        return sum(row.origin == "PRICE_CSV_TARGETED" for row in self.rows)


def _normalise(value: object) -> str:
    """Normalize a source identifier for matching without changing stored values."""
    return " ".join(str(value or "").split()).casefold()


def material_plant(values: PurchaseInfoValues) -> MaterialPlant | None:
    """Return a source-backed material/plant key, or None when either side is absent."""
    material = _normalise(values.get("material_number"))
    plant = _normalise(values.get("plant_code"))
    return (material, plant) if material and plant else None


def _purchase_setup_values(price_row: PurchaseInfoValues) -> dict[str, object]:
    """Project one price-source row into the documented Purchase Info row contract."""
    return {
        "info_record": price_row.get("info_record"),
        "vendor": price_row.get("vendor"),
        "supplier_name": price_row.get("supplier_name"),
        "material_number": price_row.get("material_number"),
        "plant_code": price_row.get("plant_code"),
        "purchasing_group": price_row.get("purchasing_group"),
        "category": None,
        "tax_code": None,
        "fixed_firm_date": None,
    }


def _source_coordinates(values: PurchaseInfoValues) -> tuple[str | None, int | None]:
    """Read optional source coordinates retained for a preview manifest."""
    sheet = values.get("sheet_name")
    row_number = values.get("source_row_number")
    return (
        str(sheet) if isinstance(sheet, str) and sheet.strip() else None,
        row_number if isinstance(row_number, int) and row_number > 0 else None,
    )


def reconcile_targeted_purchase_info(
    baseline_rows: Iterable[PurchaseInfoValues],
    price_rows: Iterable[PurchaseInfoValues],
    relevant_pairs: Iterable[MaterialPlant],
) -> PurchaseInfoReconciliation:
    """Preserve baseline rows and add only price rows for uncovered UI material/plant pairs.

    Matching is case- and whitespace-insensitive, but the emitted source values
    remain unchanged. Every baseline row is retained, including duplicates. Price
    rows are retained in their source order only where their material/plant pair
    is relevant and has no baseline Purchase Info evidence.
    """
    baseline = tuple(dict(row) for row in baseline_rows)
    targets = {
        (_normalise(material), _normalise(plant))
        for material, plant in relevant_pairs
        if _normalise(material) and _normalise(plant)
    }
    baseline_pairs = {pair for row in baseline if (pair := material_plant(row)) is not None}
    uncovered_targets = targets - baseline_pairs

    selected_price_rows = tuple(
        ReconciledPurchaseInfoRow(
            _purchase_setup_values(row),
            "PRICE_CSV_TARGETED",
            *_source_coordinates(row),
        )
        for row in price_rows
        if (pair := material_plant(row)) is not None and pair in uncovered_targets
    )
    selected_pairs = {
        pair
        for row in selected_price_rows
        if (pair := material_plant(row.values)) is not None
    }
    carried_forward = tuple(
        ReconciledPurchaseInfoRow(
            dict(row),
            "HANA_CARRIED_FORWARD",
            *_source_coordinates(row),
        )
        for row in baseline
    )
    return PurchaseInfoReconciliation(
        rows=carried_forward + selected_price_rows,
        baseline_pair_count=len(baseline_pairs),
        targeted_pair_count=len(selected_pairs),
        uncovered_target_pairs=tuple(sorted(uncovered_targets - selected_pairs)),
    )
