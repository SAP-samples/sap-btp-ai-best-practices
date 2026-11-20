"""Classify source-price coverage without choosing a commercial price record."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.ecn.reference_data.purchase_info_reconciliation import MaterialPlant, material_plant

PriceCoverageStatus = Literal[
    "SOURCE_PRICE_AVAILABLE",
    "ZERO_PRICE_REVIEW",
    "MULTIPLE_SOURCE_PRICES_REVIEW",
    "NOT_SUPPLIED",
]


@dataclass(frozen=True)
class PriceCoverageEntry:
    """One Material + Plant's source-price coverage state."""

    material_number: str
    plant_code: str
    status: PriceCoverageStatus
    matching_source_row_count: int


@dataclass(frozen=True)
class PriceOverlayCoverage:
    """Classified source-price coverage for one material/plant population."""

    entries: tuple[PriceCoverageEntry, ...]

    def count(self, status: PriceCoverageStatus) -> int:
        """Return the number of entries in one controlled status."""
        return sum(entry.status == status for entry in self.entries)


def _normalise(value: object) -> str:
    """Normalize a source identifier for matching without changing returned values."""
    return " ".join(str(value or "").split()).casefold()


def _positive_price(row: Mapping[str, object]) -> bool:
    """Return whether a parser-validated source net price is greater than zero."""
    return Decimal(str(row["net_price"])) > 0


def assess_price_overlay_coverage(
    price_rows: Iterable[Mapping[str, object]],
    relevant_pairs: Iterable[MaterialPlant],
) -> PriceOverlayCoverage:
    """Classify price evidence without selecting among competing source records.

    A single positive row is ready to display as source evidence. Zero values,
    multiple positive source rows, and missing records remain controlled review
    states until business validity, supplier, and selection rules are defined.
    """
    by_pair: dict[MaterialPlant, list[Mapping[str, object]]] = {}
    for row in price_rows:
        pair = material_plant(row)
        if pair is not None:
            by_pair.setdefault(pair, []).append(row)

    display_pairs: dict[MaterialPlant, MaterialPlant] = {}
    for material, plant in relevant_pairs:
        normalized = (_normalise(material), _normalise(plant))
        if normalized[0] and normalized[1]:
            display_pairs.setdefault(normalized, (material, plant))

    entries = []
    for normalized, (material, plant) in sorted(display_pairs.items()):
        matches = by_pair.get(normalized, [])
        if not matches:
            status: PriceCoverageStatus = "NOT_SUPPLIED"
        elif not any(_positive_price(row) for row in matches):
            status = "ZERO_PRICE_REVIEW"
        elif len(matches) > 1:
            status = "MULTIPLE_SOURCE_PRICES_REVIEW"
        else:
            status = "SOURCE_PRICE_AVAILABLE"
        entries.append(
            PriceCoverageEntry(
                material_number=material,
                plant_code=plant,
                status=status,
                matching_source_row_count=len(matches),
            )
        )
    return PriceOverlayCoverage(entries=tuple(entries))
