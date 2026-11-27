"""Wheel RoC rule: whole-day RoC for frequency and changeovers, pallets only for inventory.

This module is the single place where a subgroup's coverage (RoC, days of demand
one batch covers), its batches per week, its changeovers and its inventory days
are calculated. The optimizer (coefficients, beam ranking, business-rule facts),
the independent validation, the reporting and the SAP settings all call it, so
they agree on every number. It reproduces the customer's production wheel
workbook:

1. **Base RoC** (continuous): ``demand_days x effective batch / group demand``.
   It is stored as ``base_group_coverage_days``.
2. **Wheel RoC** ``R``: the base RoC rounded to whole days with Excel ``ROUND``
   (half away from zero), at least ``WHEEL_ROC_MINIMUM_DAYS``. This is the
   workbook's "RoC selected PV". A base RoC under half a day would round to 0
   and divide by zero, hence the minimum.
3. **Batches per week**: ``days per week / R``, with days per week =
   ``demand_days / productive_weeks`` (5 in the customer's calendar).
4. **FINI wheel RoC**: ``R x k``. ``k`` is the FINI's runner multiple
   (``production_wheel.cycle_stock``); it is 1 for every FINI of a Single RoC
   subgroup. Multiples are computed from ``R``, never from a pallet-adjusted
   coverage.
5. **Changeovers per week** (``J_CH``): ``batches per week x (sum(1 / k) - 1)``.
   A FINI with multiple ``k`` joins one batch out of ``k``; the batch itself is
   made every cycle because the fastest FINI needs it. With Single RoC this is
   ``batches per week x (FINIs - 1)``.
6. **FINI inventory RoC**: the FINI wheel RoC adjusted to cover at least one
   full pallet, with the workbook's ``fxAdjustedRoC``: when one batch fills less
   than 0.9 pallets of the FINI, the RoC becomes ``floor(R x k / pallets)``.
   Pallets per batch = nominal lot x FINI demand share / pallet litres, rounded
   to 1 decimal (the workbook's "Pallets as per subgroup", which uses the
   nominal lot without the batch factor). A FINI filled every ``k``-th batch
   still uses the pallets of one batch (confirmed by the customer). Pallets
   never change steps 2-5.

The coverage of the subgroup under each ``CoverageBasis``:

- ``BASE_GROUP``: demand-weighted mean of the FINI wheel RoC (``R`` for Single RoC);
- ``ADJUSTED_GROUP``: demand-weighted mean of the FINI inventory RoC;
- ``WORST_FINI``: largest FINI inventory RoC.

Worked example (synthetic): demand 80 days, 16 productive weeks (5 days per
week), base RoC 3.89 days, three FINIs with multiples 1 / 5 / 10. ``R = 4``,
batches per week = 1.25, FINI wheel RoC 4 / 20 / 40,
``J_CH = 1.25 x (1 + 0.2 + 0.1 - 1) = 0.375``.

Self-check::

    cd api && ../.venv/bin/python -m production_wheel.wheel_roc
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from production_wheel.schemas import CoverageBasis

# Smallest wheel RoC in days. Excel ROUND of a base RoC under 0.5 gives 0, which
# would make batches per week infinite.
WHEEL_ROC_MINIMUM_DAYS = 1
# The workbook's fxAdjustedRoC adjusts a FINI when one batch fills less than this many pallets.
PALLET_ADJUSTMENT_THRESHOLD = 0.9
# fxAdjustedRoC counts a FINI whose pallets round to 0.0 as this many pallets.
ZERO_PALLET_SUBSTITUTE = 0.09


def wheel_roc_days(base_days: Decimal | float | None) -> int | None:
    """Round a continuous base RoC to the wheel RoC ``R`` in whole days.

    Args:
        base_days: ``demand_days x effective batch / group demand``.

    Returns:
        Excel ``ROUND(base_days, 0)`` (half away from zero), at least
        ``WHEEL_ROC_MINIMUM_DAYS``; ``None`` for a missing or non-positive base.

    Example:
        ``wheel_roc_days(3.89) == 4``; ``wheel_roc_days(4.5) == 5``;
        ``wheel_roc_days(0.3) == 1``.
    """

    if base_days is None:
        return None
    value = base_days if isinstance(base_days, Decimal) else Decimal(str(base_days))
    if not value.is_finite() or value <= 0:
        return None
    return max(WHEEL_ROC_MINIMUM_DAYS, int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)))


def days_per_week(demand_days: float, productive_weeks: float) -> float:
    """Return demand days per productive week (5 for the customer calendar).

    Args:
        demand_days: Days of the demand horizon (e.g. 80).
        productive_weeks: Productive weeks of the same horizon (e.g. 16).

    Returns:
        ``demand_days / productive_weeks``.
    """

    return float(demand_days) / float(productive_weeks)


def pallets_per_batch(
    nominal_lot_litres: float,
    demand_litres: float,
    group_demand_litres: float,
    pallet_litres: float | None,
) -> float | None:
    """Return the pallets one batch fills for a FINI, as the workbook rounds them.

    Args:
        nominal_lot_litres: Nominal lot of the subgroup's PV (no batch factor).
        demand_litres: The FINI's demand over the horizon.
        group_demand_litres: The subgroup's demand over the horizon.
        pallet_litres: Litres on one pallet of the FINI.

    Returns:
        ``ROUND(nominal lot x demand / group demand / pallet litres, 1)``, or
        ``None`` when the pallet size is missing or not positive.
    """

    if pallet_litres is None or pallet_litres <= 0 or group_demand_litres <= 0:
        return None
    value = Decimal(str(nominal_lot_litres)) * Decimal(str(demand_litres)) / (
        Decimal(str(group_demand_litres)) * Decimal(str(pallet_litres)))
    return float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def inventory_roc_days(wheel_days: float | None, pallets: float | None) -> float | None:
    """Port of the workbook's ``fxAdjustedRoC``: cover at least one full pallet.

    The pallets passed in are those of one batch (the workbook's "Pallets as per
    subgroup"), also for a FINI filled only every ``k``-th batch. The customer
    confirmed this on 25 November 2026: RoC 96 days, multiple 2, 0.8 pallets per
    batch gives floor(192 / 0.8) = 240 inventory days (SAP settings then cap the
    DC lot size at 200 days).

    Args:
        wheel_days: The FINI's wheel RoC (``R x k``).
        pallets: Pallets one batch fills for the FINI (``pallets_per_batch``).

    Returns:
        ``floor(wheel_days / pallets)`` when ``pallets`` is under
        ``PALLET_ADJUSTMENT_THRESHOLD`` (0 pallets count as
        ``ZERO_PALLET_SUBSTITUTE``), else ``wheel_days`` unchanged; missing or
        negative pallets leave the RoC unchanged.

    Example:
        ``inventory_roc_days(10, 0.85) == 11.0``; ``inventory_roc_days(5, 1.2) == 5``.
    """

    if wheel_days is None:
        return None
    if pallets == 0:
        pallets = ZERO_PALLET_SUBSTITUTE
    if pallets is None or pallets <= 0 or pallets >= PALLET_ADJUSTMENT_THRESHOLD:
        return wheel_days
    return float(math.floor(wheel_days / pallets))


@dataclass(frozen=True, slots=True)
class WheelFigures:
    """RoC, frequency, changeovers and inventory of one subgroup under the wheel rule.

    Attributes:
        base_coverage_days: Continuous base RoC (``demand_days x batch / demand``).
        roc_days: Wheel RoC ``R`` in whole days.
        frequency_per_week: Batches per week, ``days per week / R``.
        fills_per_batch: Average FINIs filled per batch, ``sum(1 / k)``.
        j_ch: Changeovers per week, ``frequency x (fills_per_batch - 1)``.
        fini_ids: FINIs in input order.
        multiples: Runner multiple ``k`` of each FINI (1 = every batch).
        fini_wheel_roc_days: ``R x k`` of each FINI.
        fini_pallets_per_batch: Pallets one batch fills for each FINI.
        fini_inventory_roc_days: Pallet-adjusted wheel RoC of each FINI.
        coverage_by_basis: Subgroup coverage under every ``CoverageBasis``.
    """

    base_coverage_days: float
    roc_days: int
    frequency_per_week: float
    fills_per_batch: float
    j_ch: float
    fini_ids: tuple[str, ...]
    multiples: tuple[int, ...]
    fini_wheel_roc_days: tuple[float, ...]
    fini_pallets_per_batch: tuple[float | None, ...]
    fini_inventory_roc_days: tuple[float, ...]
    coverage_by_basis: Mapping[CoverageBasis, float]

    def coverage(self, basis: CoverageBasis) -> float:
        """Return the subgroup coverage under one coverage basis."""

        return self.coverage_by_basis[basis]

    def by_fini(self, values: Sequence) -> dict[str, object]:
        """Return ``{fini_id: value}`` for one of the per-FINI tuples."""

        return dict(zip(self.fini_ids, values))


def wheel_figures(
    fini_ids: Sequence[str],
    demands: Sequence[float],
    pallet_litres: Sequence[float | None],
    multiples: Sequence[int],
    nominal_lot_litres: float,
    effective_batch_litres: float,
    demand_days: float,
    productive_weeks: float,
) -> WheelFigures:
    """Calculate every wheel figure of one subgroup from its primitives.

    Args:
        fini_ids: The subgroup's FINIs.
        demands: Each FINI's demand over the horizon (litres), aligned with ``fini_ids``.
        pallet_litres: Each FINI's litres per pallet, aligned with ``fini_ids``.
        multiples: Each FINI's runner multiple (all 1 for Single RoC).
        nominal_lot_litres: Nominal lot of the PV (pallets per batch use it).
        effective_batch_litres: Nominal lot x batch factor (the RoC uses it).
        demand_days: Days of the demand horizon.
        productive_weeks: Productive weeks of the same horizon.

    Returns:
        ``WheelFigures`` (module docstring).

    Raises:
        ValueError: Empty subgroup, misaligned inputs, non-positive demand or
            batch, or a multiple below 1.
    """

    count = len(fini_ids)
    if count == 0 or not (len(demands) == len(pallet_litres) == len(multiples) == count):
        raise ValueError("wheel figures need aligned, non-empty FINI inputs")
    if any(demand <= 0 for demand in demands) or effective_batch_litres <= 0:
        raise ValueError("wheel figures need positive demand and batch")
    if any(multiple < 1 for multiple in multiples):
        raise ValueError("runner multiples must be at least 1")
    group_demand = sum(demands)
    base = Decimal(str(demand_days)) * Decimal(str(effective_batch_litres)) / Decimal(str(group_demand))
    roc = wheel_roc_days(base)
    frequency = days_per_week(demand_days, productive_weeks) / roc
    fills = float(count) if all(multiple == 1 for multiple in multiples) else sum(1 / k for k in multiples)
    wheel = tuple(float(roc * k) for k in multiples)
    pallets = tuple(pallets_per_batch(nominal_lot_litres, demand, group_demand, pallet)
                    for demand, pallet in zip(demands, pallet_litres))
    inventory = tuple(inventory_roc_days(days, pallet_count) for days, pallet_count in zip(wheel, pallets))
    coverage = {
        CoverageBasis.BASE_GROUP: sum(d * w for d, w in zip(demands, wheel)) / group_demand,
        CoverageBasis.ADJUSTED_GROUP: sum(d * i for d, i in zip(demands, inventory)) / group_demand,
        CoverageBasis.WORST_FINI: max(inventory),
    }
    if all(multiple == 1 for multiple in multiples):
        # Exactly R (no float drift from the weighted mean) for every Single RoC subgroup.
        coverage[CoverageBasis.BASE_GROUP] = float(roc)
    return WheelFigures(
        base_coverage_days=float(base),
        roc_days=roc,
        frequency_per_week=frequency,
        fills_per_batch=fills,
        j_ch=frequency * (fills - 1),
        fini_ids=tuple(fini_ids),
        multiples=tuple(int(k) for k in multiples),
        fini_wheel_roc_days=wheel,
        fini_pallets_per_batch=pallets,
        fini_inventory_roc_days=inventory,
        coverage_by_basis=coverage,
    )


def _self_check() -> None:
    """Assert the module docstring's worked example and the edge cases; raises on regression."""

    assert wheel_roc_days(3.8913) == 4 and wheel_roc_days(4.5) == 5 and wheel_roc_days(0.3) == 1
    assert wheel_roc_days(None) is None and wheel_roc_days(0) is None
    assert inventory_roc_days(10, 0.85) == 11.0 and inventory_roc_days(5, 1.2) == 5
    assert inventory_roc_days(4, 0) == float(math.floor(4 / 0.09))
    # Base RoC 80 x 3150.9 / 64778 = 3.89 days -> R = 4.
    figures = wheel_figures(("A", "B", "C"), (54906.0, 7854.0, 2018.0), (513.0, 456.0, 360.0),
                            (1, 5, 10), 3501.0, 3150.9, 80, 16)
    assert figures.roc_days == 4 and figures.frequency_per_week == 1.25
    assert figures.fini_wheel_roc_days == (4.0, 20.0, 40.0)
    assert abs(figures.j_ch - 0.375) < 1e-12
    single = wheel_figures(("A", "B"), (100.0, 300.0), (50.0, 50.0), (1, 1), 100.0, 90.0, 80, 16)
    assert single.coverage(CoverageBasis.BASE_GROUP) == 18.0 and single.j_ch == 5 / 18


if __name__ == "__main__":
    _self_check()
    print("wheel RoC self-check passed")
