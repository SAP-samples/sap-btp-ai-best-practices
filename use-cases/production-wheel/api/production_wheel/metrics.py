"""Deterministic pallet, coverage, recurrence, and objective calculations."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Iterable, Mapping, Sequence

from production_wheel.schemas import CoverageBasis, CoverageMode, PalletFormula, TargetBand


def decimal_value(value: Decimal | str | int | float) -> Decimal:
    """Convert a finite numeric input to a decimal through its text form."""

    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"numeric value must be finite: {value}")
    return result


def positive_decimal(value: Decimal | str | int | float, label: str) -> Decimal:
    """Return a positive decimal or raise a field-specific error."""

    result = decimal_value(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def effective_batch(
    nominal_lot_litres: Decimal | str | int | float,
    factor: Decimal | str | int | float = Decimal("0.90"),
) -> Decimal:
    """Return the RoC-effective batch litres for one PV."""

    return positive_decimal(nominal_lot_litres, "nominal lot") * positive_decimal(factor, "factor")


def proportional_allocations(
    demands: Mapping[str, Decimal | str | int | float],
    batch_litres: Decimal | str | int | float,
) -> dict[str, Decimal]:
    """Allocate an effective batch to FINIs in proportion to annual demand."""

    if not demands:
        raise ValueError("demands cannot be empty")
    normalized = {material: positive_decimal(value, f"demand {material}") for material, value in demands.items()}
    total = sum(normalized.values(), Decimal(0))
    batch = positive_decimal(batch_litres, "batch litres")
    return {material: batch * demand / total for material, demand in normalized.items()}


def pallet_allocations(
    nominal_allocations: Mapping[str, Decimal | str | int | float],
    pallet_litres: Mapping[str, Decimal | str | int | float],
    formula: PalletFormula,
) -> dict[str, Decimal]:
    """Apply the selected minimum/rounding rule to each FINI allocation."""

    if set(nominal_allocations) != set(pallet_litres):
        raise ValueError("allocation and pallet maps must contain the same materials")
    result: dict[str, Decimal] = {}
    for material in sorted(nominal_allocations):
        allocation = positive_decimal(nominal_allocations[material], f"allocation {material}")
        pallet = positive_decimal(pallet_litres[material], f"pallet {material}")
        if formula == PalletFormula.MINIMUM_ONLY:
            result[material] = max(allocation, pallet)
        elif formula == PalletFormula.WHOLE_PALLET_ROUNDING:
            multiple = (allocation / pallet).to_integral_value(rounding=ROUND_CEILING)
            result[material] = pallet * multiple
        else:  # pragma: no cover - guarded by enum typing
            raise ValueError(f"unsupported pallet formula: {formula}")
    return result


def group_coverage(
    basis: CoverageBasis,
    demands: Mapping[str, Decimal | str | int | float],
    effective_batch_litres: Decimal | str | int | float,
    adjusted_allocations: Mapping[str, Decimal | str | int | float],
    demand_days: int = 250,
) -> Decimal:
    """Calculate coverage from the selected group/FINI quantity interpretation."""

    normalized_demands = {
        material: positive_decimal(value, f"demand {material}") for material, value in demands.items()
    }
    total_demand = sum(normalized_demands.values(), Decimal(0))
    days = positive_decimal(demand_days, "demand days")
    batch = positive_decimal(effective_batch_litres, "effective batch")
    if basis == CoverageBasis.BASE_GROUP:
        return days * batch / total_demand
    if set(normalized_demands) != set(adjusted_allocations):
        raise ValueError("demand and adjusted-allocation maps must contain the same materials")
    allocations = {
        material: positive_decimal(value, f"adjusted allocation {material}")
        for material, value in adjusted_allocations.items()
    }
    if basis == CoverageBasis.ADJUSTED_GROUP:
        return days * sum(allocations.values(), Decimal(0)) / total_demand
    if basis == CoverageBasis.WORST_FINI:
        return max(days * allocations[material] / normalized_demands[material] for material in allocations)
    raise ValueError(f"unsupported coverage basis: {basis}")


def group_frequency(
    total_demand_litres: Decimal | str | int | float,
    effective_batch_litres: Decimal | str | int | float,
    productive_weeks: int = 50,
) -> Decimal:
    """Calculate production recurrence independently of pallet coverage reporting."""

    demand = positive_decimal(total_demand_litres, "total demand")
    batch = positive_decimal(effective_batch_litres, "effective batch")
    weeks = positive_decimal(productive_weeks, "productive weeks")
    return demand / (weeks * batch)


def changeover_contribution(frequency: Decimal | str | int | float, member_count: int) -> Decimal:
    """Return the agreed within-group FINI-change contribution."""

    if member_count < 1:
        raise ValueError("member_count must be positive")
    return positive_decimal(frequency, "frequency") * (member_count - 1)


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Portfolio coverage statistics in days."""

    maximum: float
    p90: float
    median: float
    demand_weighted_mean: float
    group_mean: float


def coverage_summary(
    coverages: Sequence[Decimal | str | int | float],
    demands: Sequence[Decimal | str | int | float],
) -> CoverageSummary:
    """Calculate fixed-convention portfolio coverage statistics.

    P90 uses the nearest-rank convention: sorted value at ``ceil(0.9*n)``.
    """

    if not coverages or len(coverages) != len(demands):
        raise ValueError("coverage and demand sequences must be non-empty and aligned")
    c = [float(decimal_value(value)) for value in coverages]
    d = [float(positive_decimal(value, "group demand")) for value in demands]
    ordered = sorted(c)
    p90 = ordered[max(math.ceil(0.9 * len(ordered)) - 1, 0)]
    weighted = sum(value * weight for value, weight in zip(c, d, strict=True)) / sum(d)
    return CoverageSummary(max(c), p90, statistics.median(c), weighted, statistics.fmean(c))


@dataclass(frozen=True, slots=True)
class TargetBandKey:
    """Exact lexicographic target-band objective levels."""

    violation_count: int
    worst_excess_days: float
    total_excess_days: float
    demand_weighted_mean: float

    def as_tuple(self) -> tuple[float, ...]:
        """Return the ordered minimization tuple."""

        return (
            float(self.violation_count),
            self.worst_excess_days,
            self.total_excess_days,
            self.demand_weighted_mean,
        )


def target_band_key(
    coverages: Sequence[Decimal | str | int | float],
    demands: Sequence[Decimal | str | int | float],
    band: TargetBand,
) -> TargetBandKey:
    """Return group violation count, worst/total day excess, and weighted mean."""

    summary = coverage_summary(coverages, demands)
    lower = float(band.lower_days)
    upper = float(band.upper_days)
    values = [float(decimal_value(value)) for value in coverages]
    excesses = [max(lower - value, value - upper, 0.0) for value in values]
    return TargetBandKey(
        violation_count=sum(excess > 0 for excess in excesses),
        worst_excess_days=max(excesses, default=0.0),
        total_excess_days=sum(excesses),
        demand_weighted_mean=summary.demand_weighted_mean,
    )


def objective_key(
    mode: CoverageMode,
    coverages: Sequence[Decimal | str | int | float],
    demands: Sequence[Decimal | str | int | float],
    total_j_ch: Decimal | str | int | float,
    band: TargetBand,
) -> tuple[float, ...]:
    """Return the governed ordered business objective tuple for comparison/tests."""

    summary = coverage_summary(coverages, demands)
    if mode == CoverageMode.MAX:
        return (summary.maximum,)
    if mode == CoverageMode.DEMAND_WEIGHTED_MEAN:
        return (summary.demand_weighted_mean,)
    if mode == CoverageMode.GROUP_MEAN:
        return (summary.group_mean,)
    if mode == CoverageMode.TARGET_BAND:
        return target_band_key(coverages, demands, band).as_tuple()
    if mode == CoverageMode.OPERATIONS_FIRST:
        target = target_band_key(coverages, demands, band)
        # Operations-first protects count/worst/total coverage violations, then
        # prioritizes J_CH before the weighted-coverage tie-breaker.
        return (
            float(target.violation_count),
            target.worst_excess_days,
            target.total_excess_days,
            float(decimal_value(total_j_ch)),
            target.demand_weighted_mean,
        )
    if mode == CoverageMode.BASELINE_CONSTRAINED_COVERAGE:
        return (summary.demand_weighted_mean,)
    if mode == CoverageMode.BASELINE_CONSTRAINED_OPERATIONS:
        return (float(decimal_value(total_j_ch)), summary.demand_weighted_mean)
    if mode == CoverageMode.PARETO:
        return (summary.maximum, float(decimal_value(total_j_ch)))
    raise ValueError(f"unsupported coverage mode: {mode}")
