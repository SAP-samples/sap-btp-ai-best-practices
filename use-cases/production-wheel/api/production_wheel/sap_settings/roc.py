"""Coverage (RoC, days) rules of the SAP settings template and the SAP lot-size key.

Terms used by the customer template:
- **RoC selected PV**: days one batch of the selected production version covers
  for its subgroup. It equals the wheel RoC ``R`` of the optimizer
  (``wheel_roc.wheel_roc_days(base_group_coverage_days)``: Excel ``ROUND`` to
  whole days, at least 1 day).
- **Pallets as per subgroup**: pallets one batch fills for a FINI
  (``excel_round(legacy_nominal_pallet_count, 1)``).
- **RoC 1 full SEFI batch**: days a whole SEFI batch would cover for one FINI
  alone (``reference_coverage_source_days``). It drives the runner classes.

Excel ``ROUND`` rounds half away from zero, unlike Python ``round``, so
``excel_round`` uses ``Decimal`` with ``ROUND_HALF_UP``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from production_wheel.extraction.common import as_number, as_text
# excel_round is defined with the shared runner rules and re-exported here for SAP settings callers.
from production_wheel.runners import CycleStockPolicy, RunnerThresholds, classify, excel_round, runner_multiple, sefi_ratio
from production_wheel.wheel_roc import inventory_roc_days, wheel_roc_days

# SAP lot-size keys for 100..200 days in steps of 10 (no "J"): 100 -> 9A ... 200 -> 9L.
_HUNDREDS_KEYS = "ABCDEFGHIKL"


def pallet_adjusted_roc(roc: float | None, pallets: float | None) -> float | None:
    """Port of the template's ``fxAdjustedRoC``: cover at least one full pallet.

    The rule lives in ``production_wheel.wheel_roc.inventory_roc_days`` so the
    optimizer's inventory coverage and the DC cycle stock use the same numbers.
    When a batch fills less than 0.9 pallets of a FINI, the DC cycle stock is set
    to the coverage of one full pallet: ``floor(roc / pallets)``. Zero pallets
    count as 0.09; missing or non-positive pallets leave the RoC unchanged.
    """

    return inventory_roc_days(roc, pallets)


def normalized_days(days: float | None, maximum: float = 200) -> int | None:
    """Cap days at ``maximum`` and round down to tens above 100, as the template does.

    Returns ``None`` for missing, non-positive or fractional values at or below
    100, which have no SAP lot-size key.
    """

    if days is None or days <= 0:
        return None
    if days > maximum:
        return int(maximum)
    if days > 100:
        return int(math.floor(days / 10) * 10)
    return int(days) if float(days).is_integer() else None


def sap_lot_size_code(days: int | None) -> str | None:
    """Map cycle days to the SAP lot-size key: 1-99 -> "01".."99", 100..200 -> 9A..9L."""

    if days is None:
        return None
    if 1 <= days <= 99:
        return f"{days:02d}"
    if 100 <= days <= 200 and days % 10 == 0:
        return "9" + _HUNDREDS_KEYS[(days - 100) // 10]
    return None


def runner_class(full_roc: float | None, ratio: float | None, thresholds: RunnerThresholds) -> str | None:
    """Classify a FINI as High / Medium / Low runner with the shared definition.

    The rule itself lives in ``production_wheel.runners.classify`` (also used by
    the optimizer business rules); this wrapper only keeps the template's
    capitalised labels for the audit table and export.

    Args:
        full_roc: Full-batch coverage days ("RoC 1 full SEFI batch").
        ratio: Ratio to the SEFI's fastest runner.
        thresholds: Runner cut-offs frozen with the run's plant profile.

    Returns:
        ``"High"``, ``"Medium"``, ``"Low"`` or ``None``.
    """

    label = classify(full_roc, ratio, thresholds)
    return label.capitalize() if label else None


def runner_adjusted_roc(
    base: float | None,
    ratio: float | None,
    full_roc: float | None,
    runner: str | None,
    policy: CycleStockPolicy,
) -> tuple[float | None, str]:
    """Port of the template's ``Adjusted RoC selected PV`` formula.

    Slower runners in a subgroup get a whole multiple of the base RoC, so they
    are produced every few cycles instead of every cycle. The multiple comes from
    the rule shared with the optimizer, ``production_wheel.runners.runner_multiple``.

    Args:
        base: Rounded RoC of the selected PV.
        ratio: Full-batch RoC of the FINI divided by the smallest one in its subgroup.
        full_roc: Full-batch RoC of the FINI.
        runner: Runner class from ``runner_class`` (capitalised).
        policy: Cycle stock policy frozen with the run's plant profile.

    Returns:
        ``(adjusted_roc, status)``. Status is ``adjusted``, ``reference_runner``
        (fastest of its subgroup, unchanged), ``missing_input``,
        ``invalid_input`` or ``base_above_max``. Unusable inputs return ``None``.
    """

    multiple, status = runner_multiple(base, ratio, full_roc, runner.lower() if runner else None, policy)
    return (base * multiple if multiple is not None else None), status


def runner_adjustments(
    wheel_members: list[dict[str, Any]],
    policy: CycleStockPolicy,
    thresholds: RunnerThresholds,
) -> list[dict[str, Any]]:
    """Compute base and runner-multiple RoC for every FINI of the selected wheel.

    The cycle stock strategy of each subgroup comes from the selected point
    (``members.cycle_stock_strategy``, chosen by the optimizer under the run's
    plant profile). SAP settings recompute the multiple from the wheel RoC ``R``
    like the customer template. The optimizer computes it from the same ``R``,
    so ``optimizer_runner_multiple`` (kept for comparison) matches for runs made
    with the wheel RoC rule; older runs used an unrounded base and can differ.

    Args:
        wheel_members: ``members`` rows of the selected point that have a
            proposed subgroup (HANA values may be strings).
        policy: Cycle stock policy frozen with the run's plant profile.
        thresholds: Runner cut-offs frozen with the run's plant profile.

    Returns:
        One audit row per FINI. ``coverage_days`` is the multiplied RoC when the
        subgroup uses runner multiples and the formula succeeded, else the base
        RoC. A runner-multiple FINI whose formula fails keeps its base RoC
        (multiple 1); ``status`` records why. Single-RoC subgroups have status
        ``single_roc``.
    """

    rows = []
    for member in wheel_members:
        full = as_number(member.get("reference_coverage_source_days"))
        optimizer_multiple = as_number(member.get("runner_multiple"))
        rows.append({
            "sefi": as_text(member.get("sefi")),
            "plant": as_text(member.get("plant")),
            "subgroup": as_text(member.get("proposed_subgroup_code")),
            "material": as_text(member.get("material")),
            # Rows of runs made before the cycle stock strategy have no value: single RoC.
            "cycle_stock_strategy": as_text(member.get("cycle_stock_strategy")) or "single",
            "optimizer_runner_multiple": int(optimizer_multiple) if optimizer_multiple else None,
            "full_batch_coverage_days": full,
            # The optimizer's wheel RoC R (whole days, at least 1), so multiples match the optimizer's.
            "base_coverage_days": wheel_roc_days(as_number(member.get("base_group_coverage_days"))),
        })
    sefi_minimum, subgroup_minimum = defaultdict(lambda: math.inf), defaultdict(lambda: math.inf)
    for row in rows:
        if row["full_batch_coverage_days"] and row["full_batch_coverage_days"] > 0:
            full = row["full_batch_coverage_days"]
            sefi_minimum[row["sefi"], row["plant"]] = min(sefi_minimum[row["sefi"], row["plant"]], full)
            key = (row["sefi"], row["plant"], row["subgroup"])
            subgroup_minimum[key] = min(subgroup_minimum[key], full)
    for row in rows:
        full = row["full_batch_coverage_days"]
        sefi_min = sefi_minimum.get((row["sefi"], row["plant"]))
        subgroup_min = subgroup_minimum.get((row["sefi"], row["plant"], row["subgroup"]))
        positive = full is not None and full > 0
        # Power Query rounds the SEFI ratio half-to-even; Excel rounds the subgroup ratio half-up.
        row["sefi_ratio"] = sefi_ratio(full, sefi_min)
        row["subgroup_ratio"] = excel_round(full / subgroup_min, 1) if positive and subgroup_min else None
        row["runner_class"] = runner_class(full, row["sefi_ratio"], thresholds)
        row["applied"] = False
        if row["cycle_stock_strategy"] != "runner_multiple":
            row["adjusted_coverage_days"], row["status"] = None, "single_roc"
        else:
            row["adjusted_coverage_days"], row["status"] = runner_adjusted_roc(
                row["base_coverage_days"], row["subgroup_ratio"], full, row["runner_class"], policy)
            row["applied"] = row["status"] == "adjusted"
        row["coverage_days"] = row["adjusted_coverage_days"] if row["applied"] else row["base_coverage_days"]
        base = row["base_coverage_days"]
        row["runner_multiple"] = round(row["coverage_days"] / base) if row["applied"] and base else 1
    return rows
