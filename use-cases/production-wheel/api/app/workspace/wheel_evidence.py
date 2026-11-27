"""Numerical derivations of the wheel RoC rule for the assistant's assignment explanations.

``analytics_evidence.explain_assignment`` shows, for one group, how every stored
KPI follows from stored primitives. Runs created with the wheel RoC rule
(``production_wheel.wheel_roc``) store ``wheel_roc_days`` on their group rows;
for those runs this module replaces the frequency, changeover and coverage
derivations with the rule's formulas:

- ``wheel_roc_days`` = ``max(1, ROUND(base_group_coverage_days, 0))``;
- ``frequency_per_week`` = ``(demand_days / productive_weeks) / wheel_roc_days``;
- ``j_ch_contribution`` = ``frequency_per_week * (sum(1 / runner_multiple) - 1)``;
- per FINI: wheel RoC ``wheel_roc_days * runner_multiple``, pallets per batch
  ``ROUND(nominal lot * demand share / pallet litres, 1)`` and the inventory RoC
  of the workbook's ``fxAdjustedRoC``;
- group coverage views from those per-FINI values.

Runs without ``wheel_roc_days`` keep their original derivations untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from production_wheel.wheel_roc import inventory_roc_days, pallets_per_batch, wheel_roc_days

Derivation = Callable[[str, dict[str, Any], float | None, Any, list[str]], dict[str, Any]]
Number = Callable[[Any], float | None]


def has_wheel_rule(group: Mapping[str, Any]) -> bool:
    """Return whether a stored group row was produced with the wheel RoC rule."""

    return group.get("wheel_roc_days") not in (None, "")


def _member_demand(row: Mapping[str, Any], number: Number) -> float | None:
    """Return a member's demand over the run horizon (the litres the optimizer used)."""

    period = number(row.get("forecast_litres_period"))
    return period if period is not None else number(row.get("forecast_litres_12m"))


def wheel_derivations(
    group: Mapping[str, Any],
    members: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    derivation: Derivation,
    number: Number,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return group and per-material derivations of the wheel RoC rule.

    Args:
        group: Stored ``groups`` row of the explained group.
        members: Stored ``members`` rows of the same group.
        config: Persisted run configuration (demand days, productive weeks,
            coverage basis).
        derivation: ``analytics_evidence._derivation`` (formula, inputs,
            calculated, reported, source fields).
        number: ``analytics_evidence._number``.

    Returns:
        ``(group_derivations, {material: member_derivations})``. Keys override
        the legacy derivations of the same name.
    """

    days, weeks = number(config.get("demand_days")), number(config.get("productive_weeks"))
    base = number(group.get("base_group_coverage_days"))
    nominal_lot = number(group.get("nominal_lot_litres"))
    group_demand = number(group.get("group_demand_litres"))
    roc = wheel_roc_days(base)
    frequency = days / weeks / roc if days and weeks and roc else None
    per_member: dict[str, dict[str, Any]] = {}
    rows = []
    for row in members:
        material = str(row.get("material"))
        multiple = number(row.get("runner_multiple")) or 1.0
        demand = _member_demand(row, number)
        pallet = number(row.get("pallet_litres_resolved"))
        wheel = roc * multiple if roc is not None else None
        pallets = (pallets_per_batch(nominal_lot, demand, group_demand, pallet)
                   if nominal_lot and demand and group_demand else None)
        inventory = inventory_roc_days(wheel, pallets)
        rows.append((demand, multiple, wheel, inventory))
        per_member[material] = {
            "fini_wheel_roc_days": derivation(
                "wheel_roc_days * runner_multiple",
                {"wheel_roc_days": roc, "runner_multiple": multiple},
                wheel, row.get("fini_wheel_roc_days"),
                ["groups.wheel_roc_days", "members.runner_multiple"]),
            "pallets_per_batch": derivation(
                "ROUND(nominal_lot_litres * member_demand / group_demand_litres / pallet_litres_resolved, 1)",
                {"nominal_lot_litres": nominal_lot, "member_demand": demand,
                 "group_demand_litres": group_demand, "pallet_litres_resolved": pallet},
                pallets, row.get("pallets_per_batch"),
                ["groups.nominal_lot_litres", "members.forecast_litres_period",
                 "groups.group_demand_litres", "members.pallet_litres_resolved"]),
            # Same key as the legacy pallet-allocation derivation: the FINI inventory RoC.
            "adjusted_coverage": derivation(
                "floor(fini_wheel_roc_days / pallets_per_batch) if pallets_per_batch < 0.9 "
                "(0 counts as 0.09) else fini_wheel_roc_days",
                {"fini_wheel_roc_days": wheel, "pallets_per_batch": pallets},
                inventory, row.get("fini_inventory_roc_days", row.get("fini_adjusted_coverage_days")),
                ["members.fini_wheel_roc_days", "members.pallets_per_batch"]),
        }
    complete = rows and all(item[0] and item[2] is not None and item[3] is not None for item in rows)
    total = sum(item[0] for item in rows) if complete else None
    wheel_mean = sum(d * w for d, _, w, _ in rows) / total if complete and total else None
    inventory_mean = sum(d * i for d, _, _, i in rows) / total if complete and total else None
    worst = max(item[3] for item in rows) if complete else None
    fills = sum(1 / item[1] for item in rows) if rows else None
    j_ch = frequency * (fills - 1) if frequency is not None and fills is not None else None
    basis = group.get("active_coverage_basis") or config.get("coverage_basis")
    active = {"BASE_GROUP": wheel_mean, "ADJUSTED_GROUP": inventory_mean, "WORST_FINI": worst}.get(basis)
    group_derivations = {
        "wheel_roc_days": derivation(
            "max(1, ROUND(base_group_coverage_days, 0))", {"base_group_coverage_days": base},
            roc, group.get("wheel_roc_days"), ["groups.base_group_coverage_days"]),
        "frequency_per_week": derivation(
            "(demand_days / productive_weeks) / wheel_roc_days",
            {"demand_days": days, "productive_weeks": weeks, "wheel_roc_days": roc},
            frequency, group.get("frequency_per_week"),
            ["metadata.config.demand_days", "metadata.config.productive_weeks", "groups.wheel_roc_days"]),
        "j_ch_contribution": derivation(
            "frequency_per_week * (sum(1 / runner_multiple) - 1)",
            {"frequency_per_week": frequency, "runner_multiples": [item[1] for item in rows]},
            j_ch, group.get("j_ch_contribution"), ["groups.frequency_per_week", "members.runner_multiple"]),
        "wheel_coverage_days": derivation(
            "sum(member_demand * fini_wheel_roc_days) / group demand",
            {"member_wheel_roc_days": [item[2] for item in rows]},
            wheel_mean, group.get("wheel_coverage_days"), ["members.fini_wheel_roc_days"]),
        "adjusted_group_coverage_days": derivation(
            "sum(member_demand * fini_inventory_roc_days) / group demand",
            {"member_inventory_roc_days": [item[3] for item in rows]},
            inventory_mean, group.get("adjusted_group_coverage_days"), ["members.fini_inventory_roc_days"]),
        "worst_fini_coverage_days": derivation(
            "max(fini_inventory_roc_days)", {"member_inventory_roc_days": [item[3] for item in rows]},
            worst, group.get("worst_fini_coverage_days"), ["members.fini_inventory_roc_days"]),
        "coverage_days": derivation(
            "BASE_GROUP: wheel_coverage_days; ADJUSTED_GROUP: adjusted_group_coverage_days; "
            "WORST_FINI: worst_fini_coverage_days",
            {"active_coverage_basis": basis}, active, group.get("coverage_days"),
            ["groups.active_coverage_basis", "groups.coverage_days"]),
    }
    return group_derivations, per_member
