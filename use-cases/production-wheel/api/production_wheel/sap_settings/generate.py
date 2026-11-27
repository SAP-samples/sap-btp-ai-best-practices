"""Generate every SAP settings tab for one selected frontier point.

Each tab reports a status:
- ``ready``: complete.
- ``partial``: produced, but some rows lack evidence; the message says which
  and how many.
- ``blocked``: not produced; the message names the missing input or policy
  setting.
- ``not_applicable``: does not apply to this plant.

Missing inputs never become invented values.
"""

from __future__ import annotations

import json
import math
from typing import Any

from production_wheel.extraction.common import as_number, as_text
from production_wheel.sap_settings.calendar import planning_calendar_rows
from production_wheel.sap_settings.dc_lot_sizes import dc_lot_size_rows
from production_wheel.sap_settings.kpis import changeover_rows, pv_conflict_rows, wheel_roc
from production_wheel.sap_settings.recipes import apo_rows, character_pool, recipe_rows
from production_wheel.sap_settings.requirements import (
    apo_dependency_gap,
    apo_input_gap,
    central_dc_gap,
    cycle_stock_gaps,
)
from production_wheel.sap_settings.roc import excel_round, runner_adjustments
from production_wheel.runners import CycleStockPolicy, RunnerThresholds

# Run-owned HANA result views written by every generation (and cleared on a new selection).
SAP_RESULT_VIEWS = (
    "sap_dc_lot_sizes",
    "sap_planning_calendar",
    "sap_recipe_combinations",
    "sap_apo_lot_sizes",
    "sap_changeover_kpis",
    "sap_pv_conflicts",
    "sap_runner_adjustments",
)


def _tab(rows: list | None, status: str, message: str = "") -> dict[str, Any]:
    """Summarise one tab for the run payload."""

    return {"status": status, "message": message, "rows": len(rows or [])}


def _same_changeovers(kpi: float | None, point: float | None) -> bool | None:
    """Return whether the KPI tab's changeovers equal the point's ``J_CH``.

    Args:
        kpi: Site-total changeovers per week of the KPI tab.
        point: The selected point's ``J_CH``.

    Returns:
        ``None`` when either value is missing, else equality within 1e-6
        (relative) or 1e-9 (absolute).
    """

    if kpi is None or point is None:
        return None
    return math.isclose(float(kpi), float(point), rel_tol=1e-6, abs_tol=1e-9)


def generate(
    members: list[dict[str, Any]],
    lookups: dict[str, list[dict[str, Any]] | None],
    policy: dict[str, Any],
    selection: dict[str, Any],
    plant: str,
    period_label: str,
    days_per_week: float,
    point_j_ch: float | None = None,
    runner_thresholds: RunnerThresholds | None = None,
    runner_thresholds_source: str = "defaults",
    cycle_stock_policy: CycleStockPolicy | None = None,
    cycle_stock_policy_source: str = "defaults",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build the SAP settings tables and the run summary for the selected point.

    Args:
        members: Every ``members`` row of the selected point.
        lookups: Dataset lookup views by name; ``None`` marks a view the
            dataset does not have.
        policy: Resolved plant policy.
        selection: Stored ``decision_selection`` (chosen point). The cycle
            stock strategy of each subgroup is read from ``members``.
        plant: Producing plant.
        period_label: Planning period for the APO comment (e.g. a quarter).
        days_per_week: Demand days per productive week (template uses 5).
        point_j_ch: The point's ``J_CH`` for the KPI cross-check.
        runner_thresholds: High / medium / low cut-offs frozen with the run's
            plant profile; ``None`` uses the template defaults.
        runner_thresholds_source: Where the cut-offs came from, for the summary.
        cycle_stock_policy: Runner multiple minimums, medium factor and the
            cycle-stock cap frozen with the run's plant profile; ``None`` uses
            the template defaults.
        cycle_stock_policy_source: Where that policy came from, for the summary.

    Returns:
        ``(tables, summary)``. ``tables`` maps ``sap_*`` result views to rows
        carrying ``point_index``. ``summary`` holds per-tab status, wheel RoC,
        KPI vs ``J_CH``, the policy snapshot and integration notes.
    """

    point = int(selection["point_index"])
    wheel_members = [member for member in members if member.get("proposed_subgroup_code")]
    thresholds = runner_thresholds or RunnerThresholds()
    cycle_policy = cycle_stock_policy or CycleStockPolicy()
    # The strategy of every subgroup comes from the selected point (members.cycle_stock_strategy).
    runner = runner_adjustments(wheel_members, cycle_policy, thresholds)
    by_material = {member_row["material"]: member_row for member_row in runner}
    tables: dict[str, list[dict[str, Any]]] = {"sap_runner_adjustments": runner}
    tabs: dict[str, dict[str, Any]] = {}

    conflicts = pv_conflict_rows(wheel_members)
    tables["sap_pv_conflicts"] = conflicts
    tabs["pv_check"] = _tab(conflicts, "partial" if conflicts else "ready",
                            f"{len(conflicts)} subgroups mix production versions" if conflicts else "")

    kpis = changeover_rows(wheel_members, plant, days_per_week, policy["kpi_line_breakdown"] == "yes", runner)
    tables["sap_changeover_kpis"] = kpis
    demand = {as_text(member.get("material")): as_number(member.get("forecast_litres_12m")) for member in wheel_members}
    tabs["kpis"] = _tab(kpis, "ready")

    network, pallets = lookups.get("dc_network"), lookups.get("dc_pallet_quantities")
    blocked = [gap for gap in cycle_stock_gaps(lookups) if gap.status == "blocked"]
    if blocked:
        tables["sap_dc_lot_sizes"], dropped = [], 0
        tabs["cycle_stock"] = _tab([], "blocked", blocked[0].message)
    else:
        wheel = {
            as_text(member.get("material")): {
                "coverage_days": by_material.get(as_text(member.get("material")), {}).get("coverage_days"),
                "pallets_per_subgroup": excel_round(as_number(member.get("legacy_nominal_pallet_count")), 1),
                "sales_scenario": as_text(member.get("sales_scenario")),
                "sales_network_scenario": as_text(member.get("sales_network_scenario")),
            }
            for member in wheel_members
        }
        abc = lookups.get("dc_abc_classes")
        rows, dropped = dc_lot_size_rows(wheel, network, abc or [], pallets, policy, cycle_policy.max_days)
        tables["sap_dc_lot_sizes"] = rows
        notes = [gap.message for gap in cycle_stock_gaps(lookups) if gap.status == "partial"]
        unmapped = sum(row["status"] != "ready" for row in rows)
        if unmapped:
            notes.append(f"{unmapped} rows without an SAP lot-size key")
        if dropped:
            notes.append(f"{dropped} DC links without wheel RoC or ABC class skipped")
        tabs["cycle_stock"] = _tab(rows, "partial" if abc is None or unmapped else "ready", "; ".join(notes))

    calendar, status, message = planning_calendar_rows(members, policy, plant)
    tables["sap_planning_calendar"] = calendar
    tabs["planning_calendar"] = _tab(calendar, status, message)

    pool = character_pool(lookups.get("recipe_characters"), policy)
    recipes, status, message = recipe_rows(wheel_members, pool, plant)
    tables["sap_recipe_combinations"] = recipes
    tabs["recipe_combination"] = _tab(recipes, status, message)

    masses = lookups.get("production_version_masses")
    missing_masses = apo_input_gap(lookups)
    if status != "ready":
        tables["sap_apo_lot_sizes"] = []
        tabs["apo_lot_size"] = _tab([], "blocked", apo_dependency_gap().message)
    elif missing_masses:
        tables["sap_apo_lot_sizes"] = []
        tabs["apo_lot_size"] = _tab([], "blocked", missing_masses.message)
    else:
        apo = apo_rows(wheel_members, recipes, masses, policy, period_label)
        tables["sap_apo_lot_sizes"] = apo
        missing = sum(row["status"] != "ready" for row in apo)
        tabs["apo_lot_size"] = _tab(apo, "partial" if missing else "ready",
                                    f"{missing} rows without a KG lot size" if missing else "")

    central = central_dc_gap(policy)
    tabs["central_dc_lot_size"] = _tab([], "blocked", central.message) if central else _tab([], "not_applicable")
    tabs["runner_audit"] = _tab(runner, "ready")

    for rows in tables.values():
        for row in rows:
            row["point_index"] = point
    site_total = next((row for row in kpis if row["level"] == "site_total"), {})
    summary = {
        "point_index": point,
        "decision_label": selection.get("decision_label"),
        "period_label": period_label,
        "tabs": tabs,
        "wheel_roc_days": wheel_roc(runner, demand),
        "kpi_changeovers_per_week": site_total.get("changeovers_per_week"),
        "kpi_batches_per_week": site_total.get("batches_per_week"),
        "point_j_ch": point_j_ch,
        # Both use the wheel RoC R (production_wheel.wheel_roc), so they agree for
        # runs made with that rule; False flags an older run or a data problem.
        "kpi_matches_point_j_ch": _same_changeovers(site_total.get("changeovers_per_week"), point_j_ch),
        "runner_adjusted_fini": sum(row["applied"] for row in runner),
        "runner_multiple_subgroups": sum(row.get("cycle_stock_strategy") == "runner_multiple"
                                         for row in kpis if row["level"] == "subgroup"),
        "kpi_single_changeovers_per_week": site_total.get("single_changeovers_per_week"),
        "days_per_week": days_per_week,
        "policy": json.loads(json.dumps(policy, default=list)),
        "runner_thresholds": thresholds.model_dump(),
        "runner_thresholds_source": runner_thresholds_source,
        "cycle_stock_policy": cycle_policy.model_dump(),
        "cycle_stock_policy_source": cycle_stock_policy_source,
    }
    return tables, summary
