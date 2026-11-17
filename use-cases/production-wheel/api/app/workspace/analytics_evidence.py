"""Population-aware comparisons and numerical assignment explanations."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from production_wheel.metrics import pallet_allocations
from production_wheel.schemas import PalletFormula


def _number(value: Any) -> float | None:
    """Return a finite numeric source value, or null if unavailable."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _point_rows(
    repo: Any, run_id: str, point_index: int, view: str
) -> list[dict[str, Any]]:
    """Retrieve only rows carrying the original requested point index."""
    return repo.rows(run_id, view, filters={"point_index": int(point_index)})


def _solution(repo: Any, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Require exactly one persisted summary for a run and original point index."""
    if set(reference) != {"run_id", "point_index"}:
        raise ValueError("solution reference requires run_id and point_index")
    rows = _point_rows(
        repo, str(reference["run_id"]), int(reference["point_index"]), "solutions"
    )
    if len(rows) != 1:
        raise ValueError("requested solution point does not exist or is ambiguous")
    return rows[0]


def _config(run: Mapping[str, Any]) -> dict[str, Any]:
    """Read persisted configuration without substituting unrecorded defaults."""
    return run.get("metadata", {}).get("config", {}) or run.get("config", {}) or {}


def _allocations(value: Any) -> dict[str, float]:
    """Decode persisted allocation JSON pairs or maps without evaluating code."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    try:
        return {
            str(key): number
            for key, amount in dict(value or {}).items()
            if (number := _number(amount)) is not None
        }
    except (ValueError, TypeError):
        return {}


def _derivation(
    formula: str,
    inputs: dict[str, Any],
    calculated: float | None,
    reported: Any,
    source_fields: list[str],
) -> dict[str, Any]:
    """Package numerical operands, formula and independent reported-value comparison."""
    observed = _number(reported)
    return {
        "formula": formula,
        "inputs": inputs,
        "calculated": calculated,
        "reported": observed,
        "matches_reported": math.isclose(
            calculated, observed, rel_tol=1e-7, abs_tol=1e-7
        )
        if calculated is not None and observed is not None
        else None,
        "source_fields": source_fields,
    }


def explain_assignment(
    repo: Any,
    run_id: str,
    point_index: int,
    material: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Explain a material or group's actual persisted assignment and KPI operands.

    Returns explicit source rows, PV and allocation evidence, numerical formulas,
    and eligible lines. Missing operands remain unknown; eligible lines are never
    presented as a scheduled assignment, and unselected candidates are not ranked.
    """
    if not material and not group_id:
        raise ValueError("material or group_id is required")
    summary = _solution(repo, {"run_id": run_id, "point_index": point_index})
    members = _point_rows(repo, run_id, point_index, "members")
    groups = _point_rows(repo, run_id, point_index, "groups")
    matched_members = [
        row
        for row in members
        if (not material or str(row.get("material")) == material)
        and (not group_id or row.get("group_id") == group_id)
    ]
    if material and len(matched_members) != 1:
        raise ValueError("material is missing or ambiguous; specify its group_id")
    if material:
        group_id = matched_members[0].get("group_id") or ""
        if not group_id:
            return {
                "run_id": run_id,
                "point_index": point_index,
                "assigned": False,
                "members": matched_members,
                "exclusion_reason": matched_members[0].get(
                    "run_exclusion_reason", matched_members[0].get("exclusion_reason")
                ),
                "evidence": {
                    "source": "members",
                    "model_status": matched_members[0].get(
                        "run_model_status", matched_members[0].get("model_status")
                    ),
                    "source_model_status": matched_members[0].get("model_status"),
                    "source_exclusion_reason": matched_members[0].get(
                        "exclusion_reason"
                    ),
                },
                "derivations": {},
                "caveats": ["This source FINI has no proposed assignment."],
            }
    selected_groups = [row for row in groups if row.get("group_id") == group_id]
    if len(selected_groups) != 1:
        raise ValueError("group is missing or ambiguous")
    group = selected_groups[0]
    group_members = [row for row in members if row.get("group_id") == group_id]
    config = _config(repo.get("runs", run_id))
    effective, demand = (
        _number(group.get("effective_batch_litres")),
        _number(group.get("group_demand_litres")),
    )
    days, weeks, size = (
        _number(config.get("demand_days")),
        _number(config.get("productive_weeks")),
        _number(group.get("group_size")),
    )
    base = (
        days * effective / demand
        if days is not None and effective is not None and demand and demand > 0
        else None
    )
    frequency = (
        demand / (effective * weeks)
        if demand is not None and effective and effective > 0 and weeks and weeks > 0
        else None
    )
    j_ch = (
        frequency * (size - 1) if frequency is not None and size is not None else None
    )
    adjusted = _allocations(group.get("selected_pallet_allocations"))
    adjusted_coverage = (
        days * sum(adjusted.values()) / demand
        if adjusted
        and len(adjusted) == size
        and days is not None
        and demand
        and demand > 0
        else None
    )
    nominal_lot, factor = (
        _number(group.get("nominal_lot_litres")),
        _number(config.get("canonical_factor")),
    )
    derivations = {
        "effective_batch_litres": _derivation(
            "nominal_lot_litres * canonical_factor",
            {"nominal_lot_litres": nominal_lot, "canonical_factor": factor},
            nominal_lot * factor
            if nominal_lot is not None and factor is not None
            else None,
            effective,
            ["groups.nominal_lot_litres", "metadata.config.canonical_factor"],
        ),
        "base_group_coverage_days": _derivation(
            "demand_days * effective_batch_litres / group_demand_litres",
            {
                "demand_days": days,
                "effective_batch_litres": effective,
                "group_demand_litres": demand,
            },
            base,
            group.get("base_group_coverage_days"),
            [
                "metadata.config.demand_days",
                "groups.effective_batch_litres",
                "groups.group_demand_litres",
            ],
        ),
        "adjusted_group_coverage_days": _derivation(
            "demand_days * sum(selected_pallet_allocations) / group_demand_litres",
            {
                "demand_days": days,
                "selected_pallet_allocations": adjusted,
                "group_demand_litres": demand,
            },
            adjusted_coverage,
            group.get("adjusted_group_coverage_days"),
            [
                "metadata.config.demand_days",
                "groups.selected_pallet_allocations",
                "groups.group_demand_litres",
            ],
        ),
        "frequency_per_week": _derivation(
            "group_demand_litres / (effective_batch_litres * productive_weeks)",
            {
                "group_demand_litres": demand,
                "effective_batch_litres": effective,
                "productive_weeks": weeks,
            },
            frequency,
            group.get("frequency_per_week"),
            [
                "groups.group_demand_litres",
                "groups.effective_batch_litres",
                "metadata.config.productive_weeks",
            ],
        ),
        "j_ch_contribution": _derivation(
            "frequency_per_week * (group_size - 1)",
            {"frequency_per_week": frequency, "group_size": size},
            j_ch,
            group.get("j_ch_contribution"),
            ["groups.frequency_per_week", "groups.group_size"],
        ),
    }
    member_derivations = []
    for row in group_members:
        member_demand = _number(row.get("forecast_litres_12m"))
        allocation = _number(row.get("selected_pallet_allocation_litres"))
        pallet = _number(row.get("pallet_litres_resolved"))
        nominal = (
            effective * member_demand / demand
            if effective is not None
            and member_demand is not None
            and demand
            and demand > 0
            else None
        )
        formula = config.get("pallet_formula")
        rounded = None
        if (
            nominal
            and nominal > 0
            and pallet
            and pallet > 0
            and formula in {item.value for item in PalletFormula}
        ):
            rounded = float(
                pallet_allocations(
                    {"member": nominal}, {"member": pallet}, PalletFormula(formula)
                )["member"]
            )
        member_derivations.append(
            {
                "material": row.get("material"),
                "pallet_allocation": _derivation(
                    "max(nominal_allocation, pallet_litres)"
                    if formula == "MINIMUM_ONLY"
                    else "ceil(nominal_allocation / pallet_litres) * pallet_litres"
                    if formula == "WHOLE_PALLET_ROUNDING"
                    else "unavailable pallet rule",
                    {
                        "nominal_allocation": nominal,
                        "pallet_litres": pallet,
                        "pallet_formula": formula,
                    },
                    rounded,
                    allocation,
                    [
                        "groups.effective_batch_litres",
                        "members.forecast_litres_12m",
                        "members.pallet_litres_resolved",
                        "metadata.config.pallet_formula",
                    ],
                ),
                "nominal_allocation": _derivation(
                    "effective_batch_litres * member_demand / group_demand_litres",
                    {
                        "effective_batch_litres": effective,
                        "member_demand": member_demand,
                        "group_demand_litres": demand,
                    },
                    effective * member_demand / demand
                    if effective is not None and member_demand is not None and demand
                    else None,
                    row.get("nominal_allocation_litres"),
                    [
                        "groups.effective_batch_litres",
                        "members.forecast_litres_12m",
                        "groups.group_demand_litres",
                    ],
                ),
                "adjusted_coverage": _derivation(
                    "demand_days * selected_pallet_allocation_litres / member_demand",
                    {
                        "demand_days": days,
                        "selected_pallet_allocation_litres": allocation,
                        "member_demand": member_demand,
                    },
                    days * allocation / member_demand
                    if days is not None
                    and allocation is not None
                    and member_demand
                    and member_demand > 0
                    else None,
                    row.get("fini_adjusted_coverage_days"),
                    [
                        "metadata.config.demand_days",
                        "members.selected_pallet_allocation_litres",
                        "members.forecast_litres_12m",
                    ],
                ),
                "selected_pallet_count": _derivation(
                    "selected_pallet_allocation_litres / pallet_litres_resolved",
                    {
                        "selected_pallet_allocation_litres": allocation,
                        "pallet_litres_resolved": pallet,
                    },
                    allocation / pallet
                    if allocation is not None and pallet and pallet > 0
                    else None,
                    row.get("selected_pallet_count"),
                    [
                        "members.selected_pallet_allocation_litres",
                        "members.pallet_litres_resolved",
                    ],
                ),
            }
        )
    member_coverages = [
        item["adjusted_coverage"]["calculated"] for item in member_derivations
    ]
    worst = (
        max(member_coverages)
        if member_coverages
        and len(member_coverages) == size
        and all(value is not None for value in member_coverages)
        else None
    )
    derivations["worst_fini_coverage_days"] = _derivation(
        "max(member adjusted coverage days)",
        {"member_adjusted_coverage_days": member_coverages},
        worst,
        group.get("worst_fini_coverage_days"),
        [
            "members.forecast_litres_12m",
            "members.selected_pallet_allocation_litres",
            "metadata.config.demand_days",
        ],
    )
    basis = group.get("active_coverage_basis") or config.get("coverage_basis")
    active = {
        "BASE_GROUP": base,
        "ADJUSTED_GROUP": adjusted_coverage,
        "WORST_FINI": worst,
    }.get(basis)
    derivations["coverage_days"] = _derivation(
        "select coverage view by active_coverage_basis",
        {"active_coverage_basis": basis},
        active,
        group.get("coverage_days"),
        ["groups.active_coverage_basis", "groups.coverage_days"],
    )
    matrix = [
        row
        for row in _point_rows(repo, run_id, point_index, "matrix_pairs")
        if row.get("group_id") == group_id
    ]
    pools = [
        row
        for row in _point_rows(repo, run_id, point_index, "pools")
        if row.get("plant") == group.get("plant")
        and row.get("sefi") == group.get("sefi")
    ]
    return {
        "run_id": run_id,
        "point_index": point_index,
        "assigned": True,
        "group": group,
        "members": group_members,
        "derivations": derivations,
        "member_derivations": member_derivations,
        "pv": {
            "selected_pv": group.get("selected_pv"),
            "nominal_lot_litres": group.get("nominal_lot_litres"),
            "effective_batch_litres": effective,
            "source": "groups",
        },
        "line_eligibility": {
            "common_lines": str(group.get("common_lines", "")).split("|")
            if group.get("common_lines")
            else [],
            "scheduled_assignment": False,
            "source": "groups.common_lines",
        },
        "evidence": {
            "matrix_pairs": matrix,
            "pools": pools,
            "active_coverage_basis": group.get("active_coverage_basis"),
            "pallet_formula": config.get("pallet_formula"),
            "proof_scope": summary.get("proof_scope"),
            "validation_status": summary.get("validation_status"),
            "business_acceptance_assessed": False,
        },
        "caveats": [
            "Eligible lines describe feasible line overlap; they are not a production schedule.",
            "This explains the selected solution and recorded constraints; it does not prove superiority over every unselected candidate.",
            "J_CH covers recurring within-group FINI changes only; it excludes sequencing and initial or between-group setups.",
        ],
    }
