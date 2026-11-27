"""Source-, modeled-, and assigned-population comparisons with explicit proof evidence."""

from __future__ import annotations

from typing import Any, Mapping

from .analytics_evidence import _number, _point_rows, _solution, _config

METRICS = (
    "demand_weighted_mean_coverage_days",
    "maximum_coverage_days",
    "p90_coverage_days",
    "median_coverage_days",
    "j_ch",
    "group_count",
    "singleton_group_count",
    "modeled_fini_count",
    "modeled_demand_litres",
    "matrix_exception_pair_count",
)


def _members_by_key(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the comparison population by exact plant/material without lossy joins."""
    index = {}
    for row in rows:
        key = (
            str(row.get("plant", "")),
            str(row.get("material", row.get("fini_id", ""))),
        )
        if key in index:
            raise ValueError(f"ambiguous plant/material population key: {key}")
        index[key] = row
    return index


def _membership(row: Mapping[str, Any]) -> tuple[str, ...] | str | None:
    """Identify membership independently of local subgroup numbering."""
    members = row.get("proposed_members")
    if members:
        return tuple(sorted(str(members).split("|")))
    return row.get("proposed_membership_hash") or None


def _differences(
    left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Return explicit nested request/config differences with both source values."""
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        field = f"{prefix}.{key}" if prefix else key
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            rows.extend(_differences(a, b, field))
        elif a != b:
            rows.append({"field": field, "left": a, "right": b})
    return rows


def _assigned(row: Mapping[str, Any]) -> bool:
    """Recognize an actual proposed assignment independently of source admission."""
    return bool(row.get("proposed_subgroup") or row.get("group_id"))


def _modeled(row: Mapping[str, Any]) -> bool:
    """Prefer explicit run admission, falling back to historical source evidence."""
    status = row.get("run_model_status") or row.get("model_status")
    return status == "modeled" if status else _assigned(row)


def _population(
    left: set[tuple[str, str]], right: set[tuple[str, str]], scope: str
) -> dict[str, Any]:
    """Describe exact population membership without equating source rows to modeled FINIs."""
    return {
        "key": ["plant", "material"],
        "scope": scope,
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(left & right),
        "left_only": [
            {"plant": key[0], "material": key[1]} for key in sorted(left - right)
        ],
        "right_only": [
            {"plant": key[0], "material": key[1]} for key in sorted(right - left)
        ],
    }


def _assessment_flag(value: Any) -> bool | None:
    """Read recorded assessment flags without treating missing evidence as false."""
    if value is True or value is False:
        return value
    if str(value).lower() in {"1", "1.0", "true"}:
        return True
    if str(value).lower() in {"0", "0.0", "false"}:
        return False
    return None


def _recorded_proof(
    summary: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    """Return actual point proof, gap and pool fields plus separately recorded acceptance."""
    fields = (
        "proof_scope",
        "validation_status",
        "acceptance_status",
        "result_class",
        "status",
        "termination_condition",
        "solver_method",
        "candidate_pool_completeness",
        "relative_gap",
        "primary_relative_gap",
        "primary_best_bound",
        "primary_objective",
        "objective_tier_caveats",
        "j_ch_objective_tier_status",
    )
    result = {field: summary.get(field) for field in fields}
    result["evidence_sources"] = {
        field: "solution_summary" for field in fields if field in summary
    }
    for field in ("business_acceptance_assessed", "business_acceptable"):
        value = summary.get(field)
        source = "solution_summary"
        if value is None:
            value = run.get("metadata", {}).get(field)
            source = "run_metadata"
        result[field] = _assessment_flag(value)
        if value is not None:
            result["evidence_sources"][field] = source
    result["gap_scope"] = {
        "primary_relative_gap": "reported primary point objective tier",
        "relative_gap": "reported final point solve; not a full-candidate-space proof",
    }
    return result


def _common_coverage(
    left: Mapping, right: Mapping, keys: set[tuple[str, str]]
) -> dict[str, Any]:
    """Aggregate one observed FINI coverage per common modeled assignment, without group duplication."""
    sides = {}
    complete_sides = {}
    for name, members in (("left", left), ("right", right)):
        values = [
            _number(members[key].get("fini_adjusted_coverage_days"))
            for key in sorted(keys)
        ]
        weights = [
            _number(members[key].get("forecast_litres_12m")) for key in sorted(keys)
        ]
        complete = bool(keys) and all(value is not None for value in values)
        weight_complete = complete and all(
            weight is not None and weight > 0 for weight in weights
        )
        complete_sides[name] = complete
        sides[name] = {
            "observed_member_count": sum(value is not None for value in values),
            "mean_days": sum(values) / len(values) if complete else None,
            "demand_weighted_mean_days": sum(
                value * weight for value, weight in zip(values, weights)
            )
            / sum(weights)
            if weight_complete
            else None,
            "weight_evidence_complete": weight_complete,
            "missing_coverage_members": [
                {"plant": key[0], "material": key[1]}
                for key, value in zip(sorted(keys), values)
                if value is None
            ],
        }
    return {
        "scope": "modeled_and_assigned_population_intersection",
        "population_count": len(keys),
        "coverage_field": "members.fini_adjusted_coverage_days",
        "weight_field": "members.forecast_litres_12m",
        "units": "days",
        "coverage_evidence_complete": all(complete_sides.values()),
        **sides,
        "delta_demand_weighted_mean_days": sides["right"]["demand_weighted_mean_days"]
        - sides["left"]["demand_weighted_mean_days"]
        if all(side["demand_weighted_mean_days"] is not None for side in sides.values())
        else None,
        "weighting_scope": "each side's own annual demand on the identical common material keys",
        "not_a_whole_solution_metric": True,
    }


def compare_solutions(
    repo: Any, left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Compare persisted summaries and exact shared material assignments.

    Inputs identify original run/point pairs. Outputs distinguish whole-solution
    metrics from intersection-only membership/PV changes and disclose configuration
    or population changes; no objective delta is called a business improvement.
    """
    a, b = _solution(repo, left), _solution(repo, right)
    left_run, right_run = (
        repo.get("runs", left["run_id"]),
        repo.get("runs", right["run_id"]),
    )
    left_members = _members_by_key(
        _point_rows(repo, left["run_id"], left["point_index"], "members")
    )
    right_members = _members_by_key(
        _point_rows(repo, right["run_id"], right["point_index"], "members")
    )
    shared = set(left_members) & set(right_members)
    left_modeled = {key for key, row in left_members.items() if _modeled(row)}
    right_modeled = {key for key, row in right_members.items() if _modeled(row)}
    left_assigned = {key for key, row in left_members.items() if _assigned(row)}
    right_assigned = {key for key, row in right_members.items() if _assigned(row)}
    modeled_shared = left_modeled & right_modeled
    assigned_shared = left_assigned & right_assigned
    membership_changes, pv_changes = [], []
    unknown_memberships = []
    for key in sorted(shared):
        lrow, rrow = left_members[key], right_members[key]
        identity = {
            "plant": key[0],
            "material": key[1],
            "in_modeled_intersection": key in modeled_shared,
            "in_assigned_intersection": key in assigned_shared,
        }
        before, after = _membership(lrow), _membership(rrow)
        if before != after or lrow.get("sefi") != rrow.get("sefi"):
            membership_changes.append(
                {
                    **identity,
                    "left_membership": before,
                    "right_membership": after,
                    "left_sefi": lrow.get("sefi"),
                    "right_sefi": rrow.get("sefi"),
                }
            )
        elif before is None and (
            lrow.get("proposed_subgroup") or rrow.get("proposed_subgroup")
        ):
            unknown_memberships.append(identity)
        if lrow.get("selected_pv") != rrow.get("selected_pv"):
            pv_changes.append(
                {
                    **identity,
                    "left_pv": lrow.get("selected_pv"),
                    "right_pv": rrow.get("selected_pv"),
                }
            )
    metrics = []
    for field in METRICS:
        before, after = _number(a.get(field)), _number(b.get(field))
        if field in a or field in b:
            metrics.append(
                {
                    "field": field,
                    "left": before,
                    "right": after,
                    "delta": after - before
                    if before is not None and after is not None
                    else None,
                    "scope": "whole_solution",
                }
            )
    config_differences = _differences(_config(left_run), _config(right_run))
    request_differences = _differences(
        left_run.get("request", {}), right_run.get("request", {})
    )
    caveats = [
        "Whole-solution KPI deltas use each solution's own population; assignment changes use only the exact plant/material intersection.",
        "Proof scope and structural validation do not establish business acceptance.",
    ]
    if left_run.get("dataset_id") != right_run.get("dataset_id"):
        caveats.append(
            "Different dataset snapshots: changed demand, eligibility, or source coverage can affect KPI deltas."
        )
    if set(left_members) != set(right_members):
        caveats.append(
            "Populations differ; whole-solution KPI deltas are not a controlled same-population comparison."
        )
    if left_modeled != right_modeled:
        caveats.append(
            "Modeled populations differ even when preserved source rows are identical; whole-solution KPI deltas are not a same-modeled-population comparison."
        )
    if left_assigned != right_assigned:
        caveats.append(
            "Assigned populations differ; source-intersection changes include admission/removal, while modeled-intersection changes are reported separately."
        )
    if config_differences or request_differences:
        caveats.append(
            "Solve request or configuration differs; inspect the explicit differences before attributing a metric delta."
        )
    if unknown_memberships:
        caveats.append(
            "Some assigned rows lack membership fingerprints; their membership change cannot be assessed."
        )
    return {
        "left": left,
        "right": right,
        "metrics": metrics,
        "population": _population(
            set(left_members), set(right_members), "source_population"
        ),
        "modeled_population": _population(
            left_modeled, right_modeled, "modeled_population"
        ),
        "assigned_population": _population(
            left_assigned, right_assigned, "assigned_population"
        ),
        "common_member_coverage": _common_coverage(
            left_members, right_members, modeled_shared & assigned_shared
        ),
        "change_scopes": {
            "membership_changes": "source_population_intersection",
            "pv_changes": "source_population_intersection",
            "modeled_membership_changes": "modeled_population_intersection",
            "modeled_pv_changes": "modeled_population_intersection",
        },
        "modeled_membership_changes": [
            row for row in membership_changes if row["in_modeled_intersection"]
        ],
        "modeled_pv_changes": [
            row for row in pv_changes if row["in_modeled_intersection"]
        ],
        "membership_changes": membership_changes,
        "pv_changes": pv_changes,
        "unassessed_memberships": unknown_memberships,
        "config_differences": config_differences,
        "request_differences": request_differences,
        "proof": {
            "left": _recorded_proof(a, left_run),
            "right": _recorded_proof(b, right_run),
            "business_acceptance_assessed": False,
            "comparison_business_acceptance_assessed": False,
        },
        "caveats": caveats,
    }
