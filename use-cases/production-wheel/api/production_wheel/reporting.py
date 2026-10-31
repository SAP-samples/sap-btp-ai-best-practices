"""Build deterministic CSV, JSON, and Markdown solution evidence.

The functions accept ordinary mappings so extracted CSV rows can be retained
verbatim and in source order. Example::

    validation = validate_solution(result, members, pools, config, versions)
    write_scenario_artifacts(output_dir, fini_rows, members, pools, result,
                             config, validation)
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from production_wheel.candidates import CandidateMember, CandidatePool, ProductionVersion
from production_wheel.metrics import (
    changeover_contribution,
    effective_batch,
    group_coverage,
    group_frequency,
    pallet_allocations,
    proportional_allocations,
    coverage_summary,
    target_band_key,
)
from production_wheel.matrix import (
    UNKNOWN_LINE_FEASIBLE,
    UNKNOWN_NO_CURRENT_LINE_OVERLAP,
    configured_matrix,
    configured_matrix_version,
    matrix_exceptions,
    matrix_status_lookup,
    normalize_volume,
)
from production_wheel.optimization import SolveResult
from production_wheel.schemas import (
    CoverageBasis,
    CoverageMode,
    MatrixMode,
    PVMode,
    RunConfig,
)
from production_wheel.solution_validation import (
    GroupEvidence,
    ValidationResult,
)


@dataclass(frozen=True, slots=True)
class ScenarioArtifactPaths:
    """Paths emitted for one scenario result bundle."""

    proposed_subgroups: Path
    solution_groups: Path
    scenario_summary: Path
    matrix_exception_audit: Path
    constraint_audit: Path
    candidate_pool_summary: Path
    validation_issues: Path
    run_manifest: Path
    solution_report: Path


def _material(row: Mapping[str, Any]) -> str:
    """Return the canonical FINI identifier from an extracted source row."""

    return str(row.get("material", row.get("fini_id", "")))


def _json(value: Any) -> str:
    """Serialize structured CSV evidence with stable compact formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] = (),
) -> None:
    """Write deterministic UTF-8 CSV rows using first-seen field order."""

    ordered_fields = list(fields)
    seen = set(ordered_fields)
    for row in rows:
        for field in row:
            if field not in seen:
                ordered_fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one emitted artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_proposed_subgroups_rows(
    fini_rows: Iterable[Mapping[str, Any]],
    members: Iterable[CandidateMember],
    groups: Iterable[GroupEvidence],
    result: SolveResult,
    config: RunConfig,
) -> tuple[dict[str, Any], ...]:
    """Build the primary-workbook-like output while preserving every source row.

    Args:
        fini_rows: Complete canonical FINI master in original primary-workbook row order.
        members: Modeled FINI primitives used by the optimizer.
        groups: Independently recomputed selected-group evidence.
        result: Solver proof metadata to repeat on modeled proposal rows.
        config: Scenario and configuration identifiers and active settings.

    Returns:
        One output row per input row. Excluded and out-of-scope rows retain all
        baseline fields but receive blank proposed subgroup fields.
    """

    materialized_groups = tuple(groups)
    member_index = {
        (member.plant, member.sefi, member.fini_id): member for member in members
    }
    group_by_member = {
        (*group.block_key, fini_id): group
        for group in materialized_groups
        for fini_id in group.member_ids
    }
    subgroup_codes: dict[str, str] = {}
    by_block: dict[tuple[str, str], list[GroupEvidence]] = {}
    for group in materialized_groups:
        by_block.setdefault(group.block_key, []).append(group)
    for block in sorted(by_block):
        for index, group in enumerate(
            sorted(by_block[block], key=lambda item: item.member_ids), start=1
        ):
            subgroup_codes[group.group_id] = f"G{index:03d}"
    rows: list[dict[str, Any]] = []
    for source in fini_rows:
        row = dict(source)
        if config.pv_mode is PVMode.OPTIMIZED:
            row["fixed_pv_model_status"] = source.get("model_status", "")
            row["fixed_pv_exclusion_reason"] = source.get("exclusion_reason", "")
            row["model_status"] = source.get(
                "optimized_pv_model_status", source.get("model_status", "")
            )
            row["exclusion_reason"] = source.get(
                "optimized_pv_exclusion_reason", source.get("exclusion_reason", "")
            )
        key = (
            str(source.get("plant", "")),
            str(source.get("sefi", "")),
            _material(source),
        )
        modeled = key in member_index
        group = group_by_member.get(key) if modeled else None
        member = member_index.get(key)
        nominal = dict(group.nominal_allocations).get(key[2]) if group else None
        selected = dict(group.selected_allocations).get(key[2]) if group else None
        fini_adjusted_coverage = (
            dict(group.fini_adjusted_coverage_days).get(key[2]) if group else None
        )
        pallet = member.pallet_litres if member else None
        row.update(
            {
                "scenario_id": config.scenario_id,
                "configuration_id": config.configuration_id(),
                "proposed_subgroup": group.group_id if group else "",
                "proposed_subgroup_code": subgroup_codes.get(group.group_id, "") if group else "",
                "proposed_membership_hash": group.membership_hash if group else "",
                "proposed_members": "|".join(group.member_ids) if group else "",
                "selected_pv": group.pv_id if group else "",
                "fixed_pv_evidence": member.fixed_pv if member and modeled else "",
                "nominal_lot_litres": group.nominal_lot_litres if group else "",
                "effective_batch_litres": group.effective_batch_litres if group else "",
                "group_demand_litres": group.group_demand_litres if group else "",
                "group_size": group.group_size if group else "",
                "common_lines": "|".join(group.common_lines) if group else "",
                "selected_line": group.selected_line if group else None,
                "active_coverage_basis": config.coverage_basis.value if group else "",
                "coverage_basis": config.coverage_basis.value if group else "",
                "coverage_days": group.coverage_days if group else "",
                "base_group_coverage_days": (
                    group.base_group_coverage_days if group else ""
                ),
                "adjusted_group_coverage_days": (
                    group.adjusted_group_coverage_days if group else ""
                ),
                "worst_fini_coverage_days": (
                    group.worst_fini_coverage_days if group else ""
                ),
                "fini_adjusted_coverage_days": (
                    fini_adjusted_coverage
                    if fini_adjusted_coverage is not None
                    else ""
                ),
                "frequency_per_week": group.frequency_per_week if group else "",
                "nominal_allocation_litres": nominal if nominal is not None else "",
                "legacy_nominal_lot_allocation_litres": (
                    group.nominal_lot_litres * member.demand_litres / group.group_demand_litres
                    if group and member
                    else ""
                ),
                "legacy_nominal_pallet_count": (
                    group.nominal_lot_litres
                    * member.demand_litres
                    / group.group_demand_litres
                    / pallet
                    if group and member and pallet
                    else ""
                ),
                "selected_pallet_allocation_litres": selected if selected is not None else "",
                "selected_pallet_count": selected / pallet if selected is not None and pallet else "",
                "legacy_nominal_lot_pallet_diagnostic": source.get(
                    "legacy_pallets_per_subgroup", ""
                ),
                "pallet_formula": config.pallet_formula.value if group else "",
                "j_ch_contribution": group.j_ch_contribution if group else "",
                "package_volume_evidence": member.package_volume if member and modeled else "",
                "group_package_volumes": _json(group.package_volumes) if group else "",
                "pck_evidence": member.pck_code if member and modeled else "",
                "equal_pck_diagnostic": int(group.equal_pck) if group else "",
                "matrix_pair_evidence": _json(group.matrix_pair_evidence) if group else "",
                "matrix_exception_pairs": _json(group.matrix_exception_pairs) if group else "",
                "matrix_unknown_pairs": _json(group.matrix_unknown_pairs) if group else "",
                "matrix_positive_evidence_pairs": (
                    _json(group.matrix_positive_evidence_pairs) if group else ""
                ),
                "matrix_version": configured_matrix_version(config) if group else "",
                "group_size_relaxed": int(group.relaxed_group) if group else "",
                "group_size_excess": group.size_excess if group else "",
                "baseline_change_indicator": int(group.baseline_changed) if group else "",
                "pv_change_indicator": int(group.pv_changed) if group else "",
                "solver_method": result.solver_method if group else "",
                "result_class": result.result_class if group else "",
                "relative_gap": result.relative_gap if group and result.relative_gap is not None else "",
                "primary_objective": result.primary_objective if group else "",
                "primary_best_bound": result.primary_best_bound if group else "",
                "primary_relative_gap": (
                    result.primary_relative_gap
                    if group and result.primary_relative_gap is not None
                    else ""
                ),
                "candidate_pool_completeness": result.pool_completeness if group else "",
            }
        )
        rows.append(row)
    return tuple(rows)


def build_solution_groups_rows(
    groups: Iterable[GroupEvidence], result: SolveResult, config: RunConfig
) -> tuple[dict[str, Any], ...]:
    """Build one audit row per independently recomputed selected subgroup."""

    return tuple(
        {
            "scenario_id": config.scenario_id,
            "configuration_id": config.configuration_id(),
            "proposed_subgroup": group.group_id,
            "plant": group.block_key[0],
            "sefi": group.block_key[1],
            "members": "|".join(group.member_ids),
            "membership_hash": group.membership_hash,
            "group_size": group.group_size,
            "selected_pv": group.pv_id,
            "nominal_lot_litres": group.nominal_lot_litres,
            "effective_batch_litres": group.effective_batch_litres,
            "group_demand_litres": group.group_demand_litres,
            "common_lines": "|".join(group.common_lines),
            "selected_line": group.selected_line,
            "active_coverage_basis": config.coverage_basis.value,
            "coverage_days": group.coverage_days,
            "base_group_coverage_days": group.base_group_coverage_days,
            "adjusted_group_coverage_days": group.adjusted_group_coverage_days,
            "worst_fini_coverage_days": group.worst_fini_coverage_days,
            "fini_adjusted_coverage_days": _json(
                group.fini_adjusted_coverage_days
            ),
            "frequency_per_week": group.frequency_per_week,
            "j_ch_contribution": group.j_ch_contribution,
            "nominal_allocations": _json(group.nominal_allocations),
            "selected_pallet_allocations": _json(group.selected_allocations),
            "package_volumes": _json(group.package_volumes),
            "pck_codes": "|".join(group.pck_codes),
            "equal_pck_diagnostic": int(group.equal_pck),
            "matrix_pair_evidence": _json(group.matrix_pair_evidence),
            "matrix_exception_pairs": _json(group.matrix_exception_pairs),
            "matrix_unknown_pairs": _json(group.matrix_unknown_pairs),
            "matrix_positive_evidence_pairs": _json(
                group.matrix_positive_evidence_pairs
            ),
            "matrix_version": configured_matrix_version(config),
            "group_size_relaxed": int(group.relaxed_group),
            "group_size_excess": group.size_excess,
            "baseline_change_indicator": int(group.baseline_changed),
            "pv_change_indicator": int(group.pv_changed),
            "solver_method": result.solver_method,
            "result_class": result.result_class,
            "relative_gap": result.relative_gap,
            "primary_objective": result.primary_objective,
            "primary_best_bound": result.primary_best_bound,
            "primary_relative_gap": result.primary_relative_gap,
            "candidate_pool_completeness": result.pool_completeness,
        }
        for group in sorted(groups, key=lambda item: (item.block_key, item.member_ids))
    )


def _family_evidence(row: Mapping[str, Any], side: str) -> str:
    """Return stable family evidence from either legacy or customer matrix rows."""

    value = row.get(f"families_{side}", row.get(f"family_{side}", ""))
    if isinstance(value, (tuple, list, set, frozenset)):
        return "|".join(sorted(str(item) for item in value))
    return str(value)


def build_matrix_exception_audit_rows(
    groups: Iterable[GroupEvidence],
    config: RunConfig,
    members: Iterable[CandidateMember],
    baseline_evidence_members: Iterable[CandidateMember] = (),
) -> tuple[dict[str, Any], ...]:
    """Build one row per selected explicit ``AVOID`` or ``NO`` material pair.

    Args:
        groups: Independently recomputed selected-group evidence.
        config: Active matrix version and enforcement mode.
        members: Modeled FINIs used to build the configured matrix.
        baseline_evidence_members: Historical assignments needed by empirical
            matrix versions.

    Returns:
        Deterministically ordered exception rows with material, volume, status,
        version, and source-family evidence.
    """

    selected = tuple(groups)
    if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs:
        return ()
    population = tuple(members)
    matrix_rows = configured_matrix(
        config,
        population,
        tuple(baseline_evidence_members) or population,
    )
    matrix_by_pair = {
        (normalize_volume(row["volume_a"]), normalize_volume(row["volume_b"])): row
        for row in matrix_rows
    }
    rows: list[dict[str, Any]] = []
    for group in sorted(selected, key=lambda item: (item.block_key, item.member_ids)):
        for material_a, material_b, volume_a, volume_b, status in group.matrix_pair_evidence:
            if status not in {"AVOID", "NO", "N"}:
                continue
            source = matrix_by_pair[
                (normalize_volume(volume_a), normalize_volume(volume_b))
            ]
            rows.append(
                {
                    "scenario_id": config.scenario_id,
                    "configuration_id": config.configuration_id(),
                    "plant": group.block_key[0],
                    "sefi": group.block_key[1],
                    "proposed_subgroup": group.group_id,
                    "material_a": material_a,
                    "material_b": material_b,
                    "volume_a": volume_a,
                    "volume_b": volume_b,
                    "status": status,
                    "matrix_version": configured_matrix_version(config),
                    "family_evidence_a": _family_evidence(source, "a"),
                    "family_evidence_b": _family_evidence(source, "b"),
                    "rationale": source.get("rationale", ""),
                }
            )
    return tuple(rows)


def build_candidate_pool_summary_rows(
    pools: Iterable[CandidatePool],
    config: RunConfig,
    result: SolveResult | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build pool, convergence, and per-block solver-tier evidence rows."""

    block_evidence = {
        evidence.block_key: evidence
        for evidence in (() if result is None else result.block_evidence)
    }

    return tuple(
        {
            "scenario_id": config.scenario_id,
            "configuration_id": config.configuration_id(),
            "plant": pool.block_key[0],
            "sefi": pool.block_key[1],
            "method": pool.method,
            "completeness": pool.completeness,
            "candidate_count": len(pool.candidates),
            "pool_hash": pool.pool_hash,
            "structural_fingerprint": pool.structural_fingerprint,
            "config_fingerprint": pool.config_fingerprint,
            "member_count": pool.projection.members,
            "maximum_size": pool.projection.maximum_size,
            "projected_member_subsets": pool.projection.line_feasible_member_subsets,
            "finite_pv_alternatives": pool.projection.finite_pv_alternatives,
            "projected_pv_configurations": pool.projection.projected_pv_configurations,
            "mandatory_member_sets": pool.mandatory_member_sets,
            "mandatory_pv_configurations": pool.mandatory_pv_configurations,
            "candidate_matrix_exception_pair_instances": sum(
                len(candidate.matrix_exception_pairs)
                for candidate in pool.candidates
            ),
            "candidate_matrix_unknown_pair_instances": sum(
                len(candidate.matrix_unknown_pairs) for candidate in pool.candidates
            ),
            "candidate_matrix_positive_pair_instances": sum(
                len(candidate.matrix_evidence_pairs) for candidate in pool.candidates
            ),
            "block_solver_status": (
                block_evidence[pool.block_key].status
                if pool.block_key in block_evidence
                else ""
            ),
            "block_solver_termination": (
                block_evidence[pool.block_key].termination_condition
                if pool.block_key in block_evidence
                else ""
            ),
            "block_result_class": (
                block_evidence[pool.block_key].result_class
                if pool.block_key in block_evidence
                else ""
            ),
            "block_primary_objective": (
                block_evidence[pool.block_key].primary_objective
                if pool.block_key in block_evidence
                else ""
            ),
            "block_primary_best_bound": (
                block_evidence[pool.block_key].primary_best_bound
                if pool.block_key in block_evidence
                else ""
            ),
            "block_primary_relative_gap": (
                block_evidence[pool.block_key].primary_relative_gap
                if pool.block_key in block_evidence
                else ""
            ),
            "block_max_reached_tier_gap": (
                block_evidence[pool.block_key].relative_gap
                if pool.block_key in block_evidence
                else ""
            ),
            "block_objective_levels": _json(
                [
                    {
                        "name": level.name,
                        "value": level.value,
                        "termination": level.termination_condition,
                        "best_bound": level.best_bound,
                        "relative_gap": level.relative_gap,
                        "wallclock_seconds": level.wallclock_seconds,
                    }
                    for level in block_evidence[pool.block_key].objective_levels
                ]
                if pool.block_key in block_evidence
                else []
            ),
            "nested_size_traces": _json(
                [
                    {
                        "size": trace.size,
                        "expanded_member_sets": trace.expanded_member_sets,
                        "feasible_member_sets": trace.feasible_member_sets,
                        "shortlisted_member_sets": trace.shortlisted_member_sets,
                        "selected_member_sets": trace.selected_member_sets,
                        "selected_pv_configurations": trace.selected_pv_configurations,
                        "cumulative_pv_configurations": trace.cumulative_pv_configurations,
                        "truncated": trace.truncated,
                    }
                    for trace in pool.size_traces
                ]
            ),
        }
        for pool in sorted(pools, key=lambda item: item.block_key)
    )


def build_constraint_audit_rows(
    config: RunConfig, constraints: Iterable[Any] = ()
) -> tuple[dict[str, Any], ...]:
    """Build active configuration rules plus allowlisted typed-constraint evidence."""

    settings = (
        ("coverage_mode", config.coverage_mode.value),
        ("coverage_basis", config.coverage_basis.value),
        ("pallet_formula", config.pallet_formula.value),
        ("group_size", config.group_size.model_dump(mode="json")),
        ("pv_mode", config.pv_mode.value),
        ("matrix_mode", config.matrix_mode.value),
        ("matrix_version", configured_matrix_version(config)),
        ("target_band", config.target_band.model_dump(mode="json")),
        (
            "baseline_guardrails",
            config.baseline_guardrails.model_dump(mode="json"),
        ),
        ("canonical_factor", config.canonical_factor),
        ("demand_days", config.demand_days),
        ("productive_weeks", config.productive_weeks),
        ("matrix_pairs", [pair.model_dump(mode="json") for pair in config.matrix_pairs]),
        ("assign_filling_lines", config.assign_filling_lines),
        ("high_runner_threshold_days", config.high_runner_threshold_days),
        ("runner_basis", config.runner_basis),
    )
    rows = [
        {
            "scenario_id": config.scenario_id,
            "constraint_id": f"config.{name}",
            "kind": "run_config",
            "approval_status": "active",
            "enforcement": "scenario_setting",
            "scope": "{}",
            "parameters": _json(value),
            "source_text": "",
        }
        for name, value in settings
    ]
    rule_by_id = {rule.constraint_id: rule for rule in (*tuple(constraints), *config.business_rules)}
    for constraint in sorted(
        rule_by_id.values(), key=lambda item: str(getattr(item, "constraint_id", ""))
    ):
        data = constraint.model_dump(mode="json")
        scope = data.pop("scope", {})
        source_text = data.pop("source_text", "")
        rows.append(
            {
                "scenario_id": config.scenario_id,
                "constraint_id": data.pop("constraint_id"),
                "kind": data.pop("kind"),
                "approval_status": data.pop("approval_status"),
                "enforcement": data.pop("enforcement"),
                "scope": _json(scope),
                "parameters": _json(data),
                "source_text": source_text or "",
            }
        )
    return tuple(rows)


def build_scenario_summary_row(
    groups: Iterable[GroupEvidence],
    result: SolveResult,
    config: RunConfig,
    validation: ValidationResult,
    baseline_summary: Mapping[str, Any] | None = None,
    acceptance_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the required portfolio KPIs and optional baseline deltas."""

    selected = tuple(groups)
    coverages = [group.coverage_days for group in selected]
    demands = [group.group_demand_litres for group in selected]
    summary = coverage_summary(coverages, demands) if selected else None
    band = target_band_key(coverages, demands, config.target_band) if selected else None
    coverage_views = {
        "base_group": [group.base_group_coverage_days for group in selected],
        "adjusted_group": [group.adjusted_group_coverage_days for group in selected],
        "worst_fini": [group.worst_fini_coverage_days for group in selected],
    }
    objective_tiers = [
        {
            "name": level.name,
            "value": level.value,
            "termination": level.termination_condition,
            "relative_gap": level.relative_gap,
        }
        for level in result.objective_levels
    ]
    tier_caveats = [
        level.name
        for level in result.objective_levels
        if level.termination_condition
        not in {
            "optimal",
            "decomposed_optimal",
            "not_required",
            "deterministic_selection",
            "recalculated",
        }
    ]
    j_ch_tier = next(
        (level for level in result.objective_levels if level.name == "j_ch"), None
    )
    exception_groups = tuple(
        group for group in selected if group.matrix_exception_pairs
    )
    exception_volume_pairs = {
        tuple(
            sorted(
                (
                    normalize_volume(volume_a),
                    normalize_volume(volume_b),
                )
            )
        )
        for group in exception_groups
        for _, _, volume_a, volume_b in group.matrix_exception_pairs
    }
    baseline_limits = (
        asdict(result.baseline_constraint_limits)
        if result.baseline_constraint_limits is not None
        else {}
    )
    row: dict[str, Any] = {
        "scenario_id": config.scenario_id,
        "configuration_id": config.configuration_id(),
        "status": result.status,
        "termination_condition": result.termination_condition,
        "result_class": result.result_class,
        "solver_method": result.solver_method,
        "relative_gap": result.relative_gap,
        "primary_objective": result.primary_objective,
        "primary_best_bound": result.primary_best_bound,
        "primary_relative_gap": result.primary_relative_gap,
        "objective_levels": _json(objective_tiers),
        "objective_tier_caveats": "|".join(tier_caveats),
        "j_ch_objective_tier_status": (
            j_ch_tier.termination_condition if j_ch_tier else "not_applicable"
        ),
        "candidate_pool_completeness": result.pool_completeness,
        "candidate_count": result.candidate_count,
        "baseline_constraints_applied": int(bool(baseline_limits)),
        "baseline_constraint_scope": result.baseline_constraint_scope,
        "warm_start_kind": result.warm_start_kind,
        "warm_start_group_count": result.warm_start_group_count,
        "validation_status": "valid" if validation.is_valid else "invalid",
        "validation_error_count": sum(issue.severity == "error" for issue in validation.issues),
        "validation_warning_count": sum(issue.severity == "warning" for issue in validation.issues),
        "group_count": len(selected),
        "singleton_group_count": sum(group.group_size == 1 for group in selected),
        "modeled_fini_count": sum(group.group_size for group in selected),
        "modeled_demand_litres": sum(demands),
        "coverage_mode": config.coverage_mode.value,
        "workflow": "greenfield" if config.is_greenfield else "baseline_comparison",
        "epsilon_j_ch": result.epsilon_j_ch,
        "active_coverage_basis": config.coverage_basis.value,
        "matrix_mode": config.matrix_mode.value,
        "matrix_version": configured_matrix_version(config),
        "matrix_comparison_basis": (
            "migration_zero_exception"
            if configured_matrix_version(config)
            == "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
            and config.matrix_mode is MatrixMode.HARD
            else "greenfield_operational_families"
            if config.is_greenfield
            and configured_matrix_version(config)
            == "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
            and config.matrix_mode is MatrixMode.FLEXIBLE
            else "baseline_comparable_flexible"
            if configured_matrix_version(config)
            == "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
            and config.matrix_mode is MatrixMode.FLEXIBLE
            else "standard"
        ),
        "maximum_coverage_days": summary.maximum if summary else "",
        "p90_coverage_days": summary.p90 if summary else "",
        "median_coverage_days": summary.median if summary else "",
        "demand_weighted_mean_coverage_days": summary.demand_weighted_mean if summary else "",
        "group_mean_coverage_days": summary.group_mean if summary else "",
        "target_violation_count": band.violation_count if band else "",
        "target_worst_excess_days": band.worst_excess_days if band else "",
        "target_total_excess_days": band.total_excess_days if band else "",
        "j_ch": sum(group.j_ch_contribution for group in selected),
        "relaxed_group_count": sum(group.relaxed_group for group in selected),
        "extra_fini_count": sum(group.size_excess for group in selected),
        "matrix_exception_pair_count": sum(len(group.matrix_exception_pairs) for group in selected),
        "matrix_exception_group_count": len(exception_groups),
        "matrix_exception_distinct_volume_pair_count": len(exception_volume_pairs),
        "matrix_unknown_pair_count": sum(
            len(group.matrix_unknown_pairs) for group in selected
        ),
        "matrix_positive_evidence_pair_count": sum(
            len(group.matrix_positive_evidence_pairs) for group in selected
        ),
        "pallet_adjusted_group_count": sum(
            abs(sum(value for _, value in group.selected_allocations) - group.effective_batch_litres) > 1e-7
            for group in selected
        ),
        "pallet_adjustment_litres": sum(
            sum(value for _, value in group.selected_allocations) - group.effective_batch_litres
            for group in selected
        ),
        "selected_pvs": "|".join(sorted({group.pv_id for group in selected})),
        "changed_group_count": sum(group.baseline_changed for group in selected),
        "pv_changed_group_count": sum(group.pv_changed for group in selected),
    }
    row.update(
        {
            f"in_model_limit_{name}": value
            for name, value in baseline_limits.items()
        }
    )
    for name, values in coverage_views.items():
        if not values:
            continue
        view_summary = coverage_summary(values, demands)
        view_band = target_band_key(values, demands, config.target_band)
        row.update(
            {
                f"{name}_maximum_coverage_days": view_summary.maximum,
                f"{name}_p90_coverage_days": view_summary.p90,
                f"{name}_median_coverage_days": view_summary.median,
                f"{name}_demand_weighted_mean_coverage_days": (
                    view_summary.demand_weighted_mean
                ),
                f"{name}_group_mean_coverage_days": view_summary.group_mean,
                f"{name}_target_violation_count": view_band.violation_count,
                f"{name}_target_worst_excess_days": view_band.worst_excess_days,
                f"{name}_target_total_excess_days": view_band.total_excess_days,
            }
        )
    fini_coverages = [
        coverage
        for group in selected
        for _, coverage in group.fini_adjusted_coverage_days
    ]
    fini_excesses = [
        max(
            config.target_band.lower_days - coverage,
            coverage - config.target_band.upper_days,
            0.0,
        )
        for coverage in fini_coverages
    ]
    row.update(
        {
            "pallet_adjusted_fini_violation_count": sum(
                excess > 0 for excess in fini_excesses
            ),
            "pallet_adjusted_fini_worst_excess_days": max(
                fini_excesses, default=0.0
            ),
        }
    )
    if baseline_summary:
        for name, value in sorted(baseline_summary.items()):
            row[f"baseline_{name}"] = value
            proposal = row.get(name)
            if isinstance(value, (int, float)) and isinstance(proposal, (int, float)):
                row[f"delta_{name}"] = proposal - value
    if acceptance_summary:
        row.update(dict(acceptance_summary))
    return row


def build_baseline_summary(
    members: Iterable[CandidateMember],
    production_versions: Iterable[ProductionVersion],
    config: RunConfig,
    baseline_evidence_members: Iterable[CandidateMember] = (),
) -> dict[str, float | int]:
    """Recompute comparable baseline KPIs under the active reporting settings.

    Args:
        members: Modeled FINIs carrying their historical group and fixed PV.
        production_versions: Canonical finite PV catalog used to resolve lots.
        config: Active coverage basis, pallet formula, band, and calendar.
        baseline_evidence_members: Complete historical assignments used only
            to construct empirical positive matrix evidence.

    Returns:
        Baseline KPI mapping suitable for ``build_scenario_summary_row``.

    Raises:
        ValueError: A baseline group lacks a unique common fixed PV or catalog lot.
    """

    population = tuple(members)
    historical_population = tuple(baseline_evidence_members) or population
    grouped: dict[tuple[str, str, str], list[CandidateMember]] = {}
    for member in population:
        if member.baseline_group:
            grouped.setdefault(
                (member.plant, member.sefi, member.baseline_group), []
            ).append(member)
    lots: dict[tuple[str, str, str], set[float]] = {}
    for version in production_versions:
        if version.is_positive:
            lots.setdefault(
                (version.plant, version.sefi, version.pv_id), set()
            ).add(float(version.nominal_lot_litres))  # type: ignore[arg-type]
    coverage_views: dict[CoverageBasis, list[float]] = {
        basis: [] for basis in CoverageBasis
    }
    coverages = coverage_views[config.coverage_basis]
    demands: list[float] = []
    total_j_ch = 0.0
    pallet_adjustment = 0.0
    fini_adjusted_coverages: list[float] = []
    matrix_exception_pairs = 0
    matrix_prohibited_pairs = 0
    matrix_exception_groups = 0
    matrix_exception_volume_pairs: set[tuple[Decimal, Decimal]] = set()
    matrix_unknown_pairs = 0
    matrix_positive_pairs = 0
    matrix_rows = (
        ()
        if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs
        else configured_matrix(
            config,
            population,
            historical_population,
        )
    )
    matrix_lookup = matrix_status_lookup(matrix_rows) if matrix_rows else {}
    relaxed_groups = 0
    size_excess = 0
    for key, group_members in sorted(grouped.items()):
        pvs = {member.fixed_pv for member in group_members}
        if len(pvs) != 1 or None in pvs or "" in pvs:
            raise ValueError(f"baseline group has no unique fixed PV: {key}")
        pv_id = str(next(iter(pvs)))
        catalog_lots = lots.get((key[0], key[1], pv_id), set())
        if len(catalog_lots) != 1:
            raise ValueError(f"baseline group has no unique catalog lot: {key}/{pv_id}")
        batch = float(effective_batch(next(iter(catalog_lots)), config.canonical_factor))
        demand_map = {
            member.fini_id: member.demand_litres for member in group_members
        }
        pallet_map = {
            member.fini_id: member.pallet_litres for member in group_members
        }
        nominal = proportional_allocations(demand_map, batch)
        adjusted = pallet_allocations(nominal, pallet_map, config.pallet_formula)
        demand = sum(demand_map.values())
        for basis in CoverageBasis:
            coverage_views[basis].append(
                float(
                    group_coverage(
                        basis,
                        demand_map,
                        batch,
                        adjusted,
                        config.demand_days,
                    )
                )
            )
        frequency = float(group_frequency(demand, batch, config.productive_weeks))
        demands.append(demand)
        fini_adjusted_coverages.extend(
            config.demand_days * float(adjusted[fini_id]) / float(demand_map[fini_id])
            for fini_id in sorted(adjusted)
        )
        total_j_ch += float(changeover_contribution(frequency, len(group_members)))
        pallet_adjustment += float(sum(adjusted.values())) - batch
        if config.matrix_mode is not MatrixMode.OFF or config.matrix_pairs:
            group_exceptions = matrix_exceptions(
                (
                    (member.fini_id, member.package_volume)
                    for member in group_members
                ),
                matrix_rows,
            )
            matrix_exception_pairs += len(group_exceptions)
            matrix_prohibited_pairs += sum(pair["status"] == "N" for pair in group_exceptions)
            matrix_exception_groups += int(bool(group_exceptions))
            matrix_exception_volume_pairs.update(
                tuple(
                    sorted(
                        (
                            normalize_volume(exception["volume_a"]),
                            normalize_volume(exception["volume_b"]),
                        )
                    )
                )
                for exception in group_exceptions
            )
            for left, right in combinations(group_members, 2):
                status = matrix_lookup[
                    (
                        normalize_volume(left.package_volume),
                        normalize_volume(right.package_volume),
                    )
                ]
                if status in {
                    UNKNOWN_LINE_FEASIBLE,
                    UNKNOWN_NO_CURRENT_LINE_OVERLAP,
                }:
                    matrix_unknown_pairs += 1
                elif status not in {"NO", "N", "AVOID"}:
                    matrix_positive_pairs += 1
        relaxed_groups += int(len(group_members) > config.group_size.base_limit)
        size_excess += max(len(group_members) - config.group_size.base_limit, 0)
    if not coverages:
        return {
            "group_count": 0,
            "singleton_group_count": 0,
            "target_violation_count": 0,
            "j_ch": 0.0,
            "pallet_adjustment_litres": 0.0,
        }
    summary = coverage_summary(coverages, demands)
    band = target_band_key(coverages, demands, config.target_band)
    fini_excesses = [
        max(
            config.target_band.lower_days - coverage,
            coverage - config.target_band.upper_days,
            0.0,
        )
        for coverage in fini_adjusted_coverages
    ]
    result: dict[str, float | int] = {
        "group_count": len(coverages),
        "singleton_group_count": sum(
            len(group_members) == 1 for group_members in grouped.values()
        ),
        "maximum_coverage_days": summary.maximum,
        "p90_coverage_days": summary.p90,
        "median_coverage_days": summary.median,
        "demand_weighted_mean_coverage_days": summary.demand_weighted_mean,
        "group_mean_coverage_days": summary.group_mean,
        "target_violation_count": band.violation_count,
        "target_worst_excess_days": band.worst_excess_days,
        "target_total_excess_days": band.total_excess_days,
        "j_ch": total_j_ch,
        "pallet_adjustment_litres": pallet_adjustment,
        "pallet_adjusted_fini_violation_count": sum(
            excess > 0 for excess in fini_excesses
        ),
        "pallet_adjusted_fini_worst_excess_days": max(
            fini_excesses, default=0.0
        ),
        "matrix_exception_pair_count": matrix_exception_pairs,
        "matrix_exception_group_count": matrix_exception_groups,
        "matrix_exception_distinct_volume_pair_count": len(
            matrix_exception_volume_pairs
        ),
        "matrix_unknown_pair_count": matrix_unknown_pairs,
        "matrix_positive_evidence_pair_count": matrix_positive_pairs,
        "relaxed_group_count": relaxed_groups,
        "extra_fini_count": size_excess,
        "feasible_under_active_matrix": int(
            matrix_prohibited_pairs == 0
            and (config.matrix_mode.value != "HARD" or matrix_exception_pairs == 0)
        ),
    }
    for basis, values in coverage_views.items():
        view_name = basis.value.lower()
        view_summary = coverage_summary(values, demands)
        view_band = target_band_key(values, demands, config.target_band)
        result.update(
            {
                f"{view_name}_maximum_coverage_days": view_summary.maximum,
                f"{view_name}_p90_coverage_days": view_summary.p90,
                f"{view_name}_median_coverage_days": view_summary.median,
                f"{view_name}_demand_weighted_mean_coverage_days": (
                    view_summary.demand_weighted_mean
                ),
                f"{view_name}_group_mean_coverage_days": view_summary.group_mean,
                f"{view_name}_target_violation_count": view_band.violation_count,
                f"{view_name}_target_worst_excess_days": (
                    view_band.worst_excess_days
                ),
                f"{view_name}_target_total_excess_days": (
                    view_band.total_excess_days
                ),
            }
        )
    return result


def build_solution_report(
    summary: Mapping[str, Any],
    groups: Iterable[GroupEvidence],
    pools: Iterable[CandidatePool],
    result: SolveResult,
    config: RunConfig,
    validation: ValidationResult,
    constraint_rows: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Render a dependency-free Markdown report with metrics and proof caveats."""

    selected = tuple(groups)
    pool_rows = tuple(pools)
    active_violations = sorted(
        (
            group
            for group in selected
            if group.coverage_days < config.target_band.lower_days
            or group.coverage_days > config.target_band.upper_days
        ),
        key=lambda group: max(
            config.target_band.lower_days - group.coverage_days,
            group.coverage_days - config.target_band.upper_days,
            0.0,
        ),
        reverse=True,
    )[:10]
    base_violations = sorted(
        (
            group
            for group in selected
            if group.base_group_coverage_days < config.target_band.lower_days
            or group.base_group_coverage_days > config.target_band.upper_days
        ),
        key=lambda group: max(
            config.target_band.lower_days - group.base_group_coverage_days,
            group.base_group_coverage_days - config.target_band.upper_days,
            0.0,
        ),
        reverse=True,
    )[:10]
    pallet_diagnostic_violations = sorted(
        (
            group
            for group in selected
            if group.adjusted_group_coverage_days < config.target_band.lower_days
            or group.adjusted_group_coverage_days > config.target_band.upper_days
            or group.worst_fini_coverage_days < config.target_band.lower_days
            or group.worst_fini_coverage_days > config.target_band.upper_days
        ),
        key=lambda group: max(
            config.target_band.lower_days - group.worst_fini_coverage_days,
            group.worst_fini_coverage_days - config.target_band.upper_days,
            0.0,
        ),
        reverse=True,
    )[:10]
    exact = sum(pool.completeness == "complete" for pool in pool_rows)
    restricted = len(pool_rows) - exact
    lines = [
        f"# Solution report: {config.scenario_id}",
        "",
        f"Configuration `{config.configuration_id()}` produced **{result.result_class}** status ",
        f"with independent validation **{'valid' if validation.is_valid else 'invalid'}**.",
        "",
        "## Coverage and changeover semantics",
        "",
        f"`coverage_days` is the active configured KPI using `{config.coverage_basis.value}`. "
        "Shared/base group coverage is the transcript-grounded default interpretation for "
        "the recurring inventory strategy; all three coverage views are reported regardless "
        "of the active optimization basis.",
        "",
        "Pallet-adjusted member coverage is operational evidence for guardrails, review, and "
        "sensitivity analysis. `MINIMUM_ONLY` is provisional: the transcript confirms the "
        "one-full-pallet floor but leaves allocation treatment above that floor open. "
        "A low runner in a group need not run in every group occurrence.",
        "",
        "`J_CH = sum_g F_g(n_g - 1)` counts only expected within-group FINI transitions. "
        "Singletons contribute zero, and initial or between-group setups are excluded.",
    ]
    if config.is_greenfield:
        lines.extend(
            [
                "",
                "## Greenfield validation",
                "",
                f"Status: **{summary.get('acceptance_status', 'NOT_ASSESSED')}**. "
                "Feasibility and objective bounds do not use historical groups.",
                f"Warm start: `{summary.get('warm_start_kind', 'none')}` with "
                f"{summary.get('warm_start_group_count', 0)} selected groups. "
                "A valid historical wheel may still appear as optional comparison columns.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Baseline-relative acceptance",
                "",
                f"Governance status: **{summary.get('acceptance_status', 'NOT_ASSESSED')}**. "
                f"Failed guardrails: `{summary.get('failed_baseline_guardrails', '') or 'none'}`.",
                "The assessment recalculates the frozen membership under the same coverage, pallet, "
                "PV, matrix, and target-band settings before comparing violation count, worst excess, "
                "total excess, demand-weighted/P90 coverage, group/singleton counts, `J_CH`, and "
                "matrix-exception group, pair, and distinct-volume-pair counts.",
                "",
                "## In-model baseline constraints and start",
                "",
                f"Constraints applied: **{'yes' if result.baseline_constraint_limits is not None else 'no'}**. "
                f"Scope: `{result.baseline_constraint_scope}`. "
                f"Warm start: `{summary.get('warm_start_kind', 'none')}` with "
                f"{summary.get('warm_start_group_count', 0)} selected groups.",
            ]
        )
    lines.extend(
        [
            "",
            "## KPI summary",
            "",
            "| KPI | Proposal | Baseline | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    kpis = (
        "group_count",
        "singleton_group_count",
        "maximum_coverage_days",
        "p90_coverage_days",
        "median_coverage_days",
        "demand_weighted_mean_coverage_days",
        "target_violation_count",
        "target_worst_excess_days",
        "target_total_excess_days",
        "j_ch",
        "relaxed_group_count",
        "extra_fini_count",
        "matrix_exception_group_count",
        "matrix_exception_pair_count",
        "matrix_exception_distinct_volume_pair_count",
        "matrix_unknown_pair_count",
        "matrix_positive_evidence_pair_count",
        "pv_changed_group_count",
        "pallet_adjustment_litres",
    )
    for name in kpis:
        lines.append(
            f"| {name} | {summary.get(name, '')} | {summary.get(f'baseline_{name}', '')} | {summary.get(f'delta_{name}', '')} |"
        )
    if result.baseline_constraint_limits is not None:
        lines.extend(
            [
                "",
                "| In-model limit | Value |",
                "|---|---:|",
                *[
                    f"| {name} | {value} |"
                    for name, value in asdict(
                        result.baseline_constraint_limits
                    ).items()
                ],
            ]
        )
    if (
        summary.get("acceptance_status") == "ACCEPTED_BASELINE_GUARDRAILS"
        and config.coverage_mode
        in {
            CoverageMode.OPERATIONS_FIRST,
            CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
        }
        and summary.get("j_ch_objective_tier_status")
        not in {"optimal", "decomposed_optimal"}
    ):
        lines.extend(
            [
                "",
                "The incumbent passes the numeric baseline guardrails, but the "
                f"operations `J_CH` tier is `{summary.get('j_ch_objective_tier_status')}`. "
                "Do not claim that `J_CH` was fully minimized.",
            ]
        )
    lines.extend(
        [
            "",
            "## Coverage evidence by interpretation",
            "",
            "| Coverage view | Role | Maximum days | P90 days | Median days | Demand-weighted mean | Proposal violations | Baseline violations |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    coverage_roles = {
        "base_group": "default shared-group comparison",
        "adjusted_group": "pallet group diagnostic",
        "worst_fini": "member guardrail/sensitivity",
    }
    for view, role in coverage_roles.items():
        lines.append(
            f"| {view} | {role} | {summary.get(f'{view}_maximum_coverage_days', '')} | "
            f"{summary.get(f'{view}_p90_coverage_days', '')} | "
            f"{summary.get(f'{view}_median_coverage_days', '')} | "
            f"{summary.get(f'{view}_demand_weighted_mean_coverage_days', '')} | "
            f"{summary.get(f'{view}_target_violation_count', '')} | "
            f"{summary.get(f'baseline_{view}_target_violation_count', '')} |"
        )
    lines.append(
        "Pallet-adjusted FINI violations are reported separately: "
        f"{summary.get('pallet_adjusted_fini_violation_count', 0)} FINIs; worst excess "
        f"{summary.get('pallet_adjusted_fini_worst_excess_days', 0)} days. "
        f"Baseline: {summary.get('baseline_pallet_adjusted_fini_violation_count', '')} "
        "FINIs."
    )
    lines.extend(
        [
            "",
            "## Active rules",
            "",
            f"Coverage `{config.coverage_mode.value}` / `{config.coverage_basis.value}`; pallet `{config.pallet_formula.value}`; ",
            f"cap `{config.group_size.effective_cap}`; PV `{config.pv_mode.value}`; matrix `{config.matrix_mode.value}` ",
            f"/ `{configured_matrix_version(config)}`; ",
            f"target band `{config.target_band.lower_days}-{config.target_band.upper_days}` days.",
            "",
        ]
    )
    for row in constraint_rows:
        if row.get("kind") != "run_config":
            lines.append(f"- `{row.get('constraint_id')}`: {row.get('kind')} {row.get('parameters')}")
    if config.matrix_pairs:
        lines.append(
            "The matrix is frozen from the selected plant profile. Explicit `N` pairs "
            "are prohibited in every mode; `AVOID` pairs are rejected in HARD and "
            "retained as reported exceptions in FLEXIBLE and OFF."
        )
    elif configured_matrix_version(config) == "BASELINE_EMPIRICAL_MATRIX_V1":
        lines.append(
            "The empirical matrix records historical co-membership as positive evidence. "
            "Unobserved pairs remain `UNKNOWN` and are reported without being treated as "
            "incompatible; only a future customer-approved `NO` can prohibit a pair."
        )
    elif configured_matrix_version(config) == "SYNTHETIC_VOLUME_MATRIX_V1":
        lines.append(
            "This is the provisional family-based synthetic matrix sensitivity, not a "
            "reconstruction of the rules used by the frozen baseline."
        )
    elif configured_matrix_version(config) == "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1":
        lines.append(
            "The customer operational-family rule marks shared-family pairs as preferred "
            "and outside-family pairs as `AVOID`. HARD is the zero-exception migration "
            "comparison; FLEXIBLE retains and reports exceptions for baseline-comparable "
            "trade-off review."
        )
    lines.extend(["", "## Primary shared/base target-band violations", ""])
    if base_violations:
        lines.extend(["| Group | Members | Coverage days |", "|---|---:|---:|"])
        for group in base_violations:
            lines.append(
                f"| {group.group_id} | {group.group_size} | "
                f"{group.base_group_coverage_days:.6f} |"
            )
    else:
        lines.append("No selected group violates the shared/base target band.")
    if config.coverage_basis.value != "BASE_GROUP":
        lines.extend(
            [
                "",
                f"## Active configured `{config.coverage_basis.value}` target-band violations",
                "",
            ]
        )
        if active_violations:
            lines.extend(
                ["| Group | Members | Coverage days |", "|---|---:|---:|"]
            )
            for group in active_violations:
                lines.append(
                    f"| {group.group_id} | {group.group_size} | "
                    f"{group.coverage_days:.6f} |"
                )
        else:
            lines.append("No selected group violates the active configured target band.")
    lines.extend(["", "## Pallet-adjusted coverage diagnostics", ""])
    if pallet_diagnostic_violations:
        lines.extend(
            [
                "| Group | Base group days | Adjusted group days | Worst FINI days |",
                "|---|---:|---:|---:|",
            ]
        )
        for group in pallet_diagnostic_violations:
            lines.append(
                f"| {group.group_id} | {group.base_group_coverage_days:.6f} | "
                f"{group.adjusted_group_coverage_days:.6f} | "
                f"{group.worst_fini_coverage_days:.6f} |"
            )
    else:
        lines.append("No selected group has a pallet-adjusted target-band diagnostic.")
    relaxations = [group for group in selected if group.relaxed_group]
    lines.extend(["", "## Relaxations and membership changes", ""])
    lines.append(
        f"Relaxed groups: {len(relaxations)}; extra FINIs above cap seven: {sum(group.size_excess for group in relaxations)}; "
        f"changed proposed groups: {sum(group.baseline_changed for group in selected)}; "
        f"groups with a PV change: {sum(group.pv_changed for group in selected)}."
    )
    if summary.get("baseline_feasible_under_active_matrix") in {0, "0"}:
        lines.append(
            "The frozen baseline violates the proposal's active package-volume matrix. "
            "This result is a feasibility/sensitivity study and is not eligible to be "
            "presented as an improvement over the original baseline grouping."
        )
    if config.pv_mode.value == "OPTIMIZED":
        lines.append(
            "PV selection is a sensitivity. Improvements in this scenario combine regrouping and PV changes and must not be attributed solely to regrouping."
        )
    lines.extend(["", "## Solver and candidate-library evidence", ""])
    lines.append(
        f"Solver `{result.solver_method}` reported `{result.status}` / `{result.termination_condition}`; "
        f"primary-tier gap `{result.primary_relative_gap}` and maximum reached "
        f"block-tier gap `{result.relative_gap}`. "
        f"Candidate blocks: {exact} complete and {restricted} restricted."
    )
    lines.extend(
        [
            "",
            "| Objective tier | Observed incumbent value | Reach/proof status | Relative gap |",
            "|---|---:|---|---:|",
        ]
    )
    for level in result.objective_levels:
        lines.append(
            f"| {level.name} | {level.value} | {level.termination_condition} | "
            f"{'' if level.relative_gap is None else level.relative_gap} |"
        )
    if restricted:
        lines.append(
            "Restricted blocks are optimized only over the disclosed deterministic candidate library; even a 0% within-pool gap is pool-optimal, not globally optimal."
        )
    for pool in sorted(pool_rows, key=lambda item: item.block_key):
        lines.append(
            f"- `{pool.block_key[0]}/{pool.block_key[1]}`: {pool.method}, {pool.completeness}, "
            f"{len(pool.candidates)} candidates, pool hash `{pool.pool_hash}`."
        )
    lines.extend(["", "## Independent validation", ""])
    if validation.issues:
        for issue in validation.issues[:20]:
            lines.append(
                f"- **{issue.severity}** `{issue.rule_id}` ({issue.entity_key}): {issue.message}"
            )
    else:
        lines.append("All independently recomputed structural rules and KPI coefficients passed.")
    return "\n".join(lines).rstrip() + "\n"


def write_scenario_artifacts(
    output_directory: Path,
    fini_rows: Iterable[Mapping[str, Any]],
    members: Iterable[CandidateMember],
    pools: Iterable[CandidatePool],
    result: SolveResult,
    config: RunConfig,
    validation: ValidationResult,
    constraints: Iterable[Any] = (),
    baseline_summary: Mapping[str, Any] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    production_versions: Iterable[ProductionVersion] = (),
    acceptance_summary: Mapping[str, Any] | None = None,
    baseline_evidence_members: Iterable[CandidateMember] = (),
    include_baseline_comparison: bool = True,
) -> ScenarioArtifactPaths:
    """Write one complete, deterministic per-scenario result bundle.

    Args:
        output_directory: New or existing run/scenario output directory.
        fini_rows: Complete source-ordered canonical FINI master.
        members: Modeled optimizer population.
        pools: Candidate libraries used for the solve.
        result: Completed solver outcome.
        config: Active run configuration.
        validation: Independent validation result for this exact solution.
        constraints: Optional allowlisted typed constraints for audit output.
        baseline_summary: Optional KPI mapping used for baseline deltas.
        manifest_metadata: Optional extraction/run provenance to retain in JSON.
        production_versions: Canonical PV catalog used to compute baseline deltas
            when ``baseline_summary`` is not supplied.
        acceptance_summary: Optional flattened baseline-relative governance
            assessment for the CSV, report, and manifest.
        baseline_evidence_members: Complete historical assignments used for
            comparable empirical matrix reporting.
        include_baseline_comparison: Whether to derive optional baseline KPI
            columns when no explicit summary was supplied.

    Returns:
        Typed paths to every emitted scenario artifact.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    materialized_members = tuple(members)
    materialized_pools = tuple(pools)
    materialized_versions = tuple(production_versions)
    materialized_baseline_evidence = tuple(baseline_evidence_members)
    if (
        include_baseline_comparison
        and baseline_summary is None
        and materialized_versions
    ):
        baseline_summary = build_baseline_summary(
            materialized_members,
            materialized_versions,
            config,
            materialized_baseline_evidence,
        )
    proposed = build_proposed_subgroups_rows(
        fini_rows, materialized_members, validation.groups, result, config
    )
    solution_groups = build_solution_groups_rows(validation.groups, result, config)
    matrix_exception_audit = build_matrix_exception_audit_rows(
        validation.groups,
        config,
        materialized_members,
        materialized_baseline_evidence,
    )
    if acceptance_summary:
        repeated_acceptance = {
            "acceptance_status": acceptance_summary.get("acceptance_status", ""),
            "failed_baseline_guardrails": acceptance_summary.get(
                "failed_baseline_guardrails", ""
            ),
        }
        for row in (*proposed, *solution_groups):
            row.update(repeated_acceptance)
    pool_summary = build_candidate_pool_summary_rows(
        materialized_pools, config, result
    )
    constraint_audit = build_constraint_audit_rows(config, constraints)
    summary = build_scenario_summary_row(
        validation.groups,
        result,
        config,
        validation,
        baseline_summary,
        acceptance_summary,
    )
    validation_rows = validation.issue_rows()
    paths = ScenarioArtifactPaths(
        proposed_subgroups=output_directory / "proposed_subgroups.csv",
        solution_groups=output_directory / "solution_groups.csv",
        scenario_summary=output_directory / "scenario_summary.csv",
        matrix_exception_audit=output_directory / "matrix_exception_audit.csv",
        constraint_audit=output_directory / "constraint_audit.csv",
        candidate_pool_summary=output_directory / "candidate_pool_summary.csv",
        validation_issues=output_directory / "validation_issues.csv",
        run_manifest=output_directory / "run_manifest.json",
        solution_report=output_directory / "solution_report.md",
    )
    for path, rows in (
        (paths.proposed_subgroups, proposed),
        (paths.solution_groups, solution_groups),
        (paths.scenario_summary, (summary,)),
        (paths.constraint_audit, constraint_audit),
        (paths.candidate_pool_summary, pool_summary),
    ):
        _write_csv(path, rows)
    _write_csv(
        paths.matrix_exception_audit,
        matrix_exception_audit,
        (
            "scenario_id",
            "configuration_id",
            "plant",
            "sefi",
            "proposed_subgroup",
            "material_a",
            "material_b",
            "volume_a",
            "volume_b",
            "status",
            "matrix_version",
            "family_evidence_a",
            "family_evidence_b",
            "rationale",
        ),
    )
    _write_csv(
        paths.validation_issues,
        validation_rows,
        ("rule_id", "severity", "entity", "entity_key", "observed", "expected", "message"),
    )
    paths.solution_report.write_text(
        build_solution_report(
            summary,
            validation.groups,
            materialized_pools,
            result,
            config,
            validation,
            constraint_audit,
        ),
        encoding="utf-8",
    )
    artifact_paths = (
        paths.proposed_subgroups,
        paths.solution_groups,
        paths.scenario_summary,
        paths.matrix_exception_audit,
        paths.constraint_audit,
        paths.candidate_pool_summary,
        paths.validation_issues,
        paths.solution_report,
    )
    manifest = dict(manifest_metadata or {})
    manifest.update(
        {
            "scenario_id": config.scenario_id,
            "configuration_id": config.configuration_id(),
            "config": config.model_dump(mode="json"),
            "solver": {
                "status": result.status,
                "termination_condition": result.termination_condition,
                "result_class": result.result_class,
                "method": result.solver_method,
                "gap": result.relative_gap,
                "epsilon_j_ch": result.epsilon_j_ch,
                "pool_completeness": result.pool_completeness,
                "baseline_constraint_scope": result.baseline_constraint_scope,
                "baseline_constraint_limits": (
                    asdict(result.baseline_constraint_limits)
                    if result.baseline_constraint_limits is not None
                    else None
                ),
                "warm_start": {
                    "kind": result.warm_start_kind,
                    "group_count": result.warm_start_group_count,
                },
                "objective_levels": [
                    {
                        "name": level.name,
                        "value": level.value,
                        "termination": level.termination_condition,
                        "relative_gap": level.relative_gap,
                    }
                    for level in result.objective_levels
                ],
                "block_evidence": [
                    {
                        "block": evidence.block_key,
                        "status": evidence.status,
                        "termination_condition": evidence.termination_condition,
                        "result_class": evidence.result_class,
                        "primary_objective": evidence.primary_objective,
                        "primary_best_bound": evidence.primary_best_bound,
                        "primary_relative_gap": evidence.primary_relative_gap,
                        "max_reached_tier_gap": evidence.relative_gap,
                        "candidate_count": evidence.candidate_count,
                        "objective_levels": [
                            {
                                "name": level.name,
                                "value": level.value,
                                "termination": level.termination_condition,
                                "best_bound": level.best_bound,
                                "relative_gap": level.relative_gap,
                                "wallclock_seconds": level.wallclock_seconds,
                            }
                            for level in evidence.objective_levels
                        ],
                    }
                    for evidence in result.block_evidence
                ],
            },
            "validation": {
                "valid": validation.is_valid,
                "errors": sum(issue.severity == "error" for issue in validation.issues),
                "warnings": sum(issue.severity == "warning" for issue in validation.issues),
            },
            "acceptance": dict(acceptance_summary or {}),
            "matrix_evidence": {
                "version": configured_matrix_version(config),
                "mode": config.matrix_mode.value,
                "selected_explicit_no_pairs": sum(
                    row["status"] in {"NO", "N"} for row in matrix_exception_audit
                ),
                "selected_avoid_pairs": sum(
                    row["status"] == "AVOID" for row in matrix_exception_audit
                ),
                "selected_exception_groups": summary.get(
                    "matrix_exception_group_count", 0
                ),
                "selected_exception_pair_instances": summary.get(
                    "matrix_exception_pair_count", 0
                ),
                "selected_exception_distinct_volume_pairs": summary.get(
                    "matrix_exception_distinct_volume_pair_count", 0
                ),
                "selected_unknown_pairs": summary.get(
                    "matrix_unknown_pair_count", 0
                ),
                "selected_positive_evidence_pairs": summary.get(
                    "matrix_positive_evidence_pair_count", 0
                ),
            },
            "row_counts": {
                "proposed_subgroups": len(proposed),
                "solution_groups": len(solution_groups),
                "scenario_summary": 1,
                "matrix_exception_audit": len(matrix_exception_audit),
                "constraint_audit": len(constraint_audit),
                "candidate_pool_summary": len(pool_summary),
                "validation_issues": len(validation_rows),
            },
            "output_hashes": {
                path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                for path in artifact_paths
            },
        }
    )
    paths.run_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return paths
