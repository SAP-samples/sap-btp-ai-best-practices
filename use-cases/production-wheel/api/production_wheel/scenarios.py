"""Run deterministic optimizer scenarios over canonical extracted CSV tables.

This module contains no natural-language or arbitrary-code translation.  It
loads the governed extraction contract and orchestrates existing candidate and
solver APIs.  Example::

    from pathlib import Path
    from production_wheel.scenarios import load_canonical_inputs, run_scenario_suite

    inputs = load_canonical_inputs(Path("prototype/output/<run_id>"))
    suite = run_scenario_suite(inputs)
    assert len(suite.outcomes) == 25
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from production_wheel.candidates import (
    CandidateMember,
    CandidatePool,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.optimization import (
    ObjectiveLevel,
    SelectedCandidate,
    SolveResult,
    precompute_coefficients,
    solve_candidate_pools,
    solve_decomposed_pools,
)
from production_wheel.matrix import (
    CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
    normalize_volume,
)
from production_wheel.pareto import (
    GreenfieldFrontierResult,
    ParetoPoint,
    build_greenfield_frontier,
    build_portfolio_frontier,
)
from production_wheel.schemas import (
    BaselineAcceptanceStatus,
    CoverageBasis,
    CoverageMode,
    GroupSizeMode,
    GroupSizePolicy,
    MatrixMode,
    PalletFormula,
    PVMode,
    RunConfig,
    SolverLimits,
    VersionIdentifiers,
)

ProgressCallback = Callable[[str, int, int], None]
PoolTransform = Callable[[tuple[CandidatePool, ...]], tuple[CandidatePool, ...]]
CORE_SCENARIO_IDS = (
    "core_max",
    "core_demand_weighted_mean",
    "core_group_mean",
    "core_target_band",
    "core_operations_first",
)
GREENFIELD_FRONTIER_ID = "customer_families_greenfield_pareto"


@dataclass(frozen=True, slots=True)
class CanonicalInputs:
    """Canonical optimizer records plus untouched FINI CSV rows.

    Args:
        extracted_directory: Directory containing extraction tables.
        fini_rows: Every raw FINI dictionary in source-file order.
        members: Modeled and validated candidate members only.
        production_versions: Canonical finite PV records.
        baseline_evidence_members: All historical assignments, including
            modeled exclusions, used only for empirical positive evidence.
    """

    extracted_directory: Path
    fini_rows: tuple[dict[str, str], ...]
    members: tuple[CandidateMember, ...]
    production_versions: tuple[ProductionVersion, ...]
    baseline_evidence_members: tuple[CandidateMember, ...] = ()
    optimized_pv_members: tuple[CandidateMember, ...] = ()

    def members_for(self, config: RunConfig) -> tuple[CandidateMember, ...]:
        """Return the fixed-PV default or explicit optimized-PV sensitivity scope."""

        if config.pv_mode is PVMode.OPTIMIZED and self.optimized_pv_members:
            return self.optimized_pv_members
        return self.members


@dataclass(frozen=True, slots=True)
class PoolCacheRecord:
    """Audit evidence for one structural pool generation or reuse family."""

    structural_fingerprint: str
    pool_hashes: tuple[str, ...]
    block_count: int
    scenario_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineGuardrailAssessment:
    """Comparable baseline thresholds and proposal acceptance outcome."""

    status: BaselineAcceptanceStatus
    failed_guardrails: tuple[str, ...] = ()
    baseline_target_violation_count: int | None = None
    baseline_target_worst_excess_days: float | None = None
    baseline_target_total_excess_days: float | None = None
    baseline_demand_weighted_mean_coverage_days: float | None = None
    baseline_p90_coverage_days: float | None = None
    baseline_group_count: int | None = None
    baseline_singleton_group_count: int | None = None
    baseline_j_ch: float | None = None
    baseline_matrix_exception_group_count: int | None = None
    baseline_matrix_exception_pair_count: int | None = None
    baseline_matrix_exception_distinct_volume_pair_count: int | None = None
    accepted_j_ch_limit: float | None = None
    proposal_target_violation_count: int | None = None
    proposal_target_worst_excess_days: float | None = None
    proposal_target_total_excess_days: float | None = None
    proposal_demand_weighted_mean_coverage_days: float | None = None
    proposal_p90_coverage_days: float | None = None
    proposal_group_count: int | None = None
    proposal_singleton_group_count: int | None = None
    proposal_j_ch: float | None = None
    proposal_matrix_exception_group_count: int | None = None
    proposal_matrix_exception_pair_count: int | None = None
    proposal_matrix_exception_distinct_volume_pair_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return stable flattened fields for CSV, report, and manifest outputs."""

        return {
            "acceptance_status": self.status.value,
            "acceptance_evidence_source": "independent_group_recomputation",
            "failed_baseline_guardrails": "|".join(self.failed_guardrails),
            "baseline_guardrail_target_violation_count": self.baseline_target_violation_count,
            "baseline_guardrail_target_worst_excess_days": self.baseline_target_worst_excess_days,
            "baseline_guardrail_target_total_excess_days": self.baseline_target_total_excess_days,
            "baseline_guardrail_demand_weighted_mean_coverage_days": self.baseline_demand_weighted_mean_coverage_days,
            "baseline_guardrail_p90_coverage_days": self.baseline_p90_coverage_days,
            "baseline_guardrail_group_count": self.baseline_group_count,
            "baseline_guardrail_singleton_group_count": self.baseline_singleton_group_count,
            "baseline_guardrail_j_ch": self.baseline_j_ch,
            "baseline_guardrail_matrix_exception_group_count": self.baseline_matrix_exception_group_count,
            "baseline_guardrail_matrix_exception_pair_count": self.baseline_matrix_exception_pair_count,
            "baseline_guardrail_matrix_exception_distinct_volume_pair_count": self.baseline_matrix_exception_distinct_volume_pair_count,
            "accepted_j_ch_limit": self.accepted_j_ch_limit,
            "proposal_guardrail_target_violation_count": self.proposal_target_violation_count,
            "proposal_guardrail_target_worst_excess_days": self.proposal_target_worst_excess_days,
            "proposal_guardrail_target_total_excess_days": self.proposal_target_total_excess_days,
            "proposal_guardrail_demand_weighted_mean_coverage_days": self.proposal_demand_weighted_mean_coverage_days,
            "proposal_guardrail_p90_coverage_days": self.proposal_p90_coverage_days,
            "proposal_guardrail_group_count": self.proposal_group_count,
            "proposal_guardrail_singleton_group_count": self.proposal_singleton_group_count,
            "proposal_guardrail_j_ch": self.proposal_j_ch,
            "proposal_guardrail_matrix_exception_group_count": self.proposal_matrix_exception_group_count,
            "proposal_guardrail_matrix_exception_pair_count": self.proposal_matrix_exception_pair_count,
            "proposal_guardrail_matrix_exception_distinct_volume_pair_count": self.proposal_matrix_exception_distinct_volume_pair_count,
        }


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """One baseline or optimizer scenario outcome, including failures."""

    summary_id: str
    kind: str
    scenario_id: str
    configuration_id: str | None
    config: RunConfig | None
    status: str
    result_class: str
    solve_result: SolveResult | None = None
    structural_fingerprint: str | None = None
    pool_hashes: tuple[str, ...] = ()
    pools: tuple[CandidatePool, ...] = ()
    members: tuple[CandidateMember, ...] = ()
    acceptance: BaselineGuardrailAssessment | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSuiteResult:
    """Complete demo bundle with fixed-cardinality summary results."""

    inputs: CanonicalInputs
    baseline: ScenarioOutcome
    scenarios: tuple[ScenarioOutcome, ...]
    pareto_points: tuple[ParetoPoint, ...]
    pool_cache: tuple[PoolCacheRecord, ...]
    elapsed_seconds: float

    @property
    def outcomes(self) -> tuple[object, ...]:
        """Return baseline, 19 deduplicated scenarios, and five Pareto points."""

        return (self.baseline, *self.scenarios, *self.pareto_points)

    @property
    def partial(self) -> bool:
        """Return whether any requested summary result lacks a valid incumbent."""

        scenario_failure = any(
            outcome.solve_result is None or not outcome.solve_result.has_incumbent
            for outcome in self.scenarios
        )
        pareto_failure = any(point.status not in {"optimal", "feasible_limit"} for point in self.pareto_points)
        return self.baseline.solve_result is None or scenario_failure or pareto_failure


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    """Read one canonical table while preserving row and field ordering."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    """Convert a blank CSV cell to ``None`` and a populated cell to float."""

    return None if value is None or not value.strip() else float(value)


def load_canonical_inputs(run_directory: Path) -> CanonicalInputs:
    """Load extracted FINIs and production versions for scenario execution.

    Args:
        run_directory: Extraction run root or its ``extracted`` child.

    Returns:
        Typed modeled inputs while retaining all raw FINI rows unchanged.

    Raises:
        FileNotFoundError: Either required canonical table is absent.
        ValueError: A modeled FINI lacks required positive primitives.
    """

    root = run_directory.resolve()
    extracted = root if (root / "fini_master.csv").is_file() else root / "extracted"
    fini_rows = _read_csv(extracted / "fini_master.csv")
    version_rows = _read_csv(extracted / "production_versions.csv")
    return canonical_inputs_from_tables(
        {"fini_master": fini_rows, "production_versions": version_rows}, str(extracted)
    )


def canonical_inputs_from_tables(
    tables: dict, source_label: str = "hana"
) -> CanonicalInputs:
    """Build solver inputs from in-memory canonical rows without reading files.

    Args:
        tables: Canonical FINI master and production-version row lists.
        source_label: Diagnostic origin label, never opened as a filesystem path.

    Returns:
        Typed candidate members and versions with legacy CSV-compatible raw rows.
    """

    def normalize(row: dict) -> dict[str, str]:
        """Normalize typed values to the CSV record contract without rewriting IDs.

        Boolean semantics apply only to baseline_assigned, active, pallet_conflict,
        eligible, compatible, fixed_pv_consistent and source_calculation_matches.
        Numeric HANA 0/1 and equivalent strings become canonical flag strings.
        Line arrays become pipe-separated strings; structured source evidence is
        serialized as JSON so it can be read back without Python-repr parsing.
        """
        boolean_fields = {
            "baseline_assigned", "active", "pallet_conflict", "eligible",
            "compatible", "fixed_pv_consistent", "source_calculation_matches",
        }
        result: dict[str, str] = {}
        for key, value in row.items():
            if value is None:
                result[key] = ""
            elif key == "eligible_lines" and isinstance(value, (list, tuple)):
                result[key] = "|".join(str(line) for line in value)
            elif isinstance(value, (dict, list, tuple)):
                result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            elif key in boolean_fields:
                token = str(value).strip().lower()
                if token in {"true", "yes", "active"}:
                    result[key] = "1"
                elif token in {"false", "no", "inactive"}:
                    result[key] = "0"
                else:
                    try:
                        numeric = float(token)
                    except ValueError:
                        numeric = None
                    result[key] = str(int(numeric)) if numeric in (0.0, 1.0) else str(value)
            else:
                result[key] = str(value)
        return result

    fini_rows = tuple(normalize(row) for row in tables["fini_master"])
    version_rows = tuple(normalize(row) for row in tables["production_versions"])
    extracted = Path(source_label)
    members = tuple(
        CandidateMember.from_record(row)
        for row in fini_rows
        if row.get("model_status") == "modeled"
    )
    optimized_pv_members = tuple(
        CandidateMember.from_record(row)
        for row in fini_rows
        if row.get("optimized_pv_model_status") == "modeled"
    )
    baseline_evidence_members = tuple(
        CandidateMember.from_record(row)
        for row in fini_rows
        if row.get("baseline_assigned", "").strip().lower() in {"1", "true", "yes"}
    )
    versions = tuple(
        ProductionVersion(
            plant=row["plant"],
            sefi=row["sefi"],
            pv_id=row.get("production_version", ""),
            nominal_lot_litres=_optional_float(row.get("lot_size_litres")),
            active=row.get("active", "1").strip().lower()
            not in {"0", "false", "no", "inactive"},
        )
        for row in version_rows
    )
    if not members and not optimized_pv_members:
        raise ValueError("fini_master contains no modeled FINIs")
    return CanonicalInputs(
        extracted,
        fini_rows,
        members,
        versions,
        baseline_evidence_members,
        optimized_pv_members,
    )


def _deduplicate_configs(configs: Iterable[RunConfig]) -> tuple[RunConfig, ...]:
    """Keep the first scenario label for each normalized configuration hash."""

    unique: dict[str, RunConfig] = {}
    for config in configs:
        unique.setdefault(config.configuration_id(), config)
    return tuple(unique.values())


def _build_demo_configurations() -> tuple[RunConfig, ...]:
    """Construct and freeze the governed 19 unique demo solve configurations."""

    configs = [
        RunConfig(scenario_id="core_max", coverage_mode=CoverageMode.MAX),
        RunConfig(
            scenario_id="core_demand_weighted_mean",
            coverage_mode=CoverageMode.DEMAND_WEIGHTED_MEAN,
        ),
        RunConfig(
            scenario_id="core_group_mean",
            coverage_mode=CoverageMode.GROUP_MEAN,
        ),
        RunConfig(
            scenario_id="core_target_band",
            coverage_mode=CoverageMode.TARGET_BAND,
        ),
        RunConfig(
            scenario_id="core_operations_first",
            coverage_mode=CoverageMode.OPERATIONS_FIRST,
        ),
    ]
    for basis in CoverageBasis:
        for formula in PalletFormula:
            configs.append(
                RunConfig(
                    scenario_id=f"target_{basis.value.lower()}_{formula.value.lower()}",
                    coverage_mode=CoverageMode.TARGET_BAND,
                    coverage_basis=basis,
                    pallet_formula=formula,
                )
            )
    configs.extend(
        [
            RunConfig(
                scenario_id="target_cap_8",
                group_size=GroupSizePolicy(
                    mode=GroupSizeMode.BOUNDED_RELAXATION, max_excess=1
                ),
            ),
            RunConfig(
                scenario_id="target_cap_9",
                group_size=GroupSizePolicy(
                    mode=GroupSizeMode.BOUNDED_RELAXATION, max_excess=2
                ),
            ),
            RunConfig(
                scenario_id="sensitivity_synthetic_hard_matrix",
                matrix_mode=MatrixMode.HARD,
                versions=VersionIdentifiers(
                    matrix_version="SYNTHETIC_VOLUME_MATRIX_V1"
                ),
            ),
            RunConfig(scenario_id="sensitivity_matrix_off", matrix_mode=MatrixMode.OFF),
            RunConfig(scenario_id="sensitivity_optimized_pv", pv_mode=PVMode.OPTIMIZED),
        ]
    )
    customer_versions = VersionIdentifiers(
        matrix_version=CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
    )
    configs.extend(
        [
            RunConfig(
                scenario_id="customer_families_hard_target",
                coverage_mode=CoverageMode.TARGET_BAND,
                matrix_mode=MatrixMode.HARD,
                versions=customer_versions,
            ),
            RunConfig(
                scenario_id="customer_families_flexible_target",
                coverage_mode=CoverageMode.TARGET_BAND,
                matrix_mode=MatrixMode.FLEXIBLE,
                versions=customer_versions,
            ),
            RunConfig(
                scenario_id="customer_families_hard_operations",
                coverage_mode=CoverageMode.OPERATIONS_FIRST,
                matrix_mode=MatrixMode.HARD,
                versions=customer_versions,
            ),
            RunConfig(
                scenario_id="customer_families_flexible_operations",
                coverage_mode=CoverageMode.OPERATIONS_FIRST,
                matrix_mode=MatrixMode.FLEXIBLE,
                versions=customer_versions,
            ),
        ]
    )
    unique = _deduplicate_configs(configs)
    if len(unique) != 19:
        raise AssertionError(f"demo contract requires 19 unique configurations, got {len(unique)}")
    return unique


DEMO_CONFIGURATIONS = _build_demo_configurations()


def named_configurations() -> dict[str, RunConfig]:
    """Return every reviewed CLI scenario without widening the demo suite."""

    customer_versions = VersionIdentifiers(
        matrix_version=CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
    )
    additional = (
        RunConfig(
            scenario_id="customer_families_greenfield_coverage",
            coverage_mode=CoverageMode.GREENFIELD_COVERAGE,
            matrix_mode=MatrixMode.FLEXIBLE,
            versions=customer_versions,
        ),
        RunConfig(
            scenario_id="customer_families_greenfield_operations",
            coverage_mode=CoverageMode.GREENFIELD_OPERATIONS,
            matrix_mode=MatrixMode.FLEXIBLE,
            versions=customer_versions,
        ),
        RunConfig(
            scenario_id="customer_families_baseline_constrained_coverage",
            coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
            matrix_mode=MatrixMode.FLEXIBLE,
            versions=customer_versions,
        ),
        RunConfig(
            scenario_id="customer_families_baseline_constrained_operations",
            coverage_mode=CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
            matrix_mode=MatrixMode.FLEXIBLE,
            versions=customer_versions,
        ),
    )
    return {
        config.scenario_id: config
        for config in (*DEMO_CONFIGURATIONS, *additional)
    }


def greenfield_frontier_configuration() -> RunConfig:
    """Return the recommended baseline-independent customer-family frontier."""

    return RunConfig(
        scenario_id=GREENFIELD_FRONTIER_ID,
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.FLEXIBLE,
        versions=VersionIdentifiers(
            matrix_version=CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
        ),
    )


def run_greenfield_frontier(
    inputs: CanonicalInputs,
    config: RunConfig | None = None,
    *,
    point_count: int = 17,
    block_option_count: int = 5,
    per_block_stage_seconds: float | None = None,
    per_block_total_seconds: float | None = None,
    block_worker_count: int = 2,
    large_block_candidate_threshold: int = 250_000,
    global_epsilon_exponent: float = 2.0,
    pool_transform: PoolTransform | None = None,
    progress: ProgressCallback | None = None,
) -> GreenfieldFrontierResult:
    """Generate baseline-independent pools and coordinate a Pareto frontier.

    Args:
        inputs: Canonical extracted optimizer inputs.
        config: Optional reviewed greenfield configuration.
        point_count: Requested global epsilon samples including both anchors.
        block_option_count: Requested per-block epsilon samples.
        per_block_stage_seconds: Optional independent limit for each block tier.
        per_block_total_seconds: Optional conservative total limit per block.
        block_worker_count: Maximum isolated block-solving processes.
        large_block_candidate_threshold: Pools at or above this count run alone.
        global_epsilon_exponent: Power-law density bias toward low ``J_CH``.
        progress: Optional phase callback.

    Returns:
        Validation-ready anchors, block partitions, and frontier points.
    """

    active_config = config or greenfield_frontier_configuration()
    if active_config.coverage_mode is not CoverageMode.PARETO:
        raise ValueError("greenfield frontier configuration must use PARETO mode")
    members = inputs.members_for(active_config)
    pools = generate_candidate_pools(
        members,
        inputs.production_versions,
        active_config,
        progress,
        inputs.baseline_evidence_members or None,
    )
    if pool_transform is not None:
        pools = pool_transform(pools)
    return build_greenfield_frontier(
        pools,
        members,
        active_config,
        point_count=point_count,
        block_option_count=block_option_count,
        per_block_stage_seconds=per_block_stage_seconds,
        per_block_total_seconds=per_block_total_seconds,
        block_worker_count=block_worker_count,
        large_block_candidate_threshold=large_block_candidate_threshold,
        global_epsilon_exponent=global_epsilon_exponent,
        progress=progress,
    )


def demo_configurations() -> tuple[RunConfig, ...]:
    """Return the immutable, configuration-hash-deduplicated demo settings."""

    return DEMO_CONFIGURATIONS


def run_single_scenario(
    inputs: CanonicalInputs,
    config: RunConfig,
    progress: ProgressCallback | None = None,
    pool_transform: PoolTransform | None = None,
) -> ScenarioOutcome:
    """Generate, solve, and return one named optimizer configuration."""

    try:
        active_members = inputs.members_for(config)
        active_inputs = (
            inputs
            if active_members is inputs.members
            else replace(inputs, members=active_members)
        )
        pools = generate_candidate_pools(
            active_members,
            inputs.production_versions,
            config,
            progress,
            inputs.baseline_evidence_members or inputs.members,
        )
        if pool_transform is not None:
            pools = pool_transform(pools)
        result = (
            solve_candidate_pools(pools, active_members, config)
            if config.coverage_mode is CoverageMode.GROUP_MEAN
            else solve_decomposed_pools(pools, active_members, config)
        )
        acceptance = (
            None
            if config.is_greenfield
            else assess_baseline_guardrails(active_inputs, pools, config, result)
        )
        return ScenarioOutcome(
            summary_id=config.scenario_id,
            kind="scenario",
            scenario_id=config.scenario_id,
            configuration_id=config.configuration_id(),
            config=config,
            status=result.status,
            result_class=result.result_class,
            solve_result=result,
            structural_fingerprint=config.structural_ruleset_fingerprint(),
            pool_hashes=tuple(pool.pool_hash for pool in pools),
            pools=tuple(pools),
            members=active_members,
            acceptance=acceptance,
        )
    except Exception as exc:
        return ScenarioOutcome(
            summary_id=config.scenario_id,
            kind="scenario",
            scenario_id=config.scenario_id,
            configuration_id=config.configuration_id(),
            config=config,
            status="failed",
            result_class="scenario-unavailable",
            structural_fingerprint=config.structural_ruleset_fingerprint(),
            acceptance=BaselineGuardrailAssessment(
                BaselineAcceptanceStatus.NOT_ACCEPTABLE,
                ("scenario_failed",),
            ),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _baseline_result(
    inputs: CanonicalInputs,
    pools: Sequence[CandidatePool],
    config: RunConfig,
) -> SolveResult:
    """Recalculate the frozen historical partition from canonical primitives."""

    groups: dict[tuple[str, str, str], list[CandidateMember]] = {}
    for member in inputs.members:
        if not member.baseline_group:
            raise ValueError(f"modeled FINI has no baseline group: {member.fini_id}")
        groups.setdefault((*member.block_key, member.baseline_group), []).append(member)
    candidates = {
        (candidate.block_key, candidate.member_ids, candidate.pv_id): candidate
        for pool in pools
        for candidate in pool.candidates
    }
    frozen = []
    for (plant, sefi, baseline_group), group_members in sorted(groups.items()):
        fixed_pvs = {member.fixed_pv for member in group_members}
        if len(fixed_pvs) != 1 or None in fixed_pvs or "" in fixed_pvs:
            raise ValueError(
                "baseline group has no unique planning fixed PV: "
                f"{(plant, sefi, baseline_group)}"
            )
        fixed_pv = str(next(iter(fixed_pvs)))
        member_ids = tuple(sorted(member.fini_id for member in group_members))
        key = ((plant, sefi), member_ids, fixed_pv)
        if key not in candidates:
            raise ValueError(
                "baseline group/planning PV is absent from the active candidate pool: "
                f"{key}"
            )
        frozen.append(candidates[key])
    coefficients = precompute_coefficients(frozen, inputs.members, config)
    selected = tuple(
        SelectedCandidate(candidate, coefficient)
        for candidate, coefficient in zip(frozen, coefficients, strict=True)
    )
    maximum = max(item.coverage_days for item in coefficients)
    total_j_ch = sum(item.j_ch for item in coefficients)
    completeness = (
        "complete"
        if all(pool.completeness == "complete" for pool in pools)
        else "restricted"
    )
    return SolveResult(
        status="baseline_recalculated",
        termination_condition="not_optimized",
        has_incumbent=True,
        result_class="frozen-recalculated-baseline",
        selected=selected,
        objective_levels=(
            ObjectiveLevel("maximum_coverage_days", maximum, "recalculated"),
            ObjectiveLevel("j_ch", total_j_ch, "recalculated"),
        ),
        primary_objective=maximum,
        primary_best_bound=None,
        primary_relative_gap=None,
        incumbent_objective=None,
        best_bound=None,
        relative_gap=None,
        candidate_count=sum(len(pool.candidates) for pool in pools),
        pool_completeness=completeness,
        solver_method="frozen_assignment_recalculation",
    )


def assess_baseline_guardrails(
    inputs: CanonicalInputs,
    pools: Sequence[CandidatePool],
    config: RunConfig,
    proposal: SolveResult,
) -> BaselineGuardrailAssessment:
    """Classify a proposal against the frozen baseline under identical settings.

    Args:
        inputs: Canonical modeled FINIs and finite production versions.
        pools: Candidate libraries generated for ``config``.
        config: Active matrix, coverage, pallet, PV, and tolerance settings.
        proposal: Solver result to assess.

    Returns:
        Derived thresholds, observed proposal values, and governance status.
    """

    try:
        # Candidate presence proves that the frozen assignment is feasible under
        # active structural/matrix/PV rules; its KPIs are recomputed separately.
        _baseline_result(inputs, pools, config)
    except ValueError:
        return BaselineGuardrailAssessment(
            BaselineAcceptanceStatus.BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES,
            ("frozen_baseline_cannot_be_reproduced_under_active_rules",),
        )
    if not proposal.has_incumbent:
        return BaselineGuardrailAssessment(
            BaselineAcceptanceStatus.NOT_ACCEPTABLE,
            ("proposal_has_no_incumbent",),
        )
    # Local imports avoid coupling the optimizer orchestration module to report
    # construction during module initialization.
    from production_wheel.metrics import coverage_summary, target_band_key
    from production_wheel.reporting import build_baseline_summary
    from production_wheel.solution_validation import validate_solution

    independent = validate_solution(
        proposal,
        inputs.members,
        pools,
        config,
        inputs.production_versions,
        inputs.baseline_evidence_members or inputs.members,
    )
    if not independent.is_valid:
        return BaselineGuardrailAssessment(
            BaselineAcceptanceStatus.NOT_ACCEPTABLE,
            ("proposal_independent_validation_failed",),
        )
    proposal_coverages = [group.coverage_days for group in independent.groups]
    proposal_demands = [group.group_demand_litres for group in independent.groups]
    proposal_band = target_band_key(
        proposal_coverages, proposal_demands, config.target_band
    )
    proposal_coverage_summary = coverage_summary(
        proposal_coverages, proposal_demands
    )
    proposal_exception_groups = tuple(
        group for group in independent.groups if group.matrix_exception_pairs
    )
    proposal_exception_volume_pairs = {
        tuple(sorted((normalize_volume(volume_a), normalize_volume(volume_b))))
        for group in proposal_exception_groups
        for _, _, volume_a, volume_b in group.matrix_exception_pairs
    }
    proposal_values = (
        int(proposal_band.violation_count),
        float(proposal_band.worst_excess_days),
        float(proposal_band.total_excess_days),
        float(proposal_coverage_summary.demand_weighted_mean),
        float(proposal_coverage_summary.p90),
        len(independent.groups),
        sum(group.group_size == 1 for group in independent.groups),
        float(sum(group.j_ch_contribution for group in independent.groups)),
        len(proposal_exception_groups),
        sum(len(group.matrix_exception_pairs) for group in independent.groups),
        len(proposal_exception_volume_pairs),
    )
    baseline_summary = build_baseline_summary(
        inputs.members,
        inputs.production_versions,
        config,
        inputs.baseline_evidence_members or inputs.members,
    )
    baseline_values = (
        int(baseline_summary["target_violation_count"]),
        float(baseline_summary["target_worst_excess_days"]),
        float(baseline_summary["target_total_excess_days"]),
        float(baseline_summary["demand_weighted_mean_coverage_days"]),
        float(baseline_summary["p90_coverage_days"]),
        int(baseline_summary["group_count"]),
        int(baseline_summary["singleton_group_count"]),
        float(baseline_summary["j_ch"]),
        int(baseline_summary["matrix_exception_group_count"]),
        int(baseline_summary["matrix_exception_pair_count"]),
        int(baseline_summary["matrix_exception_distinct_volume_pair_count"]),
    )
    j_ch_limit = config.baseline_guardrails.j_ch_limit(baseline_values[7])
    thresholds = (*baseline_values[:7], j_ch_limit, *baseline_values[8:])
    names = (
        "target_violation_count",
        "target_worst_excess_days",
        "target_total_excess_days",
        "demand_weighted_mean_coverage_days",
        "p90_coverage_days",
        "group_count",
        "singleton_group_count",
        "j_ch",
        "matrix_exception_group_count",
        "matrix_exception_pair_count",
        "matrix_exception_distinct_volume_pair_count",
    )
    tolerance = 1e-9
    failed = tuple(
        name
        for name, value, limit in zip(names, proposal_values, thresholds, strict=True)
        if value > limit + tolerance
    )
    improved = any(
        value < limit - tolerance
        for value, limit in zip(proposal_values, baseline_values, strict=True)
    )
    status = (
        BaselineAcceptanceStatus.ACCEPTED_BASELINE_GUARDRAILS
        if not failed
        else BaselineAcceptanceStatus.PARETO_REVIEW_REQUIRED
        if improved
        else BaselineAcceptanceStatus.NOT_ACCEPTABLE
    )
    return BaselineGuardrailAssessment(
        status=status,
        failed_guardrails=failed,
        baseline_target_violation_count=baseline_values[0],
        baseline_target_worst_excess_days=baseline_values[1],
        baseline_target_total_excess_days=baseline_values[2],
        baseline_demand_weighted_mean_coverage_days=baseline_values[3],
        baseline_p90_coverage_days=baseline_values[4],
        baseline_group_count=baseline_values[5],
        baseline_singleton_group_count=baseline_values[6],
        baseline_j_ch=baseline_values[7],
        baseline_matrix_exception_group_count=baseline_values[8],
        baseline_matrix_exception_pair_count=baseline_values[9],
        baseline_matrix_exception_distinct_volume_pair_count=baseline_values[10],
        accepted_j_ch_limit=j_ch_limit,
        proposal_target_violation_count=proposal_values[0],
        proposal_target_worst_excess_days=proposal_values[1],
        proposal_target_total_excess_days=proposal_values[2],
        proposal_demand_weighted_mean_coverage_days=proposal_values[3],
        proposal_p90_coverage_days=proposal_values[4],
        proposal_group_count=proposal_values[5],
        proposal_singleton_group_count=proposal_values[6],
        proposal_j_ch=proposal_values[7],
        proposal_matrix_exception_group_count=proposal_values[8],
        proposal_matrix_exception_pair_count=proposal_values[9],
        proposal_matrix_exception_distinct_volume_pair_count=proposal_values[10],
    )


def _timeout_outcome(config: RunConfig) -> ScenarioOutcome:
    """Create a stable suite-deadline outcome without starting more work."""

    return ScenarioOutcome(
        summary_id=config.scenario_id,
        kind="scenario",
        scenario_id=config.scenario_id,
        configuration_id=config.configuration_id(),
        config=config,
        status="suite_timeout",
        result_class="not_solved",
        structural_fingerprint=config.structural_ruleset_fingerprint(),
        acceptance=BaselineGuardrailAssessment(
            BaselineAcceptanceStatus.NOT_ACCEPTABLE,
            ("suite_timeout",),
        ),
        error_type="TimeoutError",
        error_message="complete scenario-suite time limit exhausted",
    )


def run_scenario_suite(
    inputs: CanonicalInputs,
    configs: Sequence[RunConfig] | None = None,
    progress: ProgressCallback | None = None,
) -> ScenarioSuiteResult:
    """Run deduplicated scenarios, baseline, and five portfolio frontier points.

    Pools are cached by structural-ruleset fingerprint. Objective, coverage,
    pallet, and reporting changes therefore reuse candidate libraries, while
    cap, matrix, and PV widening settings naturally build separate libraries.
    Scenario exceptions and suite-deadline exhaustion are recorded and do not
    discard successful earlier outcomes.

    Args:
        inputs: Canonical rows loaded from one extraction run.
        configs: Optional settings; defaults to the frozen 19-run demo.
        progress: Optional callback receiving phase, completed units, and total.

    Returns:
        Structured partial-or-complete suite evidence.
    """

    started = time.monotonic()
    requested = _deduplicate_configs(configs or DEMO_CONFIGURATIONS)
    suite_limit = min(config.solver_limits.suite_time_limit_seconds for config in requested)
    deadline = started + suite_limit
    pool_cache: dict[str, tuple[CandidatePool, ...]] = {}
    pool_users: dict[str, list[str]] = {}

    def pools_for(config: RunConfig) -> tuple[CandidatePool, ...]:
        """Return or generate the pool family for one structural fingerprint."""

        fingerprint = config.structural_ruleset_fingerprint()
        pool_users.setdefault(fingerprint, []).append(config.scenario_id)
        if fingerprint not in pool_cache:
            pool_cache[fingerprint] = generate_candidate_pools(
                inputs.members,
                inputs.production_versions,
                config,
                progress,
                inputs.baseline_evidence_members or inputs.members,
            )
        return pool_cache[fingerprint]

    baseline_config = RunConfig(scenario_id="baseline_recalculated")
    try:
        baseline_pools = pools_for(baseline_config)
        baseline_solve = _baseline_result(inputs, baseline_pools, baseline_config)
        baseline = ScenarioOutcome(
            summary_id="baseline_recalculated",
            kind="baseline",
            scenario_id="baseline_recalculated",
            configuration_id=None,
            config=baseline_config,
            status=baseline_solve.status,
            result_class=baseline_solve.result_class,
            solve_result=baseline_solve,
            structural_fingerprint=baseline_config.structural_ruleset_fingerprint(),
            pool_hashes=tuple(pool.pool_hash for pool in baseline_pools),
            pools=tuple(baseline_pools),
            acceptance=assess_baseline_guardrails(
                inputs, baseline_pools, baseline_config, baseline_solve
            ),
        )
    except Exception as exc:
        baseline = ScenarioOutcome(
            summary_id="baseline_recalculated",
            kind="baseline",
            scenario_id="baseline_recalculated",
            configuration_id=None,
            config=baseline_config,
            status="failed",
            result_class="baseline-unavailable",
            acceptance=BaselineGuardrailAssessment(
                BaselineAcceptanceStatus.NOT_ACCEPTABLE,
                ("baseline_recalculation_failed",),
            ),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    outcomes: list[ScenarioOutcome] = []
    total = len(requested)
    for completed, config in enumerate(requested, start=1):
        if time.monotonic() >= deadline:
            outcome = _timeout_outcome(config)
        else:
            try:
                pools = pools_for(config)
                remaining = max(deadline - time.monotonic(), 1e-3)
                runtime_config = config.model_copy(
                    update={
                        "solver_limits": config.solver_limits.model_copy(
                            update={
                                "time_limit_seconds": min(
                                    config.solver_limits.time_limit_seconds, remaining
                                )
                            }
                        )
                    }
                )
                result = (
                    solve_candidate_pools(pools, inputs.members, runtime_config)
                    if config.coverage_mode is CoverageMode.GROUP_MEAN
                    else solve_decomposed_pools(pools, inputs.members, runtime_config)
                )
                acceptance = assess_baseline_guardrails(
                    inputs, pools, config, result
                )
                outcome = ScenarioOutcome(
                    summary_id=config.scenario_id,
                    kind="scenario",
                    scenario_id=config.scenario_id,
                    configuration_id=config.configuration_id(),
                    config=config,
                    status=result.status,
                    result_class=result.result_class,
                    solve_result=result,
                    structural_fingerprint=config.structural_ruleset_fingerprint(),
                    pool_hashes=tuple(pool.pool_hash for pool in pools),
                    pools=tuple(pools),
                    acceptance=acceptance,
                )
            except Exception as exc:  # Preserve completed scenarios in partial suites.
                outcome = ScenarioOutcome(
                    summary_id=config.scenario_id,
                    kind="scenario",
                    scenario_id=config.scenario_id,
                    configuration_id=config.configuration_id(),
                    config=config,
                    status="failed",
                    result_class="scenario-unavailable",
                    structural_fingerprint=config.structural_ruleset_fingerprint(),
                    acceptance=BaselineGuardrailAssessment(
                        BaselineAcceptanceStatus.NOT_ACCEPTABLE,
                        ("scenario_failed",),
                    ),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        outcomes.append(outcome)
        if progress:
            progress("scenarios", completed, total)

    successful_core = {
        outcome.scenario_id: outcome.solve_result
        for outcome in outcomes
        if outcome.scenario_id in CORE_SCENARIO_IDS and outcome.solve_result is not None
    }
    frontier_config = next(
        (config for config in requested if config.scenario_id == "core_max"),
        RunConfig(scenario_id="pareto_portfolio", coverage_mode=CoverageMode.MAX),
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        pareto = tuple(
            ParetoPoint(
                point_index=index,
                epsilon_j_ch=None,
                status="suite_timeout",
                result_class="portfolio-unavailable",
                error_type="TimeoutError",
                error_message="complete scenario-suite time limit exhausted",
            )
            for index in range(1, 6)
        )
    else:
        frontier_config = frontier_config.model_copy(
            update={
                "solver_limits": frontier_config.solver_limits.model_copy(
                    update={
                        "time_limit_seconds": min(
                            frontier_config.solver_limits.time_limit_seconds, remaining
                        )
                    }
                )
            }
        )
        pareto = build_portfolio_frontier(successful_core, frontier_config, 5)
    if progress:
        progress("pareto", len(pareto), 5)
    cache_records = tuple(
        PoolCacheRecord(
            structural_fingerprint=fingerprint,
            pool_hashes=tuple(pool.pool_hash for pool in pool_cache[fingerprint]),
            block_count=len(pool_cache[fingerprint]),
            scenario_ids=tuple(pool_users.get(fingerprint, ())),
        )
        for fingerprint in pool_cache
    )
    return ScenarioSuiteResult(
        inputs=inputs,
        baseline=baseline,
        scenarios=tuple(outcomes),
        pareto_points=pareto,
        pool_cache=cache_records,
        elapsed_seconds=time.monotonic() - started,
    )
