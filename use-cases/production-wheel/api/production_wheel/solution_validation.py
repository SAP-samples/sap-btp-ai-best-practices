"""Independent post-solve structural and KPI validation.

This module deliberately does not call the optimizer's coefficient builder. It
reconstructs each selected group from canonical FINI primitives so a solver or
candidate-library defect cannot validate itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Any, Iterable, Mapping

from production_wheel.candidates import (
    CandidateMember,
    CandidatePool,
    ProductionVersion,
)
from production_wheel.candidates.models import (
    MatrixEvidence,
    MatrixException,
    stable_hash,
)
from production_wheel.matrix import (
    MATRIX_EXCEPTION_STATUSES,
    UNKNOWN_LINE_FEASIBLE,
    UNKNOWN_NO_CURRENT_LINE_OVERLAP,
    configured_matrix,
    matrix_status_lookup,
    normalize_volume,
)
from production_wheel.metrics import (
    changeover_contribution,
    effective_batch,
    group_coverage,
    group_frequency,
    pallet_allocations,
    proportional_allocations,
)
from production_wheel.optimization import SolveResult
from production_wheel.schemas import CoverageBasis, MatrixMode, PVMode, RunConfig

_REL_TOLERANCE = 1e-9
_ABS_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One independently observed validation finding.

    Args:
        rule_id: Stable identifier for the checked rule.
        severity: ``error`` for invalid results or ``warning`` for proof caveats.
        entity: Type of object checked, such as solution or selected group.
        entity_key: Stable key for the checked object.
        observed: Independently observed value.
        expected: Required or solver-reported value.
        message: Short action-oriented explanation.
    """

    rule_id: str
    severity: str
    entity: str
    entity_key: str
    observed: Any
    expected: Any
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic CSV/JSON-ready representation of the finding."""

        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "entity": self.entity,
            "entity_key": self.entity_key,
            "observed": self.observed,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GroupEvidence:
    """Independently recomputed evidence for one selected subgroup.

    All KPI fields come from canonical member demand, pallet, line, package,
    and PV primitives. Allocation tuples are sorted by FINI identifier.
    """

    group_id: str
    block_key: tuple[str, str]
    member_ids: tuple[str, ...]
    membership_hash: str
    pv_id: str
    nominal_lot_litres: float
    effective_batch_litres: float
    group_demand_litres: float
    common_lines: tuple[str, ...]
    coverage_days: float
    base_group_coverage_days: float
    adjusted_group_coverage_days: float
    worst_fini_coverage_days: float
    fini_adjusted_coverage_days: tuple[tuple[str, float], ...]
    frequency_per_week: float
    j_ch_contribution: float
    nominal_allocations: tuple[tuple[str, float], ...]
    selected_allocations: tuple[tuple[str, float], ...]
    matrix_pair_evidence: tuple[MatrixEvidence, ...]
    matrix_exception_pairs: tuple[MatrixException, ...]
    matrix_unknown_pairs: tuple[MatrixEvidence, ...]
    matrix_positive_evidence_pairs: tuple[MatrixEvidence, ...]
    package_volumes: tuple[tuple[str, float], ...]
    pck_codes: tuple[str, ...]
    equal_pck: bool
    relaxed_group: bool
    size_excess: int
    baseline_changed: bool
    pv_changed: bool
    selected_line: str | None = None

    @property
    def group_size(self) -> int:
        """Return the number of FINIs in the selected subgroup."""

        return len(self.member_ids)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Independent validation outcome and its recomputed selected groups."""

    issues: tuple[ValidationIssue, ...]
    groups: tuple[GroupEvidence, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether no error-severity finding was observed."""

        return not any(issue.severity == "error" for issue in self.issues)

    def issue_rows(self) -> tuple[dict[str, Any], ...]:
        """Return findings as deterministic CSV/JSON-ready dictionaries."""

        return tuple(issue.as_dict() for issue in self.issues)


def deterministic_group_id(
    block_key: tuple[str, str], member_ids: Iterable[str]
) -> str:
    """Return a stable proposed subgroup ID from a sorted block/member tuple.

    Args:
        block_key: Plant and SEFI decomposition key.
        member_ids: FINIs assigned to the group in any input order.

    Returns:
        Stable identifier that is unchanged by solver candidate ordering.
    """

    identity = {"block": tuple(block_key), "members": tuple(sorted(member_ids))}
    return f"PROPOSED-{stable_hash(identity)[:16].upper()}"


def _issue(
    issues: list[ValidationIssue],
    rule_id: str,
    entity_key: str,
    observed: Any,
    expected: Any,
    message: str,
    *,
    severity: str = "error",
    entity: str = "selected_group",
) -> None:
    """Append one normalized validation finding to an issue accumulator."""

    issues.append(
        ValidationIssue(
            rule_id, severity, entity, entity_key, observed, expected, message
        )
    )


def _same_number(observed: float, expected: float) -> bool:
    """Return whether two finite numeric values agree at audit tolerance."""

    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=_REL_TOLERANCE,
        abs_tol=_ABS_TOLERANCE,
    )


def _baseline_members(
    members: Iterable[CandidateMember],
) -> dict[tuple[str, str, str], frozenset[str]]:
    """Index each baseline-assigned FINI to its original block membership."""

    groups: dict[tuple[str, str, str], set[str]] = {}
    for member in members:
        if member.baseline_group:
            groups.setdefault(
                (member.plant, member.sefi, member.baseline_group), set()
            ).add(member.fini_id)
    return {
        (plant, sefi, fini_id): frozenset(group_members)
        for (plant, sefi, _), group_members in groups.items()
        for fini_id in group_members
    }


def _matrix_pair_evidence(
    selected_members: tuple[CandidateMember, ...],
    matrix_lookup: Mapping[tuple[Decimal, Decimal], str],
) -> tuple[MatrixEvidence, ...]:
    """Recompute every package-volume pair and its governed evidence status."""

    evidence: list[MatrixEvidence] = []
    for left, right in combinations(selected_members, 2):
        key = (
            normalize_volume(left.package_volume),
            normalize_volume(right.package_volume),
        )
        status = matrix_lookup.get(key)
        if status is None:
            raise ValueError(f"volume pair is absent from configured matrix: {key}")
        evidence.append(
            (
                left.fini_id,
                right.fini_id,
                left.package_volume,
                right.package_volume,
                status,
            )
        )
    return tuple(evidence)


def _valid_optimized_versions(
    production_versions: Iterable[ProductionVersion],
) -> dict[tuple[str, str, str], frozenset[float]]:
    """Index finite active optimized-PV alternatives by block and PV alias."""

    values: dict[tuple[str, str, str], set[float]] = {}
    for version in production_versions:
        if version.is_positive:
            values.setdefault(
                (version.plant, version.sefi, version.pv_id), set()
            ).add(float(version.nominal_lot_litres))  # type: ignore[arg-type]
    return {key: frozenset(lots) for key, lots in values.items()}


def validate_solution(
    result: SolveResult,
    members: Iterable[CandidateMember],
    pools: Iterable[CandidatePool],
    config: RunConfig,
    production_versions: Iterable[ProductionVersion] = (),
    baseline_evidence_members: Iterable[CandidateMember] = (),
) -> ValidationResult:
    """Independently validate a completed exact-cover solution.

    Args:
        result: Solver outcome whose selected candidates must be checked.
        members: Complete modeled FINI population that must be covered once.
        pools: Candidate libraries used to obtain the result.
        config: Active structural, PV, pallet, coverage, and calendar settings.
        production_versions: Finite active PV catalog required in optimized mode.
        baseline_evidence_members: Complete historical assignments used for the
            same empirical matrix evidence as candidate generation.

    Returns:
        Structured findings and independently recomputed group evidence. The
        result is invalid when any finding has ``error`` severity.
    """

    ordered_members = tuple(
        sorted(members, key=lambda member: (member.plant, member.sefi, member.fini_id))
    )
    historical_members = tuple(baseline_evidence_members) or ordered_members
    pool_rows = tuple(pools)
    issues: list[ValidationIssue] = []
    groups: list[GroupEvidence] = []
    matrix_lookup: dict[tuple[Decimal, Decimal], str] = {}
    if config.matrix_mode is not MatrixMode.OFF or config.matrix_pairs:
        try:
            matrix_lookup = matrix_status_lookup(
                configured_matrix(
                    config,
                    ordered_members,
                    historical_members,
                )
            )
        except ValueError as error:
            _issue(
                issues,
                "MATRIX_VERSION_SUPPORTED",
                "solution",
                str(error),
                config.versions.matrix_version,
                "The configured package-volume matrix must be buildable from canonical inputs.",
                entity="solution",
            )
    by_key = {
        (member.plant, member.sefi, member.fini_id): member
        for member in ordered_members
    }
    if len(by_key) != len(ordered_members):
        _issue(
            issues,
            "MODELED_MEMBER_KEYS_UNIQUE",
            "solution",
            len(ordered_members) - len(by_key),
            0,
            "Modeled plant/SEFI/FINI keys must be unique.",
            entity="solution",
        )
    if not result.has_incumbent:
        _issue(
            issues,
            "SOLUTION_HAS_INCUMBENT",
            "solution",
            result.status,
            "feasible incumbent",
            "A solution without an incumbent cannot be exported as a proposal.",
            entity="solution",
        )

    pool_candidates = {
        candidate.candidate_hash
        for pool in pool_rows
        for candidate in pool.candidates
    }
    valid_versions = _valid_optimized_versions(production_versions)
    baseline_by_fini = _baseline_members(ordered_members)
    covered: list[tuple[str, str, str]] = []

    for selected in sorted(
        result.selected,
        key=lambda item: (
            item.candidate.block_key,
            tuple(sorted(item.candidate.member_ids)),
            item.candidate.pv_id,
        ),
    ):
        candidate = selected.candidate
        member_ids = tuple(sorted(candidate.member_ids))
        group_id = deterministic_group_id(candidate.block_key, member_ids)
        entity_key = group_id
        if candidate.candidate_hash not in pool_candidates:
            _issue(
                issues,
                "SELECTED_CANDIDATE_IN_LIBRARY",
                entity_key,
                candidate.candidate_hash,
                "candidate hash present in disclosed pool",
                "Selected candidate was not found in the supplied candidate library.",
            )
        keys = tuple((*candidate.block_key, fini_id) for fini_id in member_ids)
        unknown = tuple(key for key in keys if key not in by_key)
        if unknown:
            _issue(
                issues,
                "SELECTED_MEMBERS_KNOWN",
                entity_key,
                unknown,
                "all members present in modeled population",
                "The selected group references unknown or cross-block FINIs.",
            )
            continue
        selected_members = tuple(by_key[key] for key in keys)
        covered.extend(keys)
        actual_blocks = {member.block_key for member in selected_members}
        if actual_blocks != {candidate.block_key}:
            _issue(
                issues,
                "GROUP_BLOCK_BOUNDARY",
                entity_key,
                sorted(actual_blocks),
                [candidate.block_key],
                "A selected group cannot cross plant or SEFI boundaries.",
            )

        common_lines = set(selected_members[0].eligible_lines)
        for member in selected_members[1:]:
            common_lines.intersection_update(member.eligible_lines)
        stable_lines = tuple(sorted(common_lines))
        if not stable_lines:
            _issue(
                issues,
                "COMMON_LINE_INTERSECTION",
                entity_key,
                [],
                "one or more common eligible lines",
                "Selected FINIs do not share a known eligible filling line.",
            )
        if stable_lines != tuple(candidate.common_lines):
            _issue(
                issues,
                "COMMON_LINE_EVIDENCE",
                entity_key,
                candidate.common_lines,
                stable_lines,
                "Candidate common-line evidence differs from canonical primitives.",
            )
        if (config.assign_filling_lines or candidate.selected_line is not None) and candidate.selected_line not in stable_lines:
            _issue(issues, "SELECTED_LINE_ELIGIBLE", entity_key,
                   candidate.selected_line, stable_lines,
                   "The selected line must be eligible for every canonical group member.")
        if len(member_ids) > config.cap_for_block(*candidate.block_key):
            _issue(
                issues,
                "GROUP_SIZE_CAP",
                entity_key,
                len(member_ids),
                config.cap_for_block(*candidate.block_key),
                "Selected group exceeds the active size cap.",
            )

        if config.pv_mode is PVMode.FIXED:
            fixed_pvs = {member.fixed_pv for member in selected_members}
            if fixed_pvs != {candidate.pv_id}:
                _issue(
                    issues,
                    "FIXED_PV_AGREEMENT",
                    entity_key,
                    sorted("<missing>" if value is None else value for value in fixed_pvs),
                    candidate.pv_id,
                    "Every member must agree with the selected planning-supplied PV.",
                )
            catalog_lots = valid_versions.get(
                (*candidate.block_key, candidate.pv_id), frozenset()
            )
            if catalog_lots != {candidate.nominal_lot_litres}:
                _issue(
                    issues,
                    "FIXED_PV_LOT_UNIQUENESS",
                    entity_key,
                    sorted(catalog_lots),
                    [candidate.nominal_lot_litres],
                    "Fixed mode requires one positive planning-supplied catalog lot with no fallback.",
                )
            supplied_lots = {
                float(member.fixed_lot_litres)
                for member in selected_members
                if member.fixed_lot_litres is not None
            }
            if supplied_lots and supplied_lots != {candidate.nominal_lot_litres}:
                _issue(
                    issues,
                    "FIXED_MEMBER_LOT_EVIDENCE",
                    entity_key,
                    sorted(supplied_lots),
                    candidate.nominal_lot_litres,
                    "Optional member-level lot evidence must agree with the catalog lot.",
                )
        else:
            lots = valid_versions.get((*candidate.block_key, candidate.pv_id), frozenset())
            if lots != {candidate.nominal_lot_litres}:
                _issue(
                    issues,
                    "OPTIMIZED_PV_ACTIVE_ALTERNATIVE",
                    entity_key,
                    sorted(lots),
                    [candidate.nominal_lot_litres],
                    "Optimized PV must be one unique finite active block alternative.",
                )

        try:
            pair_evidence = (
                ()
                if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs
                else _matrix_pair_evidence(selected_members, matrix_lookup)
            )
            exceptions = tuple(
                (left, right, left_volume, right_volume)
                for left, right, left_volume, right_volume, status in pair_evidence
                if status in MATRIX_EXCEPTION_STATUSES
            )
            unknown_pairs = tuple(
                pair
                for pair in pair_evidence
                if pair[4]
                in {UNKNOWN_LINE_FEASIBLE, UNKNOWN_NO_CURRENT_LINE_OVERLAP}
            )
            positive_pairs = tuple(
                pair
                for pair in pair_evidence
                if pair[4]
                not in {
                    *MATRIX_EXCEPTION_STATUSES,
                    UNKNOWN_LINE_FEASIBLE,
                    UNKNOWN_NO_CURRENT_LINE_OVERLAP,
                }
            )
        except ValueError as error:
            _issue(
                issues,
                "MATRIX_VOLUME_CATALOG",
                entity_key,
                str(error),
                config.versions.matrix_version,
                "Active matrix validation requires a governed package volume.",
            )
            pair_evidence = ()
            exceptions = ()
            unknown_pairs = ()
            positive_pairs = ()
        prohibited = tuple(pair for pair in pair_evidence if pair[4] == "N")
        if prohibited:
            _issue(issues, "PROFILE_MATRIX_PROHIBITION", entity_key, prohibited, [],
                   "Explicit profile N pairs are forbidden in every matrix mode.")
        if config.matrix_mode is MatrixMode.HARD and exceptions:
            _issue(
                issues,
                "HARD_MATRIX_COMPATIBILITY",
                entity_key,
                exceptions,
                [],
                "HARD matrix mode forbids every incompatible volume pair.",
            )
        if tuple(candidate.matrix_exception_pairs) != exceptions:
            _issue(
                issues,
                "MATRIX_EXCEPTION_EVIDENCE",
                entity_key,
                candidate.matrix_exception_pairs,
                exceptions,
                "Candidate matrix evidence differs from independent pair evaluation.",
            )
        if tuple(candidate.matrix_unknown_pairs) != unknown_pairs:
            _issue(
                issues,
                "MATRIX_UNKNOWN_EVIDENCE",
                entity_key,
                candidate.matrix_unknown_pairs,
                unknown_pairs,
                "Candidate unknown-pair evidence differs from independent evaluation.",
            )
        if tuple(candidate.matrix_evidence_pairs) != positive_pairs:
            _issue(
                issues,
                "MATRIX_POSITIVE_EVIDENCE",
                entity_key,
                candidate.matrix_evidence_pairs,
                positive_pairs,
                "Candidate positive-pair evidence differs from independent evaluation.",
            )

        try:
            effective = float(
                effective_batch(candidate.nominal_lot_litres, config.canonical_factor)
            )
        except ValueError as error:
            _issue(
                issues,
                "PV_LOT_POSITIVE_FINITE",
                entity_key,
                candidate.nominal_lot_litres,
                "positive finite nominal lot",
                str(error),
            )
            continue
        if not _same_number(candidate.effective_batch_litres, effective):
            _issue(
                issues,
                "EFFECTIVE_BATCH_FACTOR",
                entity_key,
                candidate.effective_batch_litres,
                effective,
                "Effective batch must use the canonical factor from RunConfig.",
            )
        demands = {
            member.fini_id: Decimal(str(member.demand_litres))
            for member in selected_members
        }
        pallets = {
            member.fini_id: Decimal(str(member.pallet_litres))
            for member in selected_members
        }
        nominal = proportional_allocations(demands, effective)
        adjusted = pallet_allocations(nominal, pallets, config.pallet_formula)
        demand = float(sum(demands.values(), Decimal(0)))
        coverage_views = {
            basis: float(
                group_coverage(
                    basis,
                    demands,
                    effective,
                    adjusted,
                    config.demand_days,
                )
            )
            for basis in CoverageBasis
        }
        coverage = coverage_views[config.coverage_basis]
        fini_adjusted_coverage = tuple(
            (
                fini_id,
                float(Decimal(config.demand_days) * adjusted[fini_id] / demands[fini_id]),
            )
            for fini_id in sorted(adjusted)
        )
        frequency = float(
            group_frequency(demand, effective, config.productive_weeks)
        )
        j_ch = float(changeover_contribution(frequency, len(member_ids)))
        checks = (
            ("GROUP_DEMAND_COEFFICIENT", candidate.group_demand_litres, demand),
            ("COVERAGE_COEFFICIENT", selected.coefficients.coverage_days, coverage),
            ("J_CH_COEFFICIENT", selected.coefficients.j_ch, j_ch),
            ("GROUP_DEMAND_SOLVER_COEFFICIENT", selected.coefficients.group_demand_litres, demand),
        )
        for rule_id, observed, expected in checks:
            if not _same_number(observed, expected):
                _issue(
                    issues,
                    rule_id,
                    entity_key,
                    observed,
                    expected,
                    "Stored solver/candidate coefficient differs from independent recomputation.",
                )
        expected_violation = int(
            coverage < config.target_band.lower_days
            or coverage > config.target_band.upper_days
        )
        expected_excess = max(
            config.target_band.lower_days - coverage,
            coverage - config.target_band.upper_days,
            0.0,
        )
        discrete_checks = (
            ("TARGET_VIOLATION_COEFFICIENT", selected.coefficients.target_violation, expected_violation),
            ("RELAXED_GROUP_COEFFICIENT", selected.coefficients.relaxed_group, int(len(member_ids) > 7)),
            ("GROUP_SIZE_EXCESS_COEFFICIENT", selected.coefficients.size_excess, max(len(member_ids) - 7, 0)),
            ("MATRIX_EXCEPTION_COEFFICIENT", selected.coefficients.matrix_exceptions, len(exceptions)),
        )
        for rule_id, observed, expected in discrete_checks:
            if observed != expected:
                _issue(
                    issues,
                    rule_id,
                    entity_key,
                    observed,
                    expected,
                    "Stored discrete solver coefficient differs from independent recomputation.",
                )
        if not _same_number(selected.coefficients.target_excess_days, expected_excess):
            _issue(
                issues,
                "TARGET_EXCESS_COEFFICIENT",
                entity_key,
                selected.coefficients.target_excess_days,
                expected_excess,
                "Stored target excess differs from independent recomputation.",
            )

        member_set = frozenset(member_ids)
        baseline_changed = any(
            baseline_by_fini.get((*candidate.block_key, fini_id), frozenset())
            != member_set
            for fini_id in member_ids
        )
        pck_codes = tuple(
            sorted(
                {
                    member.pck_code
                    for member in selected_members
                    if member.pck_code not in (None, "")
                }
            )
        )
        groups.append(
            GroupEvidence(
                group_id=group_id,
                block_key=candidate.block_key,
                member_ids=member_ids,
                membership_hash=stable_hash(
                    {"block": candidate.block_key, "members": member_ids}
                ),
                pv_id=candidate.pv_id,
                nominal_lot_litres=candidate.nominal_lot_litres,
                effective_batch_litres=effective,
                group_demand_litres=demand,
                common_lines=stable_lines,
                selected_line=candidate.selected_line,
                coverage_days=coverage,
                base_group_coverage_days=coverage_views[CoverageBasis.BASE_GROUP],
                adjusted_group_coverage_days=coverage_views[
                    CoverageBasis.ADJUSTED_GROUP
                ],
                worst_fini_coverage_days=coverage_views[CoverageBasis.WORST_FINI],
                fini_adjusted_coverage_days=fini_adjusted_coverage,
                frequency_per_week=frequency,
                j_ch_contribution=j_ch,
                nominal_allocations=tuple(
                    (fini_id, float(nominal[fini_id])) for fini_id in sorted(nominal)
                ),
                selected_allocations=tuple(
                    (fini_id, float(adjusted[fini_id])) for fini_id in sorted(adjusted)
                ),
                matrix_pair_evidence=pair_evidence,
                matrix_exception_pairs=exceptions,
                matrix_unknown_pairs=unknown_pairs,
                matrix_positive_evidence_pairs=positive_pairs,
                package_volumes=tuple(
                    (member.fini_id, member.package_volume)
                    for member in selected_members
                ),
                pck_codes=pck_codes,
                equal_pck=(
                    len(pck_codes) == 1
                    and all(member.pck_code not in (None, "") for member in selected_members)
                ),
                relaxed_group=len(member_ids) > 7,
                size_excess=max(len(member_ids) - 7, 0),
                baseline_changed=baseline_changed,
                pv_changed=any(
                    member.fixed_pv != candidate.pv_id
                    for member in selected_members
                ),
            )
        )

    if config.business_rules or config.assign_filling_lines:
        from production_wheel.business_rules import selected_rule_failures

        try:
            failures = selected_rule_failures(
                tuple(item.candidate for item in result.selected), ordered_members, config
            )
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            failures = [("BUSINESS_RULE_INPUTS", str(error), "valid canonical rule inputs")]
        for rule_id, observed, expected in failures:
            _issue(issues, rule_id, "solution", observed, expected,
                   "Independent reconstruction failed a configured business rule or selection bound.",
                   entity="solution")

    expected_keys = set(by_key)
    covered_counts = {key: covered.count(key) for key in set(covered)}
    missing = tuple(sorted(expected_keys - set(covered)))
    repeated = tuple(sorted(key for key, count in covered_counts.items() if count != 1))
    if missing or repeated or len(covered) != len(expected_keys):
        _issue(
            issues,
            "EXACT_COVER",
            "solution",
            {"missing": missing, "repeated": repeated, "selected_rows": len(covered)},
            {"missing": (), "repeated": (), "selected_rows": len(expected_keys)},
            "Every modeled FINI must appear in exactly one selected subgroup.",
            entity="solution",
        )

    pool_completeness = (
        "complete"
        if pool_rows and all(pool.completeness == "complete" for pool in pool_rows)
        else "restricted"
    )
    if result.pool_completeness != pool_completeness:
        _issue(
            issues,
            "POOL_COMPLETENESS_EVIDENCE",
            "solution",
            result.pool_completeness,
            pool_completeness,
            "Result completeness must match the disclosed candidate libraries.",
            entity="solution",
        )
    if pool_completeness == "restricted":
        _issue(
            issues,
            "RESTRICTED_LIBRARY_PROOF_CAVEAT",
            "solution",
            result.result_class,
            "within disclosed candidate library only",
            "A restricted-library optimum is never proof of global optimality.",
            severity="warning",
            entity="solution",
        )

    return ValidationResult(
        tuple(
            sorted(
                issues,
                key=lambda issue: (
                    issue.severity != "error",
                    issue.rule_id,
                    issue.entity_key,
                ),
            )
        ),
        tuple(sorted(groups, key=lambda group: (group.block_key, group.member_ids))),
    )
