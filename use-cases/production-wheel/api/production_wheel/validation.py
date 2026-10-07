"""Deterministic request validation and capability discovery."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import InvalidOperation
from enum import StrEnum
from typing import Iterable, Mapping

from production_wheel.matrix import (
    BASELINE_EMPIRICAL_MATRIX_VERSION,
    CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
    CUSTOMER_VOLUME_CATALOG,
    MODELED_VOLUMES,
    SYNTHETIC_MATRIX_VERSION,
    normalize_volume,
)
from production_wheel.schemas import (
    AllowedPVsConstraint,
    ApprovalStatus,
    BaselineAcceptanceStatus,
    CannotLinkConstraint,
    CoverageBoundConstraint,
    CoverageBasis,
    CoverageMode,
    Enforcement,
    FiniDispositionConstraint,
    FixedPVConstraint,
    FreezeAssignmentConstraint,
    GreenfieldValidationStatus,
    GroupSizeRelaxationConstraint,
    MatrixMode,
    MaxGroupSizeConstraint,
    MustLinkConstraint,
    PVMode,
    PalletFormula,
    RequiredLinesConstraint,
    RunConfig,
    SolveRequest,
    VolumeCompatibilityOverrideConstraint,
)


class ValidationStatus(StrEnum):
    """Outcome of deterministic request validation."""

    VALID = "valid"
    MANUAL_REVIEW = "manual_review"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Known identifiers used to resolve typed constraints."""

    materials: frozenset[str] = frozenset()
    production_versions_by_block: Mapping[tuple[str, str], frozenset[str]] | None = None
    filling_lines: frozenset[str | int] = frozenset(str(line) for line in range(1, 7))


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    """One deterministic validation finding."""

    code: str
    severity: str
    message: str
    constraint_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Validated request plus findings and regeneration guidance."""

    status: ValidationStatus
    request: SolveRequest
    messages: tuple[ValidationMessage, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the request can proceed automatically."""

        return self.status == ValidationStatus.VALID


class ChangeClass(StrEnum):
    """Candidate-library impact of a configuration change."""

    REUSE = "reuse"
    NARROWING = "narrowing"
    WIDENING = "widening"


@dataclass(frozen=True, slots=True)
class ConfigChange:
    """Summarize whether a change filters or regenerates candidates."""

    classification: ChangeClass
    reasons: tuple[str, ...]


def _unknown_materials(materials: Iterable[str], context: ValidationContext) -> set[str]:
    """Return unresolved material identifiers when master data are supplied."""

    return set(materials) - set(context.materials) if context.materials else set()


def validate_request(request: SolveRequest, context: ValidationContext | None = None) -> ValidationResult:
    """Validate identifiers, contradictions, and supported v1 limits."""

    context = context or ValidationContext()
    messages: list[ValidationMessage] = []
    constraints = (*request.config.business_rules, *request.constraints)
    constraint_ids = [constraint.constraint_id for constraint in constraints]
    duplicate_ids = sorted(
        identifier for identifier in set(constraint_ids) if constraint_ids.count(identifier) > 1
    )
    if duplicate_ids:
        messages.append(
            ValidationMessage(
                "DUPLICATE_CONSTRAINT_ID",
                "error",
                f"constraint identifiers must be unique: {duplicate_ids}",
            )
        )
    if request.config.group_size.effective_cap > 9:
        messages.append(
            ValidationMessage(
                "GROUP_CAP_V1_LIMIT",
                "manual_review",
                "v1 supports only caps 7, 8, and 9; larger requests require scale review",
            )
        )
    must_pairs: set[frozenset[str]] = set()
    cannot_pairs: set[frozenset[str]] = set()
    dispositions: dict[str, set[str]] = {}
    size_values: set[int] = set()
    fixed_by_scope: dict[tuple[str | None, str | None], set[str]] = {}
    allowed_by_scope: dict[tuple[str | None, str | None], list[set[str]]] = {}
    coverage_by_scope: dict[tuple[str | None, str | None], list[CoverageBoundConstraint]] = {}
    volume_overrides: dict[tuple[str, str], set[bool]] = {}
    parent: dict[str, str] = {}

    def find(material: str) -> str:
        parent.setdefault(material, material)
        if parent[material] != material:
            parent[material] = find(parent[material])
        return parent[material]

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for constraint in constraints:
        if constraint.approval_status == ApprovalStatus.REJECTED:
            messages.append(
                ValidationMessage(
                    "REJECTED_CONSTRAINT",
                    "error",
                    "rejected constraints cannot be executed",
                    constraint.constraint_id,
                )
            )
        elif constraint.approval_status != ApprovalStatus.APPROVED:
            messages.append(
                ValidationMessage(
                    "CONSTRAINT_APPROVAL_REQUIRED",
                    "manual_review",
                    "constraint must be approved before compilation",
                    constraint.constraint_id,
                )
            )
        if constraint.enforcement == Enforcement.BOUNDED_RELAXATION and not isinstance(
            constraint, (GroupSizeRelaxationConstraint, CoverageBoundConstraint)
        ):
            messages.append(
                ValidationMessage(
                    "UNSUPPORTED_BOUNDED_ENFORCEMENT",
                    "error",
                    "bounded relaxation is supported only for size and coverage rules",
                    constraint.constraint_id,
                )
            )
        materials: tuple[str, ...] = ()
        if isinstance(constraint, (MustLinkConstraint, FreezeAssignmentConstraint)):
            materials = constraint.materials
        elif isinstance(constraint, CannotLinkConstraint):
            materials = constraint.materials
        elif isinstance(constraint, FiniDispositionConstraint):
            materials = (constraint.material,)
        unknown = _unknown_materials(materials, context)
        if unknown:
            messages.append(
                ValidationMessage(
                    "UNKNOWN_MATERIAL",
                    "error",
                    f"unknown material identifiers: {sorted(unknown)}",
                    constraint.constraint_id,
                )
            )
        if isinstance(constraint, MustLinkConstraint):
            anchor = constraint.materials[0]
            for material in constraint.materials[1:]:
                union(anchor, material)
                must_pairs.add(frozenset((anchor, material)))
        elif isinstance(constraint, CannotLinkConstraint):
            cannot_pairs.add(frozenset(constraint.materials))
        elif isinstance(constraint, FiniDispositionConstraint):
            dispositions.setdefault(constraint.material, set()).add(constraint.action)
        elif isinstance(constraint, MaxGroupSizeConstraint):
            size_values.add(constraint.maximum)
            if constraint.maximum > 9:
                messages.append(
                    ValidationMessage(
                        "GROUP_CAP_V1_LIMIT",
                        "manual_review",
                        "maximum-size constraint above nine requires scale review",
                        constraint.constraint_id,
                    )
                )
        elif isinstance(constraint, GroupSizeRelaxationConstraint):
            size_values.add(constraint.base_limit + constraint.max_excess)
            if constraint.base_limit != 7 or constraint.max_excess > 2:
                messages.append(
                    ValidationMessage(
                        "GROUP_RELAXATION_V1_LIMIT",
                        "manual_review",
                        "v1 supports historical base seven with at most two extra members",
                        constraint.constraint_id,
                    )
                )
        elif isinstance(constraint, RequiredLinesConstraint):
            invalid = set(map(str, constraint.filling_lines)) - set(map(str, context.filling_lines))
            if invalid:
                messages.append(
                    ValidationMessage(
                        "UNKNOWN_FILLING_LINE",
                        "error",
                        f"unknown filling lines: {sorted(invalid)}",
                        constraint.constraint_id,
                    )
                )
        elif isinstance(constraint, VolumeCompatibilityOverrideConstraint):
            normalized_pair: list[str] = []
            for volume in (constraint.volume_a, constraint.volume_b):
                try:
                    normalized = normalize_volume(volume)
                except (InvalidOperation, ValueError):
                    normalized = None
                governed_volumes = (
                    CUSTOMER_VOLUME_CATALOG
                    if request.config.versions.matrix_version
                    == CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
                    else MODELED_VOLUMES
                )
                if normalized not in governed_volumes:
                    messages.append(
                        ValidationMessage(
                            "UNKNOWN_PACKAGE_VOLUME",
                            "error",
                            f"volume is outside the governed catalog: {volume}",
                            constraint.constraint_id,
                        )
                    )
                else:
                    normalized_pair.append(format(normalized, "f"))
            if len(normalized_pair) == 2:
                pair = tuple(sorted(normalized_pair))
                volume_overrides.setdefault(pair, set()).add(constraint.compatible)
        elif isinstance(constraint, (FixedPVConstraint, AllowedPVsConstraint)):
            scope_key = (constraint.scope.plant, constraint.scope.sefi)
            if isinstance(constraint, FixedPVConstraint):
                fixed_by_scope.setdefault(scope_key, set()).add(constraint.production_version)
            else:
                allowed_by_scope.setdefault(scope_key, []).append(
                    set(constraint.production_versions)
                )
            if context.production_versions_by_block is not None:
                block = (constraint.scope.plant or "", constraint.scope.sefi or "")
                allowed = context.production_versions_by_block.get(block, frozenset())
                requested = (
                    {constraint.production_version}
                    if isinstance(constraint, FixedPVConstraint)
                    else set(constraint.production_versions)
                )
                missing = requested - set(allowed)
                if missing:
                    messages.append(
                        ValidationMessage(
                            "UNKNOWN_PRODUCTION_VERSION",
                            "error",
                            f"PV aliases do not resolve in scope {block}: {sorted(missing)}",
                            constraint.constraint_id,
                        )
                    )
        elif isinstance(constraint, CoverageBoundConstraint):
            coverage_by_scope.setdefault(
                (constraint.scope.plant, constraint.scope.sefi), []
            ).append(constraint)
    contradictions = must_pairs & cannot_pairs
    if contradictions:
        messages.append(
            ValidationMessage(
                "MUST_CANNOT_CONTRADICTION",
                "error",
                f"pairs appear in both must-link and cannot-link: {sorted(map(sorted, contradictions))}",
            )
        )
    for material, actions in sorted(dispositions.items()):
        if len(actions) > 1:
            messages.append(
                ValidationMessage(
                    "FINI_DISPOSITION_CONTRADICTION",
                    "error",
                    f"FINI is both included and excluded: {material}",
                )
            )
    # Upper bounds intersect in the compiler by actual plant/SEFI scope.
    # Different maxima alone are not a contradiction (for example caps 2 and 7).
    from production_wheel.rule_models import SelectionBound
    selections = {}
    for rule in constraints:
        if isinstance(rule, SelectionBound):
            key = (rule.scope.plant, rule.scope.sefi, rule.measure.model_dump_json())
            selections.setdefault(key, []).append(rule)
    for key, rules in selections.items():
        lower = max((r.lower for r in rules if r.lower is not None), default=float('-inf'))
        upper = min((r.upper for r in rules if r.upper is not None), default=float('inf'))
        if lower > upper:
            messages.append(ValidationMessage('SELECTION_BOUND_CONTRADICTION', 'error',
                f'selection bounds have an empty interval in scope {key[:2]}'))
    for scope_key, fixed in fixed_by_scope.items():
        if len(fixed) > 1:
            messages.append(
                ValidationMessage(
                    "FIXED_PV_CONTRADICTION",
                    "error",
                    f"multiple fixed PVs in scope {scope_key}: {sorted(fixed)}",
                )
            )
        for allowed in allowed_by_scope.get(scope_key, []):
            if fixed and not (fixed & allowed):
                messages.append(
                    ValidationMessage(
                        "FIXED_ALLOWED_PV_CONTRADICTION",
                        "error",
                        f"fixed PV is outside the allowed set in scope {scope_key}",
                    )
                )
    for scope_key, bounds in coverage_by_scope.items():
        lowers = [bound.lower_days for bound in bounds if bound.lower_days is not None]
        uppers = [bound.upper_days for bound in bounds if bound.upper_days is not None]
        if lowers and uppers and max(lowers) >= min(uppers):
            messages.append(
                ValidationMessage(
                    "COVERAGE_BOUND_CONTRADICTION",
                    "error",
                    f"coverage bounds have an empty interval in scope {scope_key}",
                )
            )
    for pair, values in volume_overrides.items():
        if len(values) > 1:
            messages.append(
                ValidationMessage(
                    "VOLUME_OVERRIDE_CONTRADICTION",
                    "error",
                    f"volume pair has conflicting overrides: {pair}",
                )
            )
    components: dict[str, set[str]] = {}
    for material in parent:
        components.setdefault(find(material), set()).add(material)
    possible_cap = max([request.config.group_size.effective_cap, *size_values,
        *(override.maximum for override in request.config.group_size_overrides)])
    # Actual component/block caps are checked after scope and cap compilation.
    oversized = [sorted(component) for component in components.values() if len(component) > possible_cap]
    if oversized:
        messages.append(
            ValidationMessage(
                "MUST_LINK_EXCEEDS_CAP",
                "error",
                f"must-link component exceeds group cap: {oversized}",
            )
        )
    for pair in cannot_pairs:
        left, right = tuple(pair)
        if find(left) == find(right):
            messages.append(
                ValidationMessage(
                    "TRANSITIVE_LINK_CONTRADICTION",
                    "error",
                    f"cannot-link pair is joined transitively: {sorted(pair)}",
                )
            )
    material_constraints = any(
        isinstance(
            constraint,
            (
                MustLinkConstraint,
                CannotLinkConstraint,
                FiniDispositionConstraint,
                FreezeAssignmentConstraint,
            ),
        )
        for constraint in request.constraints
    )
    if material_constraints and not context.materials:
        messages.append(
            ValidationMessage(
                "MASTER_DATA_CONTEXT_REQUIRED",
                "manual_review",
                "material-scoped constraints require current master-data identifiers",
            )
        )
    if any(message.severity == "error" for message in messages):
        status = ValidationStatus.INVALID
    elif any(message.severity == "manual_review" for message in messages):
        status = ValidationStatus.MANUAL_REVIEW
    else:
        status = ValidationStatus.VALID
    return ValidationResult(status, request, tuple(messages))


def classify_config_change(previous: RunConfig, current: RunConfig) -> ConfigChange:
    """Classify whether a configuration change reuses, filters, or widens a pool."""

    widening: list[str] = []
    narrowing: list[str] = []
    if current.group_size.effective_cap > previous.group_size.effective_cap:
        widening.append("group cap increased")
    elif current.group_size.effective_cap < previous.group_size.effective_cap:
        narrowing.append("group cap decreased")
    matrix_rank = {
        MatrixMode.HARD: 0,
        MatrixMode.DIAGNOSTIC: 1,
        MatrixMode.FLEXIBLE: 1,
        MatrixMode.OFF: 2,
    }
    if matrix_rank[current.matrix_mode] > matrix_rank[previous.matrix_mode]:
        widening.append("matrix enforcement relaxed")
    elif matrix_rank[current.matrix_mode] < matrix_rank[previous.matrix_mode]:
        narrowing.append("matrix enforcement tightened")
    if previous.pv_mode == PVMode.FIXED and current.pv_mode == PVMode.OPTIMIZED:
        widening.append("PV alternatives enabled")
    elif previous.pv_mode == PVMode.OPTIMIZED and current.pv_mode == PVMode.FIXED:
        narrowing.append("PV alternatives disabled")
    if previous.versions.matrix_version != current.versions.matrix_version:
        widening.append("matrix version changed")
    if previous.versions.ruleset_version != current.versions.ruleset_version:
        widening.append("structural ruleset version changed")
    if previous.pool_limits != current.pool_limits:
        widening.append("candidate-pool limits changed")
    if (
        previous.uses_baseline_candidate_seeds
        != current.uses_baseline_candidate_seeds
    ):
        widening.append("baseline candidate-seed policy changed")
    if widening:
        return ConfigChange(ChangeClass.WIDENING, tuple(widening + narrowing))
    if narrowing:
        return ConfigChange(ChangeClass.NARROWING, tuple(narrowing))
    return ConfigChange(ChangeClass.REUSE, ("only objective, KPI, target, or solver settings changed",))


def get_capabilities() -> dict[str, object]:
    """Return the finite optimizer contract exposed to a future agent."""

    defaults = RunConfig()
    return {
        "coverage_modes": [mode.value for mode in CoverageMode],
        "coverage_bases": [basis.value for basis in CoverageBasis],
        "pallet_formulas": [formula.value for formula in PalletFormula],
        "pv_modes": [mode.value for mode in PVMode],
        "matrix_modes": [mode.value for mode in MatrixMode],
        "matrix_versions": {
            BASELINE_EMPIRICAL_MATRIX_VERSION: {
                "role": "default historical evidence",
                "statuses": [
                    "YES_OBSERVED_BASELINE",
                    "UNKNOWN_LINE_FEASIBLE",
                    "UNKNOWN_NO_CURRENT_LINE_OVERLAP",
                ],
                "inferred_no_pairs": 0,
            },
            SYNTHETIC_MATRIX_VERSION: {
                "role": "explicit provisional sensitivity only",
                "statuses": ["YES", "NO"],
            },
            CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1: {
                "role": "greenfield operational compatibility policy; strict comparison optional",
                "statuses": ["Y", "AVOID", "N"],
                "generated_hard_no_pairs": 0,
                "package_volumes": [
                    format(volume, "f") for volume in CUSTOMER_VOLUME_CATALOG
                ],
            },
        },
        "defaults": {
            "coverage_basis": defaults.coverage_basis.value,
            "pallet_formula": defaults.pallet_formula.value,
            "matrix_mode": defaults.matrix_mode.value,
            "matrix_version": defaults.versions.matrix_version,
            "pv_mode": defaults.pv_mode.value,
            "group_cap": defaults.group_size.effective_cap,
        },
        "baseline_acceptance": {
            "statuses": [status.value for status in BaselineAcceptanceStatus],
            "guardrails": [
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
            ],
            "threshold_source": "frozen baseline recalculated under identical settings",
        },
        "greenfield_frontier": {
            "status": GreenfieldValidationStatus.VALID_PARETO_POINT.value,
            "axes": ["demand_weighted_mean_coverage_days", "j_ch"],
            "epsilon_axis": "j_ch",
            "baseline_role": "optional comparison only",
            "proof_scopes": ["exact", "restricted-library", "runtime-limited"],
        },
        "supported_group_caps": [7, 8, 9],
        "package_volumes": [format(volume, "f") for volume in MODELED_VOLUMES],
        "constraint_kinds": [
            "max_group_size",
            "group_size_relaxation",
            "must_link",
            "cannot_link",
            "fini_disposition",
            "fixed_pv",
            "allowed_pvs",
            "required_lines",
            "volume_compatibility_override",
            "coverage_bound",
            "freeze_assignment",
            "group_rule",
            "selection_bound",
        ],
        "accepts_raw_code": False,
    }
