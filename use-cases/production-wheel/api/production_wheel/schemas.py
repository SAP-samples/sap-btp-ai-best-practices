"""Typed, solver-independent configuration and constraint contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from production_wheel.rule_models import BusinessConstraint, GroupRule, SelectionBound, MatrixPair


class CoverageMode(StrEnum):
    """Supported objective/scenario modes."""

    MAX = "MAX"
    DEMAND_WEIGHTED_MEAN = "DEMAND_WEIGHTED_MEAN"
    GROUP_MEAN = "GROUP_MEAN"
    TARGET_BAND = "TARGET_BAND"
    OPERATIONS_FIRST = "OPERATIONS_FIRST"
    BASELINE_CONSTRAINED_COVERAGE = "BASELINE_CONSTRAINED_COVERAGE"
    BASELINE_CONSTRAINED_OPERATIONS = "BASELINE_CONSTRAINED_OPERATIONS"
    GREENFIELD_COVERAGE = "GREENFIELD_COVERAGE"
    GREENFIELD_OPERATIONS = "GREENFIELD_OPERATIONS"
    PARETO = "PARETO"


class CoverageBasis(StrEnum):
    """Quantity interpretation used to calculate group coverage."""

    BASE_GROUP = "BASE_GROUP"
    ADJUSTED_GROUP = "ADJUSTED_GROUP"
    WORST_FINI = "WORST_FINI"


class PalletFormula(StrEnum):
    """Supported FINI pallet-allocation formulas."""

    MINIMUM_ONLY = "MINIMUM_ONLY"
    WHOLE_PALLET_ROUNDING = "WHOLE_PALLET_ROUNDING"


class GroupSizeMode(StrEnum):
    """Historical-cap enforcement policy."""

    HARD = "HARD"
    BOUNDED_RELAXATION = "BOUNDED_RELAXATION"


class PVMode(StrEnum):
    """Production-version ownership mode."""

    FIXED = "FIXED"
    OPTIMIZED = "OPTIMIZED"


class MatrixMode(StrEnum):
    """Package-volume compatibility enforcement mode."""

    HARD = "HARD"
    DIAGNOSTIC = "DIAGNOSTIC"
    FLEXIBLE = "FLEXIBLE"
    OFF = "OFF"


class Enforcement(StrEnum):
    """Constraint enforcement type exposed to a future agent."""

    HARD = "hard"
    BOUNDED_RELAXATION = "bounded_relaxation"


class ApprovalStatus(StrEnum):
    """Human-governance state for an interpreted rule."""

    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class BaselineAcceptanceStatus(StrEnum):
    """Governed outcome of the baseline-relative acceptance assessment."""

    ACCEPTED_BASELINE_GUARDRAILS = "ACCEPTED_BASELINE_GUARDRAILS"
    PARETO_REVIEW_REQUIRED = "PARETO_REVIEW_REQUIRED"
    BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES = (
        "BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES"
    )
    NOT_ACCEPTABLE = "NOT_ACCEPTABLE"


class GreenfieldValidationStatus(StrEnum):
    """Independent structural status for baseline-free Pareto proposals."""

    VALID_PARETO_POINT = "VALID_PARETO_POINT"
    INVALID_PARETO_POINT = "INVALID_PARETO_POINT"


class StrictModel(BaseModel):
    """Base model that rejects undeclared input fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GroupSizePolicy(StrictModel):
    """Historical size seven plus an optional bounded excess."""

    mode: GroupSizeMode = GroupSizeMode.HARD
    base_limit: int = Field(default=7, ge=1)
    max_excess: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("max_excess", "additional_members"),
    )

    @model_validator(mode="after")
    def validate_policy(self) -> "GroupSizePolicy":
        """Ensure hard and bounded modes have coherent excess values."""

        if self.base_limit != 7:
            raise ValueError("v1 uses the governed historical base limit of seven")
        if self.mode == GroupSizeMode.HARD and self.max_excess != 0:
            raise ValueError("hard group-size mode cannot carry an excess")
        if self.mode == GroupSizeMode.BOUNDED_RELAXATION and self.max_excess < 1:
            raise ValueError("bounded relaxation requires a positive excess")
        return self

    @property
    def effective_cap(self) -> int:
        """Return the largest permitted group size."""

        return self.base_limit + self.max_excess

    @property
    def maximum_size(self) -> int:
        """Compatibility alias for candidate-pool generation."""

        return self.effective_cap

    @property
    def additional_members(self) -> int:
        """Return the bounded excess above the historical limit."""

        return self.max_excess


class GroupSizeOverride(StrictModel):
    """Per-block maximum group size overriding the global cap for one (plant, SEFI).

    The historical ``group_size`` base limit of seven is plant-specific; other
    plants and SEFIs may permit a different maximum. Each override names a single
    (plant, SEFI) block and the largest group size candidate generation may build
    for it. Overrides are additive and never mutate the frozen global default, so
    an override-free run is byte-identical to prior behaviour.
    """

    plant: str = Field(min_length=1)
    sefi: str = Field(min_length=1)
    maximum: int = Field(ge=1)

    @field_validator("plant", "sefi")
    @classmethod
    def _strip_identifier(cls, value: str) -> str:
        """Strip block identifiers and reject whitespace-only values."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("group-size override identifiers cannot be blank")
        return stripped


class TargetBand(StrictModel):
    """Visible provisional coverage guidance in days."""

    lower_days: float = Field(
        default=5.0,
        ge=0,
        validation_alias=AliasChoices("lower_days", "minimum_days"),
    )
    upper_days: float = Field(
        default=365.0,
        gt=0,
        validation_alias=AliasChoices("upper_days", "maximum_days"),
    )

    @model_validator(mode="after")
    def validate_order(self) -> "TargetBand":
        """Require the upper coverage bound to exceed the lower bound."""

        if self.upper_days <= self.lower_days:
            raise ValueError("upper_days must be greater than lower_days")
        return self

    @property
    def minimum_days(self) -> float:
        """Return the lower target bound using candidate terminology."""

        return self.lower_days

    @property
    def maximum_days(self) -> float:
        """Return the upper target bound using candidate terminology."""

        return self.upper_days


class BaselineGuardrailPolicy(StrictModel):
    """No-worse-than-baseline inventory, fragmentation, and J_CH policy.

    Coverage-band, demand-weighted, P90, group-count, singleton-count, and J_CH
    thresholds are derived by recalculating the frozen assignment under the
    proposal's own settings. The optional tolerances apply only to baseline
    J_CH; leaving both unset requires every governed metric to be no worse.
    """

    j_ch_relative_tolerance: float | None = Field(default=None, ge=0)
    j_ch_absolute_tolerance: float | None = Field(default=None, ge=0)

    def j_ch_limit(self, baseline_j_ch: float) -> float:
        """Return the largest accepted J_CH for a comparable baseline value."""

        relative = self.j_ch_relative_tolerance or 0.0
        absolute = self.j_ch_absolute_tolerance or 0.0
        return baseline_j_ch * (1.0 + relative) + absolute


class PoolLimits(StrictModel):
    """Deterministic exact/restricted candidate-library limits."""

    exhaustive_subset_limit: int = Field(
        default=100_000,
        ge=1,
        validation_alias=AliasChoices("exhaustive_subset_limit", "exact_member_subsets"),
    )
    exhaustive_configuration_limit: int = Field(
        default=250_000,
        ge=1,
        validation_alias=AliasChoices(
            "exhaustive_configuration_limit", "exact_pv_configurations"
        ),
    )
    candidate_ceiling_per_block: int = Field(
        default=250_000,
        ge=1,
        validation_alias=AliasChoices(
            "candidate_ceiling_per_block", "restricted_pv_configurations"
        ),
    )
    beam_width_per_size_scorer: int = Field(
        default=2_000,
        ge=1,
        validation_alias=AliasChoices(
            "beam_width_per_size_scorer", "retained_member_sets_per_size_scorer"
        ),
    )
    forced_exact_configuration_ceiling: int = Field(default=250_000, ge=1)
    """Upper bound on projected configurations for a ``--exhaustive-block`` forced
    exact enumeration. A block explicitly requested as exhaustive is still only
    enumerated exhaustively when its projected line-feasible configuration count
    is at or below this ceiling; larger blocks fall back to the restricted beam.
    This prevents a forced block from building an enormous MILP that exhausts
    memory or stalls in solver presolve. It defaults to the same value as
    ``exhaustive_configuration_limit`` so forcing never exceeds the size the
    library already treats as exactly solvable."""

    @property
    def exact_member_subsets(self) -> int:
        """Return the exhaustive member-set threshold."""

        return self.exhaustive_subset_limit

    @property
    def exact_pv_configurations(self) -> int:
        """Return the exhaustive PV-configuration threshold."""

        return self.exhaustive_configuration_limit

    @property
    def restricted_pv_configurations(self) -> int:
        """Return the restricted per-block configuration ceiling."""

        return self.candidate_ceiling_per_block

    @property
    def retained_member_sets_per_size_scorer(self) -> int:
        """Return the beam width for each invariant scorer."""

        return self.beam_width_per_size_scorer


class SolverLimits(StrictModel):
    """Development solver limits and accepted within-model gap target."""

    time_limit_seconds: float = Field(default=600.0, gt=0)
    suite_time_limit_seconds: float = Field(default=3_600.0, gt=0)
    mip_gap: float = Field(default=0.01, ge=0, le=1)
    threads: int = Field(default=1, ge=1)
    random_seed: int = 0
    presolve_reduction_limit: int = Field(default=-1, ge=-1)
    """Maximum number of HiGHS presolve reductions; ``-1`` keeps the solver
    default (unbounded presolve). A finite value bounds presolve work so a very
    large model cannot stall indefinitely in a single presolve pass. It never
    changes the mathematical optimum, only how much reduction is attempted."""


class VersionIdentifiers(StrictModel):
    """Version identifiers carried into fingerprints and audit artifacts."""

    matrix_version: str = "BASELINE_EMPIRICAL_MATRIX_V1"
    ruleset_version: str = "PROTOTYPE_RULESET_V5"
    schema_version: str = "PROTOTYPE_SCHEMA_V9"


class RunConfig(StrictModel):
    """Complete deterministic scenario configuration."""

    scenario_id: str = Field(default="default", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    coverage_mode: CoverageMode = CoverageMode.TARGET_BAND
    coverage_basis: CoverageBasis = CoverageBasis.BASE_GROUP
    pallet_formula: PalletFormula = PalletFormula.MINIMUM_ONLY
    group_size: GroupSizePolicy = Field(default_factory=GroupSizePolicy)
    pv_mode: PVMode = PVMode.FIXED
    matrix_mode: MatrixMode = MatrixMode.DIAGNOSTIC
    target_band: TargetBand = Field(default_factory=TargetBand)
    baseline_guardrails: BaselineGuardrailPolicy = Field(
        default_factory=BaselineGuardrailPolicy
    )
    pool_limits: PoolLimits = Field(default_factory=PoolLimits)
    solver_limits: SolverLimits = Field(default_factory=SolverLimits)
    versions: VersionIdentifiers = Field(default_factory=VersionIdentifiers)
    exhaustive_blocks: tuple[tuple[str, str], ...] = ()
    group_size_overrides: tuple[GroupSizeOverride, ...] = ()
    demand_days: int = Field(default=250, gt=0)
    productive_weeks: float = Field(default=50, gt=0, allow_inf_nan=False)
    canonical_factor: float = Field(default=0.90, gt=0, le=1)
    high_runner_threshold_days: float = Field(default=15, gt=0, allow_inf_nan=False)
    runner_basis: Literal["reference", "candidate_pv"] = "reference"
    assign_filling_lines: bool = False
    preferred_line: str | None = None
    matrix_pairs: tuple[MatrixPair, ...] = ()
    business_rules: tuple[BusinessConstraint, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_canonical_parameters(self) -> "RunConfig":
        """Normalize scoped identifiers while accepting validated calendar overrides."""

        normalized = tuple(
            (plant.strip(), sefi.strip()) for plant, sefi in self.exhaustive_blocks
        )
        if any(not plant or not sefi for plant, sefi in normalized):
            raise ValueError("exhaustive block identifiers cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("exhaustive blocks must be distinct")
        object.__setattr__(self, "exhaustive_blocks", tuple(sorted(normalized)))
        override_keys = [
            (override.plant, override.sefi) for override in self.group_size_overrides
        ]
        if len(set(override_keys)) != len(override_keys):
            raise ValueError(
                "group-size overrides must target distinct (plant, SEFI) blocks"
            )
        if len({r.constraint_id for r in self.business_rules}) != len(self.business_rules):
            raise ValueError("business rule identifiers must be unique")
        if any(r.approval_status != 'approved' for r in self.business_rules):
            raise ValueError("configured business rules require approved status")
        # Selected-line predicates require actual line alternatives; common
        # eligibility alone cannot establish an assignment.
        stack = [e for r in self.business_rules for e in
            ((r.when, r.assertion) if isinstance(r, GroupRule) else (r.measure,))]
        needs_line = bool(self.preferred_line)
        while stack:
            node = stack.pop()
            needs_line |= node.field == "group.selected_line"
            stack.extend(node.args)
        if needs_line:
            object.__setattr__(self, "assign_filling_lines", True)
        object.__setattr__(
            self,
            "group_size_overrides",
            tuple(
                sorted(
                    self.group_size_overrides,
                    key=lambda item: (item.plant, item.sefi),
                )
            ),
        )
        return self

    def configuration_id(self) -> str:
        """Return a stable digest over the complete normalized configuration."""

        normalized = self.model_dump(mode="json")
        normalized.pop("scenario_id", None)
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def structural_ruleset_fingerprint(self) -> str:
        """Return a digest over settings that can change candidate membership."""

        payload = {
            "group_size": self.group_size.model_dump(mode="json"),
            "pv_mode": self.pv_mode.value,
            # DIAGNOSTIC and FLEXIBLE admit the same candidates and retain the
            # same evidence; only the solver objective treats them differently.
            "matrix_mode": (
                "ALLOW_EXCEPTIONS"
                if self.matrix_mode in {MatrixMode.DIAGNOSTIC, MatrixMode.FLEXIBLE}
                else self.matrix_mode.value
            ),
            "matrix_version": self.versions.matrix_version,
            "ruleset_version": self.versions.ruleset_version,
            "pool_limits": self.pool_limits.model_dump(mode="json"),
            "exhaustive_blocks": self.exhaustive_blocks,
            "group_size_overrides": [
                override.model_dump(mode="json")
                for override in self.group_size_overrides
            ],
            "baseline_seeded_restricted_pool": self.uses_baseline_candidate_seeds,
        }
        if self.matrix_pairs or self.business_rules or self.assign_filling_lines or (self.demand_days, self.productive_weeks, self.canonical_factor) != (250, 50, 0.9):
            payload.update(matrix_pairs=[p.model_dump() for p in self.matrix_pairs],
                business_rules=[r.model_dump() for r in self.business_rules],
                assign_filling_lines=self.assign_filling_lines,
                high_runner_threshold_days=self.high_runner_threshold_days,
                runner_basis=self.runner_basis, demand_days=self.demand_days,
                productive_weeks=self.productive_weeks, canonical_factor=self.canonical_factor)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def cap_for_block(self, plant: str, sefi: str) -> int:
        """Return the maximum group size for one (plant, SEFI) block.

        Falls back to the global ``group_size.effective_cap`` when no per-block
        override applies, so an override-free configuration behaves identically
        to before this feature existed.
        """

        for override in self.group_size_overrides:
            if override.plant == plant and override.sefi == sefi:
                return override.maximum
        return self.group_size.effective_cap

    @property
    def uses_baseline_candidate_seeds(self) -> bool:
        """Return whether restricted pools must retain historical neighborhoods."""

        return self.coverage_mode in {
            CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
            CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
        }

    @property
    def is_greenfield(self) -> bool:
        """Return whether feasibility and objectives are baseline-independent."""

        return self.coverage_mode in {
            CoverageMode.GREENFIELD_COVERAGE,
            CoverageMode.GREENFIELD_OPERATIONS,
            CoverageMode.PARETO,
        }


class ConstraintScope(StrictModel):
    """Optional plant/SEFI scope for an executable rule."""

    plant: str | None = None
    sefi: str | None = None

    @field_validator("plant", "sefi")
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        """Strip optional identifiers and reject whitespace-only values."""

        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("scope identifiers cannot be blank")
        return stripped


class ConstraintBase(StrictModel):
    """Shared audit fields for every allowlisted constraint."""

    constraint_id: str = Field(min_length=1)
    scope: ConstraintScope = Field(default_factory=ConstraintScope)
    enforcement: Enforcement = Enforcement.HARD
    source_text: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_identifier(cls, value: str) -> str:
        """Strip and validate the audit identifier."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("constraint_id cannot be blank")
        return stripped


class MaxGroupSizeConstraint(ConstraintBase):
    """Override the absolute maximum group size for a scope."""

    kind: Literal["max_group_size"]
    maximum: int = Field(ge=1)


class GroupSizeRelaxationConstraint(ConstraintBase):
    """Permit a bounded excess above the historical size seven."""

    kind: Literal["group_size_relaxation"]
    base_limit: int = Field(default=7, ge=1)
    max_excess: int = Field(ge=1)


class MustLinkConstraint(ConstraintBase):
    """Require all listed FINIs to share a group."""

    kind: Literal["must_link"]
    materials: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_distinct_materials(self) -> "MustLinkConstraint":
        """Require at least two distinct FINI identifiers."""

        if len(set(self.materials)) != len(self.materials):
            raise ValueError("must_link materials must be distinct")
        return self


class CannotLinkConstraint(ConstraintBase):
    """Forbid a pair of FINIs from sharing a group."""

    kind: Literal["cannot_link"]
    materials: tuple[str, str]

    @model_validator(mode="after")
    def validate_distinct_pair(self) -> "CannotLinkConstraint":
        """Reject a FINI being declared incompatible with itself."""

        if self.materials[0] == self.materials[1]:
            raise ValueError("cannot_link materials must be distinct")
        return self


class FiniDispositionConstraint(ConstraintBase):
    """Include or exclude one FINI from modeled scope."""

    kind: Literal["fini_disposition"]
    material: str
    action: Literal["include", "exclude"]
    reason: str


class FixedPVConstraint(ConstraintBase):
    """Fix a scoped block to one planning-supplied PV alias."""

    kind: Literal["fixed_pv"]
    production_version: str


class AllowedPVsConstraint(ConstraintBase):
    """Restrict optimized PV sensitivity to an explicit finite allowlist."""

    kind: Literal["allowed_pvs"]
    production_versions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_versions(self) -> "AllowedPVsConstraint":
        """Reject duplicate PV aliases in an allowlist."""

        if len(set(self.production_versions)) != len(self.production_versions):
            raise ValueError("allowed PV aliases must be distinct")
        return self


class RequiredLinesConstraint(ConstraintBase):
    """Restrict a scope to an explicit set of eligible line witnesses."""

    kind: Literal["required_lines"]
    filling_lines: tuple[str, ...] = Field(min_length=1)

    @field_validator("filling_lines", mode="before")
    @classmethod
    def normalize_line_identifiers(cls, value: object) -> tuple[str, ...]:
        """Normalize legacy integer or string line inputs to nonblank strings."""

        if not isinstance(value, (list, tuple)):
            raise ValueError("filling_lines must be a list or tuple")
        if any(isinstance(line, bool) or not isinstance(line, (str, int)) for line in value):
            raise ValueError("line identifiers must be strings or integers")
        lines = tuple(str(line).strip() for line in value)
        if any(not line for line in lines):
            raise ValueError("line identifiers cannot be blank")
        return lines

    @model_validator(mode="after")
    def validate_distinct_lines(self) -> "RequiredLinesConstraint":
        """Reject duplicate line identifiers."""

        if len(set(self.filling_lines)) != len(self.filling_lines):
            raise ValueError("required lines must be distinct")
        return self


class VolumeCompatibilityOverrideConstraint(ConstraintBase):
    """Override one package-volume pair in a governed matrix."""

    kind: Literal["volume_compatibility_override"]
    volume_a: str
    volume_b: str
    compatible: bool


class CoverageBoundConstraint(ConstraintBase):
    """Apply scoped coverage lower/upper guidance."""

    kind: Literal["coverage_bound"]
    lower_days: float | None = Field(default=None, ge=0)
    upper_days: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_bound(self) -> "CoverageBoundConstraint":
        """Require at least one bound and a coherent interval."""

        if self.lower_days is None and self.upper_days is None:
            raise ValueError("coverage bound requires lower_days or upper_days")
        if (
            self.lower_days is not None
            and self.upper_days is not None
            and self.upper_days <= self.lower_days
        ):
            raise ValueError("coverage upper_days must exceed lower_days")
        return self


class FreezeAssignmentConstraint(ConstraintBase):
    """Freeze listed FINIs to their baseline co-membership."""

    kind: Literal["freeze_assignment"]
    materials: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_frozen_materials(self) -> "FreezeAssignmentConstraint":
        """Reject duplicate FINIs in a frozen-assignment request."""

        if len(set(self.materials)) != len(self.materials):
            raise ValueError("frozen materials must be distinct")
        return self


ConstraintSpec = Annotated[
    Union[
        MaxGroupSizeConstraint,
        GroupSizeRelaxationConstraint,
        MustLinkConstraint,
        CannotLinkConstraint,
        FiniDispositionConstraint,
        FixedPVConstraint,
        AllowedPVsConstraint,
        RequiredLinesConstraint,
        VolumeCompatibilityOverrideConstraint,
        CoverageBoundConstraint,
        FreezeAssignmentConstraint,
        GroupRule,
        SelectionBound,
    ],
    Field(discriminator="kind"),
]


class SolveRequest(StrictModel):
    """Validated request passed from a UI or future language-model adapter."""

    config: RunConfig = Field(default_factory=RunConfig)
    constraints: tuple[ConstraintSpec, ...] = ()
    scope: tuple[ConstraintScope, ...] = ()
    """Optional solve scope. When non-empty, only FINIs whose (plant, SEFI) block
    matches at least one entry are modeled (a scope with ``sefi`` unset matches
    every SEFI of that plant). Empty means the full modeled population, exactly
    as before this field existed."""

    def request_id(self) -> str:
        """Return a stable digest over normalized configuration and constraints."""

        set_fields = {"materials", "production_versions", "filling_lines"}

        def canonicalize(value: object, key: str | None = None) -> object:
            """Normalize set-like arrays and mapping order for hashing."""

            if isinstance(value, dict):
                return {
                    item_key: canonicalize(item_value, item_key)
                    for item_key, item_value in sorted(value.items())
                    if item_key != "scenario_id"
                }
            if isinstance(value, list):
                normalized = [canonicalize(item) for item in value]
                if key in set_fields:
                    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
                return normalized
            return value

        normalized = canonicalize(self.model_dump(mode="json"))
        if isinstance(normalized, dict) and isinstance(normalized.get("constraints"), list):
            normalized["constraints"] = sorted(
                normalized["constraints"], key=lambda item: json.dumps(item, sort_keys=True)
            )
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
