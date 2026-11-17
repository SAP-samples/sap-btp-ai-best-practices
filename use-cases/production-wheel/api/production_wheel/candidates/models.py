"""Immutable records used by deterministic candidate-pool generation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BlockKey = tuple[str, str]
MemberKey = tuple[str, ...]
MatrixException = tuple[str, str, float, float]
MatrixEvidence = tuple[str, str, float, float, str]


def stable_hash(value: object) -> str:
    """Return a SHA-256 digest for a JSON-compatible value.

    Args:
        value: JSON-compatible value whose ordering must not affect the digest.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value(record: Mapping[str, Any] | object, *names: str, default: Any = None) -> Any:
    """Read the first present mapping key or object attribute from a record."""

    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _lines(value: object) -> frozenset[str]:
    """Normalize pipe-delimited or iterable line evidence to stable identifiers."""

    if value in (None, ""):
        return frozenset()
    values = str(value).split("|") if isinstance(value, str) else value
    return frozenset(str(item).strip() for item in values if str(item).strip())


@dataclass(frozen=True, slots=True)
class CandidateMember:
    """Canonical FINI primitives needed to construct subgroup candidates.

    Args:
        fini_id: Stable material or FINI identifier.
        plant: Plant identifier used in the decomposition boundary.
        sefi: SEFI identifier used in the decomposition boundary.
        eligible_lines: Known positive line evidence; an empty set is unknown.
        demand_litres: Canonical demand in litres over the source planning horizon.
        pallet_litres: Resolved pallet conversion in litres.
        package_volume: Governed package volume used by the matrix.
        fixed_pv: Planning-supplied PV used by fixed mode.
        fixed_lot_litres: Optional member-level evidence for that fixed PV lot.
        baseline_group: Historical group identifier used for neighborhoods/stability.
        pck_code: Optional PCK diagnostic; it is never a matrix key.
    """

    fini_id: str
    plant: str
    sefi: str
    eligible_lines: frozenset[str]
    demand_litres: float
    pallet_litres: float
    package_volume: float
    fixed_pv: str | None = None
    fixed_lot_litres: float | None = None
    baseline_group: str | None = None
    pck_code: str | None = None
    lot_size_considered_litres: float | None = None
    xyz_x_dc_count: int | None = None
    xyz_y_dc_count: int | None = None
    xyz_z_dc_count: int | None = None
    primary_dc_count: int | None = None
    secondary_dc_count: int | None = None
    sales_scenario: str | None = None
    sales_network_scenario: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous identifiers and invalid KPI primitives."""

        if not self.fini_id.strip() or not self.plant.strip() or not self.sefi.strip():
            raise ValueError("fini_id, plant, and sefi must be non-blank")
        for name in ("demand_litres", "pallet_litres", "package_volume"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.fixed_lot_litres is not None and not math.isfinite(
            float(self.fixed_lot_litres)
        ):
            raise ValueError("fixed_lot_litres must be finite when supplied")
        for name in ("xyz_x_dc_count", "xyz_y_dc_count", "xyz_z_dc_count", "primary_dc_count", "secondary_dc_count"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0 or value != int(value)):
                raise ValueError(f"{name} must be a non-negative integer or unknown")
        if self.lot_size_considered_litres is not None and (not math.isfinite(self.lot_size_considered_litres) or self.lot_size_considered_litres <= 0):
            raise ValueError("lot_size_considered_litres must be finite and positive or unknown")

    @property
    def block_key(self) -> BlockKey:
        """Return the exact plant-SEFI decomposition key."""

        return self.plant, self.sefi

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | object) -> CandidateMember:
        """Create a member from a canonical extraction mapping or typed record.

        Args:
            record: Record exposing canonical extraction names or equivalent attrs.

        Returns:
            Validated immutable candidate member.
        """

        return cls(
            fini_id=str(_value(record, "fini_id", "material")),
            plant=str(_value(record, "plant")),
            sefi=str(_value(record, "sefi")),
            eligible_lines=_lines(_value(record, "eligible_lines")),
            demand_litres=float(_value(record, "forecast_litres_period") or _value(record, "demand_litres", "forecast_litres_12m")),
            pallet_litres=float(
                _value(record, "pallet_litres", "pallet_litres_resolved")
            ),
            package_volume=float(_value(record, "package_volume")),
            fixed_pv=_value(record, "fixed_pv"),
            fixed_lot_litres=_value(record, "fixed_lot_litres"),
            baseline_group=_value(record, "baseline_group", "baseline_group_key"),
            pck_code=_value(record, "pck_code"),
            **{name: (None if _value(record, name) in (None, "") else float(_value(record, name)))
               for name in ("lot_size_considered_litres", "xyz_x_dc_count", "xyz_y_dc_count",
                            "xyz_z_dc_count", "primary_dc_count", "secondary_dc_count")},
            sales_scenario=_value(record, "sales_scenario"),
            sales_network_scenario=_value(record, "sales_network_scenario"),
        )


@dataclass(frozen=True, slots=True)
class ProductionVersion:
    """Finite production-version alternative for one plant-SEFI block."""

    plant: str
    sefi: str
    pv_id: str
    nominal_lot_litres: float | None
    active: bool = True

    @property
    def block_key(self) -> BlockKey:
        """Return the exact plant-SEFI decomposition key."""

        return self.plant, self.sefi

    @property
    def is_positive(self) -> bool:
        """Return whether this is a finite active positive-lot alternative."""

        value = self.nominal_lot_litres
        return bool(
            self.active
            and self.pv_id.strip()
            and value is not None
            and math.isfinite(float(value))
            and float(value) > 0
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | object) -> ProductionVersion:
        """Create a production version from canonical extraction fields."""

        return cls(
            plant=str(_value(record, "plant")),
            sefi=str(_value(record, "sefi")),
            pv_id=str(_value(record, "pv_id", "production_version")),
            nominal_lot_litres=_value(record, "nominal_lot_litres", "lot_size_litres"),
            active=bool(_value(record, "active", default=True)),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One feasible member/PV configuration with full structural evidence."""

    block_key: BlockKey
    member_ids: MemberKey
    pv_id: str
    nominal_lot_litres: float
    effective_batch_litres: float
    common_lines: tuple[str, ...]
    group_demand_litres: float
    matrix_exception_pairs: tuple[MatrixException, ...]
    equal_pck: bool
    source_tiers: tuple[str, ...]
    membership_hash: str
    candidate_hash: str
    matrix_unknown_pairs: tuple[MatrixEvidence, ...] = ()
    matrix_evidence_pairs: tuple[MatrixEvidence, ...] = ()
    selected_line: str | None = None


@dataclass(frozen=True, slots=True)
class BlockProjection:
    """Conservative line-only scale estimate used to select a pool method."""

    block_key: BlockKey
    members: int
    maximum_size: int
    line_feasible_member_subsets: int
    finite_pv_alternatives: int
    projected_pv_configurations: int
    method: str


@dataclass(frozen=True, slots=True)
class PoolSizeTrace:
    """Deterministic convergence evidence for one restricted beam size."""

    size: int
    expanded_member_sets: int
    feasible_member_sets: int
    shortlisted_member_sets: int
    selected_member_sets: int
    selected_pv_configurations: int
    cumulative_pv_configurations: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Candidate library and evidence for one independently solvable block."""

    block_key: BlockKey
    method: str
    completeness: str
    structural_fingerprint: str
    projection: BlockProjection
    candidates: tuple[Candidate, ...]
    mandatory_member_sets: int
    mandatory_pv_configurations: int
    size_traces: tuple[PoolSizeTrace, ...]
    pool_hash: str
    config_fingerprint: str


class PoolBudgetExceeded(RuntimeError):
    """Raised when mandatory restricted tiers alone exceed the pool ceiling."""

    def __init__(self, block_key: BlockKey, configurations: int, limit: int) -> None:
        """Create an exception carrying the exact disclosed lower bound and limit."""

        self.block_key = block_key
        self.configurations = configurations
        self.limit = limit
        super().__init__(
            f"mandatory PV configurations exceed pool limit for {block_key}: "
            f"{configurations} > {limit}"
        )
