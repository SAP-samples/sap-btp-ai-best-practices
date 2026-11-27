"""Versioned package-volume matrices and their shared pair semantics."""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from production_wheel.candidates.models import CandidateMember
    from production_wheel.schemas import RunConfig

SYNTHETIC_MATRIX_VERSION = "SYNTHETIC_VOLUME_MATRIX_V1"
BASELINE_EMPIRICAL_MATRIX_V1 = "BASELINE_EMPIRICAL_MATRIX_V1"
BASELINE_EMPIRICAL_MATRIX_VERSION = BASELINE_EMPIRICAL_MATRIX_V1
CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1 = "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
MATRIX_VERSION = SYNTHETIC_MATRIX_VERSION
MATRIX_EXCEPTION_STATUSES = frozenset({"NO", "N", "AVOID"})
YES_OBSERVED_BASELINE = "YES_OBSERVED_BASELINE"
UNKNOWN_LINE_FEASIBLE = "UNKNOWN_LINE_FEASIBLE"
UNKNOWN_NO_CURRENT_LINE_OVERLAP = "UNKNOWN_NO_CURRENT_LINE_OVERLAP"

VOLUME_FAMILIES: dict[str, tuple[Decimal, ...]] = {
    "SMALL": tuple(map(Decimal, ("0.25", "0.5", "0.75", "0.9", "1"))),
    "SMALL_MEDIUM": tuple(map(Decimal, ("2.18", "2.5", "3"))),
    "MEDIUM": tuple(map(Decimal, ("4", "4.44", "5", "6"))),
    "LARGE": tuple(map(Decimal, ("7.5", "8", "9", "10"))),
    "EXTRA_LARGE": tuple(
        map(Decimal, ("12", "12.5", "12.92", "13.351", "13.755", "14", "15"))
    ),
}
MODELED_VOLUMES: tuple[Decimal, ...] = tuple(
    sorted(volume for family in VOLUME_FAMILIES.values() for volume in family)
)
CUSTOMER_VOLUME_CATALOG: tuple[Decimal, ...] = tuple(
    map(
        Decimal,
        (
            "0.25",
            "0.5",
            "0.75",
            "0.9",
            "0.93",
            "1",
            "2",
            "2.18",
            "2.325",
            "2.35",
            "2.5",
            "3",
            "4",
            "4.44",
            "4.5",
            "4.65",
            "5",
            "6",
            "7.5",
            "8",
            "9",
            "9.3",
            "9.4",
            "10",
            "12",
            "12.5",
            "12.666",
            "12.92",
            "13.02",
            "13.351",
            "13.755",
            "13.95",
            "14",
            "15",
        ),
    )
)
CUSTOMER_VOLUME_FAMILIES: tuple[tuple[str, Decimal, Decimal], ...] = (
    ("LE_1_L", Decimal("0"), Decimal("1")),
    ("2_TO_5_L", Decimal("2"), Decimal("5")),
    ("4_TO_10_L", Decimal("4"), Decimal("10")),
    ("10_TO_15_L", Decimal("10"), Decimal("15")),
)


def normalize_volume(value: Decimal | str | int | float) -> Decimal:
    """Normalize a volume using its decimal text representation."""

    return value if isinstance(value, Decimal) else Decimal(str(value))


def volume_family(value: Decimal | str | int | float) -> str:
    """Return the provisional family for an observed package volume.

    Raises:
        ValueError: If the volume is outside the governed 23-value catalog.
    """

    normalized = normalize_volume(value)
    for family, volumes in VOLUME_FAMILIES.items():
        if normalized in volumes:
            return family
    raise ValueError(f"volume is not in {MATRIX_VERSION}: {normalized}")


def matrix_compatible(
    volume_a: Decimal | str | int | float,
    volume_b: Decimal | str | int | float,
) -> bool:
    """Return whether two governed package volumes share a synthetic family."""

    return volume_family(volume_a) == volume_family(volume_b)


def customer_volume_families(
    value: Decimal | str | int | float,
) -> tuple[str, ...]:
    """Return every inclusive customer family containing a governed volume.

    The overlaps at 4--5 L and 10 L are intentional: two volumes are preferred
    when they share at least one operational family.

    Raises:
        ValueError: If the volume is outside the complete 34-value catalog.
    """

    normalized = normalize_volume(value)
    if normalized not in CUSTOMER_VOLUME_CATALOG:
        raise ValueError(
            f"volume is not in {CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1}: "
            f"{normalized}"
        )
    return tuple(
        name
        for name, lower, upper in CUSTOMER_VOLUME_FAMILIES
        if lower <= normalized <= upper
    )


def build_customer_operational_volume_matrix() -> tuple[dict[str, object], ...]:
    """Return the explicit symmetric 34-by-34 customer preference matrix."""

    rows: list[dict[str, object]] = []
    for volume_a in CUSTOMER_VOLUME_CATALOG:
        families_a = customer_volume_families(volume_a)
        for volume_b in CUSTOMER_VOLUME_CATALOG:
            families_b = customer_volume_families(volume_b)
            preferred = bool(set(families_a) & set(families_b))
            status = "Y" if preferred else "AVOID"
            rows.append(
                {
                    "volume_a": format(volume_a, "f"),
                    "volume_b": format(volume_b, "f"),
                    "family_a": "|".join(families_a),
                    "family_b": "|".join(families_b),
                    "status": status,
                    "compatible": "YES" if preferred else "NO",
                    "matrix_version": CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
                    "synthetic": 0,
                    "rationale": (
                        "shares an inclusive operational volume family"
                        if preferred
                        else "outside the preferred operational volume families"
                    ),
                }
            )
    return tuple(rows)


CUSTOMER_OPERATIONAL_VOLUME_FAMILY_MATRIX_V1 = (
    build_customer_operational_volume_matrix()
)


def build_synthetic_matrix() -> tuple[dict[str, object], ...]:
    """Return all 529 explicit, symmetric matrix records."""

    rows: list[dict[str, object]] = []
    for volume_a in MODELED_VOLUMES:
        for volume_b in MODELED_VOLUMES:
            family_a = volume_family(volume_a)
            family_b = volume_family(volume_b)
            compatible = family_a == family_b
            rows.append(
                {
                    "volume_a": format(volume_a, "f"),
                    "volume_b": format(volume_b, "f"),
                    "family_a": family_a,
                    "family_b": family_b,
                    "compatible": "YES" if compatible else "NO",
                    "matrix_version": MATRIX_VERSION,
                    "synthetic": 1,
                    "rationale": (
                        "same provisional package-volume family"
                        if compatible
                        else "different provisional package-volume families"
                    ),
                }
            )
    return tuple(rows)


SYNTHETIC_VOLUME_MATRIX_V1 = build_synthetic_matrix()


def build_baseline_empirical_matrix(
    modeled_members: Iterable[CandidateMember],
    baseline_evidence_members: Iterable[CandidateMember] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build tri-state evidence from modeled lines and all baseline assignments.

    Args:
        modeled_members: Global modeled population used only for current
            shared-line witnesses.
        baseline_evidence_members: All historical baseline assignments used for
            positive co-membership evidence. Defaults to the modeled population.

    Returns:
        All 529 ordered volume-pair records. Historical co-membership is positive
        evidence, while every unobserved pair remains explicitly unknown.
    """

    modeled = tuple(
        sorted(modeled_members, key=lambda item: (*item.block_key, item.fini_id))
    )
    historical = tuple(
        sorted(
            baseline_evidence_members if baseline_evidence_members is not None else modeled,
            key=lambda item: (*item.block_key, item.fini_id),
        )
    )
    by_volume: dict[Decimal, list[CandidateMember]] = {
        volume: [] for volume in MODELED_VOLUMES
    }
    baseline_groups: dict[tuple[str, str, str], set[Decimal]] = {}
    for member in modeled:
        volume = normalize_volume(member.package_volume)
        if volume not in by_volume:
            raise ValueError(
                f"volume is not in {BASELINE_EMPIRICAL_MATRIX_VERSION}: {volume}"
            )
        by_volume[volume].append(member)
    for member in historical:
        volume = normalize_volume(member.package_volume)
        if volume not in by_volume:
            raise ValueError(
                f"volume is not in {BASELINE_EMPIRICAL_MATRIX_VERSION}: {volume}"
            )
        if member.baseline_group:
            key = (*member.block_key, member.baseline_group)
            baseline_groups.setdefault(key, set()).add(volume)

    observed = {
        tuple(sorted((volume_a, volume_b)))
        for volumes in baseline_groups.values()
        for volume_a, volume_b in combinations(sorted(volumes), 2)
    }
    # Identity is always compatible; it also gives a stable diagonal when a
    # small unit-test population does not contain all 23 governed volumes.
    observed.update((volume, volume) for volume in MODELED_VOLUMES)

    def has_line_evidence(volume_a: Decimal, volume_b: Decimal) -> bool:
        """Return whether any distinct FINI pair shares a known filling line."""

        left = by_volume[volume_a]
        right = by_volume[volume_b]
        return any(
            member_a.fini_id != member_b.fini_id
            and member_a.block_key == member_b.block_key
            and member_a.eligible_lines & member_b.eligible_lines
            for member_a in left
            for member_b in right
        )

    rows: list[dict[str, object]] = []
    for volume_a in MODELED_VOLUMES:
        for volume_b in MODELED_VOLUMES:
            pair = tuple(sorted((volume_a, volume_b)))
            if pair in observed:
                status = YES_OBSERVED_BASELINE
                rationale = "observed baseline co-membership or identity diagonal"
            elif has_line_evidence(volume_a, volume_b):
                status = UNKNOWN_LINE_FEASIBLE
                rationale = "unobserved pair with current shared-line evidence"
            else:
                status = UNKNOWN_NO_CURRENT_LINE_OVERLAP
                rationale = "unobserved pair without current shared-line evidence"
            rows.append(
                {
                    "volume_a": format(volume_a, "f"),
                    "volume_b": format(volume_b, "f"),
                    "family_a": volume_family(volume_a),
                    "family_b": volume_family(volume_b),
                    "status": status,
                    "compatible": "YES" if status == YES_OBSERVED_BASELINE else "UNKNOWN",
                    "matrix_version": BASELINE_EMPIRICAL_MATRIX_VERSION,
                    "synthetic": 0,
                    "rationale": rationale,
                }
            )
    return tuple(rows)


def build_matrix_for_version(
    matrix_version: str,
    modeled_members: Iterable[CandidateMember],
    baseline_evidence_members: Iterable[CandidateMember] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return the configured matrix using modeled and historical populations."""

    if matrix_version == SYNTHETIC_MATRIX_VERSION:
        return SYNTHETIC_VOLUME_MATRIX_V1
    if matrix_version == BASELINE_EMPIRICAL_MATRIX_VERSION:
        return build_baseline_empirical_matrix(
            modeled_members, baseline_evidence_members
        )
    if matrix_version == CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1:
        return CUSTOMER_OPERATIONAL_VOLUME_FAMILY_MATRIX_V1
    raise ValueError(f"unsupported package-volume matrix version: {matrix_version}")


def configured_matrix_version(config: RunConfig) -> str:
    """Return the matrix contract label; exact profile pairs live in the run snapshot."""
    return "PLANT_PROFILE_MATRIX_V1" if config.matrix_pairs else config.versions.matrix_version


class ProfileMatrixCoverageError(ValueError):
    """Raised when modeled products use package volumes absent from the profile matrix.

    The optimizer never invents a compatibility status, so a profile matrix must
    list every package volume of the modeled population.

    Attributes:
        coverage: The ``profile_matrix_coverage`` summary that failed, with the
            missing volumes and how many modeled products use each one.
    """

    def __init__(self, coverage: Mapping[str, object]):
        self.coverage = dict(coverage)
        super().__init__(describe_matrix_coverage_gap(self.coverage))


def profile_matrix_coverage(
    config: RunConfig, members: Iterable[CandidateMember]
) -> dict[str, object] | None:
    """Compare the modeled package volumes with the volumes of the profile matrix.

    Args:
        config: Effective run configuration. Only explicit profile pairs
            (``config.matrix_pairs``) are checked; legacy matrix versions build
            their rows from the population itself and always cover it.
        members: Modeled candidate members of the run or draft.

    Returns:
        ``None`` when the configuration has no profile matrix. Otherwise a dict
        with ``matrix_volume_count``, ``modeled_volume_count``,
        ``modeled_member_count``, ``missing_volumes`` (a list of
        ``{"volume", "member_count"}`` sorted by volume),
        ``affected_member_count``, ``complete`` (True when nothing is missing) and
        ``volume_matching`` (the profile mode, so the message can name classes).
    """

    # A class-matched profile whose classes place no volume still has a matrix:
    # report every volume as missing instead of falling back to a legacy matrix.
    if not config.matrix_pairs and getattr(config, "volume_matching", "exact") != "nearest_class":
        return None
    matrix_volumes = {
        normalize_volume(volume)
        for pair in config.matrix_pairs
        for volume in (pair.volume_a, pair.volume_b)
    }
    usage: dict[Decimal, int] = {}
    for member in members:
        volume = normalize_volume(member.package_volume)
        usage[volume] = usage.get(volume, 0) + 1
    missing = sorted(volume for volume in usage if volume not in matrix_volumes)
    return {
        "matrix_volume_count": len(matrix_volumes),
        "modeled_volume_count": len(usage),
        "modeled_member_count": sum(usage.values()),
        "missing_volumes": [
            {"volume": format(volume, "f"), "member_count": usage[volume]}
            for volume in missing
        ],
        "affected_member_count": sum(usage[volume] for volume in missing),
        "complete": not missing,
        "volume_matching": getattr(config, "volume_matching", "exact"),
    }


def describe_matrix_coverage_gap(coverage: Mapping[str, object]) -> str:
    """Return a planner-readable sentence for an incomplete profile matrix.

    Args:
        coverage: Output of ``profile_matrix_coverage`` with missing volumes.

    Returns:
        One message naming the missing volumes (largest product counts first,
        at most 20) and the actions that resolve the gap.
    """

    missing = list(coverage.get("missing_volumes") or [])
    ranked = sorted(missing, key=lambda item: (-int(item["member_count"]), float(item["volume"])))
    listed = ", ".join(
        f"{item['volume']} L ({item['member_count']} product{'' if item['member_count'] == 1 else 's'})"
        for item in ranked[:20]
    )
    more = f" and {len(ranked) - 20} more" if len(ranked) > 20 else ""
    if coverage.get("volume_matching") == "nearest_class":
        return (
            f"{len(missing)} package volumes of the modeled products fall outside every class of the "
            f"plant profile's volume matrix ({coverage.get('affected_member_count')} of "
            f"{coverage.get('modeled_member_count')} products): {listed}{more}. They lie below the "
            "smallest or above the largest reference volume and the matrix has no '<n' / '>n' class "
            "for them. Add that class (or a reference volume) to the profile's volume compatibility "
            "matrix in Settings, or ask the Assistant to propose it."
        )
    return (
        f"The plant profile volume matrix lists {coverage.get('matrix_volume_count')} volumes, "
        f"but {len(missing)} package volumes of the modeled products are missing from it "
        f"({coverage.get('affected_member_count')} of {coverage.get('modeled_member_count')} products): "
        f"{listed}{more}. Add these volumes to the profile's volume compatibility matrix in "
        "Settings (or ask the Assistant to propose them), or select a profile whose matrix "
        "covers them."
    )


def configured_matrix(
    config: RunConfig,
    modeled_members: Iterable[CandidateMember],
    baseline_evidence_members: Iterable[CandidateMember] | None = None,
) -> tuple[dict[str, object], ...]:
    """Resolve validated symmetric profile pairs or the unchanged legacy matrix.

    Profile pairs may be triangular or fully ordered. Conflicting reverse pairs
    and missing observed-volume pairs (including diagonals) fail closed. Explicit
    N is a prohibition in every mode; AVOID is preference evidence outside HARD.
    """
    members = tuple(modeled_members)
    if not config.matrix_pairs and getattr(config, "volume_matching", "exact") != "nearest_class":
        return build_matrix_for_version(
            config.versions.matrix_version, members, baseline_evidence_members
        )
    pairs: dict[tuple[Decimal, Decimal], str] = {}
    for pair in config.matrix_pairs:
        left, right = normalize_volume(pair.volume_a), normalize_volume(pair.volume_b)
        for key in ((left, right), (right, left)):
            if key in pairs and pairs[key] != pair.status:
                raise ValueError(f"conflicting profile matrix pair: {key}")
            pairs[key] = pair.status
    # A volume absent from the whole matrix is a profile gap the planner can fix;
    # raise the typed error so validation and failure reports can name it.
    coverage = profile_matrix_coverage(config, members)
    if coverage and not coverage["complete"]:
        raise ProfileMatrixCoverageError(coverage)
    volumes = {normalize_volume(member.package_volume) for member in members}
    missing = [(left, right) for left in sorted(volumes) for right in sorted(volumes)
               if (left, right) not in pairs]
    if missing:
        raise ValueError(f"observed volume pairs absent from profile matrix: {missing[:10]}")
    return tuple({
        "volume_a": format(left, "f"), "volume_b": format(right, "f"),
        "status": status, "compatible": "YES" if status == "Y" else "NO",
        "matrix_version": configured_matrix_version(config), "synthetic": 0,
        "rationale": "explicit plant profile pair frozen in run configuration",
    } for (left, right), status in sorted(pairs.items()))


def matrix_status_lookup(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[Decimal, Decimal], str]:
    """Index explicit matrix rows by normalized ordered volume pair."""

    return {
        (normalize_volume(row["volume_a"]), normalize_volume(row["volume_b"])): str(
            row.get("status", row["compatible"])
        )
        for row in rows
    }


def matrix_exceptions(
    material_volumes: Iterable[tuple[str, Decimal | str | int | float]],
    matrix_rows: Sequence[Mapping[str, object]] = SYNTHETIC_VOLUME_MATRIX_V1,
) -> tuple[dict[str, str], ...]:
    """Return explicit ``NO`` or ``AVOID`` FINI pairs for one candidate group.

    Args:
        material_volumes: Material identifier and package volume pairs.
        matrix_rows: Complete configured matrix. The synthetic matrix remains
            the backwards-compatible default for direct callers.

    Returns:
        Unordered material pairs whose configured status is ``NO`` or ``AVOID``.
        Empirical ``UNKNOWN`` statuses are never converted into exceptions.
    """

    normalized = sorted(
        ((str(material), normalize_volume(volume)) for material, volume in material_volumes),
        key=lambda item: item[0],
    )
    lookup = {
        (normalize_volume(row["volume_a"]), normalize_volume(row["volume_b"])): row
        for row in matrix_rows
    }
    versions = {str(row["matrix_version"]) for row in matrix_rows}
    if len(versions) != 1:
        raise ValueError("matrix rows must carry one matrix version")
    matrix_version = next(iter(versions))
    exceptions: list[dict[str, str]] = []
    for (material_a, volume_a), (material_b, volume_b) in combinations(normalized, 2):
        row = lookup.get((volume_a, volume_b))
        if row is None:
            raise ValueError(
                f"volume pair is absent from configured matrix: {(volume_a, volume_b)}"
            )
        status = str(row.get("status", row["compatible"]))
        if status not in MATRIX_EXCEPTION_STATUSES:
            continue
        exceptions.append(
            {
                "material_a": material_a,
                "material_b": material_b,
                "volume_a": format(volume_a, "f"),
                "volume_b": format(volume_b, "f"),
                "status": status,
                "matrix_version": matrix_version,
                "family_a": str(row.get("family_a", "")),
                "family_b": str(row.get("family_b", "")),
                "rationale": str(row.get("rationale", "")),
            }
        )
    return tuple(exceptions)
