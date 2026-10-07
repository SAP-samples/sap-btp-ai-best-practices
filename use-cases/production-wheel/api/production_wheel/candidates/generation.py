"""Deterministic exact and nested restricted candidate-pool generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from itertools import combinations
from typing import TypeAlias

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
from production_wheel.schemas import (
    CoverageBasis,
    MatrixMode,
    PalletFormula,
    PVMode,
    RunConfig,
    TargetBand,
)
from production_wheel.business_rules import group_allowed

from .models import (
    BlockKey,
    BlockProjection,
    Candidate,
    CandidateMember,
    CandidatePool,
    MemberKey,
    PoolBudgetExceeded,
    PoolSizeTrace,
    ProductionVersion,
    stable_hash,
)

ProgressCallback: TypeAlias = Callable[[str, int, int], None]
MatrixLookup: TypeAlias = Mapping[tuple[Decimal, Decimal], str]
_SOURCE_ORDER = {
    "singleton": 0,
    "current_group": 1,
    "pair": 2,
    "neighborhood_add": 3,
    "neighborhood_drop": 4,
    "neighborhood_swap": 5,
    "triple": 6,
    "exact": 7,
}
_BEAM_TARGET_BAND = TargetBand()


def partition_blocks(
    members: Iterable[CandidateMember],
) -> dict[BlockKey, tuple[CandidateMember, ...]]:
    """Partition FINIs by plant and SEFI with stable member ordering.

    Args:
        members: Candidate inputs from the canonical modeled population.

    Returns:
        Insertion-ordered mapping sorted by block and FINI identifier.
    """

    grouped: dict[BlockKey, list[CandidateMember]] = defaultdict(list)
    seen: set[tuple[BlockKey, str]] = set()
    for member in members:
        key = (member.block_key, member.fini_id)
        if key in seen:
            raise ValueError(f"duplicate FINI in block: {key}")
        seen.add(key)
        grouped[member.block_key].append(member)
    return {
        block: tuple(sorted(grouped[block], key=lambda item: item.fini_id))
        for block in sorted(grouped)
    }


def line_feasible_subset_count(
    members: Sequence[CandidateMember], maximum_size: int
) -> int:
    """Count line-feasible member subsets with a structural dynamic program.

    The projection deliberately ignores PV and matrix restrictions. This makes
    exact/restricted method classification conservative and stable when an
    objective or reporting setting changes.
    """

    if maximum_size < 1:
        raise ValueError("maximum_size must be positive")
    known = [member.eligible_lines for member in members if member.eligible_lines]
    if not known:
        return 0
    limit = min(maximum_size, len(known))
    universe = frozenset().union(*known)
    states: list[dict[frozenset[str], int]] = [
        defaultdict(int) for _ in range(limit + 1)
    ]
    states[0][universe] = 1
    for lines in known:
        for size in range(limit - 1, -1, -1):
            for common, count in tuple(states[size].items()):
                if intersection := common & lines:
                    states[size + 1][intersection] += count
    return sum(sum(counts.values()) for counts in states[1:])


def _version_lots(
    versions: Sequence[ProductionVersion], block_key: BlockKey
) -> dict[str, tuple[float, ...]]:
    """Return sorted distinct positive lots by active PV identifier."""

    lots: dict[str, set[float]] = defaultdict(set)
    for version in versions:
        if version.block_key == block_key and version.is_positive:
            lots[version.pv_id].add(float(version.nominal_lot_litres))  # type: ignore[arg-type]
    return {pv: tuple(sorted(values)) for pv, values in sorted(lots.items())}


def _optimized_versions(
    versions: Sequence[ProductionVersion], block_key: BlockKey
) -> tuple[tuple[str, float], ...]:
    """Return finite unambiguous active PV alternatives for optimized mode."""

    return tuple(
        (pv, lots[0])
        for pv, lots in _version_lots(versions, block_key).items()
        if len(lots) == 1
    )


def _block_fingerprint(
    members: Sequence[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    matrix_lookup: MatrixLookup,
) -> str:
    """Hash structural inputs and scorer primitives for one block library."""

    ordered = tuple(sorted(members, key=lambda item: item.fini_id))
    block_key = ordered[0].block_key
    payload = {
        "config": config.structural_ruleset_fingerprint(),
        "block": block_key,
        "members": [
            {
                "fini_id": member.fini_id,
                "eligible_lines": sorted(member.eligible_lines),
                "demand_litres": member.demand_litres,
                "pallet_litres": member.pallet_litres,
                "package_volume": member.package_volume,
                "fixed_pv": member.fixed_pv,
                "fixed_lot_litres": member.fixed_lot_litres,
                **(
                    {"baseline_group": member.baseline_group}
                    if config.uses_baseline_candidate_seeds
                    else {}
                ),
                "pck_code": member.pck_code,
                **({name: getattr(member, name) for name in (
                    "lot_size_considered_litres", "xyz_x_dc_count", "xyz_y_dc_count", "xyz_z_dc_count",
                    "primary_dc_count", "secondary_dc_count", "sales_scenario", "sales_network_scenario")}
                    if config.business_rules else {}),
            }
            for member in ordered
        ],
        "versions": [
            {
                "pv_id": version.pv_id,
                "nominal_lot_litres": version.nominal_lot_litres,
                "active": version.active,
            }
            for version in sorted(
                (item for item in versions if item.block_key == block_key),
                key=lambda item: (item.pv_id, item.nominal_lot_litres or 0.0, item.active),
            )
        ],
        "matrix_evidence": [
            (format(volume_a, "f"), format(volume_b, "f"), status)
            for (volume_a, volume_b), status in sorted(matrix_lookup.items())
        ],
    }
    return stable_hash(payload)


def project_block(
    members: Sequence[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
) -> BlockProjection:
    """Project block scale and select exact or restricted generation.

    Args:
        members: Members from exactly one plant-SEFI block.
        versions: Finite production-version catalog, possibly for many blocks.
        config: Validated run settings and pool limits.

    Returns:
        Conservative line-only scale projection and method classification.
    """

    ordered = tuple(sorted(members, key=lambda item: item.fini_id))
    if not ordered:
        raise ValueError("a block must contain at least one member")
    block_key = ordered[0].block_key
    if any(member.block_key != block_key for member in ordered):
        raise ValueError("project_block cannot cross plant-SEFI boundaries")
    maximum_size = config.cap_for_block(*block_key)
    subsets = line_feasible_subset_count(ordered, maximum_size)
    pv_count = (
        len(_optimized_versions(versions, block_key))
        if config.pv_mode is PVMode.OPTIMIZED
        else 1
    )
    configurations = subsets * pv_count
    if config.assign_filling_lines:
        configurations *= len(set().union(*(m.eligible_lines for m in ordered)))
    limits = config.pool_limits
    # A block explicitly requested as exhaustive is still capped: forcing exact
    # enumeration on a block whose projected configuration count is enormous
    # would build an intractable MILP (memory blow-up and unbounded solver
    # presolve). Such a block falls back to the restricted beam instead.
    forced = block_key in config.exhaustive_blocks
    method = (
        "exact"
        if (
            forced and configurations <= limits.forced_exact_configuration_ceiling
        )
        or (
            not forced
            and subsets <= limits.exact_member_subsets
            and configurations <= limits.exact_pv_configurations
        )
        else "restricted"
    )
    return BlockProjection(
        block_key=block_key,
        members=len(ordered),
        maximum_size=maximum_size,
        line_feasible_member_subsets=subsets,
        finite_pv_alternatives=pv_count,
        projected_pv_configurations=configurations,
        method=method,
    )


def _common_lines(
    member_ids: MemberKey, by_id: dict[str, CandidateMember]
) -> tuple[str, ...]:
    """Return sorted common lines, or an empty tuple for infeasible evidence."""

    common: set[str] | None = None
    for fini_id in member_ids:
        lines = by_id[fini_id].eligible_lines
        if not lines:
            return ()
        common = set(lines) if common is None else common & lines
        if not common:
            return ()
    return tuple(sorted(common or ()))


def _matrix_pairs(
    member_ids: MemberKey,
    by_id: dict[str, CandidateMember],
    matrix_lookup: MatrixLookup,
) -> tuple[
    tuple[tuple[str, str, float, float], ...],
    tuple[tuple[str, str, float, float, str], ...],
    tuple[tuple[str, str, float, float, str], ...],
]:
    """Classify explicit exceptions, unknowns, and positive FINI-pair evidence."""

    # Diagonal lookup validates singleton volumes against the governed catalog.
    for fini_id in member_ids:
        volume = normalize_volume(by_id[fini_id].package_volume)
        if (volume, volume) not in matrix_lookup:
            raise ValueError(f"volume is absent from configured matrix: {volume}")
    exceptions = []
    unknown = []
    evidence = []
    for left_id, right_id in combinations(member_ids, 2):
        left = by_id[left_id]
        right = by_id[right_id]
        key = (
            normalize_volume(left.package_volume),
            normalize_volume(right.package_volume),
        )
        status = matrix_lookup.get(key)
        if status is None:
            raise ValueError(f"volume pair is absent from configured matrix: {key}")
        if status in MATRIX_EXCEPTION_STATUSES:
            exceptions.append(
                (left_id, right_id, left.package_volume, right.package_volume)
            )
        else:
            row = (
                left_id,
                right_id,
                left.package_volume,
                right.package_volume,
                status,
            )
            if status in {
                UNKNOWN_LINE_FEASIBLE,
                UNKNOWN_NO_CURRENT_LINE_OVERLAP,
            }:
                unknown.append(row)
            else:
                evidence.append(row)
    return tuple(exceptions), tuple(unknown), tuple(evidence)


def _fixed_version(
    member_ids: MemberKey,
    by_id: dict[str, CandidateMember],
    version_lots: dict[str, tuple[float, ...]],
) -> tuple[tuple[str, float], ...]:
    """Resolve exactly one common planning PV and exact positive lot."""

    aliases = {by_id[fini_id].fixed_pv for fini_id in member_ids}
    if None in aliases or "" in aliases or len(aliases) != 1:
        return ()
    pv_id = str(next(iter(aliases)))
    lots = version_lots.get(pv_id, ())
    if len(lots) != 1:
        return ()
    lot = lots[0]
    for fini_id in member_ids:
        supplied = by_id[fini_id].fixed_lot_litres
        if supplied is not None and (supplied <= 0 or float(supplied) != lot):
            return ()
    return ((pv_id, lot),)


def _candidate_configurations(
    member_ids: MemberKey,
    by_id: dict[str, CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    sources: Iterable[str],
    structural_fingerprint: str,
    matrix_lookup: MatrixLookup,
) -> tuple[Candidate, ...]:
    """Expand one feasible member set into its governed PV configurations."""

    common_lines = _common_lines(member_ids, by_id)
    if not common_lines:
        return ()
    exceptions, unknown_pairs, evidence_pairs = (
        ((), (), ())
        if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs
        else _matrix_pairs(member_ids, by_id, matrix_lookup)
    )
    if config.matrix_mode is MatrixMode.HARD and exceptions:
        return ()
    if any(matrix_lookup.get((normalize_volume(by_id[a].package_volume),
            normalize_volume(by_id[b].package_volume))) == "N"
            for a, b in combinations(member_ids, 2)):
        return ()
    block_key = by_id[member_ids[0]].block_key
    lots = _version_lots(versions, block_key)
    alternatives = (
        _fixed_version(member_ids, by_id, lots)
        if config.pv_mode is PVMode.FIXED
        else _optimized_versions(versions, block_key)
    )
    source_tiers = tuple(
        sorted(set(sources), key=lambda value: (_SOURCE_ORDER.get(value, 100), value))
    )
    membership_hash = stable_hash({"block": block_key, "members": member_ids})
    group_demand = sum(by_id[fini_id].demand_litres for fini_id in member_ids)
    pcks = {by_id[fini_id].pck_code for fini_id in member_ids}
    equal_pck = len(pcks) == 1 and None not in pcks and "" not in pcks
    result = []
    for pv_id, lot in alternatives:
        identity = {
            "fingerprint": structural_fingerprint,
            "block": block_key,
            "members": member_ids,
            "pv": pv_id,
            "nominal_lot_litres": lot,
        }
        for line in common_lines if config.assign_filling_lines else (None,):
            candidate_identity = {**identity, "selected_line": line} if line is not None else identity
            candidate = Candidate(
                block_key=block_key,
                member_ids=member_ids,
                pv_id=pv_id,
                nominal_lot_litres=lot,
                effective_batch_litres=float(effective_batch(lot, config.canonical_factor)),
                common_lines=common_lines,
                group_demand_litres=group_demand,
                matrix_exception_pairs=exceptions,
                equal_pck=equal_pck,
                source_tiers=source_tiers,
                membership_hash=membership_hash,
                candidate_hash=stable_hash(candidate_identity),
                matrix_unknown_pairs=unknown_pairs,
                matrix_evidence_pairs=evidence_pairs,
                selected_line=line,
            )
            result.append(candidate)
    return tuple(result)


def _candidate_sort_key(candidate: Candidate) -> tuple[object, ...]:
    """Return the canonical output ordering for candidate configurations."""

    return candidate.member_ids, candidate.pv_id, candidate.nominal_lot_litres, candidate.selected_line or ""


def _exact_pool(
    members: Sequence[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    projection: BlockProjection,
    progress: ProgressCallback | None,
    matrix_lookup: MatrixLookup,
) -> CandidatePool:
    """Enumerate every structurally and operationally feasible configuration."""

    by_id = {member.fini_id: member for member in members}
    structural_fingerprint = _block_fingerprint(
        members, versions, config, matrix_lookup
    )
    ids = tuple(sorted(by_id))
    candidates: list[Candidate] = []
    maximum_size = min(config.cap_for_block(*projection.block_key), len(ids))
    for size in range(1, maximum_size + 1):
        for member_ids in combinations(ids, size):
            candidates.extend(
                _candidate_configurations(
                    member_ids,
                    by_id,
                    versions,
                    config,
                    ("exact",),
                    structural_fingerprint,
                    matrix_lookup,
                )
            )
        if progress:
            progress(f"exact:{projection.block_key}", size, maximum_size)
    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    return _pool_result(
        projection,
        structural_fingerprint,
        config.structural_ruleset_fingerprint(),
        ordered,
        0,
        0,
        (),
    )


def _record_source(
    memberships: dict[MemberKey, set[str]],
    member_ids: Iterable[str],
    source: str,
    cap: int,
) -> None:
    """Add one normalized non-empty membership if it respects the active cap."""

    key = tuple(sorted(set(member_ids)))
    if key and len(key) <= cap:
        memberships[key].add(source)


def _mandatory_memberships(
    members: Sequence[CandidateMember], cap: int, include_baseline: bool
) -> dict[MemberKey, set[str]]:
    """Build baseline-independent small sets and optional historical seeds."""

    ids = tuple(member.fini_id for member in members)
    result: dict[MemberKey, set[str]] = defaultdict(set)
    for fini_id in ids:
        _record_source(result, (fini_id,), "singleton", cap)
    for pair in combinations(ids, 2):
        _record_source(result, pair, "pair", cap)
    if cap >= 3:
        for triple in combinations(ids, 3):
            _record_source(result, triple, "triple", cap)

    if not include_baseline:
        return result

    current: dict[str, list[str]] = defaultdict(list)
    for member in members:
        if member.baseline_group:
            current[member.baseline_group].append(member.fini_id)
    for group_id in sorted(current):
        group = tuple(sorted(current[group_id]))
        _record_source(result, group, "current_group", cap)
        for removed in group:
            _record_source(
                result,
                (item for item in group if item != removed),
                "neighborhood_drop",
                cap,
            )
        outside = tuple(item for item in ids if item not in group)
        for added in outside:
            _record_source(result, group + (added,), "neighborhood_add", cap)
        for removed in group:
            for added in outside:
                _record_source(
                    result,
                    tuple(item for item in group if item != removed) + (added,),
                    "neighborhood_swap",
                    cap,
                )
    return result


def _coverage(
    candidate: Candidate,
    by_id: dict[str, CandidateMember],
    basis: CoverageBasis,
    formula: PalletFormula,
    demand_days: int = 250,
) -> float:
    """Calculate one governed coverage view for beam ranking."""

    demands = {item: by_id[item].demand_litres for item in candidate.member_ids}
    pallets = {item: by_id[item].pallet_litres for item in candidate.member_ids}
    nominal = proportional_allocations(demands, candidate.effective_batch_litres)
    allocations = pallet_allocations(nominal, pallets, formula)
    return float(
        group_coverage(
            basis,
            demands,
            candidate.effective_batch_litres,
            allocations,
            demand_days,
        )
    )


def _candidate_jch(candidate: Candidate, productive_weeks: float = 50) -> float:
    """Calculate the governed changeover contribution for beam ranking."""

    frequency = group_frequency(
        candidate.group_demand_litres, candidate.effective_batch_litres, productive_weeks
    )
    return changeover_contribution(frequency, len(candidate.member_ids))


def _baseline_changes(candidate: Candidate, by_id: dict[str, CandidateMember]) -> int:
    """Count members outside the candidate's dominant historical group."""

    counts: dict[str | None, int] = defaultdict(int)
    for fini_id in candidate.member_ids:
        counts[by_id[fini_id].baseline_group] += 1
    return len(candidate.member_ids) - max(counts.values())


def _scorer_keys(
    configurations: Sequence[Candidate],
    by_id: dict[str, CandidateMember],
    include_baseline: bool,
    config: RunConfig,
) -> dict[str, tuple[object, ...]]:
    """Return best invariant score per scorer for one member set.

    Beam scoring uses the versioned provisional 5--365 day guidance instead of
    the requested objective band. Objective-only changes can therefore reuse an
    identical structural pool, as promised by the ruleset fingerprint.
    """

    scores: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for candidate in configurations:
        identity = (candidate.member_ids, candidate.pv_id, candidate.nominal_lot_litres)
        for basis in CoverageBasis:
            for formula in PalletFormula:
                coverage = _coverage(candidate, by_id, basis, formula, config.demand_days)
                suffix = f"{basis.value.lower()}:{formula.value.lower()}"
                scores[f"coverage:{suffix}"].append((coverage, identity))
                minimum = _BEAM_TARGET_BAND.minimum_days
                maximum = _BEAM_TARGET_BAND.maximum_days
                excess = max(minimum - coverage, coverage - maximum, 0.0)
                scores[f"target_band:{suffix}"].append(
                    (int(excess > 0), excess, coverage, identity)
                )
        scores["j_ch"].append((_candidate_jch(candidate, config.productive_weeks), identity))
        scores["matrix_exceptions"].append(
            (len(candidate.matrix_exception_pairs), identity)
        )
        if include_baseline:
            scores["baseline_stability"].append(
                (_baseline_changes(candidate, by_id), identity)
            )
    return {name: min(values) for name, values in sorted(scores.items())}


def _rank_member_sets(
    configurations: dict[MemberKey, tuple[Candidate, ...]],
    by_id: dict[str, CandidateMember],
    config: RunConfig,
) -> dict[str, tuple[MemberKey, ...]]:
    """Retain the configured number of member sets independently per scorer."""

    scored: dict[str, list[tuple[tuple[object, ...], MemberKey]]] = defaultdict(list)
    for member_ids, candidates in configurations.items():
        for scorer, key in _scorer_keys(
            candidates, by_id, config.uses_baseline_candidate_seeds, config
        ).items():
            scored[scorer].append((key, member_ids))
    retain = config.pool_limits.retained_member_sets_per_size_scorer
    return {
        scorer: tuple(member_ids for _, member_ids in sorted(values)[:retain])
        for scorer, values in sorted(scored.items())
    }


def _round_robin(
    rankings: dict[str, tuple[MemberKey, ...]],
    costs: dict[MemberKey, int],
    capacity: int,
) -> tuple[MemberKey, ...]:
    """Select shortlisted member sets through stable scorer round-robin."""

    chosen: list[MemberKey] = []
    seen: set[MemberKey] = set()
    positions = {name: 0 for name in rankings}
    while capacity > 0:
        advanced = False
        for scorer in sorted(rankings):
            ranking = rankings[scorer]
            while (
                positions[scorer] < len(ranking) and ranking[positions[scorer]] in seen
            ):
                positions[scorer] += 1
            if positions[scorer] >= len(ranking):
                continue
            member_ids = ranking[positions[scorer]]
            positions[scorer] += 1
            advanced = True
            if costs[member_ids] <= capacity:
                seen.add(member_ids)
                chosen.append(member_ids)
                capacity -= costs[member_ids]
        if not advanced:
            break
    return tuple(chosen)


def _restricted_pool(
    members: Sequence[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    projection: BlockProjection,
    progress: ProgressCallback | None,
    matrix_lookup: MatrixLookup,
) -> CandidatePool:
    """Construct mandatory tiers followed by deterministic nested beam sizes."""

    by_id = {member.fini_id: member for member in members}
    structural_fingerprint = _block_fingerprint(
        members, versions, config, matrix_lookup
    )
    cap = config.cap_for_block(*projection.block_key)
    mandatory_sources = _mandatory_memberships(
        members, cap, config.uses_baseline_candidate_seeds
    )
    mandatory: list[Candidate] = []
    feasible_sources: dict[MemberKey, set[str]] = {}
    for member_ids in sorted(mandatory_sources, key=lambda item: (len(item), item)):
        candidates = _candidate_configurations(
            member_ids,
            by_id,
            versions,
            config,
            mandatory_sources[member_ids],
            structural_fingerprint,
            matrix_lookup,
        )
        if candidates:
            mandatory.extend(candidates)
            feasible_sources[member_ids] = mandatory_sources[member_ids]
    limit = config.pool_limits.restricted_pv_configurations
    if len(mandatory) > limit:
        raise PoolBudgetExceeded(projection.block_key, len(mandatory), limit)

    selected = list(mandatory)
    traces: list[PoolSizeTrace] = []
    previous = tuple(key for key in feasible_sources if len(key) == 3)
    all_ids = tuple(sorted(by_id))
    for size in range(4, min(cap, len(all_ids)) + 1):
        expanded = {
            tuple(sorted(member_ids + (added,)))
            for member_ids in previous
            for added in all_ids
            if added not in member_ids
        }
        configurations: dict[MemberKey, tuple[Candidate, ...]] = {}
        for member_ids in sorted(expanded):
            if member_ids in feasible_sources:
                continue
            candidates = _candidate_configurations(
                member_ids,
                by_id,
                versions,
                config,
                (f"beam_size_{size}",),
                structural_fingerprint,
                matrix_lookup,
            )
            if candidates:
                configurations[member_ids] = candidates
        rankings = _rank_member_sets(configurations, by_id, config)
        shortlisted = set().union(*rankings.values()) if rankings else set()
        costs = {
            member_ids: len(configurations[member_ids]) for member_ids in shortlisted
        }
        capacity = limit - len(selected)
        total_shortlisted = sum(costs.values())
        if total_shortlisted <= capacity:
            chosen = tuple(sorted(shortlisted))
        else:
            chosen = _round_robin(rankings, costs, capacity)
        for member_ids in chosen:
            scorers = tuple(
                f"scorer:{name}"
                for name, ranking in rankings.items()
                if member_ids in ranking
            )
            # Rebuild only to attach complete provenance; identity hashes are unchanged.
            selected.extend(
                _candidate_configurations(
                    member_ids,
                    by_id,
                    versions,
                    config,
                    (f"beam_size_{size}", *scorers),
                    structural_fingerprint,
                    matrix_lookup,
                )
            )
        previous = tuple(
            sorted(
                set(chosen)
                | {
                    member_ids
                    for member_ids in feasible_sources
                    if len(member_ids) == size
                }
            )
        )
        traces.append(
            PoolSizeTrace(
                size=size,
                expanded_member_sets=len(expanded),
                feasible_member_sets=len(configurations),
                shortlisted_member_sets=len(shortlisted),
                selected_member_sets=len(chosen),
                selected_pv_configurations=sum(costs[item] for item in chosen),
                cumulative_pv_configurations=len(selected),
                truncated=total_shortlisted > capacity,
            )
        )
        if progress:
            progress(
                f"restricted:{projection.block_key}",
                size - 3,
                max(min(cap, len(all_ids)) - 3, 1),
            )
        if not previous or len(selected) >= limit:
            break
    ordered = tuple(sorted(selected, key=_candidate_sort_key))
    return _pool_result(
        projection,
        structural_fingerprint,
        config.structural_ruleset_fingerprint(),
        ordered,
        len(feasible_sources),
        len(mandatory),
        tuple(traces),
    )


def _pool_result(
    projection: BlockProjection,
    structural_fingerprint: str,
    config_fingerprint: str,
    candidates: tuple[Candidate, ...],
    mandatory_member_sets: int,
    mandatory_configurations: int,
    traces: tuple[PoolSizeTrace, ...],
) -> CandidatePool:
    """Create a pool with a stable digest over ordered candidate identities."""

    payload = {
        "block": projection.block_key,
        "method": projection.method,
        "fingerprint": structural_fingerprint,
        "candidate_hashes": [candidate.candidate_hash for candidate in candidates],
    }
    return CandidatePool(
        block_key=projection.block_key,
        method=projection.method,
        completeness="complete" if projection.method == "exact" else "restricted",
        structural_fingerprint=structural_fingerprint,
        projection=projection,
        candidates=candidates,
        mandatory_member_sets=mandatory_member_sets,
        mandatory_pv_configurations=mandatory_configurations,
        size_traces=traces,
        pool_hash=stable_hash(payload),
        config_fingerprint=config_fingerprint,
    )


def rebuild_pool(
    pool: CandidatePool, candidates: Iterable[Candidate]
) -> CandidatePool:
    """Return ``pool`` restricted to a subset of its candidates, re-hashed.

    Applied when agent-supplied constraints drop candidates after generation
    (for example must-link or cannot-link). The pool digest is recomputed over
    the retained candidates so downstream provenance matches exactly, and the
    library is reported as ``restricted`` whenever any candidate was removed,
    because it is no longer a full enumeration of the block. Passing the pool's
    own candidates back reproduces the original ``pool_hash`` unchanged.

    Args:
        pool: Original generated pool for one block.
        candidates: Retained subset of that pool's candidates, in any order.

    Returns:
        A pool carrying only the retained candidates with a consistent digest.
    """

    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    unchanged = len(ordered) == len(pool.candidates)
    payload = {
        "block": pool.block_key,
        "method": pool.method,
        "fingerprint": pool.structural_fingerprint,
        "candidate_hashes": [candidate.candidate_hash for candidate in ordered],
    }
    return CandidatePool(
        block_key=pool.block_key,
        method=pool.method,
        completeness=pool.completeness if unchanged else "restricted",
        structural_fingerprint=pool.structural_fingerprint,
        projection=pool.projection,
        candidates=ordered,
        mandatory_member_sets=pool.mandatory_member_sets,
        mandatory_pv_configurations=pool.mandatory_pv_configurations,
        size_traces=pool.size_traces,
        pool_hash=stable_hash(payload),
        config_fingerprint=pool.config_fingerprint,
    )


def generate_block_pool(
    members: Sequence[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    progress: ProgressCallback | None = None,
    *,
    matrix_rows: Sequence[Mapping[str, object]] | None = None,
    baseline_evidence_members: Iterable[CandidateMember] | None = None,
) -> CandidatePool:
    """Generate the selected exact or restricted pool for one block.

    Args:
        members: Candidate members belonging to exactly one plant-SEFI block.
        versions: Canonical finite production versions, possibly for all blocks.
        config: Validated structural settings and pool budgets.
        progress: Optional callback receiving phase, completed units, and total.
        matrix_rows: Optional global matrix evidence. When omitted, empirical
            evidence is derived from the supplied block population for tests.
        baseline_evidence_members: Optional complete historical assignment
            population used only for positive empirical evidence.

    Returns:
        Deterministic candidate pool with completeness and convergence evidence.

    Raises:
        PoolBudgetExceeded: Mandatory restricted configurations exceed the cap.
    """

    ordered = tuple(sorted(members, key=lambda item: item.fini_id))
    rows = (
        ()
        if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs
        else matrix_rows
        or configured_matrix(config, ordered, baseline_evidence_members)
    )
    lookup = matrix_status_lookup(rows) if rows else {}
    projection = project_block(ordered, versions, config)
    if projection.method == "exact":
        pool = _exact_pool(
            ordered, versions, config, projection, progress, lookup
        )
    else:
        pool = _restricted_pool(ordered, versions, config, projection, progress, lookup)
    # A conditional rule need not hold for a partial group. Apply it after beam
    # expansion so a minimum-size rule cannot prune the seeds of valid groups.
    by_id = {m.fini_id: m for m in ordered}
    retained = tuple(c for c in pool.candidates if group_allowed(c,
        [by_id[fid] for fid in c.member_ids], config))
    from dataclasses import replace
    return replace(rebuild_pool(pool, retained), completeness=pool.completeness)


def generate_candidate_pools(
    members: Iterable[CandidateMember],
    versions: Sequence[ProductionVersion],
    config: RunConfig,
    progress: ProgressCallback | None = None,
    baseline_evidence_members: Iterable[CandidateMember] | None = None,
) -> tuple[CandidatePool, ...]:
    """Generate pools using modeled lines and complete historical evidence."""

    population = tuple(members)
    blocks = partition_blocks(population)
    matrix_rows = (
        None
        if config.matrix_mode is MatrixMode.OFF and not config.matrix_pairs
        else configured_matrix(
            config,
            population,
            baseline_evidence_members,
        )
    )
    result = []
    for completed, block_key in enumerate(blocks, start=1):
        result.append(
            generate_block_pool(
                blocks[block_key],
                versions,
                config,
                progress,
                matrix_rows=matrix_rows,
            )
        )
        if progress:
            progress("blocks", completed, len(blocks))
    return tuple(result)
