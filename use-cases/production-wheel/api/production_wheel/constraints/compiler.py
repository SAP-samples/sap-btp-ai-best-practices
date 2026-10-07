"""Compile a validated ``SolveRequest`` into concrete optimizer behaviour.

This module is the bridge the orchestrating agent relies on. It turns the finite,
typed ``ConstraintSpec`` vocabulary (``schemas.py``) -- already
contradiction-checked by :func:`production_wheel.validation.validate_request` --
into three concrete effects applied around the existing deterministic solve:

1. **Member pre-filter.** ``SolveRequest.scope`` and ``fini_disposition``
   include/exclude restrict the modeled FINI population *before* candidate
   generation, by rebuilding a scoped :class:`CanonicalInputs`.
2. **Per-block group-size caps.** ``max_group_size`` and
   ``group_size_relaxation`` merge into ``RunConfig.group_size_overrides``; the
   caps are enforced during generation and solver eligibility through
   ``RunConfig.cap_for_block`` (see the group-size-override feature).
3. **Candidate filter.** ``must_link``, ``cannot_link``, ``fixed_pv``,
   ``allowed_pvs`` and ``required_lines`` drop generated candidates that violate
   the rule, applied to the pools between generation and solving.

Constraint kinds that cannot yet be compiled in this phase raise
:class:`ConstraintCompileError` so a request is never silently under-applied:

- ``coverage_bound`` -- needs a scoped linear row in the block/portfolio MILP
  masters;
- ``freeze_assignment`` -- baseline co-membership, and greenfield runs carry no
  baseline;
- ``volume_compatibility_override`` -- needs package-volume matrix injection at
  generation time.

No natural language and no arbitrary code are handled here; only the finite,
validated typed vocabulary. Compilation is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from production_wheel.candidates import Candidate, CandidateMember, CandidatePool, rebuild_pool
from production_wheel.rule_models import GroupRule, SelectionBound
from production_wheel.schemas import (
    AllowedPVsConstraint,
    CannotLinkConstraint,
    ConstraintScope,
    FiniDispositionConstraint,
    FixedPVConstraint,
    GroupSizeOverride,
    GroupSizeRelaxationConstraint,
    MaxGroupSizeConstraint,
    MustLinkConstraint,
    RequiredLinesConstraint,
    RunConfig,
    SolveRequest,
)

if TYPE_CHECKING:  # CanonicalInputs lives in scenarios; annotations only, no cycle.
    from production_wheel.scenarios import CanonicalInputs

BlockKey = tuple[str, str]
CandidatePredicate = Callable[[Candidate], bool]

# Typed kinds validated by validate_request but not yet applicable here. The
# message names what each one needs so the deferral is explicit to the caller.
DEFERRED_CONSTRAINT_KINDS: dict[str, str] = {
    "coverage_bound": "a scoped coverage row in the block/portfolio MILP masters",
    "freeze_assignment": "baseline co-membership (greenfield runs carry no baseline)",
    "volume_compatibility_override": "package-volume matrix injection at generation time",
}


class ConstraintCompileError(ValueError):
    """Raised when a validated request cannot be fully applied by this compiler."""


def _scope_matches(block: BlockKey, scope: ConstraintScope) -> bool:
    """Return whether one (plant, SEFI) block falls inside an optional scope.

    An unset ``plant`` or ``sefi`` is a wildcard, so a plant-only scope matches
    every SEFI of that plant and an empty scope matches every block.
    """

    plant, sefi = block
    if scope.plant is not None and scope.plant != plant:
        return False
    if scope.sefi is not None and scope.sefi != sefi:
        return False
    return True


@dataclass(frozen=True)
class CompiledRequest:
    """A validated request lowered into inputs, config, and a candidate filter.

    Attributes:
        inputs: Scope/disposition-filtered canonical inputs to solve over.
        config: Run configuration with per-block group-size caps merged in.
        applied: Ordered audit records describing every effect applied, for
            provenance in the run manifest.
    """

    inputs: "CanonicalInputs"
    config: RunConfig
    applied: tuple[dict[str, object], ...]
    _predicates: tuple[CandidatePredicate, ...]

    def filter_pools(
        self, pools: Iterable[CandidatePool]
    ) -> tuple[CandidatePool, ...]:
        """Return the pools restricted to candidates satisfying every predicate.

        Each pool is re-hashed over its retained candidates so downstream
        provenance stays consistent. With no candidate-level constraints this is
        an identity transform (pools are returned unchanged).
        """

        pools = tuple(pools)
        if not self._predicates:
            return pools
        rebuilt: list[CandidatePool] = []
        for pool in pools:
            kept = tuple(
                candidate
                for candidate in pool.candidates
                if all(predicate(candidate) for predicate in self._predicates)
            )
            rebuilt.append(rebuild_pool(pool, kept))
        return tuple(rebuilt)


def compile_request(request: SolveRequest, inputs: "CanonicalInputs") -> CompiledRequest:
    """Lower a validated request into scoped inputs, config, and a pool filter.

    Args:
        request: A request that has already passed ``validate_request``.
        inputs: Canonical extracted inputs for the full modeled population.

    Returns:
        A :class:`CompiledRequest` ready to drive ``run_single_scenario`` or
        ``run_greenfield_frontier`` (via their ``pool_transform`` hook).

    Raises:
        ConstraintCompileError: A constraint kind is not yet supported, a scope
            or group-size rule matches no modeled block, an ``include`` names an
            unmodeled FINI, or the resulting scope is empty.
    """

    deferred = sorted({c.kind for c in request.constraints} & DEFERRED_CONSTRAINT_KINDS.keys())
    if deferred:
        detail = ", ".join(f"{kind} (needs {DEFERRED_CONSTRAINT_KINDS[kind]})" for kind in deferred)
        raise ConstraintCompileError(f"constraint kinds not yet supported by the compiler: {detail}")

    applied: list[dict[str, object]] = []

    # Scope the active PV population, including optimized-only eligible FINIs.
    active_members = inputs.members_for(request.config)
    if active_members is not inputs.members:
        inputs = replace(inputs, members=active_members)
    kept_ids = _resolve_member_scope(request, inputs, applied)
    filtered_inputs = _filter_inputs(inputs, kept_ids)
    if not filtered_inputs.members:
        raise ConstraintCompileError("solve scope and dispositions removed every modeled FINI")

    config = _merge_group_size_caps(request, filtered_inputs, applied)
    _check_link_components(request, filtered_inputs, config)
    rules = (*config.business_rules, *(c for c in request.constraints if isinstance(c, (GroupRule, SelectionBound))))
    if len({r.constraint_id for r in rules}) != len(rules):
        raise ConstraintCompileError("business rule identifiers must be unique across profile and scenario")
    for rule in rules:
        if rule.approval_status != "approved":
            raise ConstraintCompileError(f"business rule {rule.constraint_id} requires approval")
        if not any(_scope_matches(m.block_key, rule.scope) for m in filtered_inputs.members):
            raise ConstraintCompileError(f"business rule {rule.constraint_id} matches no modeled block")
        applied.append({"effect": rule.kind, "constraint_id": rule.constraint_id,
            "rule": rule.model_dump(mode="json")})
    config = RunConfig.model_validate({**config.model_dump(), "business_rules": rules})
    predicates = _candidate_predicates(request, applied)

    return CompiledRequest(
        inputs=filtered_inputs,
        config=config,
        applied=tuple(applied),
        _predicates=tuple(predicates),
    )


def _check_link_components(request, inputs, config):
    """Reject linked components crossing scope or exceeding their actual block cap."""
    groups = []
    for rule in request.constraints:
        if isinstance(rule, MustLinkConstraint):
            component = set(rule.materials)
            for group in groups[:]:
                if component & group:
                    component |= group
                    groups.remove(group)
            groups.append(component)
    by_id = {m.fini_id: m for m in inputs.members}
    for component in groups:
        if not component <= by_id.keys():
            raise ConstraintCompileError('must-link component contains FINIs outside the active scope')
        blocks = {by_id[fid].block_key for fid in component}
        if len(blocks) != 1:
            raise ConstraintCompileError('must-link component crosses plant/SEFI blocks')
        block = next(iter(blocks))
        if len(component) > config.cap_for_block(*block):
            raise ConstraintCompileError(f'must-link component exceeds group cap for {block}')


def _resolve_member_scope(
    request: SolveRequest, inputs: "CanonicalInputs", applied: list[dict[str, object]]
) -> frozenset[str]:
    """Return the FINI identifiers that survive scope and disposition filtering."""

    modeled_ids = {member.fini_id for member in inputs.members}
    modeled_blocks = {member.block_key for member in inputs.members}

    scoped_ids = set(modeled_ids)
    if request.scope:
        for entry in request.scope:
            if not any(_scope_matches(block, entry) for block in modeled_blocks):
                raise ConstraintCompileError(
                    f"solve scope {entry.plant or '*'}/{entry.sefi or '*'} matches no modeled block"
                )
        scoped_ids = {
            member.fini_id
            for member in inputs.members
            if any(_scope_matches(member.block_key, entry) for entry in request.scope)
        }
        applied.append(
            {
                "effect": "scope",
                "kept_fini_count": len(scoped_ids),
                "scope": [
                    {"plant": entry.plant, "sefi": entry.sefi} for entry in request.scope
                ],
            }
        )

    excluded: set[str] = set()
    included: set[str] = set()
    for constraint in request.constraints:
        if isinstance(constraint, FiniDispositionConstraint):
            if constraint.action == "exclude":
                excluded.add(constraint.material)
            else:
                included.add(constraint.material)

    kept = scoped_ids - excluded
    for material in sorted(included):
        if material not in modeled_ids:
            raise ConstraintCompileError(f"cannot include FINI outside the modeled population: {material}")
        if material not in kept:
            raise ConstraintCompileError(
                f"FINI {material} is marked include but removed by solve scope or an exclude"
            )

    if excluded:
        applied.append({"effect": "exclude_finis", "finis": sorted(excluded & modeled_ids)})
    if included:
        applied.append({"effect": "include_finis", "finis": sorted(included)})
    return frozenset(kept)


def _filter_inputs(inputs: "CanonicalInputs", kept_ids: frozenset[str]) -> "CanonicalInputs":
    """Return inputs restricted to the retained FINIs, or the originals unchanged."""

    if kept_ids == {member.fini_id for member in inputs.members}:
        return inputs

    def _keep(members: Iterable[CandidateMember]) -> tuple[CandidateMember, ...]:
        return tuple(member for member in members if member.fini_id in kept_ids)

    def _row_id(row: dict[str, str]) -> str | None:
        return row.get("fini_id") or row.get("material")

    # Filter the raw FINI rows too so downstream reporting reflects the scoped
    # population rather than the full workbook.
    return replace(
        inputs,
        fini_rows=tuple(row for row in inputs.fini_rows if _row_id(row) in kept_ids),
        members=_keep(inputs.members),
        optimized_pv_members=_keep(inputs.optimized_pv_members),
        baseline_evidence_members=_keep(inputs.baseline_evidence_members),
    )


def _merge_group_size_caps(
    request: SolveRequest, inputs: "CanonicalInputs", applied: list[dict[str, object]]
) -> RunConfig:
    """Merge scoped max-size / relaxation constraints into per-block cap overrides."""

    active_blocks = {member.block_key for member in inputs.members}
    overrides: dict[BlockKey, int] = {
        (override.plant, override.sefi): override.maximum
        for override in request.config.group_size_overrides
    }

    for constraint in request.constraints:
        if isinstance(constraint, MaxGroupSizeConstraint):
            cap = constraint.maximum
        elif isinstance(constraint, GroupSizeRelaxationConstraint):
            cap = constraint.base_limit + constraint.max_excess
        else:
            continue
        targets = sorted(block for block in active_blocks if _scope_matches(block, constraint.scope))
        if not targets:
            raise ConstraintCompileError(
                f"group-size constraint {constraint.constraint_id} matches no modeled block in scope"
            )
        for block in targets:
            overrides[block] = min(overrides.get(block, cap), cap)
        applied.append(
            {
                "effect": "group_size_cap",
                "constraint_id": constraint.constraint_id,
                "maximum": cap,
                "blocks": [f"{plant}/{sefi}" for plant, sefi in targets],
            }
        )

    if not overrides:
        return request.config
    override_models = tuple(
        GroupSizeOverride(plant=plant, sefi=sefi, maximum=maximum)
        for (plant, sefi), maximum in sorted(overrides.items())
    )
    return request.config.model_copy(update={"group_size_overrides": override_models})


def _candidate_predicates(
    request: SolveRequest, applied: list[dict[str, object]]
) -> list[CandidatePredicate]:
    """Build per-candidate admissibility predicates for structural constraints."""

    predicates: list[CandidatePredicate] = []
    for constraint in request.constraints:
        if isinstance(constraint, MustLinkConstraint):
            linked = frozenset(constraint.materials)

            def must_link(candidate: Candidate, linked: frozenset[str] = linked) -> bool:
                # A candidate is admissible only if it holds all of the linked
                # FINIs or none of them; partial overlap would split the group.
                overlap = linked.intersection(candidate.member_ids)
                return not overlap or overlap == linked

            predicates.append(must_link)
            applied.append(
                {"effect": "must_link", "constraint_id": constraint.constraint_id, "materials": sorted(linked)}
            )
        elif isinstance(constraint, CannotLinkConstraint):
            first, second = constraint.materials

            def cannot_link(candidate: Candidate, first: str = first, second: str = second) -> bool:
                members = set(candidate.member_ids)
                return not (first in members and second in members)

            predicates.append(cannot_link)
            applied.append(
                {"effect": "cannot_link", "constraint_id": constraint.constraint_id, "materials": [first, second]}
            )
        elif isinstance(constraint, FixedPVConstraint):
            pv = constraint.production_version
            scope = constraint.scope

            def fixed_pv(candidate: Candidate, pv: str = pv, scope: ConstraintScope = scope) -> bool:
                return not _scope_matches(candidate.block_key, scope) or candidate.pv_id == pv

            predicates.append(fixed_pv)
            applied.append(
                {"effect": "fixed_pv", "constraint_id": constraint.constraint_id, "production_version": pv}
            )
        elif isinstance(constraint, AllowedPVsConstraint):
            allowed = frozenset(constraint.production_versions)
            scope = constraint.scope

            def allowed_pvs(candidate: Candidate, allowed: frozenset[str] = allowed, scope: ConstraintScope = scope) -> bool:
                return not _scope_matches(candidate.block_key, scope) or candidate.pv_id in allowed

            predicates.append(allowed_pvs)
            applied.append(
                {"effect": "allowed_pvs", "constraint_id": constraint.constraint_id, "production_versions": sorted(allowed)}
            )
        elif isinstance(constraint, RequiredLinesConstraint):
            lines = frozenset(str(line) for line in constraint.filling_lines)
            scope = constraint.scope

            def required_lines(candidate: Candidate, lines: frozenset[str] = lines, scope: ConstraintScope = scope) -> bool:
                # The group must be producible on at least one permitted line.
                return not _scope_matches(candidate.block_key, scope) or bool(
                    set(candidate.common_lines) & lines
                )

            predicates.append(required_lines)
            applied.append(
                {"effect": "required_lines", "constraint_id": constraint.constraint_id, "filling_lines": sorted(lines)}
            )
    return predicates
