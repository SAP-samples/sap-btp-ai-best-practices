"""Tests for compiling typed constraint requests into optimizer behaviour.

Covers the member pre-filter (scope + disposition), per-block group-size cap
merging, the candidate-admissibility predicates for structural constraints, the
explicit deferral of not-yet-supported kinds, and the pool_transform hook wired
into the solve path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_wheel.candidates import (
    Candidate,
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.constraints import (
    DEFERRED_CONSTRAINT_KINDS,
    ConstraintCompileError,
    compile_request,
)
from production_wheel.scenarios import CanonicalInputs, run_single_scenario
from production_wheel.schemas import (
    AllowedPVsConstraint,
    CannotLinkConstraint,
    ConstraintScope,
    CoverageBoundConstraint,
    CoverageMode,
    FiniDispositionConstraint,
    FixedPVConstraint,
    FreezeAssignmentConstraint,
    GroupSizeRelaxationConstraint,
    MaxGroupSizeConstraint,
    MustLinkConstraint,
    RequiredLinesConstraint,
    RunConfig,
    SolveRequest,
    VolumeCompatibilityOverrideConstraint,
)


def _member(
    fini_id: str,
    *,
    plant: str = "P1",
    sefi: str = "S1",
    lines: tuple[str, ...] = ("1",),
    pv: str = "PV1",
) -> CandidateMember:
    """One synthetic modeled member."""

    return CandidateMember(
        fini_id=fini_id,
        plant=plant,
        sefi=sefi,
        eligible_lines=frozenset(lines),
        demand_litres=1_000.0,
        pallet_litres=25.0,
        package_volume=0.25,
        fixed_pv=pv,
        fixed_lot_litres=100.0,
        baseline_group=None,
        pck_code="PCK-A",
    )


def _inputs(members: list[CandidateMember]) -> CanonicalInputs:
    """Build minimal canonical inputs from synthetic members."""

    rows = tuple(
        {"fini_id": m.fini_id, "plant": m.plant, "sefi": m.sefi, "model_status": "modeled"}
        for m in members
    )
    return CanonicalInputs(
        extracted_directory=Path("."),
        fini_rows=rows,
        members=tuple(members),
        production_versions=(),
        baseline_evidence_members=(),
        optimized_pv_members=(),
    )


def _candidate(
    member_ids: tuple[str, ...],
    *,
    block: tuple[str, str] = ("P1", "S1"),
    pv: str = "PV1",
    lines: tuple[str, ...] = ("1",),
) -> Candidate:
    """A lightweight candidate carrying only the fields predicates read."""

    tag = "-".join(member_ids)
    return Candidate(
        block_key=block,
        member_ids=member_ids,
        pv_id=pv,
        nominal_lot_litres=100.0,
        effective_batch_litres=90.0,
        common_lines=lines,
        group_demand_litres=1_000.0,
        matrix_exception_pairs=(),
        equal_pck=True,
        source_tiers=("test",),
        membership_hash=f"m-{tag}",
        candidate_hash=f"c-{tag}-{pv}-{'|'.join(lines)}",
    )


def _survives(compiled, candidate: Candidate) -> bool:
    """Return whether a candidate passes every compiled candidate predicate."""

    return all(predicate(candidate) for predicate in compiled._predicates)


def test_empty_request_is_identity() -> None:
    """No constraints and no scope leaves inputs, config, and pools untouched."""

    inputs = _inputs([_member("1"), _member("2")])
    compiled = compile_request(SolveRequest(), inputs)
    assert compiled.inputs is inputs
    assert compiled.config == RunConfig()
    assert compiled.applied == ()
    pool_stub = ("sentinel",)
    assert compiled.filter_pools(pool_stub) == pool_stub


def test_scope_filters_members_and_rows() -> None:
    """A plant-only scope keeps just that plant's members and raw rows."""

    inputs = _inputs(
        [_member("1"), _member("2"), _member("3", plant="P2", sefi="S2")]
    )
    request = SolveRequest(scope=(ConstraintScope(plant="P1"),))
    compiled = compile_request(request, inputs)
    assert {m.fini_id for m in compiled.inputs.members} == {"1", "2"}
    assert {row["fini_id"] for row in compiled.inputs.fini_rows} == {"1", "2"}
    assert any(entry["effect"] == "scope" for entry in compiled.applied)


def test_scope_matching_no_block_raises() -> None:
    """A scope that matches no modeled block is a hard error, never silent."""

    inputs = _inputs([_member("1")])
    request = SolveRequest(scope=(ConstraintScope(plant="NOPE"),))
    with pytest.raises(ConstraintCompileError):
        compile_request(request, inputs)


def test_disposition_exclude_and_include() -> None:
    """Exclude drops a FINI; include of a present FINI is accepted."""

    inputs = _inputs([_member("1"), _member("2"), _member("3")])
    request = SolveRequest(
        constraints=(
            FiniDispositionConstraint(
                kind="fini_disposition", constraint_id="d1", material="2",
                action="exclude", reason="obsolete",
            ),
            FiniDispositionConstraint(
                kind="fini_disposition", constraint_id="d2", material="1",
                action="include", reason="keep",
            ),
        )
    )
    compiled = compile_request(request, inputs)
    assert {m.fini_id for m in compiled.inputs.members} == {"1", "3"}


def test_include_unmodeled_raises() -> None:
    """Including a FINI outside the modeled population is rejected."""

    inputs = _inputs([_member("1")])
    request = SolveRequest(
        constraints=(
            FiniDispositionConstraint(
                kind="fini_disposition", constraint_id="d1", material="999",
                action="include", reason="not modeled",
            ),
        )
    )
    with pytest.raises(ConstraintCompileError):
        compile_request(request, inputs)


def test_max_group_size_sets_scoped_override() -> None:
    """A scoped max_group_size becomes a per-block cap override on the config."""

    inputs = _inputs([_member("1"), _member("2", plant="P2", sefi="S2")])
    request = SolveRequest(
        constraints=(
            MaxGroupSizeConstraint(
                kind="max_group_size", constraint_id="g1",
                scope=ConstraintScope(plant="P1", sefi="S1"), maximum=3,
            ),
        )
    )
    compiled = compile_request(request, inputs)
    assert compiled.config.cap_for_block("P1", "S1") == 3
    assert compiled.config.cap_for_block("P2", "S2") == 7


def test_group_size_relaxation_override() -> None:
    """A relaxation constraint folds base_limit + max_excess into the block cap."""

    inputs = _inputs([_member("1")])
    request = SolveRequest(
        constraints=(
            GroupSizeRelaxationConstraint(
                kind="group_size_relaxation", constraint_id="g1",
                scope=ConstraintScope(plant="P1", sefi="S1"),
                base_limit=7, max_excess=2,
            ),
        )
    )
    compiled = compile_request(request, inputs)
    assert compiled.config.cap_for_block("P1", "S1") == 9


def test_group_size_constraint_no_block_raises() -> None:
    """A cap whose scope matches no modeled block is a hard error."""

    inputs = _inputs([_member("1")])
    request = SolveRequest(
        constraints=(
            MaxGroupSizeConstraint(
                kind="max_group_size", constraint_id="g1",
                scope=ConstraintScope(plant="P9", sefi="S9"), maximum=3,
            ),
        )
    )
    with pytest.raises(ConstraintCompileError):
        compile_request(request, inputs)


@pytest.mark.parametrize(
    "constraint",
    [
        CoverageBoundConstraint(
            kind="coverage_bound", constraint_id="x", upper_days=20.0
        ),
        FreezeAssignmentConstraint(
            kind="freeze_assignment", constraint_id="x", materials=("1",)
        ),
        VolumeCompatibilityOverrideConstraint(
            kind="volume_compatibility_override", constraint_id="x",
            volume_a="0.25", volume_b="1.0", compatible=True,
        ),
    ],
)
def test_deferred_kinds_raise(constraint) -> None:
    """Kinds not yet compilable raise rather than silently no-op."""

    assert constraint.kind in DEFERRED_CONSTRAINT_KINDS
    inputs = _inputs([_member("1")])
    with pytest.raises(ConstraintCompileError):
        compile_request(SolveRequest(constraints=(constraint,)), inputs)


def test_must_link_predicate() -> None:
    """Must-link admits only candidates holding all or none of the linked set."""

    inputs = _inputs([_member("1"), _member("2"), _member("3")])
    request = SolveRequest(
        constraints=(
            MustLinkConstraint(kind="must_link", constraint_id="m1", materials=("1", "2")),
        )
    )
    compiled = compile_request(request, inputs)
    assert _survives(compiled, _candidate(("1", "2")))
    assert _survives(compiled, _candidate(("1", "2", "3")))
    assert _survives(compiled, _candidate(("3",)))
    assert not _survives(compiled, _candidate(("1",)))
    assert not _survives(compiled, _candidate(("1", "3")))


def test_cannot_link_predicate() -> None:
    """Cannot-link drops any candidate that holds both forbidden FINIs."""

    inputs = _inputs([_member("1"), _member("2"), _member("3")])
    request = SolveRequest(
        constraints=(
            CannotLinkConstraint(kind="cannot_link", constraint_id="c1", materials=("1", "2")),
        )
    )
    compiled = compile_request(request, inputs)
    assert not _survives(compiled, _candidate(("1", "2")))
    assert not _survives(compiled, _candidate(("1", "2", "3")))
    assert _survives(compiled, _candidate(("1", "3")))
    assert _survives(compiled, _candidate(("2",)))


def test_pv_predicates_are_scoped() -> None:
    """Fixed/allowed PV filters apply only inside their scope, by pv_id."""

    inputs = _inputs([_member("1"), _member("2", plant="P2", sefi="S2")])
    request = SolveRequest(
        constraints=(
            FixedPVConstraint(
                kind="fixed_pv", constraint_id="p1",
                scope=ConstraintScope(plant="P1", sefi="S1"), production_version="PV1",
            ),
        )
    )
    compiled = compile_request(request, inputs)
    assert _survives(compiled, _candidate(("1",), pv="PV1"))
    assert not _survives(compiled, _candidate(("1",), pv="PV2"))
    # Out of scope (different block) -> untouched regardless of PV.
    assert _survives(compiled, _candidate(("2",), block=("P2", "S2"), pv="PV2"))

    allowed = SolveRequest(
        constraints=(
            AllowedPVsConstraint(
                kind="allowed_pvs", constraint_id="p2",
                scope=ConstraintScope(plant="P1", sefi="S1"),
                production_versions=("PV1", "PV3"),
            ),
        )
    )
    compiled_allowed = compile_request(allowed, inputs)
    assert _survives(compiled_allowed, _candidate(("1",), pv="PV3"))
    assert not _survives(compiled_allowed, _candidate(("1",), pv="PV2"))


def test_required_lines_predicate() -> None:
    """Required-lines keeps only candidates producible on a permitted line."""

    inputs = _inputs([_member("1")])
    request = SolveRequest(
        constraints=(
            RequiredLinesConstraint(
                kind="required_lines", constraint_id="l1",
                scope=ConstraintScope(plant="P1", sefi="S1"), filling_lines=(1,),
            ),
        )
    )
    compiled = compile_request(request, inputs)
    assert _survives(compiled, _candidate(("1",), lines=("1", "2")))
    assert not _survives(compiled, _candidate(("1",), lines=("2", "3")))


def test_must_link_filter_pools_on_generated_pool() -> None:
    """filter_pools rebuilds a real generated pool with only admissible candidates."""

    members = [_member("1"), _member("2"), _member("3")]
    versions = (ProductionVersion(plant="P1", sefi="S1", pv_id="PV1", nominal_lot_litres=100.0),)
    config = RunConfig(coverage_mode=CoverageMode.GREENFIELD_COVERAGE)
    pools = generate_candidate_pools(members, versions, config)

    request = SolveRequest(
        constraints=(
            MustLinkConstraint(kind="must_link", constraint_id="m1", materials=("1", "2")),
        )
    )
    compiled = compile_request(request, _inputs(members))
    filtered = compiled.filter_pools(pools)

    surviving = {
        frozenset(candidate.member_ids)
        for pool in filtered
        for candidate in pool.candidates
    }
    # No surviving candidate splits the linked pair.
    for members_set in surviving:
        overlap = members_set & {"1", "2"}
        assert overlap in (frozenset(), frozenset({"1", "2"}))
    # A uniting candidate still exists, so an exact cover remains feasible.
    assert frozenset({"1", "2"}) in surviving
    # Pools were rebuilt (no longer a complete enumeration).
    assert all(pool.completeness == "restricted" for pool in filtered)


def test_run_single_scenario_accepts_pool_transform() -> None:
    """The solve path honours the pool_transform hook and still finds an incumbent."""

    members = [_member("1"), _member("2"), _member("3")]
    versions = (ProductionVersion(plant="P1", sefi="S1", pv_id="PV1", nominal_lot_litres=100.0),)
    inputs = CanonicalInputs(
        extracted_directory=Path("."),
        fini_rows=(),
        members=tuple(members),
        production_versions=versions,
        baseline_evidence_members=(),
        optimized_pv_members=(),
    )
    config = RunConfig(coverage_mode=CoverageMode.GREENFIELD_COVERAGE)
    request = SolveRequest(
        constraints=(
            MustLinkConstraint(kind="must_link", constraint_id="m1", materials=("1", "2")),
        )
    )
    compiled = compile_request(request, inputs)
    outcome = run_single_scenario(inputs, config, pool_transform=compiled.filter_pools)
    assert outcome.solve_result is not None
    assert outcome.solve_result.has_incumbent


def test_optimized_population_obeys_scope(tmp_path):
    """Optimized-only FINIs cannot leak around request scope filtering."""
    from production_wheel.candidates import CandidateMember, ProductionVersion
    from production_wheel.scenarios import CanonicalInputs
    from production_wheel.schemas import SolveRequest
    from production_wheel.constraints import compile_request
    fixed=CandidateMember('F1','P','S',frozenset({'L1'}),1000,10,1,'PV1')
    optimized=CandidateMember('F2','P','T',frozenset({'L2'}),1000,10,1,None)
    inputs=CanonicalInputs(tmp_path,(),(fixed,),(),(),(fixed,optimized))
    request=SolveRequest.model_validate({'config':{'coverage_mode':'PARETO','pv_mode':'OPTIMIZED'},'scope':[{'plant':'P','sefi':'S'}]})
    result=compile_request(request,inputs)
    assert {m.fini_id for m in result.inputs.members_for(result.config)}=={'F1'}
