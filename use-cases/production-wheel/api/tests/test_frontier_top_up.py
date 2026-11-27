"""Tests for the retained-point top-up of the greenfield Pareto frontier.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_frontier_top_up.py -q
"""

from __future__ import annotations

import math

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.pareto import _next_top_up_budget, build_greenfield_frontier
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig


def _member(fini_id: str, plant: str, demand: float) -> CandidateMember:
    """Create one modeled FINI in its own plant block."""

    return CandidateMember(
        fini_id=fini_id,
        plant=plant,
        sefi="S1",
        eligible_lines=frozenset({"L1"}),
        demand_litres=demand,
        pallet_litres=25.0,
        package_volume=0.25,
        fixed_pv="PV1",
        baseline_group=None,
        pck_code="PCK1",
    )


def _six_block_frontier(minimum_point_count: int):
    """Solve six independent two-FINI blocks whose frontier has 7 trade-offs.

    A steep exponent packs the 7 requested budgets near the lowest ``J_CH``,
    so the first pass alone retains only 3 distinct points.
    """

    members = tuple(
        member
        for index in range(6)
        for member in (
            _member(f"A{index}", f"P{index}", 1_000.0 * (index + 1)),
            _member(f"B{index}", f"P{index}", 700.0 * (index + 1)),
        )
    )
    versions = tuple(
        ProductionVersion(f"P{index}", "S1", "PV1", 100.0) for index in range(6)
    )
    config = RunConfig(coverage_mode=CoverageMode.PARETO, matrix_mode=MatrixMode.OFF)
    pools = generate_candidate_pools(members, versions, config)
    return build_greenfield_frontier(
        pools,
        members,
        config,
        point_count=7,
        global_epsilon_exponent=10,
        block_worker_count=1,
        minimum_point_count=minimum_point_count,
    )


def test_first_split_is_the_schedule_midpoint() -> None:
    """With only the anchors retained, the split sits at schedule position 0.5."""

    position, epsilon = _next_top_up_budget([0.0, 100.0], 0.0, 100.0, 2.0, [])
    assert position == 0.5
    assert math.isclose(epsilon, 25.0)


def test_equal_j_ch_gaps_prefer_the_low_changeover_side() -> None:
    """Gaps of equal J_CH width are wider in schedule space near low J_CH."""

    position, epsilon = _next_top_up_budget([0.0, 50.0, 100.0], 0.0, 100.0, 2.0, [])
    assert position < math.sqrt(0.5)
    assert epsilon < 50.0


def test_cleared_probe_bisects_the_rest_of_the_gap() -> None:
    """A fruitless probe at 0.5 leaves only (0.5, 1] open, so the next is 0.75."""

    position, epsilon = _next_top_up_budget([0.0, 100.0], 0.0, 100.0, 2.0, [0.5])
    assert position == 0.75
    assert math.isclose(epsilon, 56.25)


def test_no_budget_when_anchors_coincide_or_gaps_are_closed() -> None:
    """Degenerate anchors and fully cleared gaps stop the top-up."""

    assert _next_top_up_budget([5.0], 5.0, 5.0, 2.0, []) is None
    assert _next_top_up_budget([0.0, 100.0], 0.0, 100.0, 2.0, [1.0 - 1e-7]) is None


def test_top_up_recovers_every_distinct_trade_off() -> None:
    """The top-up reaches the target the first pass missed, keeping a Pareto set."""

    first_pass = _six_block_frontier(minimum_point_count=2)
    topped_up = _six_block_frontier(minimum_point_count=7)

    assert len(first_pass.points) < 7
    assert len(topped_up.points) == 7
    assert not topped_up.frontier_shortfall
    assert topped_up.minimum_point_count == 7
    kinds = [audit.request_kind for audit in topped_up.global_solve_audits]
    assert kinds[:7] == ["schedule"] * 7
    assert set(kinds[7:]) == {"top_up"}
    assert len(kinds) - 7 <= 2 * 7
    j_ch = [point.j_ch for point in topped_up.points]
    coverage = [point.demand_weighted_mean_coverage_days for point in topped_up.points]
    assert j_ch == sorted(j_ch)
    assert coverage == sorted(coverage, reverse=True)


def test_target_is_clamped_to_requested_points() -> None:
    """An explicit small request is honoured instead of forcing the default 15."""

    frontier = _six_block_frontier(minimum_point_count=15)
    assert frontier.minimum_point_count == 7
