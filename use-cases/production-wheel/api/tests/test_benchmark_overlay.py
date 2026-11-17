"""Tests for the isolated post-hoc benchmark overlay.

The overlay is a comparison aid only and must never influence optimization, so
these tests exercise the pure comparison logic directly on synthetic points.
"""

from __future__ import annotations

from production_wheel.benchmark_overlay import (
    BenchmarkPoint,
    FrontierPoint,
    build_benchmark_overlay,
)


def test_overlay_flags_domination_matching_and_gaps() -> None:
    """Domination, exact match, non-domination, and signed gaps are reported."""

    frontier = [
        FrontierPoint(1, 46.0, 0.0),
        FrontierPoint(2, 20.55, 25.35),
        FrontierPoint(3, 19.59, 30.40),
    ]
    benchmarks = [
        BenchmarkPoint("witness", 21.0, 33.0),
        BenchmarkPoint("tough", 10.0, 5.0),
        BenchmarkPoint("exact_match", 20.55, 25.35),
    ]
    rows = {row["label"]: row for row in build_benchmark_overlay(frontier, benchmarks)}

    # A (21, 33) witness is beaten on both axes by points 2 and 3.
    assert rows["witness"]["dominated_by_frontier"] is True
    assert set(rows["witness"]["dominating_point_indexes"].split("|")) == {"2", "3"}
    # Nearest by J_CH is point 3 (30.40 vs 33); positive gaps mean the frontier
    # point is leaner and cheaper than the witness.
    assert rows["witness"]["nearest_point_index"] == 3
    assert rows["witness"]["coverage_gap_days"] > 0
    assert rows["witness"]["j_ch_gap"] > 0
    # An aggressive target the frontier cannot reach is not dominated.
    assert rows["tough"]["dominated_by_frontier"] is False
    # A benchmark equal to a produced point matches within tolerance.
    assert rows["exact_match"]["matched_by_frontier"] is True


def test_overlay_handles_empty_frontier() -> None:
    """An empty frontier yields no domination and blank nearest fields."""

    rows = build_benchmark_overlay([], [BenchmarkPoint("w", 21.0, 33.0)])
    assert rows[0]["dominated_by_frontier"] is False
    assert rows[0]["matched_by_frontier"] is False
    assert rows[0]["nearest_point_index"] == ""
