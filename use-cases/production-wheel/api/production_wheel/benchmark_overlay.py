"""Post-hoc comparison of a produced greenfield frontier against benchmarks.

This module is deliberately isolated from the optimizer. External benchmark
policies (a historical wheel, a previously delivered witness, etc.) are only
ever compared against an *already produced* frontier here. They are never read
by candidate generation, solving, epsilon selection, warm starts, or
acceptance. In production there are no benchmarks at all and this module is
simply not invoked; it exists to let us confirm, during testing, that the
baseline-free optimizer discovers policies at least as good as known-good ones.

The comparison is on the two reported axes, both minimized:
- ``demand_weighted_mean_coverage_days`` (lower is leaner inventory), and
- ``j_ch`` (lower is fewer/larger-group changeovers).

A produced frontier point *dominates* a benchmark when it is at least as good
on both axes; it *matches* when both axes are within tolerance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class BenchmarkPoint:
    """One external reference policy on the (coverage, J_CH) plane.

    Args:
        label: Human-readable identifier for the benchmark policy.
        demand_weighted_mean_coverage_days: Reference inventory-coverage axis.
        j_ch: Reference changeover-proxy axis.
    """

    label: str
    demand_weighted_mean_coverage_days: float
    j_ch: float


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One produced frontier point reduced to its two comparable axes."""

    point_index: int
    demand_weighted_mean_coverage_days: float
    j_ch: float


def build_benchmark_overlay(
    frontier: Sequence[FrontierPoint],
    benchmarks: Sequence[BenchmarkPoint],
    tolerance: float = _TOLERANCE,
) -> tuple[dict[str, Any], ...]:
    """Flag, per external benchmark, whether the frontier matches or dominates it.

    Args:
        frontier: Produced frontier points (coverage, J_CH), typically loaded
            from a finished bundle's ``frontier_summary.csv``.
        benchmarks: External reference policies to compare against.
        tolerance: Absolute tolerance applied on both axes.

    Returns:
        One row per benchmark (sorted by label) recording whether any produced
        point dominates or matches it, the dominating point indexes, and the
        nearest produced point by J_CH with its signed gaps. This function reads
        only the supplied numbers and changes no optimization decision.
    """

    rows: list[dict[str, Any]] = []
    for bench in sorted(benchmarks, key=lambda item: item.label):
        dominators = [
            point.point_index
            for point in frontier
            if point.demand_weighted_mean_coverage_days
            <= bench.demand_weighted_mean_coverage_days + tolerance
            and point.j_ch <= bench.j_ch + tolerance
        ]
        matches = [
            point.point_index
            for point in frontier
            if abs(
                point.demand_weighted_mean_coverage_days
                - bench.demand_weighted_mean_coverage_days
            )
            <= tolerance
            and abs(point.j_ch - bench.j_ch) <= tolerance
        ]
        nearest = (
            min(frontier, key=lambda point: abs(point.j_ch - bench.j_ch))
            if frontier
            else None
        )
        rows.append(
            {
                "label": bench.label,
                "benchmark_coverage_days": bench.demand_weighted_mean_coverage_days,
                "benchmark_j_ch": bench.j_ch,
                "dominated_by_frontier": bool(dominators),
                "matched_by_frontier": bool(matches),
                "dominating_point_indexes": "|".join(
                    str(index) for index in dominators
                ),
                "nearest_point_index": nearest.point_index if nearest else "",
                "nearest_coverage_days": (
                    nearest.demand_weighted_mean_coverage_days if nearest else ""
                ),
                "nearest_j_ch": nearest.j_ch if nearest else "",
                # Positive gaps mean the produced point is better than the
                # benchmark on that axis.
                "coverage_gap_days": (
                    bench.demand_weighted_mean_coverage_days
                    - nearest.demand_weighted_mean_coverage_days
                    if nearest
                    else ""
                ),
                "j_ch_gap": (bench.j_ch - nearest.j_ch if nearest else ""),
            }
        )
    return tuple(rows)


def load_frontier_points(frontier_directory: Path) -> tuple[FrontierPoint, ...]:
    """Read produced frontier points from a finished frontier bundle.

    Args:
        frontier_directory: Directory holding a written ``frontier_summary.csv``.

    Returns:
        Frontier points in file order.

    Raises:
        FileNotFoundError: The bundle has no ``frontier_summary.csv``.
    """

    path = frontier_directory / "frontier_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            FrontierPoint(
                point_index=int(row["point_index"]),
                demand_weighted_mean_coverage_days=float(
                    row["demand_weighted_mean_coverage_days"]
                ),
                j_ch=float(row["j_ch"]),
            )
            for row in csv.DictReader(handle)
        )


def write_benchmark_overlay(
    frontier_directory: Path,
    benchmarks: Sequence[BenchmarkPoint],
) -> Path:
    """Compare a finished frontier bundle to benchmarks and write a sidecar CSV.

    The output ``benchmark_overlay.csv`` is a post-hoc sidecar; it is not part of
    the frontier bundle's hashed manifest and is never consumed by the optimizer.

    Args:
        frontier_directory: Directory holding a finished frontier bundle.
        benchmarks: External reference policies.

    Returns:
        Path to the written ``benchmark_overlay.csv``.
    """

    frontier = load_frontier_points(frontier_directory)
    rows = build_benchmark_overlay(frontier, benchmarks)
    output = frontier_directory / "benchmark_overlay.csv"
    fieldnames = list(rows[0].keys()) if rows else ["label"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output
