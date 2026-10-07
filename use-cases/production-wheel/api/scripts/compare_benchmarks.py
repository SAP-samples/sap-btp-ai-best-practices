"""Compare a produced greenfield frontier against external benchmark policies.

This is a post-hoc testing aid only. It reads an already-produced frontier
bundle and a list of external reference policies and reports whether the
frontier matches or dominates each. It never runs or influences the optimizer,
and in production (where no benchmarks exist) it is simply not used.

Example:

    .venv/bin/python api/scripts/compare_benchmarks.py \\
        --frontier-directory prototype/output/demo/solutions/greenfield-quick \\
        --benchmark-point historical:20.0:30.0 \\
        --benchmark-point coverage_witness:19.5:31.0 \\
        --benchmark-point operations_witness:20.5:29.0

Writes ``<frontier-directory>/benchmark_overlay.csv`` and prints a JSON summary.
Each ``--benchmark-point`` is ``LABEL:COVERAGE_DAYS:J_CH`` and may be repeated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from production_wheel.benchmark_overlay import (
    BenchmarkPoint,
    build_benchmark_overlay,
    load_frontier_points,
    write_benchmark_overlay,
)


def _parse_benchmark(value: str) -> BenchmarkPoint:
    """Parse one ``LABEL:COVERAGE_DAYS:J_CH`` command-line benchmark point."""

    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected LABEL:COVERAGE_DAYS:J_CH, got {value!r}"
        )
    label, coverage, j_ch = parts
    try:
        return BenchmarkPoint(
            label=label,
            demand_weighted_mean_coverage_days=float(coverage),
            j_ch=float(j_ch),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Read a frontier bundle, write the overlay CSV, and print a JSON summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-directory", required=True, type=Path)
    parser.add_argument(
        "--benchmark-point",
        required=True,
        action="append",
        dest="benchmarks",
        type=_parse_benchmark,
        metavar="LABEL:COVERAGE_DAYS:J_CH",
        help="external reference policy; repeatable",
    )
    args = parser.parse_args(argv)

    frontier = load_frontier_points(args.frontier_directory)
    rows = build_benchmark_overlay(frontier, args.benchmarks)
    output = write_benchmark_overlay(args.frontier_directory, args.benchmarks)
    print(
        json.dumps(
            {
                "frontier_directory": str(args.frontier_directory),
                "frontier_points": len(frontier),
                "overlay_csv": str(output),
                "benchmarks": list(rows),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
