"""Focused artifact tests for the baseline-independent Pareto workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.greenfield_reporting import (
    validate_greenfield_frontier_artifacts,
    write_greenfield_frontier_artifacts,
)
from production_wheel.pareto import GreenfieldBlockFailure, build_greenfield_frontier
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig


def test_greenfield_bundle_validates_without_historical_groups(tmp_path) -> None:
    """Blank baseline fields produce valid points and no baseline delta columns."""

    members = tuple(
        CandidateMember(
            fini_id=f"F{index}",
            plant="P1",
            sefi="S1",
            eligible_lines=frozenset({"1"}),
            demand_litres=1_000.0,
            pallet_litres=25.0,
            package_volume=0.25,
            fixed_pv="PV1",
            baseline_group=None,
        )
        for index in (1, 2)
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 100.0),)
    config = RunConfig(
        scenario_id="greenfield-artifact-test",
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(members, versions, config)
    frontier = build_greenfield_frontier(pools, members, config)
    fini_rows = tuple(
        {
            "source_row": index + 1,
            "plant": "P1",
            "sefi": "S1",
            "material": f"F{index}",
            "model_status": "modeled",
            "baseline_group_key": "",
            "fixed_pv": "PV1",
        }
        for index in (1, 2)
    )

    output = tmp_path / "frontier"
    write_greenfield_frontier_artifacts(
        output,
        fini_rows,
        versions,
        frontier,
    )
    verification = validate_greenfield_frontier_artifacts(output)

    assert verification["valid"]
    assert not verification["business_acceptable"]
    assert not verification["business_acceptance_assessed"]
    with (output / "frontier_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["acceptance_status"] == "VALID_PARETO_POINT" for row in rows)
    assert not any(key.startswith("delta_") for key in rows[0])
    assert "baseline_group_count" not in rows[0]
    with (output / "block_solve_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        solve_rows = list(csv.DictReader(handle))
    with (output / "block_session_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        session_rows = list(csv.DictReader(handle))
    with (output / "global_solve_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        global_rows = list(csv.DictReader(handle))
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert solve_rows
    assert {row["request_kind"] for row in solve_rows} >= {
        "operations_anchor",
        "coverage_anchor",
    }
    assert session_rows[0]["model_build_count"] == "1"
    assert manifest["config"]["versions"]["schema_version"] == "PROTOTYPE_SCHEMA_V9"
    assert manifest["counts"]["structural_model_builds"] == 1
    assert manifest["counts"]["block_solver_calls"] < 10
    assert len(global_rows) == 17
    assert manifest["global_epsilon_schedule"] == {
        "kind": "power_law_low_epsilon_dense",
        "exponent": 2.0,
        "requested_points": 17,
        "retained_points": len(frontier.points),
    }
    global_steps = [
        float(right["requested_epsilon_j_ch"])
        - float(left["requested_epsilon_j_ch"])
        for left, right in zip(global_rows, global_rows[1:])
    ]
    assert global_steps[0] < global_steps[-1]

    labeled_members = tuple(
        replace(member, baseline_group=f"B{index}")
        for index, member in enumerate(members, start=1)
    )
    comparable = replace(frontier, members=labeled_members)
    comparable_output = tmp_path / "frontier-with-baseline"
    write_greenfield_frontier_artifacts(
        comparable_output,
        fini_rows,
        versions,
        comparable,
    )
    with (comparable_output / "frontier_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        comparable_rows = list(csv.DictReader(handle))
    assert comparable_rows[0]["baseline_group_count"] == "2"
    assert "delta_demand_weighted_mean_coverage_days" in comparable_rows[0]


def test_greenfield_bundle_retains_and_rejects_partial_worker_failure(tmp_path) -> None:
    """Successful audit evidence is written, but a failed block invalidates the bundle."""

    members = tuple(
        CandidateMember(
            fini_id=f"F{index}",
            plant="P1",
            sefi="S1",
            eligible_lines=frozenset({"1"}),
            demand_litres=1_000.0,
            pallet_litres=25.0,
            package_volume=0.25,
            fixed_pv="PV1",
        )
        for index in (1, 2)
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 100.0),)
    config = RunConfig(
        scenario_id="greenfield-partial-test",
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(members, versions, config)
    complete = build_greenfield_frontier(
        pools, members, config, block_worker_count=1
    )
    partial = replace(
        complete,
        points=(),
        block_failures=(
            GreenfieldBlockFailure(
                block_key=("P2", "S1"),
                candidate_count=10,
                error_type="RuntimeError",
                error_message="fixture worker failure",
            ),
        ),
    )
    output = tmp_path / "partial-frontier"

    write_greenfield_frontier_artifacts(output, (), versions, partial)
    verification = validate_greenfield_frontier_artifacts(output)

    assert not verification["valid"]
    assert "one or more block frontier workers failed" in verification["errors"]
    with (output / "block_failures.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        failure_rows = list(csv.DictReader(handle))
    assert failure_rows[0]["plant"] == "P2"
    assert failure_rows[0]["error_type"] == "RuntimeError"
