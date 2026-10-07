"""Focused command-line contract tests."""

from __future__ import annotations

import csv
import json

import pytest

from production_wheel.cli import main


def test_build_matrix_cli_writes_529_explicit_rows(tmp_path, capsys) -> None:
    """The matrix command emits the complete versioned CSV."""

    output = tmp_path / "matrix.csv"
    assert (
        main(
            [
                "build-matrix",
                "--matrix-version",
                "SYNTHETIC_VOLUME_MATRIX_V1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 529
    assert json.loads(capsys.readouterr().out)["rows"] == 529


def test_build_matrix_cli_requires_an_explicit_policy() -> None:
    """Matrix export never silently selects synthetic or empirical semantics."""

    with pytest.raises(SystemExit):
        main(["build-matrix"])


def test_capabilities_cli_is_machine_readable(capsys) -> None:
    """The future-agent boundary exposes enums without executable syntax."""

    assert main(["capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepts_raw_code"] is False
    assert payload["supported_group_caps"] == [7, 8, 9]
    assert len(payload["baseline_acceptance"]["guardrails"]) == 11
    assert {
        "matrix_exception_group_count",
        "matrix_exception_pair_count",
        "matrix_exception_distinct_volume_pair_count",
    } <= set(payload["baseline_acceptance"]["guardrails"])
    assert payload["greenfield_frontier"] == {
        "status": "VALID_PARETO_POINT",
        "axes": ["demand_weighted_mean_coverage_days", "j_ch"],
        "epsilon_axis": "j_ch",
        "baseline_role": "optional comparison only",
        "proof_scopes": ["exact", "restricted-library", "runtime-limited"],
    }


def test_cli_accepts_reviewed_constrained_scenarios() -> None:
    """The two CLI-only modes are parser choices but not suite additions."""

    with pytest.raises(FileNotFoundError):
        main(
            [
                "solve",
                "--run-directory",
                "/does/not/exist",
                "--scenario",
                "customer_families_baseline_constrained_coverage",
            ]
        )


def test_cli_accepts_reviewed_greenfield_frontier() -> None:
    """The primary frontier has an explicit parser path and point controls."""

    with pytest.raises(FileNotFoundError):
        main(
            [
                "solve",
                "--run-directory",
                "/does/not/exist",
                "--frontier",
                "customer_families_greenfield_pareto",
                "--frontier-points",
                "17",
                "--global-epsilon-exponent",
                "2",
                "--block-options-per-block",
                "5",
                "--block-workers",
                "2",
                "--large-block-candidate-threshold",
                "250000",
            ]
        )
