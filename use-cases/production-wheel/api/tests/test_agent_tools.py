"""Tests for the agent's optimizer tool implementations.

These exercise the plain ``*_impl`` functions, which import only the optimizer and
the job layer (no LangChain/LLM stack), so they run without the model runtime
installed. The LangChain wrapping and the full agent graph are verified separately
(a live model call needs SAP AI Core credentials).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from production_wheel.candidates import CandidateMember

from app.agent.tools import (
    capabilities_impl,
    extract_inputs_impl,
    job_status_impl,
    launch_solve_impl,
    read_results_impl,
    wait_for_job_impl,
)
from app.jobs import InMemoryJobStore, JobRecord, JobStatus
from app.jobs import runner as runner_module

# Import the submodule object (not the re-exported function of the same name) so
# monkeypatching its module-level extract_snapshot / load_canonical_inputs works.
import importlib

tools_module = importlib.import_module("app.agent.tools.optimizer_tools")


def _member(fini_id: str, plant: str, sefi: str) -> CandidateMember:
    """Minimal synthetic member for block-inventory aggregation."""

    return CandidateMember(
        fini_id=fini_id, plant=plant, sefi=sefi, eligible_lines=frozenset({"1"}),
        demand_litres=1_000.0, pallet_litres=25.0, package_volume=0.25,
        fixed_pv="PV1", fixed_lot_litres=100.0,
    )


def test_capabilities_impl_returns_contract() -> None:
    """The capabilities tool surfaces the typed contract the agent must obey."""

    caps = capabilities_impl()
    assert "must_link" in caps["constraint_kinds"]
    assert caps["accepts_raw_code"] is False
    assert caps["supported_group_caps"] == [7, 8, 9]


def test_extract_inputs_impl_summarizes_blocks(tmp_path, monkeypatch) -> None:
    """extract_inputs aggregates members into a per-block inventory."""

    primary = tmp_path / "m.xlsx"
    primary.write_bytes(b"stub")
    fake_result = types.SimpleNamespace(
        run_directory=tmp_path / "run", manifest={"status": "ready"}
    )
    fake_inputs = types.SimpleNamespace(
        members=[
            _member("1", "P1", "S1"),
            _member("2", "P1", "S1"),
            _member("3", "P2", "S2"),
        ]
    )
    monkeypatch.setattr(tools_module, "extract_snapshot", lambda **kwargs: fake_result)
    monkeypatch.setattr(tools_module, "load_canonical_inputs", lambda run_directory: fake_inputs)

    result = extract_inputs_impl(str(primary), str(tmp_path / "out"))
    assert result["status"] == "ready"
    assert result["modeled_fini_count"] == 3
    assert result["block_count"] == 2
    assert {"plant": "P1", "sefi": "S1", "fini_count": 2} in result["blocks"]


def test_launch_and_status_impls(tmp_path, monkeypatch) -> None:
    """launch_solve registers a job and job_status polls it to completion."""

    monkeypatch.setattr(runner_module, "solve_argv", lambda *a, **k: "echo ok")
    store = InMemoryJobStore()
    launched = launch_solve_impl(
        store, str(tmp_path / "ws"), str(tmp_path), '{"config": {}}'
    )
    assert launched["status"] == JobStatus.RUNNING.value
    job_id = launched["job_id"]

    import time

    deadline = time.monotonic() + 15.0
    status = job_status_impl(store, job_id)
    while status["status"] == JobStatus.RUNNING.value and time.monotonic() < deadline:
        time.sleep(0.05)
        status = job_status_impl(store, job_id)
    assert status["status"] == JobStatus.DONE.value
    assert status["return_code"] == 0


def test_read_results_impl_parses_bundle(tmp_path) -> None:
    """read_results extracts per-point KPIs, applied constraints, and the report."""

    bundle = tmp_path / "solution"
    bundle.mkdir()
    (bundle / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "greenfield_frontier",
                "scenario_id": "agent_run",
                "elapsed_seconds": 12.5,
                "counts": {"pareto_points": 2},
                "pool_completeness": {"restricted": 1},
                "solve_request": {
                    "applied_constraints": [{"effect": "scope", "kept_fini_count": 5}]
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle / "frontier_summary.csv").write_text(
        "demand_weighted_mean_coverage_days,j_ch,group_count,singleton_group_count\n"
        "46.5,0.0,5,5\n"
        "18.0,3.2,3,1\n",
        encoding="utf-8",
    )
    (bundle / "solution_report.md").write_text("# Frontier\nTwo points.", encoding="utf-8")

    summary = read_results_impl(str(bundle))
    assert summary["artifact_type"] == "greenfield_frontier"
    assert summary["applied_constraints"][0]["effect"] == "scope"
    assert len(summary["frontier"]) == 2
    assert summary["frontier"][0]["coverage_days"] == 46.5
    assert summary["frontier"][1]["j_ch"] == 3.2
    assert summary["frontier"][1]["group_count"] == 3
    assert "Frontier" in summary["report_markdown"]
    assert summary["report_truncated"] is False


def test_extract_inputs_impl_autodiscovers_enrichment(tmp_path, monkeypatch) -> None:
    """An enrichment sibling is found and passed to extraction when enrichment_path is omitted."""

    primary = tmp_path / "PL01 wheel.xlsx"
    primary.write_bytes(b"x")
    enrichment = tmp_path / "Enrichment inputs.xlsx"
    enrichment.write_bytes(b"y")
    captured: dict[str, object] = {}

    def fake_extract(*, primary_path, enrichment_path, run_directory):
        captured["enrichment_path"] = enrichment_path
        return types.SimpleNamespace(run_directory=run_directory, manifest={"status": "ready"})

    monkeypatch.setattr(tools_module, "extract_snapshot", fake_extract)
    monkeypatch.setattr(tools_module, "load_canonical_inputs", lambda rd: types.SimpleNamespace(members=[]))

    result = extract_inputs_impl(str(primary), str(tmp_path / "out"))
    assert result["enrichment_path"] == str(enrichment)
    assert str(captured["enrichment_path"]) == str(enrichment)


def test_discover_enrichment_is_none_when_ambiguous(tmp_path) -> None:
    """Two enrichment candidates return None so extraction never guesses between them."""

    (tmp_path / "PL01.xlsx").write_bytes(b"x")
    (tmp_path / "enrichment one.xlsx").write_bytes(b"y")
    (tmp_path / "enrichment two.xlsx").write_bytes(b"z")
    assert tools_module._discover_enrichment(tmp_path / "PL01.xlsx") is None


def test_wait_for_job_impl_returns_when_done(tmp_path, monkeypatch) -> None:
    """wait_for_job blocks in one call and returns the terminal record."""

    monkeypatch.setattr(runner_module, "solve_argv", lambda *a, **k: "echo ok")
    store = InMemoryJobStore()
    launched = launch_solve_impl(store, str(tmp_path / "ws"), str(tmp_path), "{}")
    final = wait_for_job_impl(
        store, launched["job_id"], timeout_seconds=15, poll_interval_seconds=0.05
    )
    assert final["status"] == JobStatus.DONE.value
    assert final["return_code"] == 0


def test_wait_for_job_impl_times_out_while_running(tmp_path) -> None:
    """A job with no terminal signal returns status running once the timeout hits."""

    import os

    store = InMemoryJobStore()
    log_path = tmp_path / "solve.log"
    log_path.write_text("", encoding="utf-8")
    # A live PID with no rc sentinel stays RUNNING, so the wait must time out.
    store.insert(
        JobRecord(
            job_id="run", status=JobStatus.RUNNING, mode="m", run_dir="r",
            output_dir="o", request_json="{}", pid=os.getpid(), log_path=str(log_path),
        )
    )
    final = wait_for_job_impl(store, "run", timeout_seconds=0.1, poll_interval_seconds=0.02)
    assert final["status"] == JobStatus.RUNNING.value
