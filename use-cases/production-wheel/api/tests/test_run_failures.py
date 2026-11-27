"""Failed solves leave durable, bounded evidence for the UI and the Workspace Assistant.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_run_failures.py -q
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.workspace.repository import MemoryRepository
from app.workspace.run_failures import (
    FAILURE_REPORT_ARTIFACT,
    LOG_TAIL_ARTIFACT,
    LogTail,
    build_failure_report,
    describe_exit,
    public_failure_message,
    read_failure_report,
    record_failure,
)
from app.workspace.service import WorkspaceService
from production_wheel.matrix import ProfileMatrixCoverageError, profile_matrix_coverage


def coverage_error():
    """Raise and catch a matrix coverage error so it carries a real traceback."""
    config = SimpleNamespace(matrix_pairs=[SimpleNamespace(volume_a="1", volume_b="1")])
    members = [SimpleNamespace(package_volume=1.0), SimpleNamespace(package_volume=0.465),
               SimpleNamespace(package_volume=0.465)]
    coverage = profile_matrix_coverage(config, members)
    try:
        raise ProfileMatrixCoverageError(coverage)
    except ProfileMatrixCoverageError as exc:
        return exc


def test_matrix_coverage_counts_missing_volumes_and_explains_them():
    """Coverage lists each missing volume with its product count and a planner action."""
    exc = coverage_error()
    assert exc.coverage["missing_volumes"] == [{"volume": "0.465", "member_count": 2}]
    assert exc.coverage["affected_member_count"] == 2 and not exc.coverage["complete"]
    assert "0.465 L (2 products)" in str(exc) and "Settings" in str(exc)
    assert profile_matrix_coverage(SimpleNamespace(matrix_pairs=()), []) is None


def test_failure_report_classifies_matrix_gap_and_keeps_relative_trace():
    """The report names the failure type, details and API-relative traceback frames."""
    report = build_failure_report(coverage_error(), "optimizing")
    assert report["failure_type"] == "profile_matrix_incomplete"
    assert report["details"]["matrix_coverage"]["missing_volumes"][0]["volume"] == "0.465"
    assert report["stage"] == "optimizing" and "1 package volumes" in report["summary"]
    assert report["traceback"] and not report["traceback"][-1]["file"].startswith("/")
    other = build_failure_report(RuntimeError("boom"), None)
    assert other["failure_type"] == "unhandled_exception" and other["summary"] == "RuntimeError: boom"


def test_record_failure_round_trips_and_never_raises():
    """A saved report is readable; a storage outage is logged, not raised."""
    service = WorkspaceService(MemoryRepository())
    service.repo.insert("runs", "r", {"run_id": "r", "revision": 1, "status": "running", "stage": "optimizing"})
    saved = record_failure(service, "r", coverage_error())
    assert read_failure_report(service, "r") == json.loads(json.dumps(saved))
    assert read_failure_report(service, "missing") is None
    broken = SimpleNamespace(get_run=lambda _: {}, repo=SimpleNamespace(
        put_artifact=lambda *a: (_ for _ in ()).throw(RuntimeError("HANA down"))))
    assert record_failure(broken, "r", RuntimeError("x")) is None


def test_exit_descriptions_and_public_message():
    """Exit codes and signals read as plain words; a report summary wins."""
    assert describe_exit(1) == "exited with code 1"
    assert "SIGKILL" in describe_exit(-9) and "memory" in describe_exit(-9)
    assert public_failure_message({"summary": "Missing volumes."}, 1).startswith("Solve failed: Missing volumes.")
    assert "exited with code 1" in public_failure_message(None, 1)


def test_log_tail_forwards_output_and_keeps_last_lines():
    """Progress refreshes and tracebacks stay live in the terminal and in the tail."""
    read_fd, write_fd = os.pipe()
    sink = SimpleNamespace(data=bytearray())
    sink.write = sink.data.extend
    sink.flush = lambda: None
    tail = LogTail(os.fdopen(read_fd, "rb"), sink=sink, limit=1000)
    os.write(write_fd, b"blocks 0%\rblocks 50%\rblocks 100%\nTraceback\nValueError: bad\n")
    os.close(write_fd)
    text = tail.text(max_lines=3)
    assert text.splitlines() == ["blocks 100%", "Traceback", "ValueError: bad"]
    assert bytes(sink.data).endswith(b"ValueError: bad\n")
    assert LogTail(None).text() == ""


@pytest.mark.parametrize("with_report", [False, True])
def test_supervisor_saves_log_tail_and_meaningful_error(monkeypatch, with_report):
    """A failing child leaves its output in HANA and a readable error on the run."""
    from app.workspace import worker

    service = WorkspaceService(MemoryRepository())
    service.repo.insert("runs", "r", {"run_id": "r", "revision": 1, "status": "running",
        "stage": "optimizing", "results_ready": False, "worker_id": "w",
        "budget": {"wall_time_seconds": 60}})
    if with_report:
        record_failure(service, "r", coverage_error())
    real_popen = subprocess.Popen
    script = "import sys; sys.stderr.write('progress\\rTraceback\\nValueError: bad\\n'); sys.exit(1)"
    monkeypatch.setattr(worker, "claim_next", lambda *a, **k: service.get_run("r"))
    monkeypatch.setattr(worker.subprocess, "Popen",
        lambda *a, **k: real_popen([sys.executable, "-c", script], stderr=subprocess.PIPE, start_new_session=True))
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "stderr", SimpleNamespace(buffer=SimpleNamespace(write=lambda _: None, flush=lambda: None)))
    assert worker.supervise_once(service, "w")
    run = service.get_run("r")
    assert run["status"] == "failed" and run["stage"] == "solver_failed"
    log = service.repo.artifact("r", LOG_TAIL_ARTIFACT).decode()
    assert log.startswith("Solve process exited with code 1.") and "ValueError: bad" in log
    if with_report:
        assert run["error"].startswith("Solve failed: 1 package volumes")
    else:
        assert "exited with code 1" in run["error"] and "inspect worker log" not in run["error"]


def test_failure_diagnostics_expose_report_and_log_tail():
    """Diagnostics merge the solve report, its failure type and the bounded log tail."""
    service = WorkspaceService(MemoryRepository())
    service.repo.insert("runs", "r", {"run_id": "r", "revision": 1, "status": "failed",
        "stage": "solver_failed", "error": "Solve failed"})
    record_failure(service, "r", coverage_error())
    service.repo.put_artifact("r", LOG_TAIL_ARTIFACT, ("\n".join(f"line {i}" for i in range(500))).encode())
    diagnostics = service.run_failure_diagnostics("r")
    assert diagnostics["worker_report_available"] and diagnostics["worker_log_available"]
    assert diagnostics["failure_types"] == ["profile_matrix_incomplete"]
    failure = diagnostics["worker_failure"]
    assert failure["exception_type"] == "ProfileMatrixCoverageError"
    assert failure["matrix_coverage"]["missing_volumes"] == [{"volume": "0.465", "member_count": 2}]
    tail = diagnostics["worker_log_tail"].splitlines()
    assert len(tail) == 60 and tail[-1] == "line 499"
    service.repo.insert("runs", "old", {"run_id": "old", "revision": 1, "status": "failed"})
    old = service.run_failure_diagnostics("old")
    assert old["worker_failure"] is None and old["worker_log_tail"] is None
    assert FAILURE_REPORT_ARTIFACT == "failure-report.json"
