"""Tests for the async optimizer job runner and job registry.

The detached-subprocess mechanism is exercised with trivial shell commands
(monkeypatching the solve command) so no real multi-minute solve is needed. The
HANA store's DDL/DML is exercised through a fake connection, so these tests need
no live database.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.jobs import (
    HanaJobStore,
    InMemoryJobStore,
    JobRecord,
    JobStatus,
    launch,
    poll,
    solve_argv,
)
from app.jobs import runner as runner_module


def _wait_terminal(store, job_id: str, timeout: float = 15.0) -> JobRecord:
    """Poll a job until it leaves RUNNING or the timeout elapses."""

    deadline = time.monotonic() + timeout
    record = poll(store, job_id)
    while record.status is JobStatus.RUNNING and time.monotonic() < deadline:
        time.sleep(0.05)
        record = poll(store, job_id)
    return record


def test_in_memory_store_crud() -> None:
    """Insert/get/update/list behave and duplicate inserts are rejected."""

    store = InMemoryJobStore()
    record = JobRecord(
        job_id="j1", status=JobStatus.RUNNING, mode="m", run_dir="r",
        output_dir="o", request_json="{}", submitted_at="2026-01-01T00:00:00+00:00",
    )
    store.insert(record)
    assert store.get("j1") == record
    with pytest.raises(ValueError):
        store.insert(record)
    store.update(
        JobRecord(job_id="j1", status=JobStatus.DONE, mode="m", run_dir="r", output_dir="o", request_json="{}")
    )
    assert store.get("j1").status is JobStatus.DONE
    assert [r.job_id for r in store.list()] == ["j1"]
    assert store.get("missing") is None


def test_solve_argv_includes_flags() -> None:
    """The built command carries the CLI subcommand and every knob."""

    command = solve_argv(
        "/run", "/req.json", "/out",
        {"--frontier-points": 5, "--per-block-total-seconds": 30},
    )
    assert "-m production_wheel.cli solve" in command
    assert "--run-directory /run" in command
    assert "--request /req.json" in command
    assert "--output-directory /out" in command
    assert "--frontier-points 5" in command
    assert "--per-block-total-seconds 30" in command


def test_launch_and_poll_success(tmp_path, monkeypatch) -> None:
    """A trivially-succeeding job launches, then polls to DONE with code 0."""

    monkeypatch.setattr(runner_module, "solve_argv", lambda *a, **k: "echo ok")
    store = InMemoryJobStore()
    job_id = launch(
        store, run_directory=str(tmp_path), request_json="{}",
        workspace=str(tmp_path / "jobs"),
    )
    assert store.get(job_id).status is JobStatus.RUNNING
    record = _wait_terminal(store, job_id)
    assert record.status is JobStatus.DONE
    assert record.return_code == 0
    assert "ok" in Path(record.log_path).read_text(encoding="utf-8")


def test_poll_marks_failed_on_nonzero_rc(tmp_path, monkeypatch) -> None:
    """A command exiting non-zero (and not 2) polls to FAILED with its code."""

    monkeypatch.setattr(runner_module, "solve_argv", lambda *a, **k: "false")
    store = InMemoryJobStore()
    job_id = launch(
        store, run_directory=str(tmp_path), request_json="{}",
        workspace=str(tmp_path / "jobs"),
    )
    record = _wait_terminal(store, job_id)
    assert record.status is JobStatus.FAILED
    assert record.return_code == 1


def test_poll_marks_failed_when_no_rc_and_dead(tmp_path) -> None:
    """A RUNNING record with a dead launcher and no sentinel becomes FAILED."""

    store = InMemoryJobStore()
    log_path = tmp_path / "solve.log"
    log_path.write_text("", encoding="utf-8")
    store.insert(
        JobRecord(
            job_id="dead", status=JobStatus.RUNNING, mode="m", run_dir="r",
            output_dir="o", request_json="{}", pid=None, log_path=str(log_path),
        )
    )
    record = poll(store, "dead")
    assert record.status is JobStatus.FAILED
    assert "return code" in (record.error or "")


class _FakeResult:
    """Minimal SQLAlchemy-style result over a fixed row list."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Records executed SQL and answers the table-existence probe."""

    def __init__(self, exists_count: int) -> None:
        self.executed: list[str] = []
        self._exists = exists_count

    def execute(self, statement, params=None):
        sql = str(statement)
        self.executed.append(sql)
        if "SYS.TABLES" in sql:
            return _FakeResult([(self._exists,)])
        return _FakeResult([])

    def commit(self) -> None:
        self.executed.append("COMMIT")


class _FakeHelper:
    """Stand-in for HANAConnection exposing a shared fake connection."""

    def __init__(self, conn: _FakeConn) -> None:
        self.connection = conn

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass


def test_hana_store_creates_table_when_absent() -> None:
    """ensure_table issues CREATE TABLE only when the catalog lacks the table."""

    conn = _FakeConn(exists_count=0)
    store = HanaJobStore(connection_factory=lambda: _FakeHelper(conn))
    store.ensure_table()
    assert any("CREATE TABLE PRODUCTION_WHEEL_JOBS" in sql for sql in conn.executed)


def test_hana_store_skips_create_when_present() -> None:
    """ensure_table issues no CREATE when the table already exists."""

    conn = _FakeConn(exists_count=1)
    store = HanaJobStore(connection_factory=lambda: _FakeHelper(conn))
    store.ensure_table()
    assert not any("CREATE TABLE" in sql for sql in conn.executed)


def test_hana_store_insert_emits_insert() -> None:
    """insert emits an INSERT against the prefixed jobs table."""

    conn = _FakeConn(exists_count=1)
    store = HanaJobStore(connection_factory=lambda: _FakeHelper(conn))
    store.insert(
        JobRecord(
            job_id="j1", status=JobStatus.RUNNING, mode="m", run_dir="r",
            output_dir="o", request_json="{}",
        )
    )
    assert any("INSERT INTO PRODUCTION_WHEEL_JOBS" in sql for sql in conn.executed)
