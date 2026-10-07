"""Durable job lifecycle and transaction behavior without external services."""

import pytest
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from app.workspace.worker import claim_next, recover_lost_workers
from datetime import datetime, timezone, timedelta


def job(repo, key="job", **patch):
    """Insert a minimal queued job with optional lifecycle overrides."""
    repo.insert(
        "runs",
        key,
        {
            "run_id": key,
            "revision": 1,
            "status": "queued",
            "stage": "queued",
            "created_at": "2026-01-01",
            "results_ready": False,
            **patch,
        },
    )


def test_only_one_worker_claims_a_queued_job():
    """Two supervisors cannot both own the same solve."""
    repo = MemoryRepository()
    job(repo)
    s = WorkspaceService(repo)
    assert claim_next(s, "a")["worker_id"] == "a"
    assert claim_next(s, "b") is None


def test_lost_worker_is_explicit_and_never_automatically_rerun():
    """An expired owner is marked lost; repeating expensive work needs new intent."""
    repo = MemoryRepository()
    past = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    job(repo, status="running", worker_id="dead", heartbeat_at=past)
    s = WorkspaceService(repo)
    recover_lost_workers(s)
    assert s.get_run("job")["status"] == "worker_lost"
    assert claim_next(s, "new") is None


def test_publication_rollback_does_not_expose_partial_results():
    """Failed metadata publication rolls back analytical rows and readiness together."""
    from types import SimpleNamespace

    class FailingRepo(MemoryRepository):
        """Fail the final state transition after writing relational tables."""

        def cas(self, *args, **kwargs):
            """Simulate unavailable metadata persistence."""
            if args[3].get("results_ready"):
                raise RuntimeError("database interrupted")
            return super().cas(*args, **kwargs)

    repo = FailingRepo()
    job(repo, status="running")
    s = WorkspaceService(repo)
    with pytest.raises(RuntimeError):
        s.publish_results(
            "job",
            SimpleNamespace(
                tables={"solutions": [{"point_index": 1}]},
                metadata={"integrity_valid": True, "complete": True},
                artifacts={},
            ),
        )
    assert repo.rows("job", "solutions") == []
    assert not s.get_run("job")["results_ready"]


def test_invalid_or_cancelled_bundle_cannot_publish():
    """Publication does not resurrect cancellation or turn broken evidence into ready data."""
    from types import SimpleNamespace

    for status in ("cancelled", "worker_lost"):
        repo = MemoryRepository()
        job(repo, status=status)
        s = WorkspaceService(repo)
        with pytest.raises(ValueError):
            s.publish_results(
                "job",
                SimpleNamespace(
                    metadata={"integrity_valid": True, "complete": True},
                    tables={"solutions": [{"point_index": 1}]},
                    artifacts={},
                ),
            )
        assert not s.get_run("job")["results_ready"]
    repo = MemoryRepository()
    job(repo, status="running")
    s = WorkspaceService(repo)
    with pytest.raises(ValueError):
        s.publish_results(
            "job",
            SimpleNamespace(
                metadata={"integrity_valid": False, "complete": False},
                tables={"solutions": []},
                artifacts={},
            ),
        )


def test_cancel_rechecks_terminal_state_when_completion_wins_race(monkeypatch):
    """A stale cancellation read cannot overwrite a concurrent completed result."""
    repo = MemoryRepository()
    job(repo)
    service = WorkspaceService(repo)
    original_get = service.get_run
    first_read = True

    def complete_after_read(run_id):
        """Complete the run after returning the first stale queued snapshot."""
        nonlocal first_read
        value = original_get(run_id)
        if first_read:
            first_read = False
            repo.cas(
                "runs",
                run_id,
                value["revision"],
                {"status": "completed", "results_ready": True},
            )
        return value

    monkeypatch.setattr(service, "get_run", complete_after_read)
    result = service.cancel_run("job")
    assert result["status"] == "completed"
    assert result["results_ready"] is True
    assert not result.get("cancel_requested")


def test_local_deadline_stops_child_while_database_is_unavailable(monkeypatch):
    """Resource enforcement still kills owned work when every HANA read fails."""
    from types import SimpleNamespace
    from app.workspace import worker

    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    run = {"run_id": "job", "budget": {"wall_time_seconds": 10}}
    clock = iter([0.0, 0.0, 20.0])
    reads = []

    def unavailable(run_id):
        """Simulate an outage lasting beyond the local job deadline."""
        reads.append(run_id)
        raise RuntimeError("HANA unavailable")

    def stop_child(child):
        """Record child termination without spawning a process or sending signals."""
        child.returncode = -15

    def unavailable_finish(*args):
        """Final state can fail to persist, but resource release must precede it."""
        assert process.returncode == -15
        raise RuntimeError("HANA unavailable during finalization")

    service = SimpleNamespace(get_run=unavailable, finish_run=unavailable_finish)
    monkeypatch.setattr(worker, "claim_next", lambda *_, **kwargs: run)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(worker, "_stop_child", stop_child)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="finalization"):
        worker.supervise_once(service, "owner")
    assert reads == ["job"]
    assert process.returncode == -15


def test_expired_or_replaced_owner_cannot_publish_or_advance_progress():
    """A previous worker is fenced from both active replacements and lost runs."""
    from types import SimpleNamespace

    bundle = SimpleNamespace(
        metadata={"integrity_valid": True, "complete": True},
        tables={"solutions": [{"point_index": 1}]},
        artifacts={},
    )
    for status, owner in [("running", "replacement"), ("worker_lost", "old")]:
        repo = MemoryRepository()
        job(repo, status=status, worker_id=owner)
        service = WorkspaceService(repo)
        with pytest.raises(ValueError):
            service.publish_results("job", bundle, worker_id="old")
        with pytest.raises(ValueError):
            service.patch_run("job", {"stage": "optimizing"}, worker_id="old")
        assert repo.rows("job", "solutions") == []
        assert service.get_run("job")["status"] == status


def test_finalization_does_not_overwrite_published_result():
    """A late supervisor failure cannot replace the completed transaction state."""
    repo = MemoryRepository()
    job(repo, status="completed", results_ready=True)
    service = WorkspaceService(repo)
    assert (
        service.finish_run("job", "failed", "wall_time_limit")["status"] == "completed"
    )


@pytest.mark.parametrize("prior_status", ["failed", "worker_lost"])
def test_failed_publication_retry_returns_to_retryable_state(monkeypatch, prior_status):
    """A failed retry rolls back its temporary publication claim and keeps checkpoint."""
    import json
    from app.workspace.worker import retry_publication

    repo = MemoryRepository()
    job(repo, status=prior_status, worker_id="old")
    payload = json.dumps(
        {
            "metadata": {"integrity_valid": True, "complete": True},
            "tables": {"solutions": [{"point_index": 1}]},
            "artifacts": {},
        }
    ).encode()
    repo.put_artifact("job", "publication-checkpoint.json", payload)
    service = WorkspaceService(repo)

    def fail_publication(run_id, bundle):
        """Fail after the retry has entered its transient persisting state."""
        assert service.get_run(run_id)["status"] == "persisting"
        raise RuntimeError("database write failed")

    monkeypatch.setattr(service, "publish_results", fail_publication)
    with pytest.raises(RuntimeError, match="database write failed"):
        retry_publication(service, "job")
    assert service.get_run("job")["status"] == prior_status
    assert service.get_run("job")["revision"] == 1
    assert repo.artifact("job", "publication-checkpoint.json") == payload


def test_completed_child_race_is_not_logged_as_database_failure(monkeypatch, caplog):
    """Expected lease fencing after publication produces no misleading outage error."""
    from types import SimpleNamespace
    from app.workspace import worker

    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    run = {"run_id": "job", "budget": {"wall_time_seconds": 10}}

    def get_run(run_id):
        """Return completion after the simulated publication/heartbeat race."""
        return {
            "status": "completed" if process.returncode == 0 else "running",
            "results_ready": process.returncode == 0,
            "stage": "optimizing",
        }

    def heartbeat_race(*args, **kwargs):
        """Publish and exit just before the supervisor tries to renew its lease."""
        process.returncode = 0
        raise ValueError("worker lease no longer owns active run")

    def unexpected_finish(*args):
        """Published work must never be reclassified as failed."""
        raise AssertionError("completed run was reclassified")

    service = SimpleNamespace(
        get_run=get_run, patch_run=heartbeat_race, finish_run=unexpected_finish
    )
    monkeypatch.setattr(worker, "claim_next", lambda *_, **kwargs: run)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(worker, "_stop_child", lambda _: None)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    assert worker.supervise_once(service, "owner") is True
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


def test_targeted_worker_claims_only_requested_run():
    """A diagnostic execution must not drain older unrelated queued work."""
    repo = MemoryRepository()
    job(repo, "old")
    job(repo, "requested")
    service = WorkspaceService(repo)
    claimed = claim_next(service, "diagnostic", run_id="requested")
    assert claimed["run_id"] == "requested"
    assert service.get_run("old")["status"] == "queued"
    assert claim_next(service, "second", run_id="requested") is None


@pytest.mark.parametrize("published", [False, True])
def test_supervisor_shutdown_stops_child_without_overwriting_results(monkeypatch, published):
    """API shutdown releases its process group and preserves committed result authority."""
    from types import SimpleNamespace
    from app.workspace import worker
    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    finished = []
    service = SimpleNamespace(
        get_run=lambda _: {"results_ready": published},
        finish_run=lambda *args: finished.append(args),
    )
    monkeypatch.setattr(worker, "claim_next", lambda *a, **k: {"run_id":"r", "budget":{"wall_time_seconds":60}})
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(worker, "_stop_child", lambda child: setattr(child,"returncode",-15))
    assert worker.supervise_once(service,"owner",stop=SimpleNamespace(is_set=lambda:True))
    assert process.returncode == -15
    assert len(finished) == (0 if published else 1)
    if not published:
        assert finished[0][1:3] == ("worker_lost","worker_stopped")
