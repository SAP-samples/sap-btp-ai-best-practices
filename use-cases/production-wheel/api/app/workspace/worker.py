"""Durable optimizer supervisor and isolated deterministic solve process.

Examples (from api/):
    ../.venv/bin/python -m app.workspace worker
    ../.venv/bin/python -m app.workspace execute RUN_ID
    ../.venv/bin/python -m app.workspace retry-publication RUN_ID

One supervisor runs one job at a time. HANA owns claims, heartbeats, cancellation
and completion; process groups are used only by the supervisor which created them.
"""

from __future__ import annotations
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .datasets_service import now

logger = logging.getLogger(__name__)
ACTIVE = ("running", "persisting")


def claim_next(service, worker_id, run_id=None):
    """Claim the specified queued run, or the oldest queued job; return owned metadata or None."""
    candidates = ([service.get_run(run_id)] if run_id else reversed(service.list_runs({"status": "queued"})))
    for run in candidates:
        if run["status"] != "queued":
            continue
        try:
            return service.repo.cas(
                "runs",
                run["run_id"],
                run["revision"],
                {
                    "status": "running",
                    "stage": "loading_snapshot",
                    "worker_id": worker_id,
                    "heartbeat_at": now(),
                    "started_at": now(),
                },
            )
        except ValueError:
            continue
    return None


def recover_lost_workers(service, expiry_seconds=120):
    """Mark expired leases explicitly; never silently repeat completed solver work."""
    for run in service.list_runs():
        if run["status"] not in ACTIVE or not run.get("heartbeat_at"):
            continue
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(run["heartbeat_at"])
        ).total_seconds()
        if age > expiry_seconds:
            try:
                service.repo.cas(
                    "runs",
                    run["run_id"],
                    run["revision"],
                    {
                        "status": "worker_lost",
                        "stage": "worker_lost",
                        "error": "Worker heartbeat expired; inspect or retry saved publication.",
                        "finished_at": now(),
                    },
                )
            except ValueError:
                pass


def retry_publication(service, run_id):
    """Publish a durable checkpoint without launching the solver again."""
    from production_wheel.result_bundle import ResultBundle

    run = service.get_run(run_id)
    if run.get("results_ready"):
        return run
    if run["status"] not in ("failed", "worker_lost"):
        raise ValueError("publication retry requires failed or lost worker state")
    payload = json.loads(service.repo.artifact(run_id, "publication-checkpoint.json"))
    # Keep the retry claim and publication together: any write failure restores
    # the prior failed/lost state and leaves the checkpoint immediately retryable.
    with service.repo.transaction():
        service.repo.cas(
            "runs",
            run_id,
            run["revision"],
            {
                "status": "persisting",
                "stage": "retrying_publication",
                "worker_id": None,
                "cancel_requested": False,
            },
        )
        return service.publish_results(run_id, ResultBundle(**payload))


def execute_run(service, run_id):
    """Load inputs from HANA, solve in memory, checkpoint, then publish results."""
    from production_wheel.scenarios import run_greenfield_frontier
    from production_wheel.result_bundle import build_result_bundle
    from tqdm import tqdm

    run = service.get_run(run_id)
    if run["status"] != "running":
        raise ValueError("job must be claimed before execution")
    compiled = service.compiled_draft(run)
    from production_wheel.scenarios import canonical_inputs_from_tables

    original_inputs = canonical_inputs_from_tables(
        service.input_tables(run["dataset_id"], ("fini_master", "production_versions"))
    )
    owner = run["worker_id"]
    budget = run["budget"]
    bar = tqdm(total=budget["frontier_points"], desc="Optimize production wheel")
    last_progress = 0.0

    def progress(stage, current, total):
        """Persist stage progress at most once per second and update CLI progress."""
        nonlocal last_progress
        bar.total = total
        bar.n = current
        bar.set_description(stage)
        bar.refresh()
        if time.monotonic() - last_progress >= 1:
            service.patch_run(
                run_id,
                {"stage": stage, "progress": {"current": current, "total": total}},
                worker_id=owner,
            )
            last_progress = time.monotonic()

    try:
        service.patch_run(run_id, {"stage": "optimizing"}, worker_id=owner)
        result = run_greenfield_frontier(
            compiled.inputs,
            compiled.config,
            point_count=budget["frontier_points"],
            block_option_count=budget["block_options_per_block"],
            per_block_total_seconds=budget["per_block_total_seconds"],
            block_worker_count=budget["block_workers"],
            global_epsilon_exponent=budget["global_epsilon_exponent"],
            pool_transform=compiled.filter_pools,
            progress=progress,
        )
        bundle = build_result_bundle(
            result,
            original_inputs,
            {
                "source_extraction": service.inspect_dataset(run["dataset_id"])[
                    "metadata"
                ],
                "dataset_id": run["dataset_id"],
                "plant_profile": run.get("plant_profile"),
                "solve_request": {
                    "request_id": digest_request(run),
                    "scope": run["request"].get("scope", []),
                    "applied_constraints": list(compiled.applied),
                    "source_request": run.get("source_request", run["request"]),
                },
            },
        )
        service.patch_run(
            run_id,
            {
                "stage": "persisting",
                "status": "persisting",
                "solver_finished_at": now(),
            },
            worker_id=owner,
        )
        # A checkpoint is an opaque restart artifact. Queries read only the
        # normalized tables after publication. Failed publication is retryable.
        service.repo.put_artifact(
            run_id,
            "publication-checkpoint.json",
            json.dumps(asdict(bundle), allow_nan=False).encode(),
        )
        service.publish_results(run_id, bundle, worker_id=owner)
    finally:
        bar.close()


def digest_request(run):
    """Return the optimizer's deterministic typed request fingerprint."""
    from production_wheel.schemas import SolveRequest

    return SolveRequest.model_validate(run["request"]).request_id()


def _stop_child(process):
    """Terminate only this supervisor's child group and wait for resource release."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise_once(service, worker_id, run_id=None, stop=None):
    """Supervise an optional target run ID, renewing its lease until finish, cancel or stop."""
    run = claim_next(service, worker_id, run_id=run_id)
    if run is None:
        return False
    run_id = run["run_id"]
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "app.workspace", "execute", run_id],
        start_new_session=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        while process.poll() is None:
            if stop is not None and stop.is_set():
                _stop_child(process)
                current = service.get_run(run_id)
                if not current.get("results_ready"):
                    service.finish_run(run_id, "worker_lost", "worker_stopped", "Worker stopped before the run completed.")
                return True
            # Enforce local resource limits even while HANA cannot be reached.
            if time.monotonic() - started > run["budget"]["wall_time_seconds"]:
                _stop_child(process)
                service.finish_run(
                    run_id,
                    "failed",
                    "wall_time_limit",
                    "Total job wall time limit reached.",
                )
                return True
            try:
                current = service.get_run(run_id)
                if current["status"] == "worker_lost":
                    _stop_child(process)
                    return True
                if current.get("cancel_requested") and not current.get("results_ready"):
                    _stop_child(process)
                    service.finish_run(run_id, "cancelled", "cancelled")
                    return True
                if current["status"] in ACTIVE:
                    try:
                        service.patch_run(
                            run_id, {"heartbeat_at": now()}, worker_id=worker_id
                        )
                    except ValueError:
                        # Publication can commit between the state read and CAS.
                        # An expected ownership fence is not a database outage.
                        logger.debug(
                            "Worker lease changed while refreshing progress; awaiting child state"
                        )
            except Exception:
                logger.exception(
                    "Heartbeat unavailable; retaining owned child while HANA recovers"
                )
            time.sleep(2)
        current = service.get_run(run_id)
        if not current.get("results_ready") and current["status"] not in (
            "cancelled",
            "failed",
        ):
            service.finish_run(
                run_id,
                "failed",
                "publication_failed"
                if current["stage"] == "persisting"
                else "solver_failed",
                f"Solve process exited with code {process.returncode}; inspect worker log.",
            )
        return True
    finally:
        _stop_child(process)


def worker_loop(service, stop=None):
    """Poll the durable queue; at most one solve is owned by this worker."""
    worker_id = uuid.uuid4().hex
    while stop is None or not stop.is_set():
        try:
            recover_lost_workers(service)
            if supervise_once(service, worker_id, stop=stop):
                continue
        except Exception:
            logger.exception("Worker iteration failed; retrying durable queue")
        if stop is None:
            time.sleep(3)
        else:
            stop.wait(3)
