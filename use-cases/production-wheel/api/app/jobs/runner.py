"""Launch optimizer solves as detached subprocesses and poll their status.

A solve is CPU-bound, forks worker processes internally, and can run for minutes,
so it must never execute on the event loop or inside the agent's process. Each job
is a detached child of the existing CLI::

    python -m production_wheel.cli solve --request ... --run-directory ... --output-directory ...

wrapped so the shell records the exit code to a sentinel file when the solve ends::

    <command> > solve.log 2>&1; echo $? > solve.rc

That sentinel decouples completion from the launching process: :func:`poll`
reconstructs the terminal state from the filesystem, so even a restarted agent can
finalize a job it did not itself start.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .store import JobRecord, JobStatus, JobStore

# api/ directory: `python -m production_wheel.cli` must run with this as cwd so the
# top-level production_wheel package resolves without an install step.
_API_DIR = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _process_alive(pid: int | None) -> bool:
    """Return whether a PID is still live (best-effort, POSIX)."""

    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def solve_argv(
    run_directory: str,
    request_path: str,
    output_directory: str,
    knobs: dict[str, object] | None = None,
) -> str:
    """Build the shell-quoted CLI command that runs one constrained solve.

    Args:
        run_directory: Extraction run the solve reads.
        request_path: Path to the SolveRequest JSON on disk.
        output_directory: Fresh directory the solve writes its bundle to.
        knobs: Optional CLI flag -> value pairs, e.g. ``{"--frontier-points": 17}``.

    Returns:
        A single shell-safe command string.
    """

    parts = [
        shlex.quote(sys.executable),
        "-m",
        "production_wheel.cli",
        "solve",
        "--run-directory",
        shlex.quote(str(run_directory)),
        "--request",
        shlex.quote(str(request_path)),
        "--output-directory",
        shlex.quote(str(output_directory)),
    ]
    for flag, value in (knobs or {}).items():
        parts.extend([str(flag), shlex.quote(str(value))])
    return " ".join(parts)


def _spawn_detached(command: str, log_path: Path, rc_path: Path, cwd: Path) -> int:
    """Run ``command`` detached, teeing output to a log and its exit code to ``rc_path``.

    Returns the launcher shell PID. The shell writes the command's exit code to
    ``rc_path`` after it finishes, which :func:`poll` uses as the completion signal.
    """

    wrapped = (
        f"{command} > {shlex.quote(str(log_path))} 2>&1; "
        f"echo $? > {shlex.quote(str(rc_path))}"
    )
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell injection surface
        ["/bin/sh", "-c", wrapped],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def launch(
    store: JobStore,
    *,
    run_directory: str,
    request_json: str,
    workspace: str,
    mode: str = "greenfield_pareto",
    knobs: dict[str, object] | None = None,
) -> str:
    """Launch a constrained solve as a detached job and register it.

    Args:
        store: The :class:`JobStore` to record the job in.
        run_directory: Extraction run directory the solve reads.
        request_json: Serialized SolveRequest JSON string to solve.
        workspace: Directory under which this job's artifacts are created.
        mode: Short solve descriptor stored on the record.
        knobs: Optional CLI flag -> value pairs forwarded to the solver.

    Returns:
        The new job id. The solve bundle appears under
        ``<workspace>/<job_id>/solution`` once the job reaches DONE.
    """

    job_id = uuid.uuid4().hex
    job_dir = Path(workspace) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_path = job_dir / "request.json"
    request_path.write_text(request_json, encoding="utf-8")
    solution_dir = job_dir / "solution"  # the CLI creates this; it must not pre-exist
    log_path = job_dir / "solve.log"
    rc_path = job_dir / "solve.rc"

    command = solve_argv(
        str(Path(run_directory).resolve()), str(request_path), str(solution_dir), knobs
    )
    pid = _spawn_detached(command, log_path, rc_path, cwd=_API_DIR)

    now = _utc_now()
    store.insert(
        JobRecord(
            job_id=job_id,
            status=JobStatus.RUNNING,
            mode=mode,
            run_dir=str(run_directory),
            output_dir=str(solution_dir),
            request_json=request_json,
            pid=pid,
            submitted_at=now,
            started_at=now,
            log_path=str(log_path),
        )
    )
    return job_id


def poll(store: JobStore, job_id: str) -> JobRecord:
    """Refresh and return a job's status from its return-code sentinel.

    A job is DONE once the sentinel exists (exit code 0 = valid, 2 =
    invalid/partial are both terminal solver outcomes); any other code, or a dead
    launcher with no sentinel, is FAILED. A live launcher with no sentinel stays
    RUNNING.
    """

    record = store.get(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status is not JobStatus.RUNNING:
        return record

    rc_path = Path(record.log_path).with_name("solve.rc") if record.log_path else None
    if rc_path is not None and rc_path.is_file():
        raw = rc_path.read_text(encoding="utf-8").strip()
        return_code = int(raw) if raw.lstrip("-").isdigit() else None
        if return_code in (0, 2):
            status, error = JobStatus.DONE, None
        else:
            status, error = JobStatus.FAILED, f"solve exited with code {return_code}"
        updated = replace(
            record,
            status=status,
            return_code=return_code,
            error=error,
            finished_at=_utc_now(),
        )
        store.update(updated)
        return updated

    if not _process_alive(record.pid):
        updated = replace(
            record,
            status=JobStatus.FAILED,
            error="launcher exited without writing a return code",
            finished_at=_utc_now(),
        )
        store.update(updated)
        return updated

    return record
