"""Durable evidence for optimizer runs whose solve process fails.

Before this module, a failed solve only left "exited with code 1; inspect worker
log" on the run: the traceback went to the API terminal and was lost. This module
saves two bounded HANA artifacts per failed run so the UI and the Workspace
Assistant can explain the failure:

* ``failure-report.json``: written by the solve process itself when it raises.
  It holds the exception type and message, the run stage, a classified
  ``failure_type`` with planner-actionable details, and the last traceback frames.
* ``worker-log-tail.txt``: written by the supervisor for every non-zero exit.
  It holds the last lines the solve process printed, which is the only evidence
  when the process dies without Python (killed by a signal, out of memory).

Nothing here runs the solver. Every save is best effort: a failure to persist
evidence is logged and never hides the original error.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

FAILURE_REPORT_ARTIFACT = "failure-report.json"
LOG_TAIL_ARTIFACT = "worker-log-tail.txt"
MESSAGE_LIMIT = 2000
TRACEBACK_FRAME_LIMIT = 8
LOG_TAIL_BYTES = 64_000
LOG_TAIL_LINES = 200
# Paths in saved tracebacks are made relative to the API root so they do not
# expose the host directory layout.
API_ROOT = Path(__file__).resolve().parents[2]


def classify_failure(exc: BaseException) -> tuple[str, dict[str, Any]]:
    """Map a solve exception to a stable failure type and planner-actionable details.

    Args:
        exc: Exception raised by the solve process.

    Returns:
        ``(failure_type, details)``. Known types:
        ``profile_matrix_incomplete`` (details: the matrix coverage summary with
        missing volumes and product counts), ``invalid_input`` (any other
        ``ValueError``), ``solver_inconsistent`` (HiGHS reported a result
        that the independent check contradicts; a solver library defect),
        ``out_of_memory`` and ``unhandled_exception``.
    """
    from production_wheel.matrix import ProfileMatrixCoverageError
    from production_wheel.optimization.solver_integrity import (
        SolverResultInconsistentError,
        installed_highspy_version,
    )

    if isinstance(exc, ProfileMatrixCoverageError):
        return "profile_matrix_incomplete", {"matrix_coverage": exc.coverage}
    if isinstance(exc, SolverResultInconsistentError):
        return "solver_inconsistent", {"highspy_version": installed_highspy_version()}
    if isinstance(exc, MemoryError):
        return "out_of_memory", {}
    if isinstance(exc, ValueError):
        return "invalid_input", {}
    return "unhandled_exception", {}


def _relative(path: str) -> str:
    """Return a traceback file path relative to the API root when it lies inside it."""
    try:
        return str(Path(path).resolve().relative_to(API_ROOT))
    except ValueError:
        return Path(path).name


def traceback_tail(exc: BaseException, limit: int = TRACEBACK_FRAME_LIMIT) -> list[dict[str, Any]]:
    """Return the innermost traceback frames as ``{"file", "line", "function", "code"}``.

    Args:
        exc: Exception with an attached traceback.
        limit: Maximum number of innermost frames kept.

    Returns:
        Frames ordered outermost to innermost, with API-relative file paths.
    """
    frames = traceback.extract_tb(exc.__traceback__)[-limit:]
    return [
        {
            "file": _relative(frame.filename),
            "line": frame.lineno,
            "function": frame.name,
            "code": (frame.line or "")[:300],
        }
        for frame in frames
    ]


def build_failure_report(exc: BaseException, stage: str | None) -> dict[str, Any]:
    """Build the JSON failure report saved by a solve process that raised.

    Args:
        exc: Exception that stopped the solve.
        stage: Run stage at the time of the failure (e.g. ``optimizing``).

    Returns:
        JSON-ready dict with ``failure_type``, ``exception_type``, ``message``,
        ``summary`` (one short line for the run's public error), ``stage``,
        ``details``, ``traceback`` and ``recorded_at``.
    """
    failure_type, details = classify_failure(exc)
    message = str(exc)[:MESSAGE_LIMIT]
    if failure_type == "profile_matrix_incomplete":
        coverage = details["matrix_coverage"]
        summary = (
            f"{len(coverage['missing_volumes'])} package volumes "
            f"({coverage['affected_member_count']} products) are missing from the plant "
            "profile volume matrix."
        )
    else:
        summary = f"{type(exc).__name__}: {message.splitlines()[0] if message else 'no message'}"[:300]
    return {
        "failure_type": failure_type,
        "exception_type": type(exc).__name__,
        "message": message,
        "summary": summary,
        "stage": stage,
        "details": details,
        "traceback": traceback_tail(exc),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def record_failure(service: Any, run_id: str, exc: BaseException) -> dict[str, Any] | None:
    """Save the failure report of a solve process; never raise.

    Args:
        service: Workspace service with ``get_run`` and ``repo.put_artifact``.
        run_id: Run whose solve raised.
        exc: The exception that stopped the solve.

    Returns:
        The saved report, or ``None`` when it could not be saved.
    """
    try:
        try:
            stage = service.get_run(run_id).get("stage")
        except Exception:
            stage = None
        report = build_failure_report(exc, stage)
        service.repo.put_artifact(
            run_id, FAILURE_REPORT_ARTIFACT, json.dumps(report, default=str).encode()
        )
        return report
    except Exception:
        logger.exception("Could not save the failure report of run %s", run_id)
        return None


def read_artifact(service: Any, run_id: str, name: str) -> bytes | None:
    """Read one run artifact, returning ``None`` when it was never saved or cannot be read."""
    try:
        return service.repo.artifact(run_id, name)
    except KeyError:
        return None
    except Exception:
        logger.exception("Could not read artifact %s of run %s", name, run_id)
        return None


def read_failure_report(service: Any, run_id: str) -> dict[str, Any] | None:
    """Return the saved failure report of a run, or ``None`` when absent or unreadable."""
    raw = read_artifact(service, run_id, FAILURE_REPORT_ARTIFACT)
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def describe_exit(returncode: int | None) -> str:
    """Explain a child process exit code in plain words.

    Args:
        returncode: ``Popen.returncode``; negative values are POSIX signals.

    Returns:
        For example ``"exited with code 1"`` or
        ``"was killed by signal SIGKILL (9; often the system ran out of memory)"``.
    """
    if returncode is None:
        return "ended without an exit code"
    if returncode >= 0:
        return f"exited with code {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = "an unknown signal"
    hint = "; often the system ran out of memory" if -returncode == signal.SIGKILL else ""
    return f"was killed by signal {name} ({-returncode}{hint})"


class LogTail:
    """Forward a child's output pipe to this process's stderr and keep its last bytes.

    The solve process prints tqdm progress and tracebacks to stderr. Reading raw
    chunks (not lines) keeps carriage-return progress bars live in the terminal
    while the last ``LOG_TAIL_BYTES`` stay available for the failure artifact.
    """

    def __init__(self, stream: Any, sink: Any = None, limit: int = LOG_TAIL_BYTES):
        """Start forwarding ``stream`` (a binary pipe or ``None``) in a daemon thread.

        Args:
            stream: Child's binary stderr pipe; ``None`` disables capture.
            sink: Binary writable destination; defaults to ``sys.stderr.buffer``.
            limit: Maximum number of trailing bytes retained.
        """
        self.limit = limit
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.sink = sink if sink is not None else getattr(sys.stderr, "buffer", None)
        self.thread = None
        if stream is not None and hasattr(stream, "fileno"):
            self.stream = stream
            self.thread = threading.Thread(target=self._pump, name="solve-log-tail", daemon=True)
            self.thread.start()

    def _pump(self):
        """Copy chunks until the child closes its pipe."""
        fd = self.stream.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            if self.sink is not None:
                try:
                    self.sink.write(chunk)
                    self.sink.flush()
                except Exception:
                    pass
            with self.lock:
                self.buffer.extend(chunk)
                if len(self.buffer) > self.limit:
                    del self.buffer[: len(self.buffer) - self.limit]

    def text(self, max_lines: int = LOG_TAIL_LINES) -> str:
        """Return the retained output as text, one line per progress refresh or newline.

        Args:
            max_lines: Number of trailing non-empty lines returned.
        """
        if self.thread is not None:
            # The pipe closes when the child exits; wait briefly for the last chunk.
            self.thread.join(timeout=2)
        with self.lock:
            raw = bytes(self.buffer)
        lines = [
            line.rstrip()
            for line in raw.decode("utf-8", errors="replace").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        return "\n".join(lines[-max_lines:])


def save_log_tail(service: Any, run_id: str, tail: LogTail, returncode: int | None) -> None:
    """Save the solve process output tail of a failed run; never raise.

    Args:
        service: Workspace service with ``repo.put_artifact``.
        run_id: Failed run.
        tail: Capture of the child's stderr.
        returncode: Child exit code, written as the first line of the artifact.
    """
    try:
        body = f"Solve process {describe_exit(returncode)}.\n{tail.text()}"
        service.repo.put_artifact(run_id, LOG_TAIL_ARTIFACT, body.encode("utf-8"))
    except Exception:
        logger.exception("Could not save the solve log tail of run %s", run_id)


def public_failure_message(report: Mapping[str, Any] | None, returncode: int | None) -> str:
    """Return the short error stored on a failed run record.

    Args:
        report: Failure report saved by the solve process, if any.
        returncode: Child exit code.

    Returns:
        The report summary when available; otherwise how the process ended,
        pointing to the saved log tail.
    """
    if report and report.get("summary"):
        return f"Solve failed: {report['summary']} See failure diagnostics."
    return f"Solve process {describe_exit(returncode)}; see the saved log in failure diagnostics."
