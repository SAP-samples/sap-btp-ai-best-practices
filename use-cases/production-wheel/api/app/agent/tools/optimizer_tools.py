"""LangChain tools exposing the production-wheel optimizer to the ReAct agent.

Design:
- Each capability is a plain implementation function (``*_impl``) that imports only
  the optimizer and the job layer, so it is importable and unit-testable without
  the LangChain/LLM stack.
- ``optimizer_tools(store, workspace)`` lazily wraps the implementations as
  LangChain tools for ``AgentRuntime.create(extra_tools=...)``. It binds the job
  store and a workspace directory by closure, so the model only ever supplies
  domain arguments.

The agent never imports the optimizer solve path directly: long solves run as
detached CLI subprocesses tracked in the HANA job store (see app.jobs).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from production_wheel.extraction.common import sha256_file
from production_wheel.extraction.pipeline import extract_snapshot, make_run_id
from production_wheel.scenarios import load_canonical_inputs
from production_wheel.validation import get_capabilities as _get_capabilities

from app.jobs import JobStatus, launch, poll

# Bound so a single tool result never floods the model context with a whole report.
_REPORT_CHAR_LIMIT = 12_000


def _to_float(value: object) -> float | None:
    """Best-effort float coercion returning None on empty or non-numeric cells."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    """Best-effort int coercion returning None on empty or non-numeric cells."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def capabilities_impl() -> dict[str, Any]:
    """Return the optimizer's typed capability contract."""

    return _get_capabilities()


def _discover_enrichment(primary: Path) -> str | None:
    """Find a single enrichment workbook beside the primary production workbook.

    Primary-workbook extraction needs the enrichment workbook to resolve blank
    pallet conversions, but a user typically names only the primary file. Look in
    the same directory for exactly one ``*.xlsx`` whose name contains "enrichment"
    (case-insensitive), ignoring Excel lock files and the primary workbook itself.

    Args:
        primary: Path to the primary production workbook.

    Returns:
        The enrichment workbook path, or None on zero or multiple matches so the
        caller never guesses between siblings.
    """

    candidates = [
        path
        for path in primary.parent.glob("*.xlsx")
        if "enrichment" in path.name.lower()
        and not path.name.startswith("~$")
        and path.resolve() != primary.resolve()
    ]
    return str(candidates[0]) if len(candidates) == 1 else None


def extract_inputs_impl(
    primary_path: str, output_root: str, enrichment_path: str | None = None
) -> dict[str, Any]:
    """Extract a workbook into a canonical run and summarize its block inventory.

    Args:
        primary_path: Path to the primary production Excel workbook.
        output_root: Directory under which the extraction run is written.
        enrichment_path: Enrichment workbook. When omitted, a single enrichment
            workbook beside the primary file is auto-detected; it is required to
            resolve the blank pallet conversions in the primary source.

    Returns:
        The run directory, the enrichment workbook used, and the per-(plant, SEFI) block
        inventory the agent uses to choose solve scope and per-plant caps.
    """

    primary = Path(primary_path)
    if enrichment_path is None:
        enrichment_path = _discover_enrichment(primary)
    run_directory = Path(output_root) / make_run_id(sha256_file(primary))
    result = extract_snapshot(
        primary_path=primary,
        enrichment_path=Path(enrichment_path) if enrichment_path else None,
        run_directory=run_directory,
    )
    inputs = load_canonical_inputs(result.run_directory)
    counts: dict[tuple[str, str], int] = {}
    for member in inputs.members:
        counts[member.block_key] = counts.get(member.block_key, 0) + 1
    return {
        "run_directory": str(result.run_directory),
        "status": result.manifest.get("status"),
        "enrichment_path": str(enrichment_path) if enrichment_path else None,
        "modeled_fini_count": len(inputs.members),
        "block_count": len(counts),
        "blocks": [
            {"plant": plant, "sefi": sefi, "fini_count": count}
            for (plant, sefi), count in sorted(counts.items())
        ],
    }


def launch_solve_impl(
    store: Any,
    workspace: str,
    run_directory: str,
    request_json: str,
    knobs: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Launch a constrained solve as an async job and return its identifier."""

    job_id = launch(
        store,
        run_directory=run_directory,
        request_json=request_json,
        workspace=workspace,
        knobs=knobs,
    )
    record = store.get(job_id)
    return {
        "job_id": job_id,
        "status": record.status.value,
        "output_dir": record.output_dir,
    }


def job_status_impl(store: Any, job_id: str) -> dict[str, Any]:
    """Return the current status of a launched solve job (single, non-blocking check)."""

    return poll(store, job_id).as_dict()


def wait_for_job_impl(
    store: Any,
    job_id: str,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    """Block until a job reaches a terminal state or the timeout elapses.

    The sleep-poll loop runs here (not in the agent graph), so a multi-minute solve
    costs the agent a single tool call instead of dozens of instant status polls
    that would otherwise exhaust the graph recursion limit.

    Args:
        store: The job store to read from.
        job_id: The job to await.
        timeout_seconds: Maximum wall-clock time to wait before returning while
            still running.
        poll_interval_seconds: Delay between status checks.

    Returns:
        The final job record; ``status`` is ``running`` only if the timeout was hit.
    """

    deadline = time.monotonic() + timeout_seconds
    record = poll(store, job_id)
    while record.status is JobStatus.RUNNING and time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        record = poll(store, job_id)
    return record.as_dict()


def read_results_impl(output_dir: str) -> dict[str, Any]:
    """Read a finished frontier bundle into a compact explain-back summary.

    Args:
        output_dir: The job's output directory (a greenfield frontier bundle).

    Returns:
        Per-point coverage-days and J_CH, structure counts, applied constraints,
        run metadata, and the human-readable solution report (bounded in length).
    """

    out = Path(output_dir)
    summary: dict[str, Any] = {"output_dir": str(out)}

    manifest_path = out / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["artifact_type"] = manifest.get("artifact_type")
        summary["scenario_id"] = manifest.get("scenario_id")
        summary["elapsed_seconds"] = manifest.get("elapsed_seconds")
        summary["counts"] = manifest.get("counts")
        summary["pool_completeness"] = manifest.get("pool_completeness")
        request_meta = manifest.get("solve_request") or {}
        summary["applied_constraints"] = request_meta.get("applied_constraints")

    frontier_path = out / "frontier_summary.csv"
    if frontier_path.is_file():
        points: list[dict[str, Any]] = []
        with frontier_path.open(newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                points.append(
                    {
                        "point": index,
                        "coverage_days": _to_float(row.get("demand_weighted_mean_coverage_days")),
                        "j_ch": _to_float(row.get("j_ch")),
                        "group_count": _to_int(row.get("group_count")),
                        "singleton_group_count": _to_int(row.get("singleton_group_count")),
                    }
                )
        summary["frontier"] = points

    report_path = out / "solution_report.md"
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        summary["report_markdown"] = text[:_REPORT_CHAR_LIMIT]
        summary["report_truncated"] = len(text) > _REPORT_CHAR_LIMIT

    return summary


def optimizer_tools(store: Any, workspace: str) -> list:
    """Build the LangChain tools bound to a job store and workspace directory.

    Args:
        store: A job store (app.jobs.JobStore) recording launched solves.
        workspace: Directory under which extraction runs and job artifacts live.

    Returns:
        A list of LangChain tools suitable for ``AgentRuntime.create(extra_tools=...)``.
    """

    from langchain_core.tools import tool

    @tool
    def get_optimizer_capabilities() -> dict:
        """Return the optimizer capability contract: coverage modes, matrix modes,
        the finite constraint kinds the agent may use, supported group caps, and
        governed package volumes. Call this before composing a SolveRequest."""

        return capabilities_impl()

    @tool
    def extract_inputs(primary_path: str, enrichment_path: str | None = None) -> dict:
        """Extract a primary production Excel workbook into a canonical run directory
        and return its block inventory (plants/SEFIs with per-block FINI counts) plus
        totals. The required enrichment workbook is auto-detected next to the primary file;
        pass enrichment_path only to override. Use the returned run_directory for
        launch_solve and the block inventory to choose scope and per-plant caps."""

        return extract_inputs_impl(primary_path, workspace, enrichment_path)

    @tool
    def launch_solve(
        run_directory: str, request_json: str, knobs: dict | None = None
    ) -> dict:
        """Launch a constrained PARETO frontier solve as an async background job.
        request_json is a full SolveRequest JSON string (config + typed constraints
        + optional scope). knobs maps CLI flags to values, e.g.
        {"--frontier-points": 17, "--per-block-total-seconds": 600}. Returns a
        job_id; pass it to wait_for_job to await completion (the solve can take
        several minutes)."""

        return launch_solve_impl(store, workspace, run_directory, request_json, knobs)

    @tool
    def get_job_status(job_id: str) -> dict:
        """Return the current status of a launched solve job in a single, non-blocking
        check: running, done (return_code 0 valid / 2 invalid-or-partial), or failed.
        To wait for completion, call wait_for_job instead of calling this in a loop."""

        return job_status_impl(store, job_id)

    @tool
    def wait_for_job(job_id: str, timeout_seconds: int = 900) -> dict:
        """Block until a launched solve finishes (done or failed) or the timeout
        elapses, polling in the background. Call this once, right after launch_solve,
        instead of calling get_job_status repeatedly — a full-population solve can take
        several minutes. Returns the final job record (status is running only on timeout)."""

        return wait_for_job_impl(store, job_id, timeout_seconds)

    @tool
    def read_results(output_dir: str) -> dict:
        """Read a finished frontier bundle (the job's output_dir) into a compact
        summary: per-point coverage-days and J_CH, group/singleton counts, the
        applied constraints, run metadata, and the human-readable solution report.
        Use this to explain the results to the user."""

        return read_results_impl(output_dir)

    return [
        get_optimizer_capabilities,
        extract_inputs,
        launch_solve,
        get_job_status,
        wait_for_job,
        read_results,
    ]
