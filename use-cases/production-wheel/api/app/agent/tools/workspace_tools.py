"""Bind finite, ID-based optimizer operations to the shared workspace service."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.workspace.models import QuerySpec, RunSubmission
from .result_overview import result_overview

WAIT_POLL_SECONDS = 4
WAIT_QUEUE_TIMEOUT_SECONDS = 60


def workspace_tools(
    service: Any,
    context_id: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[BaseTool]:
    """Create tools using one service and conversation selection.

    Args:
        service: Shared workspace service used by the manual HTTP interface.
        context_id: Process-local page or CLI session selection identifier.
        on_event: Optional thread-safe observer for successful draft/run changes.

    Returns:
        Finite LangChain tools accepting explicit dataset, draft and run IDs.
    """

    def emit(event: dict) -> None:
        """Notify the view of a successful mutation without changing its outcome."""
        if on_event is not None:
            on_event(event)

    @tool
    def get_optimizer_capabilities() -> dict:
        """Get supported modes, constraints and the current selected workspace IDs."""
        return dict(service.capabilities()) | {
            "workspace_context": service.context(context_id)
        }

    def profile_summary(profile):
        """Keep matrix metadata concise; the service retains all frozen matrix rows."""
        return {key: value for key, value in profile.items() if key != "matrix_rows"} | {"matrix_pair_count": len(profile.get("matrix_rows", []))}

    @tool
    def list_plant_profiles() -> list:
        """Discover active named plant profiles and their fixed settings and rules."""
        return [profile_summary(profile) for profile in service.list_plant_profiles()]

    @tool
    def select_plant_profile(profile_id: str) -> dict:
        """Select an active profile for future drafts; existing drafts remain frozen."""
        profile = service.get_plant_profile(profile_id)
        service.context(context_id, {"plant_profile_id": profile_id})
        return profile_summary(profile)

    @tool
    def list_datasets(filters: dict | None = None) -> list:
        """List datasets with optional plant, name substring, status and creation bounds.

        created_from/created_to are inclusive ISO datetimes; omitted timezones
        mean UTC. Unsupported filter keys are rejected."""
        return service.list_datasets(filters)

    @tool
    def inspect_dataset(dataset_id: str) -> dict:
        """Inspect one dataset's plant, metadata, source issues and scope inventory."""
        return service.inspect_dataset(dataset_id)

    @tool
    def get_optimizer_reference(topic: str) -> dict:
        """Read a registered optimizer reference topic; arbitrary paths are forbidden."""
        return service.reference(topic)

    @tool
    def update_run_draft(
        draft_id: str | None = None,
        revision: int | None = None,
        patch: dict | None = None,
    ) -> dict:
        """Create, inspect or revise a shared optimization draft.

        Omit draft_id to create a draft for patch.dataset_id or the explicitly
        selected context dataset; optional request/budget patches are then applied.
        For an existing draft, an empty patch reads without changing its revision.
        A nonempty existing-draft patch requires its exact current revision.
        """
        changes = dict(patch or {})
        if draft_id is None:
            dataset_id = changes.pop("dataset_id", None) or service.context(
                context_id
            ).get("dataset_id")
            if not dataset_id:
                raise ValueError(
                    "Select a dataset or provide patch.dataset_id before creating a draft"
                )
            profile_id = changes.pop("plant_profile_id", None) or service.context(context_id).get("plant_profile_id")
            result = service.create_draft(dataset_id, profile_id, changes.pop("title", ""))
            draft_id = result["draft_id"]
            if changes:
                result = service.update_draft(draft_id, result["revision"], changes)
            service.context(
                context_id, {"dataset_id": dataset_id, "draft_id": draft_id}
            )
        elif not changes:
            return service.get_draft(draft_id)
        else:
            if revision is None:
                raise ValueError(
                    "An existing draft update requires its current revision"
                )
            result = service.update_draft(draft_id, revision, changes)
        emit(
            {
                "type": "draft_changed",
                "draft_id": draft_id,
                "revision": result.get("revision"),
                "draft": result,
            }
        )
        return result

    @tool
    def validate_run_draft(draft_id: str) -> dict:
        """Validate a persisted draft and return capability/data errors before launch."""
        return service.validate_draft(draft_id)

    @tool
    def launch_optimization(
        draft_id: str,
        revision: int,
        idempotency_key: str,
        parent_run_id: str | None = None,
    ) -> dict:
        """Launch the exact draft revision as an independent persisted run.

        Use one stable idempotency key for retries of the same intended launch.
        The run continues independently of this conversation or its connection.
        """
        submitted = service.submit(
            RunSubmission(
                draft_id=draft_id,
                revision=revision,
                idempotency_key=idempotency_key,
                parent_run_id=parent_run_id,
            )
        )
        reader = getattr(service, "run_status", None)
        result = reader(submitted["run_id"]) if callable(reader) else submitted
        emit({"type": "run_created", "run_id": result.get("run_id"), "run": result})
        return result

    @tool
    def list_runs(filters: dict | None = None) -> list:
        """List runs filtered by dataset_id, status, parent_run_id or creation dates.

        created_from/created_to are inclusive ISO datetimes; omitted timezones
        mean UTC. Unknown IDs match no records. Unsupported keys are rejected."""
        return service.list_run_summaries(filters)

    status_reads: set[str] = set()

    async def compact_run_status(run_id: str, fallback: dict | None = None) -> dict:
        """Read the public run status, preserving compatibility for lightweight fakes."""
        reader = getattr(service, "run_status", None)
        if callable(reader):
            return await asyncio.to_thread(reader, run_id)
        if fallback is not None:
            return fallback
        return await asyncio.to_thread(service.get_run, run_id)

    async def await_run(run_id: str) -> dict:
        """Wait cooperatively for a terminal persisted job; emit progress without LLM calls.

        Cancellation ends this wait only. The independent optimizer owns its job.
        Reads run in a worker thread so HANA I/O cannot block the response loop.
        """
        waiting_since = time.monotonic()
        while True:
            run = await asyncio.to_thread(service.get_run, run_id)
            emit(
                {
                    "type": "run_progress",
                    "run_id": run_id,
                    "status": run.get("status"),
                    "stage": run.get("stage"),
                    "progress": run.get("progress"),
                    "created_at": run.get("created_at"),
                }
            )
            if run.get("status") not in {"queued", "running", "persisting"}:
                return await compact_run_status(run_id, run)
            if run.get("status") == "queued" and time.monotonic() - waiting_since >= WAIT_QUEUE_TIMEOUT_SECONDS:
                summary = await compact_run_status(run_id, run)
                return {**summary, "wait_status": "queued_not_started", "message": (
                    "This run is still queued: no worker has claimed it. Check that an optimizer worker "
                    "is running and whether it is busy with another run. Solver execution has not started. "
                    "Report this existing run ID to the user and stop waiting in this turn; do not relaunch."
                )}
            await asyncio.sleep(WAIT_POLL_SECONDS)

    @tool
    async def wait_for_run(run_id: str) -> dict:
        """Wait until a run completes, fails, is cancelled or loses its worker.

        Call ONCE when the user asks for analysis when a long solve finishes.
        If wait_status is queued_not_started, explain the queue delay and end
        this turn without another wait or launch; the queued job remains available.
        Progress streams to the browser without spending model/recursion steps.
        After completion call get_run_results. Never relaunch while waiting.
        """
        return await await_run(run_id)

    @tool
    async def get_run_status(run_id: str) -> dict:
        """Read status once. Repeated reads in this turn wait for job termination.

        For a report on completion prefer wait_for_run. An explicit one-off
        status question returns immediately on its first read.
        """
        if run_id in status_reads:
            return await await_run(run_id)
        status_reads.add(run_id)
        return await compact_run_status(run_id)

    @tool
    def get_run_configuration(run_id: str) -> dict:
        """Read the frozen compact configuration for one explicit optimizer run."""
        return service.run_status(run_id)["configuration"]

    @tool
    def get_run_failure_diagnostics(run_id: str) -> dict:
        """Read bounded failure evidence for a failed or incomplete run.

        Call this automatically after get_run_status reports a failed status,
        before explaining a failure or proposing another optimization.
        """
        return service.run_failure_diagnostics(run_id)

    @tool
    def get_run_matrix_page(
        run_id: str,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
    ) -> dict:
        """Read one filtered page of a run's frozen compatibility matrix evidence."""
        return service.run_matrix_page(run_id, offset=offset, limit=limit, status=status)

    @tool
    def cancel_run(run_id: str) -> dict:
        """Request cancellation of the explicitly identified independent run."""
        return service.cancel_run(run_id)

    @tool
    def get_run_results(run_id: str) -> dict:
        """Read a persisted run's result overview, frontier points and metadata."""
        return result_overview(service.results(run_id))

    @tool
    def query_optimizer_data(spec: dict) -> dict:
        """Query a dataset or result view using a validated semantic QuerySpec.

        Provide an explicit dataset_id for input views or run_id for result views,
        plus optional point_index, filters, fields, metrics, sort and pagination.
        Raw SQL and filesystem paths are not accepted.
        """
        return service.query(QuerySpec.model_validate(spec))

    @tool
    def get_solution_details(
        run_id: str,
        point_index: int,
        view: str = "groups",
        filters: list[dict] | None = None,
        limit: int = 50,
    ) -> dict:
        """Read a page of persisted groups or members for one explicit frontier point.

        Filters use QuerySpec predicates. Use query_optimizer_data to paginate,
        select fields, sort or calculate full-population aggregates.
        """
        return service.query(
            QuerySpec(
                run_id=run_id,
                point_index=point_index,
                view=view,
                filters=filters or [],
                limit=limit,
            )
        )

    @tool
    def compare_solutions(left: dict, right: dict) -> dict:
        """Compare two explicit solution selections, each {run_id, point_index}."""
        for selection in (left, right):
            if set(selection) != {"run_id", "point_index"}:
                raise ValueError("Each selection needs run_id and point_index")
            QuerySpec(
                view="groups",
                run_id=selection["run_id"],
                point_index=selection["point_index"],
            )
        return service.compare(left, right)

    @tool
    def explain_assignment(
        run_id: str,
        point_index: int,
        material: str | None = None,
        group_id: str | None = None,
    ) -> dict:
        """Explain FINI or group membership from persisted evidence for one point."""
        return service.explain(
            run_id, point_index, material=material, group_id=group_id
        )

    return [
        get_optimizer_capabilities,
        list_plant_profiles,
        select_plant_profile,
        list_datasets,
        inspect_dataset,
        get_optimizer_reference,
        update_run_draft,
        validate_run_draft,
        launch_optimization,
        list_runs,
        get_run_status,
        wait_for_run,
        get_run_configuration,
        get_run_failure_diagnostics,
        get_run_matrix_page,
        cancel_run,
        get_run_results,
        query_optimizer_data,
        get_solution_details,
        compare_solutions,
        explain_assignment,
    ]
