"""Bind finite, ID-based optimizer operations to the shared workspace service."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.workspace.import_settings import build_settings_proposal, pending_import_summary
from app.workspace.models import QuerySpec, RunSubmission
from production_wheel.extraction.frequencies import ALL_AVAILABLE
from app.workspace.plant_profiles import apply_profile_change, upgrade_settings
from production_wheel.volume_classes import NEAREST_CLASS, label_sort_key
from app.workspace.matrix_description import expand_matrix_description, summarize_matrix_change
from .dataset_views import dataset_list_entry
from .draft_views import compact_draft, compact_validation
from .result_overview import result_overview
from .sap_tools import sap_tools

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

    def selected_profile(profile_id):
        """Read an explicit profile or the one selected in this conversation."""
        profile_id = profile_id or service.context(context_id).get("plant_profile_id")
        if not profile_id:
            raise ValueError("Select a plant profile (select_plant_profile) before proposing profile changes")
        return service.get_plant_profile(profile_id)

    def resulting_volume_matching(profile, change):
        """Volume matching the profile will have after a settings change (patch first)."""
        patch = (change.get("settings") or {}).get("volume_matching")
        return patch or upgrade_settings(profile.get("settings") or {})["volume_matching"]

    def propose(profile, change, summary):
        """Validate a change against the stored profile and show it to the user for approval.

        Nothing is saved here: the browser card applies it only after the user clicks
        Approve, through the same apply_profile_change validation and revision check.
        """
        if not summary or not summary.strip():
            raise ValueError("Provide a plain-language summary of the proposed change for the user")
        apply_profile_change(profile, change)
        proposal = {"type": "profile_proposal", "proposal_id": uuid.uuid4().hex,
            "profile_id": profile["profile_id"], "profile_name": profile["name"],
            "plant": profile["plant"], "profile_revision": profile["revision"],
            "summary": summary.strip(), "change": change}
        if change.get("matrix_rows") is not None:
            # Reviewable difference shown on the card and returned to the model,
            # so neither has to read every ordered pair.
            proposal["matrix_summary"] = summarize_matrix_change(profile.get("matrix_rows", []), change["matrix_rows"])
            matching = resulting_volume_matching(profile, change)
            proposal["matrix_summary"]["volume_matching"] = matching
            dataset_id = service.context(context_id).get("dataset_id")
            if matching == NEAREST_CLASS and dataset_id:
                # Which products of the conversation's dataset change class, so the
                # planner sees the effect of a new or removed stop before approving.
                stops = sorted({row["volume_a"] for row in change["matrix_rows"]}, key=label_sort_key)
                try:
                    preview = service.dataset_volume_coverage(dataset_id, profile["profile_id"], stops)
                except ValueError:
                    preview = None
                if preview:
                    proposal["matrix_summary"].update(
                        dataset_id=dataset_id,
                        reclassified_volumes=preview["reclassified_volumes"],
                        unclassified_volumes=[item["volume"] for item in preview["dataset_volumes"]
                                              if item.get("proposed_class") is None])
        emit(proposal)
        result = {"status": "awaiting_user_approval", "proposal_id": proposal["proposal_id"],
            "profile_revision": profile["revision"],
            "note": "Shown to the user as an approval card. It is NOT saved until the user approves it in the UI."}
        if "matrix_summary" in proposal:
            result["matrix_summary"] = proposal["matrix_summary"]
        return result

    @tool
    def get_plant_profile_details(profile_id: str | None = None) -> dict:
        """Read one plant profile's revision, fixed settings and approved constraints.

        Defaults to the profile selected in this conversation. Matrix rows are
        summarized as volumes and pair counts per status; volume_matching says
        whether they are every package volume (exact) or reference volumes /
        stops with optional "<n" / ">n" classes (nearest_class)."""
        profile = selected_profile(profile_id)
        rows = profile.get("matrix_rows", [])
        statuses = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        return profile_summary(profile) | {
            "volume_matching": upgrade_settings(profile.get("settings") or {})["volume_matching"],
            "matrix_volumes": sorted({r["volume_a"] for r in rows}, key=label_sort_key),
            "matrix_pairs_by_status": statuses}

    @tool
    def propose_profile_constraint(
        action: str,
        summary: str,
        constraint: dict | None = None,
        constraint_id: str | None = None,
        profile_id: str | None = None,
    ) -> dict:
        """Propose adding, replacing or removing one fixed constraint of a plant profile.

        action: "add_constraint", "replace_constraint" or "remove_constraint".
        constraint (add/replace): a group_rule or selection_bound in the business-rule
            grammar, including constraint_id, source_text (planner wording) and
            description (plain-language meaning). Scope plant is set to the profile plant.
        constraint_id (remove): the rule to remove.
        summary: what the user is approving, in plain language.
        The change is validated now but only saved after the user approves the card."""
        if constraint is not None:
            for key in ("source_text", "description"):
                if not str(constraint.get(key) or "").strip():
                    raise ValueError(f"constraint.{key} is required so the user can review it in Settings")
        change = {"action": action}
        if constraint is not None:
            change["constraint"] = constraint
        if constraint_id is not None:
            change["constraint_id"] = constraint_id
        return propose(selected_profile(profile_id), change, summary)

    @tool
    def propose_profile_settings(
        summary: str,
        settings_patch: dict | None = None,
        volume_compatibility: dict | None = None,
        profile_id: str | None = None,
    ) -> dict:
        """Propose changing fixed profile settings and/or replacing the volume matrix.

        settings_patch: subset of horizon_days, demand_days_per_week,
            runner_thresholds (high_max_days, low_min_days, high_max_ratio, low_min_ratio),
            cycle_stock_policy (mode single / runner_multiple / optimizer_choice,
            high_min_days, medium_min_days, low_min_days, medium_batch_factor,
            max_days, mixed_groups_only), runner_basis, matrix_mode, maximum_group_size,
            sap_settings (rules of the SAP settings generated after a run; read live, so
            no re-run is needed): calendar_odd_profile / calendar_even_profile
            ("<profile>-<calendar>" plant customizing codes, must differ),
            calendar_frequency_code, calendar_lot_size, calendar_reset_lot_size,
            calendar_reset_text, recipe_characters (fallback string), apo_default_pv,
            kpi_line_breakdown (bool), multi_tier_dc (bool),
            dc_min_lot / dc_rounding ({all_classes|A|B|C: {rule, number}}; a class
            set to null means "same as all classes"), abc_cycle_days (A, B, C,
            unclassified). Nested sections are merged key by key: send only what
            changes. For SAP settings pass the profile_id of the run being discussed
            (get_sap_settings_status returns it); load the sap-settings skill first.
        volume_compatibility: description compiled into the complete new matrix.
            The profile's volume_matching (settings_patch may change it in the
            same card) decides what the matrix rows are:
              exact: every package volume of the products is a row.
              nearest_class: rows are reference volumes ("stops", e.g. 1, 2.5, 5,
                10, 15) plus optional "<n" (below the smallest stop) and ">n"
                (above the largest stop) classes; each product takes the row of
                its nearest stop (ties: the smaller stop). "<n"/">n" must name the
                smallest/largest stop: adding a stop beyond them means renaming.
            volumes: explicit volumes or class labels (strings, e.g. "2", "<1").
            dataset_id: exact matrices only: also include every package volume of
                this dataset's optimizable products (see get_dataset_volume_coverage).
            include_profile_volumes: also include the current matrix rows.
            keep_existing_pairs: keep every pair of the current matrix unchanged
                and generate only pairs involving a new label (use this to add
                volumes or stops without retyping the matrix).
            remove_volumes: labels of the current matrix to drop (a stop, ">15").
            default_status: Y, AVOID or N for generated pairs.
            inherit_from: {new label: existing label}: the new row copies the
                existing one and the two are Y together. "Add a 2 L stop that
                behaves like 2.5" = volumes ["2"], keep_existing_pairs true,
                inherit_from {"2": "2.5"}.
            compatible_groups: lists of labels that are all mutually Y.
            rules: ordered relational rules on numeric labels, later ones win
                (pairs with "<n"/">n" are set by default, inherit, groups, overrides):
                {"kind": "range_group", "min_litres", "max_litres", "within_status",
                 "outside_status"}: volumes in the range together; a pair with one
                 volume inside and one outside gets outside_status.
                {"kind": "max_difference_litres", "litres", "within_status", "outside_status"}
                {"kind": "max_difference_percent", "percent", "within_status",
                 "outside_status"}: difference as % of the smaller volume
                 (2 L vs 2.5 L = 25 %).
                Leave outside_status out to only grant within_status.
            overrides: explicit {volume_a, volume_b, status} pairs, applied last.
            Without keep_existing_pairs the result replaces the whole matrix.
        The change is validated now but only saved after the user approves the card.
        The result's matrix_summary lists added/removed volumes, pair counts per
        status and changed pairs; for a nearest_class matrix with a dataset in the
        conversation also reclassified_volumes ({volume, product_count, from, to})
        and unclassified_volumes (a run would stop on them): describe it to the user."""
        profile = selected_profile(profile_id)
        change = {"action": "update_settings"}
        if settings_patch:
            change["settings"] = settings_patch
        if volume_compatibility:
            dataset_volumes = []
            if volume_compatibility.get("dataset_id"):
                coverage = service.dataset_volume_coverage(volume_compatibility["dataset_id"], profile["profile_id"])
                dataset_volumes = [item["volume"] for item in coverage["dataset_volumes"]]
            change["matrix_rows"] = expand_matrix_description(
                volume_compatibility, profile.get("matrix_rows", []), dataset_volumes,
                resulting_volume_matching(profile, change))
        return propose(profile, change, summary)

    @tool
    def get_dataset_volume_coverage(dataset_id: str | None = None, profile_id: str | None = None,
                                    proposed_stops: list[str] | None = None) -> dict:
        """Compare a dataset's package volumes with a plant profile's volume matrix.

        Defaults to the dataset and profile selected in this conversation. Returns
        each dataset volume with its product count and whether the profile matrix
        places it, the missing volumes (a run stops on them), and matrix rows the
        dataset does not use. In a nearest_class profile each volume also has its
        matrix_class and class_product_counts totals the products per class.
        proposed_stops (e.g. ["<1", "1", "2", "2.5", "5", "10", "15", ">15"])
        previews another class axis without proposing anything: each volume gets
        proposed_class and reclassified_volumes lists the volumes that move.
        Call it before proposing a matrix for a dataset or a change of stops."""
        context = service.context(context_id)
        dataset_id = dataset_id or context.get("dataset_id")
        if not dataset_id:
            raise ValueError("Select a dataset or pass dataset_id")
        return service.dataset_volume_coverage(dataset_id, profile_id or context.get("plant_profile_id"), proposed_stops)

    @tool
    def get_pending_import() -> dict:
        """Read the workbook chosen on the Datasets page but not yet extracted.

        Returns its file names, plant, the production frequency codes found in the
        workbook with FINI row counts (plus values without a weekly code), the
        overrides already applied, the defaults and what each setting means.
        Call this before discussing or proposing extraction settings; never ask the
        user to type the frequency codes."""
        return pending_import_summary(service.context(context_id).get("pending_import"))

    @tool
    def propose_extraction_settings(
        summary: str,
        modeled_frequencies: list[str] | None = None,
        all_available_frequencies: bool = False,
        demand_days: int | None = None,
        productive_weeks: int | None = None,
        canonical_factor: float | None = None,
        reset_to_default: list[str] | None = None,
    ) -> dict:
        """Propose extraction setting overrides for the pending workbook import.

        Only pass the settings the user wants to change; current overrides are kept.
        modeled_frequencies: complete list of weekly codes to model (e.g. ["01W", "02W", "04W"]).
        all_available_frequencies: true to model every weekly code present in the
            workbook (use this for "all frequencies"; leave modeled_frequencies empty).
        reset_to_default: setting names whose override is removed.
        summary: what the user is approving, in plain language.
        The change is validated now and shown as a card; it applies only when the
        user clicks Apply, and only to the next "Upload and extract"."""
        if all_available_frequencies:
            modeled_frequencies = ALL_AVAILABLE
        changes = {key: value for key, value in {
            "modeled_frequencies": modeled_frequencies, "demand_days": demand_days,
            "productive_weeks": productive_weeks, "canonical_factor": canonical_factor,
        }.items() if value is not None}
        proposal = build_settings_proposal(
            service.context(context_id).get("pending_import"), summary, changes, reset_to_default
        )
        emit(proposal)
        return {"status": "awaiting_user_approval", "proposal_id": proposal["proposal_id"],
            "overrides": proposal["overrides"], "modeled_codes": proposal["modeled_codes"],
            "warnings": proposal["warnings"],
            "note": "Shown to the user as an Apply card. Nothing changes until the user applies it."}

    @tool
    def list_datasets(filters: dict | None = None) -> list:
        """List datasets with optional plant, name substring, status and creation bounds.

        created_from/created_to are inclusive ISO datetimes; omitted timezones
        mean UTC. Unsupported filter keys are rejected. Each entry is compact:
        identity, plant, settings, admission counts and finding counts per
        severity and check. Call inspect_dataset for the full record."""
        return [dataset_list_entry(record) for record in service.list_datasets(filters)]

    @tool
    def inspect_dataset(dataset_id: str) -> dict:
        """Inspect one dataset's plant, metadata, source issues and scope inventory."""
        return service.inspect_dataset(dataset_id)

    @tool
    def explain_dataset_finding(
        rule_id: str, reason: str | None = None, dataset_id: str | None = None
    ) -> dict:
        """Explain one extraction finding or scope note of a dataset review.

        Use this when the user asks why a finding appears, what it means, which
        products it affects, how it affects optimization runs or how to fix it.
        rule_id: the check, e.g. FINI_DATA_QUALITY, FINI_OPTIMIZED_PV_ONLY,
            PRODUCTION_FREQUENCY_UNDETERMINED or FREQUENCY_NOT_SELECTED.
        reason: the reason code (e.g. missing_fixed_pv), or for frequency notes the
            source value or weekly code; omit to cover every reason.
        dataset_id: defaults to the dataset selected on the page.
        Returns the cause, workbook source, effect and fix, the affected product
        rows (first 25, with totals), how many each production version mode
        (FIXED / OPTIMIZED) includes and, for production-version reasons, the
        production versions and lot sizes of each affected SEFI."""
        dataset_id = dataset_id or service.context(context_id).get("dataset_id")
        if not dataset_id:
            raise ValueError("Select a dataset on the Datasets page or pass dataset_id")
        return service.explain_dataset_finding(dataset_id, rule_id, reason)

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
        Profile-owned lists (matrix_pairs, group_size_overrides, the frozen
        profile matrix) are returned as summaries; never patch them.
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
            return compact_draft(service.get_draft(draft_id))
        else:
            if revision is None:
                raise ValueError(
                    "An existing draft update requires its current revision"
                )
            result = service.update_draft(draft_id, revision, changes)
        # The browser receives the full draft to refresh the form; the model
        # receives a compact view that fits the per-tool context budget.
        emit(
            {
                "type": "draft_changed",
                "draft_id": draft_id,
                "revision": result.get("revision"),
                "draft": result,
            }
        )
        return compact_draft(result)

    @tool
    def validate_run_draft(draft_id: str) -> dict:
        """Validate a persisted draft and return capability/data errors before launch.

        Always call it before launch_optimization. Its pv_mode_scope lists the FINIs
        the draft's production version mode (FIXED or OPTIMIZED) leaves out but the
        other mode would include (only_in_other_mode), and the reverse. When either
        list is not empty, tell the user which products and why before launching.
        Its profile_status.outdated means the draft froze an older plant profile
        revision: profile edits since (e.g. a new volume matrix) do not apply to it;
        offer recreate_draft_with_current_profile.
        effective_request.config summarizes profile-owned lists (matrix_pairs,
        group_size_overrides) by counts; all check fields are complete."""
        return compact_validation(service.validate_draft(draft_id))

    @tool
    def recreate_draft_with_current_profile(draft_id: str) -> dict:
        """Create a new draft from the current plant profile revision of an outdated draft.

        Keeps the dataset, title and the scenario's own explicit choices (production
        version mode, compatibility mode, budget, scenario constraints); values the
        profile owns come from the new revision. The old draft stays unchanged.
        Returns the new draft, carried_over paths and skipped paths with reasons;
        tell the user about skipped values. Validate the new draft before launch."""
        result = service.recreate_draft_with_current_profile(draft_id)
        created = result["draft"]
        service.context(context_id, {"dataset_id": created["dataset_id"], "draft_id": created["draft_id"]})
        emit({"type": "draft_changed", "draft_id": created["draft_id"],
              "revision": created.get("revision"), "draft": created})
        return {**result, "draft": compact_draft(created)}

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
        Validate the same revision first. If its pv_mode_scope reports products the
        chosen production version mode leaves out and the user has not yet decided
        about them in this conversation, ask the user before launching.
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
        A completed run carries sap_settings_offer: after the results summary,
        offer to select a scenario and generate SAP settings, naming any blocked
        tab; do not select or generate unless the user asks.
        """
        result = await await_run(run_id)
        if str(result.get("status", "")).lower() in {"completed", "succeeded"}:
            result = {**result, "sap_settings_offer": await asyncio.to_thread(sap_offer, run_id)}
        return result

    def sap_offer(run_id: str) -> dict:
        """Compact SAP settings readiness for the end-of-run report (inputs only, no generation)."""
        reader = getattr(service, "sap_settings_readiness", None)
        if not callable(reader):
            return {"available": False}
        try:
            readiness = reader(run_id)
        except (KeyError, ValueError) as error:
            return {"available": False, "reason": str(error)}
        if not readiness.get("available"):
            return readiness
        return {
            "available": True,
            "selection": readiness.get("selection"),
            "blocked_inputs": [{"tab": gap["tab"], "message": gap["message"], "fix": gap["fix"]}
                               for gap in readiness.get("gaps", []) if gap["status"] == "blocked"],
            "suggestions": [{"values": item.get("values"), "dataset_name": item.get("dataset_name")}
                            for item in readiness.get("suggestions", []) if item.get("status") == "suggested"],
            "hint": "Offer: select a scenario (Balanced B by default) and generate SAP settings.",
        }

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
        Returns: public_error; block failures from the saved checkpoint;
        worker_failure (the solve process's own report: failure_type,
        exception_type, message, stage, innermost traceback frames and, for
        failure_type profile_matrix_incomplete, matrix_coverage with the missing
        package volumes and product counts); worker_log_tail (last lines the solve
        process printed, including how it exited). When worker_report_available
        and worker_log_available are both false, the run predates saved failure
        evidence: call recheck_run_inputs before concluding the cause is unknown.
        """
        return service.run_failure_diagnostics(run_id)

    @tool
    def recheck_run_inputs(run_id: str) -> dict:
        """Re-run the pre-launch input checks on one run's frozen configuration.

        Use for a failed run without saved failure evidence (older runs), or to
        confirm the inputs of a run are still valid. Read-only; takes several
        seconds. Returns valid, errors (planner-readable), matrix_coverage
        (profile matrix volumes versus the modeled package volumes, with the
        missing volumes and how many products use each) and counts."""
        return service.recheck_run_inputs(run_id)

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
        get_plant_profile_details,
        propose_profile_constraint,
        propose_profile_settings,
        get_dataset_volume_coverage,
        get_pending_import,
        propose_extraction_settings,
        list_datasets,
        inspect_dataset,
        explain_dataset_finding,
        get_optimizer_reference,
        update_run_draft,
        validate_run_draft,
        recreate_draft_with_current_profile,
        launch_optimization,
        list_runs,
        get_run_status,
        wait_for_run,
        get_run_configuration,
        get_run_failure_diagnostics,
        recheck_run_inputs,
        get_run_matrix_page,
        cancel_run,
        get_run_results,
        query_optimizer_data,
        get_solution_details,
        compare_solutions,
        explain_assignment,
        *sap_tools(service, context_id, emit),
    ]
