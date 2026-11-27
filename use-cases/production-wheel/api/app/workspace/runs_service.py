"""Revisioned manual/agent drafts, finite request validation, and durable jobs."""

from __future__ import annotations

import uuid

from production_wheel.schemas import SolveRequest

from .datasets_service import digest, now
from .discovery import filter_records
from .models import ExecutionBudget, RunDraft, RunSubmission
from .plant_profiles import PROFILE_DEFAULT_KEYS, optimizer_view, profile_config, check_profile_source
from .pv_mode_scope import pv_mode_differences


def merge_dict(original, patch):
    """Merge object fields while replacing list-valued scope and constraints."""
    value = dict(original)
    for key, item in patch.items():
        value[key] = (
            merge_dict(value[key], item)
            if isinstance(item, dict) and isinstance(value.get(key), dict)
            else item
        )
    return value


def supplied_paths(value, prefix=""):
    """Yield explicit leaf paths, treating lists and empty objects as one value."""
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            yield from supplied_paths(item, path)
        else:
            yield path


_MISSING = object()

# Profile-owned config keys added after drafts were first stored, with the value an
# older draft implicitly used (RunConfig defaults, JSON form).
LEGACY_PROFILE_KEY_DEFAULTS = {"volume_matching": "exact", "matrix_volume_classes": []}


def _path_value(record, path):
    """Return the value at a dotted path of a nested dict, or ``_MISSING``."""
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _path_patch(path, value):
    """Build the nested draft patch that sets one dotted path to ``value``."""
    patch = value
    for key in reversed(path.split(".")):
        patch = {key: patch}
    return patch


class RunService:
    """Shared run behavior; no tool or UI path can bypass these validations."""

    def create_draft(self, dataset_id, plant_profile_id=None, title=""):
        """Create a draft using published snapshot parameters and pilot defaults."""
        dataset = self.inspect_dataset(dataset_id)
        if dataset.get("removed_at"):
            raise ValueError("This snapshot was previously removed; import its workbook as a new snapshot")
        if dataset["status"] != "published":
            raise ValueError("publish the dataset before optimizing")
        settings = dataset["metadata"].get("settings", {})
        config = {
            "scenario_id": "workspace",
            "coverage_mode": "PARETO",
            "coverage_basis": "BASE_GROUP",
            "pallet_formula": "MINIMUM_ONLY",
            "pv_mode": "FIXED",
            "matrix_mode": "FLEXIBLE",
            "versions": {"matrix_version": "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"},
        }
        config.update(
            {
                k: settings[k]
                for k in ("demand_days", "productive_weeks", "canonical_factor")
                if k in settings
            }
        )
        profile = self.get_plant_profile(plant_profile_id) if plant_profile_id else None
        if profile:
            rows = self.repo.rows(dataset_id, "fini_master")
            check_profile_source(profile, rows)
            config.update(profile_config(profile, rows))
        draft = RunDraft(
            draft_id=uuid.uuid4().hex,
            dataset_id=dataset_id,
            title=title or "",
            plant_profile_id=plant_profile_id,
            plant_profile=profile,
            request=SolveRequest(config=config, scope=[{"plant": profile["plant"]}] if profile else []),
        ).model_dump(mode="json")
        draft["created_at"] = now()
        draft["parameter_sources"] = {
            path: {"origin": "default"}
            for path in supplied_paths(
                {"request": draft["request"], "budget": draft["budget"]}
            )
        }
        for key in ("demand_days", "productive_weeks", "canonical_factor"):
            if key in settings:
                draft["parameter_sources"][f"request.config.{key}"] = {
                    "origin": "dataset",
                    "dataset_id": dataset_id,
                    "dataset_parameter_origin": dataset["metadata"]
                    .get("parameter_sources", {})
                    .get(key, "default"),
                }
        if profile:
            for key in profile_config(profile, rows):
                draft["parameter_sources"][f"request.config.{key}"] = {
                    "origin": "plant_profile", "profile_id": profile["profile_id"],
                    "revision": profile["revision"],
                }
        # Serialize draft creation with snapshot deletion to prevent orphan drafts.
        with self.repo.transaction():
            if profile:
                active = self.repo.lock_entity("profiles", profile["profile_id"])
                if active["status"] != "active" or active["revision"] != profile["revision"]:
                    raise ValueError("profile revision changed or was removed; select the profile again")
            current = self.repo.get("datasets", dataset_id)
            self.repo.cas("datasets", dataset_id, current["revision"], {})
            self.repo.insert("drafts", draft["draft_id"], draft)
        return draft

    def get_draft(self, draft_id):
        """Read the authoritative draft revision shown by dashboard and chat."""
        return self.repo.get("drafts", draft_id)

    def draft_profile_status(self, draft):
        """Compare the plant profile frozen in a draft with the profile's current revision.

        A draft keeps the profile revision it was created with, so profile edits
        made afterwards (settings, rules, volume matrix) do not reach it.

        Args:
            draft: Draft record, or a draft id.

        Returns:
            ``None`` for a draft without a profile; otherwise ``profile_id``,
            ``profile_name``, ``draft_profile_revision``, ``current_profile_revision``
            (``None`` when the profile no longer exists), ``removed`` and
            ``outdated`` (a newer active revision changes optimization inputs), plus
            ``message`` when outdated or removed, or ``note`` when the newer
            revision only changed SAP settings rules (not outdated).
        """
        if isinstance(draft, str):
            draft = self.get_draft(draft)
        frozen = draft.get("plant_profile") or {}
        profile_id = draft.get("plant_profile_id")
        if not profile_id or not frozen:
            return None
        try:
            current = self.repo.get("profiles", profile_id)
        except KeyError:
            current = None
        removed = current is None or current.get("status") != "active"
        draft_revision = frozen.get("revision")
        current_revision = None if current is None else current.get("revision")
        outdated = not removed and current_revision != draft_revision
        sap_only = False
        if outdated:
            # SAP settings rules do not change optimization results and are read live
            # when SAP settings are generated, so a revision that only edits them
            # leaves the draft current.
            latest = self.get_plant_profile(profile_id)
            sap_only = optimizer_view(latest) == optimizer_view(frozen)
            outdated = not sap_only
        status = {
            "profile_id": profile_id,
            "profile_name": frozen.get("name"),
            "draft_profile_revision": draft_revision,
            "current_profile_revision": current_revision,
            "removed": removed,
            "outdated": outdated,
        }
        if outdated:
            status["message"] = (
                f"This draft uses plant profile \"{frozen.get('name')}\" revision {draft_revision}; "
                f"the profile is now at revision {current_revision}. Profile changes made since "
                "(settings, rules, volume matrix) do not apply to this draft. Recreate the draft "
                "to use the current profile."
            )
        elif removed:
            status["message"] = (
                f"Plant profile \"{frozen.get('name')}\" was removed; select an active profile."
            )
        elif sap_only:
            status["note"] = (
                f"Plant profile \"{frozen.get('name')}\" changed only its SAP settings since this draft "
                f"(revision {draft_revision} -> {current_revision}); they are read live when SAP settings "
                "are generated, so the draft needs no update."
            )
        return status

    def recreate_draft_with_current_profile(self, draft_id):
        """Create a new draft from the current profile revision, keeping the scenario's own choices.

        The old draft is left unchanged (runs may reference it). Values the user
        set explicitly on the old draft (``parameter_sources`` origin ``explicit``,
        e.g. production version mode, compatibility mode, budget, scenario
        constraints) are re-applied one by one; a value the current profile no
        longer accepts is skipped and reported.

        Args:
            draft_id: Draft whose dataset, profile and title are reused.

        Returns:
            ``{"draft", "previous_draft_id", "profile_status", "carried_over", "skipped"}``
            where ``skipped`` lists ``{"path", "reason"}``.

        Raises:
            ValueError: the draft has no plant profile or the profile was removed.
        """
        previous = self.get_draft(draft_id)
        status = self.draft_profile_status(previous)
        if status is None:
            raise ValueError("this draft has no plant profile to refresh")
        if status["removed"]:
            raise ValueError(status["message"])
        created = self.create_draft(previous["dataset_id"], previous["plant_profile_id"], previous.get("title", ""))
        # Values owned by the profile (matrix pairs, runner thresholds, rules, ...)
        # must come from the new revision; the UI's Save draft marks them explicit too.
        profile_keys = set(profile_config(created["plant_profile"], [])) - PROFILE_DEFAULT_KEYS
        changed = {}
        for path, source in sorted((previous.get("parameter_sources") or {}).items()):
            parts = path.split(".")
            if not isinstance(source, dict) or source.get("origin") != "explicit":
                continue
            if parts[0] not in ("request", "budget"):
                continue
            if parts[:2] == ["request", "config"] and len(parts) > 2 and parts[2] in profile_keys:
                continue
            value = _path_value(previous, path)
            if value is not _MISSING and value != _path_value(created, path):
                changed[path] = value
        skipped = []
        if changed:
            patch = {}
            for path, value in changed.items():
                patch = merge_dict(patch, _path_patch(path, value))
            try:
                created = self.update_draft(created["draft_id"], created["revision"], patch)
            except ValueError:
                # One value is no longer accepted: apply them one by one and report it.
                for path, value in changed.items():
                    try:
                        created = self.update_draft(created["draft_id"], created["revision"], _path_patch(path, value))
                    except ValueError as exc:
                        skipped.append({"path": path, "reason": str(exc)})
        return {
            "draft": created,
            "previous_draft_id": draft_id,
            "profile_status": self.draft_profile_status(created),
            "carried_over": sorted(set(changed) - {item["path"] for item in skipped}),
            "skipped": skipped,
        }

    def update_draft(self, draft_id, revision, patch):
        """Apply a typed patch without overwriting concurrently edited drafts."""
        current = self.get_draft(draft_id)
        if revision != current["revision"]:
            raise ValueError("stale revision; reload current draft")
        if set(patch) - {"dataset_id", "request", "budget", "title"}:
            raise ValueError("unknown draft field")
        merged = merge_dict(
            {
                k: current[k]
                for k in ("draft_id", "dataset_id", "revision", "request", "budget", "title", "plant_profile_id", "plant_profile") if k in current
            },
            patch,
        )
        selected = self.inspect_dataset(merged["dataset_id"])
        if selected["status"] != "published":
            raise ValueError("dataset must be published")
        sources = dict(current.get("parameter_sources", {}))
        explicit_paths = set(supplied_paths(patch))
        if merged["dataset_id"] != current["dataset_id"]:
            for key, fallback in (
                ("demand_days", 250),
                ("productive_weeks", 50),
                ("canonical_factor", 0.9),
            ):
                path = f"request.config.{key}"
                prior = sources.get(path, {})
                if path not in explicit_paths and (
                    not isinstance(prior, dict) or prior.get("origin") != "explicit"
                ):
                    merged["request"]["config"][key] = (
                        selected["metadata"].get("settings", {}).get(key, fallback)
                    )
                    sources[path] = {
                        "origin": "dataset",
                        "dataset_id": selected["dataset_id"],
                        "dataset_parameter_origin": selected["metadata"]
                        .get("parameter_sources", {})
                        .get(key, "default"),
                    }
        sources.update({path: {"origin": "explicit"} for path in explicit_paths})
        self._apply_profile(merged, reject_overrides=True)
        value = RunDraft.model_validate(merged).model_dump(mode="json")
        return self.repo.cas(
            "drafts",
            draft_id,
            revision,
            {
                **value,
                "parameter_sources": sources,
            },
        )

    def _apply_profile(self, draft, reject_overrides=False):
        """Apply only the frozen snapshot; reject attempts to alter its fixed controls."""
        profile = draft.get("plant_profile")
        if not profile:
            return
        rows = self.repo.rows(draft["dataset_id"], "fini_master")
        check_profile_source(profile, rows)
        fixed = {
            key: value for key, value in profile_config(profile, rows).items()
            if key not in PROFILE_DEFAULT_KEYS
        }
        config = draft["request"]["config"]
        if reject_overrides:
            for key, value in fixed.items():
                if key == "assign_filling_lines" and not value:
                    continue
                # Drafts saved before volume classes lack these keys; read them as the
                # RunConfig defaults (exact matching, no class map).
                current = config.get(key, LEGACY_PROFILE_KEY_DEFAULTS.get(key))
                if current != value:
                    raise ValueError(f"{key} is fixed by the plant profile; change the profile instead")
        fixed["assign_filling_lines"] = fixed["assign_filling_lines"] or config.get("assign_filling_lines", False)
        config.update(fixed)
        scope = draft["request"].get("scope", [])
        if any(item.get("plant") != profile["plant"] for item in scope):
            raise ValueError("scenario plant scope contradicts the plant profile")
        draft["request"]["scope"] = scope or [{"plant": profile["plant"]}]
        fixed_ids = {r["constraint_id"] for r in profile["rules"]}
        if any(r.get("constraint_id") in fixed_ids for r in draft["request"].get("constraints", [])):
            raise ValueError("scenario constraint cannot replace a fixed profile rule")

    def compiled_draft(self, draft):
        """Validate and compile a finite request against actual snapshot identifiers."""
        from production_wheel.constraints import compile_request
        from production_wheel.scenarios import canonical_inputs_from_tables
        from production_wheel.validation import ValidationContext, validate_request

        import copy
        draft = copy.deepcopy(draft)
        if draft.get("source_request") is not None:
            draft["request"] = draft["source_request"]
        self._apply_profile(draft)
        request = SolveRequest.model_validate(draft["request"])
        if request.config.coverage_mode.value != "PARETO":
            raise ValueError("workspace runs support PARETO only")
        inputs = canonical_inputs_from_tables(
            self.input_tables(
                draft["dataset_id"], ("fini_master", "production_versions")
            )
        )
        members = inputs.members_for(request.config)
        versions = {}
        for version in inputs.production_versions:
            versions.setdefault((version.plant, version.sefi), set()).add(version.pv_id)
        context = ValidationContext(
            materials=frozenset(m.fini_id for m in members),
            production_versions_by_block=versions,
            filling_lines=frozenset(line for m in members for line in m.eligible_lines),
        )
        result = validate_request(request, context)
        if not result.is_valid:
            raise ValueError("; ".join(message.message for message in result.messages))
        compiled = compile_request(request, inputs)
        if draft.get("plant_profile"):
            fixed = profile_config(draft["plant_profile"], self.repo.rows(draft["dataset_id"], "fini_master"))
            effective = compiled.config.model_dump(mode="json")
            for key, value in fixed.items():
                if key in PROFILE_DEFAULT_KEYS or (key == "assign_filling_lines" and not value):
                    continue
                if key == "group_size_overrides":
                    original_caps = {(row["plant"], row["sefi"]): row["maximum"] for row in value}
                    effective_caps = {(row["plant"], row["sefi"]): row["maximum"] for row in effective[key]}
                    if original_caps.keys() == effective_caps.keys() and all(effective_caps[block] <= cap for block, cap in original_caps.items()):
                        continue
                if key != "business_rules" and effective.get(key) != value:
                    raise ValueError(f"scenario constraints contradict fixed profile {key}")
        if not compiled.inputs.members_for(compiled.config):
            raise ValueError("scope contains no eligible FINIs")
        return compiled

    def pv_mode_scope(self, draft):
        """Compare the draft's production version mode with the other mode.

        Args:
            draft: Persisted draft with ``dataset_id`` and ``request`` (config.pv_mode, scope).
        Returns:
            ``pv_mode_scope.pv_mode_differences`` for the draft's dataset and scope.
        """
        request = draft.get("source_request") or draft["request"]
        rows = self.repo.rows(draft["dataset_id"], "fini_master")
        return pv_mode_differences(rows, request.get("scope") or [], request.get("config", {}).get("pv_mode", "FIXED"))

    def validate_draft(self, draft_id):
        """Report effective request, modeled population, warnings, and blocking errors."""
        draft = self.get_draft(draft_id)
        # Products the chosen production version mode leaves out but the other mode
        # would include; the agent must tell the user before launching.
        pv_scope = self.pv_mode_scope(draft)
        # Warn when the profile changed after the draft was created; its frozen
        # copy (and so this validation) does not include those changes.
        profile_status = self.draft_profile_status(draft)
        profile_warnings = [profile_status["message"]] if profile_status and profile_status.get("message") else []
        try:
            compiled, members, coverage = self.checked_inputs(draft)
            return {
                "valid": True,
                "errors": [],
                "warnings": [
                    *profile_warnings,
                    *pv_scope["warnings"],
                    "Per-block solver seconds do not guarantee total elapsed time.",
                    "Business acceptance is not automatically assessed.",
                ],
                "pv_mode_scope": pv_scope,
                "effective_request": {
                    **draft["request"],
                    "config": compiled.config.model_dump(mode="json"),
                },
                "counts": {
                    "modeled_fini_count": len(members),
                    "block_count": len({m.block_key for m in members}),
                },
                "applied_constraints": list(compiled.applied),
                "matrix_coverage": coverage,
                "profile_status": profile_status,
                "revision": draft["revision"],
            }
        except ValueError as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [*profile_warnings, *pv_scope["warnings"]],
                "pv_mode_scope": pv_scope,
                "effective_request": draft["request"],
                "counts": {},
                "profile_status": profile_status,
                # Present only when the profile matrix misses modeled volumes.
                "matrix_coverage": getattr(exc, "coverage", None),
                "revision": draft["revision"],
            }

    def checked_inputs(self, record):
        """Compile a draft or run record and verify its profile matrix covers the population.

        Args:
            record: Draft or run record accepted by ``compiled_draft``.

        Returns:
            ``(compiled, members, coverage)`` where ``coverage`` is the
            ``profile_matrix_coverage`` summary (``None`` without a profile matrix).

        Raises:
            ProfileMatrixCoverageError: modeled package volumes are missing from
                the profile matrix (the solve would stop on them).
            ValueError: any other compile error of the request.
        """
        from production_wheel.matrix import ProfileMatrixCoverageError, profile_matrix_coverage

        compiled = self.compiled_draft(record)
        members = compiled.inputs.members_for(compiled.config)
        coverage = profile_matrix_coverage(compiled.config, members)
        if coverage and not coverage["complete"]:
            raise ProfileMatrixCoverageError(coverage)
        return compiled, members, coverage

    def recheck_run_inputs(self, run_id):
        """Re-run the pre-launch input checks against one run's frozen configuration.

        Used to explain failed runs that predate saved failure reports. Reads only;
        the run record is never changed. Takes several seconds on a full dataset.

        Args:
            run_id: Existing run identifier.

        Returns:
            ``{"run_id", "valid", "errors", "matrix_coverage", "counts"}``.
        """
        run = self.get_run(run_id)
        try:
            _, members, coverage = self.checked_inputs(run)
            return {"run_id": run_id, "valid": True, "errors": [], "matrix_coverage": coverage,
                    "counts": {"modeled_fini_count": len(members),
                               "block_count": len({m.block_key for m in members})}}
        except ValueError as exc:
            return {"run_id": run_id, "valid": False, "errors": [str(exc)],
                    "matrix_coverage": getattr(exc, "coverage", None), "counts": {}}

    def submit(self, submission):
        """Register a queued job before execution, deduplicating exact launch retries."""
        submission = RunSubmission.model_validate(submission)
        run_id = digest(
            {"draft": submission.draft_id, "key": submission.idempotency_key}
        )[:32]
        try:
            previous = self.get_run(run_id)
            if (
                previous["draft_revision"] != submission.revision
                or previous.get("parent_run_id") != submission.parent_run_id
            ):
                raise ValueError(
                    "idempotency key already used for a different submission"
                )
            return previous
        except KeyError:
            pass
        with self.repo.transaction():
            draft = self.get_draft(submission.draft_id)
            if draft.get("plant_profile_id"):
                profile = self.repo.lock_entity("profiles", draft["plant_profile_id"])
                if profile["status"] != "active":
                    raise ValueError("plant profile was removed; select an active profile")
            dataset = self.repo.get("datasets", draft["dataset_id"])
            if dataset.get("removed_at"):
                raise ValueError(
                    "This snapshot was previously removed; import its workbook as a new snapshot"
                )
            # Lock visibility through publication so snapshot removal cannot race launch.
            self.repo.cas("datasets", draft["dataset_id"], dataset["revision"], {})
            if draft["revision"] != submission.revision:
                raise ValueError("stale revision; validate current draft")
            validation = self.validate_draft(submission.draft_id)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            if submission.parent_run_id:
                self.get_run(submission.parent_run_id)
            # CAS locks the draft revision across concurrent launch/edit transactions.
            self.repo.cas(
                "drafts",
                submission.draft_id,
                submission.revision,
                {"last_submitted_at": now()},
            )
            value = {
                "run_id": run_id,
                "revision": 1,
                "dataset_id": draft["dataset_id"],
                "draft_id": submission.draft_id,
                "title": draft.get("title", ""),
                "plant_profile_id": draft.get("plant_profile_id"),
                "plant_profile": draft.get("plant_profile"),
                "plant_profile_revision": (draft.get("plant_profile") or {}).get("revision"),
                "draft_revision": submission.revision,
                "request": validation["effective_request"],
                "source_request": draft["request"],
                "budget": ExecutionBudget.model_validate(draft["budget"]).model_dump(),
                "parent_run_id": submission.parent_run_id,
                "status": "queued",
                "stage": "queued",
                "created_at": now(),
                "results_ready": False,
                "cancel_requested": False,
                "worker_id": None,
                "parameter_sources": draft.get("parameter_sources", {}),
                "validation": validation,
                "metadata": {},
            }
            self.repo.insert("runs", run_id, value)
        return value

    def submit_summary(self, submission):
        """Create or reuse a durable run, returning only its compact public status."""
        return self.run_status(self.submit(submission)["run_id"])

    def list_runs(self, filters=None, include_removed=False):
        """List runs by dataset, lifecycle, parent and inclusive ISO creation bounds."""
        return filter_records(
            [
                row
                for row in self.repo.list("runs")
                if include_removed or not row.get("removed_at")
            ],
            filters,
            "runs",
        )

    def get_run(self, run_id):
        """Read durable state; never probe another worker's local PID."""
        return self.repo.get("runs", run_id)

    def list_run_summaries(self, filters=None, include_removed=False):
        """Return compact public run cards while preserving raw records for workers."""
        from .run_views import run_summary

        return [
            run_summary(run)
            for run in self.list_runs(filters, include_removed=include_removed)
        ]

    def run_status(self, run_id):
        """Return one compact lifecycle/configuration view for UI and agent reads."""
        from .run_views import run_status

        return run_status(self.get_run(run_id))

    def run_failure_diagnostics(self, run_id):
        """Return bounded failure evidence without exposing raw artifacts.

        Combines the publication checkpoint (block failures), the solve process
        failure report and the saved solve log tail, whichever exist.
        """
        from .run_failures import LOG_TAIL_ARTIFACT, read_artifact, read_failure_report
        from .run_views import failure_diagnostics

        run = self.get_run(run_id)
        try:
            checkpoint = self.repo.artifact(run_id, "publication-checkpoint.json")
        except KeyError:
            checkpoint = None
        return failure_diagnostics(
            run,
            checkpoint,
            failure_report=read_failure_report(self, run_id),
            log_tail=read_artifact(self, run_id, LOG_TAIL_ARTIFACT),
        )

    def run_matrix_page(self, run_id, offset=0, limit=50, status=None):
        """Return one filtered page of the run's frozen compatibility matrix."""
        from .run_views import matrix_page

        return matrix_page(
            self.get_run(run_id), offset=offset, limit=limit, status=status
        )

    def patch_run(self, run_id, changes, worker_id=None):
        """Retry an optimistic status patch when progress/cancellation races it."""
        for _ in range(5):
            run = self.get_run(run_id)
            if worker_id is not None and (
                run.get("worker_id") != worker_id
                or run["status"] not in ("running", "persisting")
            ):
                raise ValueError("worker lease no longer owns active run")
            try:
                return self.repo.cas("runs", run_id, run["revision"], changes)
            except ValueError:
                continue
        raise ValueError("run state changed concurrently; retry")

    def cancel_run(self, run_id):
        """Request cancellation without rewriting a completed result."""
        for _ in range(5):
            run = self.get_run(run_id)
            if run.get("results_ready") or run["status"] in (
                "completed",
                "failed",
                "cancelled",
                "worker_lost",
            ):
                return run
            changes = {"cancel_requested": True}
            if run["status"] == "queued":
                changes.update(status="cancelled", stage="cancelled", finished_at=now())
            try:
                return self.repo.cas("runs", run_id, run["revision"], changes)
            except ValueError:
                continue
        raise ValueError("run changed concurrently; retry cancellation")

    def finish_run(self, run_id, status, stage, error=None):
        """Finish owned work without overwriting published results or cancellation."""
        for _ in range(5):
            run = self.get_run(run_id)
            if run.get("results_ready") or run["status"] in (
                "completed",
                "cancelled",
                "worker_lost",
            ):
                return run
            try:
                return self.repo.cas(
                    "runs",
                    run_id,
                    run["revision"],
                    {
                        "status": status,
                        "stage": stage,
                        "error": error,
                        "finished_at": now(),
                    },
                )
            except ValueError:
                continue
        raise ValueError("run changed concurrently; retry finalization")

    def results(self, run_id):
        """Expose results only after an atomic, verified relational publication."""
        run = self.get_run(run_id)
        if not run.get("results_ready"):
            raise ValueError("results are not available; inspect run status")
        return {
            "run": run,
            "points": self.repo.rows(run_id, "solutions"),
            "metadata": run["metadata"],
        }

    def run_results_view(self, run_id):
        """Return public completed-result data without the durable replay snapshot."""
        from .run_views import run_results_view

        return run_results_view(self.results(run_id))

    def publish_results(self, run_id, bundle, worker_id=None):
        """Publish all structured tables/artifacts and readiness in one transaction."""
        if (
            not bundle.metadata.get("integrity_valid")
            or not bundle.metadata.get("complete")
            or not bundle.tables.get("solutions")
        ):
            raise ValueError(
                "result bundle is incomplete or failed independent validation"
            )
        with self.repo.transaction():
            run = self.get_run(run_id)
            if run.get("results_ready"):
                return run
            if run["status"] not in ("running", "persisting") or run.get(
                "cancel_requested"
            ):
                raise ValueError("run is not eligible for result publication")
            if run.get("worker_id") and run.get("worker_id") != worker_id:
                raise ValueError("worker lease no longer owns active run")
            self.repo.cas(
                "runs",
                run_id,
                run["revision"],
                {"status": "persisting", "stage": "publishing"},
            )
            self.repo.replace_tables(run_id, bundle.tables)
            for name, text in bundle.artifacts.items():
                self.repo.put_artifact(run_id, name, text.encode())
            return self.patch_run(
                run_id,
                {
                    "status": "completed",
                    "stage": "completed",
                    "results_ready": True,
                    "finished_at": now(),
                    "metadata": bundle.metadata,
                    "error": None,
                },
            )
