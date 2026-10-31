"""Revisioned manual/agent drafts, finite request validation, and durable jobs."""

from __future__ import annotations

import uuid

from production_wheel.schemas import SolveRequest

from .datasets_service import digest, now
from .discovery import filter_records
from .models import ExecutionBudget, RunDraft, RunSubmission
from .plant_profiles import profile_config, check_profile_source


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
        fixed = profile_config(profile, rows)
        config = draft["request"]["config"]
        if reject_overrides:
            for key, value in fixed.items():
                if key == "assign_filling_lines" and not value:
                    continue
                if config.get(key) != value:
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
                if key == "assign_filling_lines" and not value:
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

    def validate_draft(self, draft_id):
        """Report effective request, modeled population, warnings, and blocking errors."""
        draft = self.get_draft(draft_id)
        try:
            compiled = self.compiled_draft(draft)
            members = compiled.inputs.members_for(compiled.config)
            return {
                "valid": True,
                "errors": [],
                "warnings": [
                    "Per-block solver seconds do not guarantee total elapsed time.",
                    "Business acceptance is not automatically assessed.",
                ],
                "effective_request": {
                    **draft["request"],
                    "config": compiled.config.model_dump(mode="json"),
                },
                "counts": {
                    "modeled_fini_count": len(members),
                    "block_count": len({m.block_key for m in members}),
                },
                "applied_constraints": list(compiled.applied),
                "revision": draft["revision"],
            }
        except ValueError as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
                "effective_request": draft["request"],
                "counts": {},
                "revision": draft["revision"],
            }

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
        """Return bounded checkpoint diagnostics without exposing the checkpoint blob."""
        from .run_views import failure_diagnostics

        run = self.get_run(run_id)
        try:
            checkpoint = self.repo.artifact(run_id, "publication-checkpoint.json")
        except KeyError:
            checkpoint = None
        return failure_diagnostics(run, checkpoint)

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
