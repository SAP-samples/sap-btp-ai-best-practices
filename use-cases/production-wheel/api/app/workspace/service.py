"""Single application facade used by the UI, typed agent tools and CLI."""

from __future__ import annotations

import copy
import threading

from .datasets_service import DatasetService
from .models import QuerySpec
from .removal import RemovalService
from .runs_service import RunService
from .plant_profiles import PlantProfileService
from .ai_model_settings import AIModelSettingsService


class WorkspaceService(DatasetService, RunService, RemovalService, PlantProfileService, AIModelSettingsService):
    """Compose independently testable dataset, run, query and context services."""

    def __init__(self, repository):
        """Bind durable repository; tests explicitly inject MemoryRepository."""
        self.repo = repository
        self._session_contexts = {}
        self._session_context_lock = threading.RLock()

    def capabilities(self):
        """Describe actual compiled constraints plus relational analytical views."""
        from production_wheel.constraints.compiler import DEFERRED_CONSTRAINT_KINDS
        from production_wheel.validation import get_capabilities

        from .models import INPUT_VIEWS, RESULT_VIEWS, ExecutionBudget

        contract = get_capabilities()
        contract.update(
            workspace_modes=["PARETO"],
            input_views=sorted(INPUT_VIEWS),
            result_views=sorted(RESULT_VIEWS),
            deferred_constraints=DEFERRED_CONSTRAINT_KINDS,
            budget_defaults=ExecutionBudget().model_dump(),
            accepts_raw_sql=False,
            accepts_raw_code=False,
            line_identifiers="plant-scoped strings",
            query_contract=QuerySpec.model_json_schema(),
        )
        return contract

    def query(self, spec):
        """Execute a bounded semantic query without code generation."""
        from .analytics import execute_query

        spec = QuerySpec.model_validate(spec)
        from .responses import AnalyticalResponse

        if spec.dataset_id:
            source = self.inspect_dataset(spec.dataset_id)
            basis = source["metadata"].get("settings", {})
        else:
            source = self.get_run(spec.run_id)
            if not source.get("results_ready"):
                raise ValueError("results are not available; inspect run status")
            basis = source.get("metadata", {}).get("config") or source.get(
                "request", {}
            ).get("config", {})
        result = execute_query(self.repo, spec)
        result["evidence"]["calculation_basis"] = {
            key: basis.get(key)
            for key in (
                "coverage_basis",
                "pallet_formula",
                "demand_days",
                "productive_weeks",
                "canonical_factor",
            )
        }
        result["evidence"]["source_version"] = source.get("dataset_id")
        result["evidence"]["parameter_sources"] = source.get(
            "parameter_sources"
        ) or source.get("metadata", {}).get("parameter_sources", {})
        return AnalyticalResponse.model_validate(result).model_dump()

    def compare(self, left, right):
        """Compare solution evidence with population and configuration caveats."""
        from .analytics import compare_solutions

        return compare_solutions(self.repo, left, right)

    def explain(self, run_id, point_index, material=None, group_id=None):
        """Retrieve assignment evidence and tested formula derivations."""
        from .analytics import explain_assignment

        return explain_assignment(self.repo, run_id, point_index, material, group_id)

    def reference(self, topic):
        """Retrieve a relevant versioned reference section and metric code provenance."""
        from .reference import reference

        return reference(topic)

    def context(self, context_id, patch=None):
        """Keep selected IDs only for this API process and page-session key."""
        if len(context_id) > 64:
            raise ValueError("context_id maximum length is 64")
        if patch is not None:
            if set(patch) - {"dataset_id", "draft_id", "run_id", "point_index", "plant_profile_id"}:
                raise ValueError("unknown context field")
        with self._session_context_lock:
            value = self._session_contexts.setdefault(
                context_id, {"context_id": context_id}
            )
            if patch:
                value.update(copy.deepcopy(patch))
            return copy.deepcopy(value)
