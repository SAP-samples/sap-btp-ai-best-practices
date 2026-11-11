"""Typed public contracts shared by HTTP, manual controls, workers, and tools."""

from __future__ import annotations

from typing import Any, Literal

from production_wheel.schemas import SolveRequest
from pydantic import BaseModel, ConfigDict, Field, model_validator

INPUT_VIEWS = frozenset(
    "fini_master production_versions fini_line_eligibility pck_line_eligibility package_volume_catalog pallet_resolution enrichment_pallet_candidates validation_issues field_dictionary site_parameters baseline_assignments baseline_group_metrics baseline_groups_5_june legacy_fini_roc_by_pv sap_character_catalog sap_character_capacity sap_character_compatibility fini_new_recipe_combination fini_cycle_stock_at_dc fini_cycle_stock_without_pallet_conversion planning_calendar_inform planning_calendar_no_calendar apo_sefi_lot_size query_catalog external_dependencies sales_forecast_summary legacy_kpi line_corrections cycle_stock_lot_size_01 legacy_abc_cycle_stock legacy_average_batch_size legacy_baseline_groups legacy_changeover_estimation legacy_recipe_alternatives legacy_sales_forecast_summary legacy_solution_summary recipe_characters sefi_filling_line_evidence sefi_zone selling_dcs_per_material".split()
)
RESULT_VIEWS = frozenset(
    "solutions groups members pools constraints validation matrix_pairs block_options block_audits block_sessions block_failures global_audits option_groups option_members point_options".split()
)


class StrictModel(BaseModel):
    """Reject misspelled or unsupported interface arguments."""

    model_config = ConfigDict(extra="forbid")


class ExecutionBudget(StrictModel):
    """Bounded solver controls; seconds are per block unless stated otherwise."""

    frontier_points: int = Field(default=17, ge=2, le=100)
    block_options_per_block: int = Field(default=5, ge=2, le=100)
    per_block_total_seconds: float = Field(default=120, gt=0, le=86400)
    block_workers: int = Field(default=2, ge=1, le=16)
    global_epsilon_exponent: float = Field(default=2, ge=1, le=10)
    wall_time_seconds: int = Field(default=3600, ge=10, le=172800)


class DatasetSnapshot(StrictModel):
    """Immutable source version with separately controlled publication state."""

    dataset_id: str
    name: str
    status: Literal["review", "published", "invalid"]
    metadata: dict[str, Any]
    issues: list[dict[str, Any]] = Field(default_factory=list)


class RunDraft(StrictModel):
    """Revisioned request shared by a manual editor and an agent conversation."""

    draft_id: str
    dataset_id: str
    revision: int = 1
    request: SolveRequest
    title: str = Field(default="", max_length=255)
    plant_profile_id: str | None = None
    plant_profile: dict[str, Any] | None = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)


class RunSubmission(StrictModel):
    """Idempotent launch of one exact draft revision."""

    draft_id: str
    revision: int
    idempotency_key: str = Field(min_length=1, max_length=128)
    parent_run_id: str | None = None


class QueryFilter(StrictModel):
    """Finite predicate vocabulary applied to a registered relational field."""

    field: str
    op: Literal["eq", "ne", "contains", "gt", "gte", "lt", "lte", "in", "is_null"] = (
        "eq"
    )
    value: Any = None


class QueryMetric(StrictModel):
    """Server-calculated aggregate, with explicit weights for weighted means."""

    field: str = "*"
    operation: Literal[
        "count", "sum", "mean", "min", "max", "median", "p90", "weighted_mean"
    ]
    alias: str | None = None
    weight_field: str | None = None


class QuerySort(StrictModel):
    """One ordering term; null values consistently sort last."""

    field: str
    direction: Literal["asc", "desc"] = "asc"


class QuerySpec(StrictModel):
    """Read-only semantic query; exposes neither SQL nor arbitrary code."""

    view: str
    dataset_id: str | None = None
    run_id: str | None = None
    point_index: int | None = Field(default=None, ge=1)
    fields: list[str] = Field(default_factory=list)
    filters: list[QueryFilter] = Field(default_factory=list, max_length=30)
    group_by: list[str] = Field(default_factory=list, max_length=5)
    metrics: list[QueryMetric] = Field(default_factory=list, max_length=20)
    sort: list[QuerySort] = Field(default_factory=list, max_length=5)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

    @model_validator(mode="after")
    def validate_scope(self):
        """Require the matching scope and a known input or result view."""
        if self.view not in INPUT_VIEWS | RESULT_VIEWS:
            raise ValueError("unknown analytical view")
        if self.view in INPUT_VIEWS and (not self.dataset_id or self.run_id):
            raise ValueError("input views require dataset_id only")
        if self.view in RESULT_VIEWS and (not self.run_id or self.dataset_id):
            raise ValueError("result views require run_id only")
        return self
