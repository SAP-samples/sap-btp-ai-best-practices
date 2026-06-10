"""Pydantic models for the Credit Optimizer API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessStatus(str, Enum):
    created = "created"
    configuring = "configuring"
    running = "running"
    completed = "completed"
    failed = "failed"


class PlanningMode(str, Enum):
    single_week = "single_week"
    multi_week = "multi_week"


class SourceProfile(str, Enum):
    offer_file = "offer_file"
    extraction_file = "extraction_file"
    hybrid = "hybrid"


class CohortInfo(BaseModel):
    date: str
    invoice_count: int


class CreateProcessResponse(BaseModel):
    process_id: str
    status: ProcessStatus
    created_at: datetime
    extraction_filename: str
    available_cohorts: List[CohortInfo] = []


class ProcessSummary(BaseModel):
    process_id: str
    status: ProcessStatus
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    extraction_filename: Optional[str] = None
    cohort: Optional[str] = None
    planning_mode: Optional[PlanningMode] = None
    source_profile: Optional[SourceProfile] = None
    candidate_count: Optional[int] = None
    selected_count: Optional[int] = None
    excluded_count: Optional[int] = None
    selected_amount: Optional[float] = None
    candidate_amount: Optional[float] = None
    optimizer_status: Optional[str] = None
    error_message: Optional[str] = None


class SolverSettings(BaseModel):
    max_time_seconds: int = 60
    random_seed: int = 0
    num_search_workers: int = 1


class ProcessDetail(BaseModel):
    process_id: str
    status: ProcessStatus
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    extraction_filename: Optional[str] = None
    cohort: Optional[str] = None
    cohort_match_granularity: Optional[str] = None
    sheet_name: Optional[str] = None
    release_event: Optional[str] = None
    release_event_mode: Optional[str] = None
    planning_mode: Optional[PlanningMode] = None
    planning_start_date: Optional[str] = None
    horizon_weeks: Optional[int] = None
    attempt_cap: Optional[int] = None
    source_profile: Optional[SourceProfile] = None
    lifecycle_input_path: Optional[str] = None
    solver_settings: Optional[SolverSettings] = None
    candidate_count: Optional[int] = None
    selected_count: Optional[int] = None
    excluded_count: Optional[int] = None
    candidate_amount: Optional[float] = None
    selected_amount: Optional[float] = None
    optimizer_status: Optional[str] = None
    error_message: Optional[str] = None
    run_metadata: Optional[Dict[str, Any]] = None


class LimitsConfig(BaseModel):
    facility_limits_by_company_code: Dict[str, float] = Field(default_factory=dict)
    customer_limits: Dict[str, float] = Field(default_factory=dict)
    group_limits: Dict[str, float] = Field(default_factory=dict)
    customer_to_group: Dict[str, str] = Field(default_factory=dict)
    base_exposure: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    defaults: Dict[str, float] = Field(default_factory=dict)
    synthetic_generation: Dict[str, Any] = Field(default_factory=dict)


class RuleDefinition(BaseModel):
    name: str
    type: str
    enabled: bool = True
    column: Optional[str] = None
    value: Optional[Any] = None
    value_from_context: Optional[str] = None
    values: Optional[List[Any]] = None
    values_from_context: Optional[str] = None
    min_value: Optional[Any] = None
    min_value_from_context: Optional[str] = None
    include_equal: Optional[bool] = None
    other_column: Optional[str] = None
    pattern: Optional[str] = None
    ignore_case: Optional[bool] = None
    columns: Optional[List[str]] = None
    fallback_columns: Optional[List[str]] = None
    match_granularity: Optional[str] = None
    match_granularity_from_context: Optional[str] = None
    enabled_from_context: Optional[str] = None


class RulesConfig(BaseModel):
    rules: List[RuleDefinition] = Field(default_factory=list)


class FacilityUsage(BaseModel):
    company_code: str
    used: float
    limit: float
    utilization_pct: float


class CustomerUsage(BaseModel):
    customer: str
    used: float
    limit: float
    utilization_pct: float


class OptimizationResults(BaseModel):
    cohort: Optional[str] = None
    candidate_count: int = 0
    selected_count: int = 0
    excluded_count: int = 0
    rule_excluded_count: int = 0
    optimizer_not_selected_count: int = 0
    candidate_amount: float = 0.0
    selected_amount: float = 0.0
    selected_amount_ratio_pct: float = 0.0
    planning_mode: Optional[PlanningMode] = None
    source_profile: Optional[SourceProfile] = None
    horizon_weeks: Optional[int] = None
    optimizer_status: Optional[str] = None
    top3_customer_concentration_pct: float = 0.0
    facility_usage: List[FacilityUsage] = Field(default_factory=list)
    customer_usage: List[CustomerUsage] = Field(default_factory=list)
    rule_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    weekly_plan: List[Dict[str, Any]] = Field(default_factory=list)
    weekly_exposure: List[Dict[str, Any]] = Field(default_factory=list)
    deferred_reasons: Dict[str, int] = Field(default_factory=dict)
    lifecycle_profile: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class InvoiceRow(BaseModel):
    invoice_ref: Optional[str] = None
    company_code: Optional[str] = None
    customer: Optional[str] = None
    purchase_price: Optional[float] = None
    due_date: Optional[str] = None
    status: Optional[str] = None
    excluded_stage: Optional[str] = None
    excluded_reason: Optional[str] = None
    excluded_reason_detail: Optional[str] = None


class InvoiceListResponse(BaseModel):
    invoices: List[InvoiceRow] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0
