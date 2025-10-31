"""Response Models for Procurement Assistant API"""

from typing import Dict, List, Optional, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class BaseResponse(BaseModel):
    """Base response model"""
    status: Literal["success", "error", "accepted"] = Field(..., description="Response status")


class PipelineOutputs(BaseModel):
    """Pipeline output file references"""
    vendor_ranking: str
    optimized_allocation: str
    comparison: str


class PipelineMetrics(BaseModel):
    """Pipeline execution metrics"""
    materials_evaluated: int
    vendors_analyzed: int
    total_savings: float
    savings_percentage: float


class PipelineResponse(BaseResponse):
    """Response for complete pipeline execution"""
    profile_id: str
    outputs: PipelineOutputs
    metrics: PipelineMetrics


class ResultEndpoints(BaseModel):
    """Endpoints for accessing job results"""
    status: str
    results: Optional[str] = None
    summary: Optional[str] = None
    allocations: Optional[str] = None
    download: str


class AsyncJobResponse(BaseResponse):
    """Response for async job creation"""
    job_id: str
    estimated_duration_seconds: int
    result_endpoints: Dict[str, str]


class PaginationInfo(BaseModel):
    """Pagination information"""
    current_page: int
    total_pages: int
    page_size: int
    total_records: int


class EvaluationMetadata(BaseModel):
    """Metadata for vendor evaluation results"""
    total_vendors_evaluated: int
    total_materials_evaluated: int
    total_combinations: int
    filters_applied: bool
    evaluation_mode: str
    pagination: PaginationInfo


class VendorMetrics(BaseModel):
    """Vendor performance metrics"""
    avg_unit_price: float
    tariff_impact_percent: float
    logistics_cost: float
    lead_time_days: float
    on_time_rate: float
    in_full_rate: float


class CostComponentBreakdown(BaseModel):
    """Cost component breakdown"""
    cost_BasePrice: float
    cost_Tariff: float
    cost_Logistics: float
    cost_Holding_LeadTime: float
    cost_Holding_LTVariability: float
    cost_Holding_Lateness: float
    cost_Risk_PriceVolatility: float
    cost_Impact_PriceTrend: float


class VendorEvaluation(BaseModel):
    """Individual vendor evaluation result"""
    rank: int
    supplier_id: str
    supplier_name: str
    material_id: str
    material_description: str
    country: str
    effective_cost_per_unit: float
    final_score: float
    po_line_item_count: int
    metrics: VendorMetrics
    cost_components: Optional[CostComponentBreakdown] = None


class InlineEvaluationResponse(BaseResponse):
    """Response for inline vendor evaluation"""
    metadata: EvaluationMetadata
    vendors: List[VendorEvaluation]


class OptimizationSummary(BaseModel):
    """Summary of optimization results"""
    total_effective_cost: float
    total_materials_optimized: int
    total_allocation_changes: int
    total_quantity_allocated: float
    solver_time_seconds: float
    optimality_gap: float


class ConstraintsSatisfied(BaseModel):
    """Status of constraint satisfaction"""
    demand_met: bool
    capacity_respected: bool
    multi_supplier_enforced: bool


class MaterialChange(BaseModel):
    """Top material allocation change"""
    material_id: str
    description: str
    old_primary_supplier: str
    new_primary_supplier: str
    quantity_shifted: float
    cost_impact: float


class OptimizationSummaryResponse(BaseResponse):
    """Response for optimization summary"""
    optimization_status: Literal["optimal", "feasible", "infeasible", "unbounded"]
    summary: OptimizationSummary
    constraints_satisfied: ConstraintsSatisfied
    top_changes: List[MaterialChange]


class AllocationDetail(BaseModel):
    """Detailed allocation information"""
    supplier_id: str
    supplier_name: str
    material_id: str
    material_description: str
    allocated_quantity: float
    effective_cost_per_unit: float
    total_effective_cost: float
    average_unit_price: float


class AllocationMetadata(BaseModel):
    """Allocation results metadata"""
    total_allocations: int
    current_page: int
    total_pages: int
    page_size: int


class AllocationResponse(BaseResponse):
    """Response for allocation details"""
    metadata: AllocationMetadata
    allocations: List[AllocationDetail]


class CostBreakdownComponent(BaseModel):
    """Individual cost component breakdown"""
    historical: float
    optimized: float
    savings: float


class ComparisonSummary(BaseModel):
    """Summary of policy comparison"""
    total_historical_cost: float
    total_optimized_cost: float
    total_savings: float
    savings_percentage: float
    allocation_changes: int


class ComparisonCostBreakdown(BaseModel):
    """Cost breakdown for comparison"""
    cost_BasePrice: CostBreakdownComponent
    cost_Tariff: CostBreakdownComponent
    cost_Logistics: CostBreakdownComponent


class TopImprovement(BaseModel):
    """Top improvement opportunity"""
    material_id: str
    material_description: str
    historical_supplier: str
    optimized_supplier: str
    quantity_shift: float
    cost_reduction: float


class SummaryComparisonResponse(BaseResponse):
    """Response for summary comparison"""
    summary: ComparisonSummary
    cost_breakdown: ComparisonCostBreakdown
    top_improvements: List[TopImprovement]


class JobStatus(BaseModel):
    """Job status information"""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: Optional[float] = Field(None, ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result_endpoints: Optional[ResultEndpoints] = None


class JobStatusResponse(BaseResponse):
    """Response for job status query"""
    job: JobStatus


class JobResultsResponse(BaseResponse):
    """Response for job results"""
    job_id: str
    metadata: Dict[str, Any]
    data: List[Dict[str, Any]]