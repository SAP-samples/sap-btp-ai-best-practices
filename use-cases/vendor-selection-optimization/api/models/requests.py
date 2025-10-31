"""Request Models for Procurement Assistant API"""

from typing import Dict, List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, ConfigDict


class DateRange(BaseModel):
    """Date range model for filtering"""
    start: date = Field(..., description="Start date (inclusive)")
    end: date = Field(..., description="End date (inclusive)")


class VendorFilters(BaseModel):
    """Filters for vendor evaluation"""
    material_groups: Optional[List[str]] = Field(
        default=None, 
        description="Filter by material group codes (MATKL)"
    )
    materials: Optional[List[str]] = Field(
        default=None, 
        description="Filter by material IDs (MATNR)"
    )
    suppliers: Optional[List[str]] = Field(
        default=None, 
        description="Filter by supplier IDs (LIFNR)"
    )
    date_range: Optional[DateRange] = Field(
        default=None, 
        description="Historical data range for evaluation"
    )


class MetricWeights(BaseModel):
    """Metric weights for vendor scoring"""
    AvgUnitPriceUSD_Norm: float = Field(default=0.20, ge=0, le=1)
    PriceVolatility_Norm: float = Field(default=0.15, ge=0, le=1)
    PriceTrend_Norm: float = Field(default=0.10, ge=0, le=1)
    TariffImpact_Norm: float = Field(default=0.15, ge=0, le=1)
    AvgLeadTimeDays_Norm: float = Field(default=0.10, ge=0, le=1)
    LeadTimeVariability_Norm: float = Field(default=0.10, ge=0, le=1)
    OnTimeRate_Norm: float = Field(default=0.10, ge=0, le=1)
    InFullRate_Norm: float = Field(default=0.10, ge=0, le=1)


class CostComponents(BaseModel):
    """Cost component toggles"""
    cost_BasePrice: bool = Field(default=True)
    cost_Tariff: bool = Field(default=True)
    cost_Holding_LeadTime: bool = Field(default=True)
    cost_Holding_LTVariability: bool = Field(default=True)
    cost_Holding_Lateness: bool = Field(default=True)
    cost_Risk_PriceVolatility: bool = Field(default=True)
    cost_Impact_PriceTrend: bool = Field(default=True)
    cost_Logistics: bool = Field(default=True)


class RunPipelineRequest(BaseModel):
    """Request model for running complete optimization pipeline"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    profile_id: str = Field(..., description="Profile identifier")
    mode: Literal["matkl", "matnr", "maktx"] = Field(..., description="Grouping mode")
    force_regenerate: bool = Field(
        default=False, 
        description="Force regeneration of all derived files"
    )
    metric_weights: Optional[MetricWeights] = Field(
        default=None,
        description="Optional metric weights for vendor evaluation. Uses defaults if not provided"
    )
    demand_period_days: int = Field(
        default=365,
        ge=1,
        le=730,
        description="Demand calculation period in days"
    )
    filters: Optional[VendorFilters] = Field(
        default=None,
        description="Optional filters for vendor evaluation"
    )
    clean_previous_results: bool = Field(
        default=False,
        description="Clean previous result files before running pipeline"
    )


class EvaluateVendorsRequest(BaseModel):
    """Request model for vendor evaluation"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    profile_id: str = Field(..., description="Profile identifier")
    mode: Literal["matkl", "matnr", "maktx"] = Field(..., description="Grouping mode")
    filters: Optional[VendorFilters] = Field(
        default=None, 
        description="Optional filters. If omitted, evaluates ALL materials and vendors"
    )
    metric_weights: Optional[MetricWeights] = Field(
        default=None, 
        description="Optional metric weights. Uses defaults if not provided"
    )
    cost_components: Optional[CostComponents] = Field(
        default=None, 
        description="Optional cost component toggles. All enabled by default"
    )
    output_format: Literal["inline", "async"] = Field(
        default="async", 
        description="Output format - inline for small datasets, async for large"
    )
    include_details: bool = Field(
        default=True, 
        description="Include detailed cost components in results"
    )


class OptimizationConstraints(BaseModel):
    """Constraints for optimization"""
    enforce_multi_supplier: bool = Field(default=True)
    min_suppliers_per_material: int = Field(default=2, ge=1)
    max_supplier_share: float = Field(default=0.80, gt=0, le=1)


class SolverOptions(BaseModel):
    """Solver configuration options"""
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    gap_tolerance: float = Field(default=0.01, ge=0, le=0.1)


class OptimizeAllocationRequest(BaseModel):
    """Request model for procurement optimization"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    profile_id: str = Field(..., description="Profile identifier")
    mode: Literal["matkl", "matnr", "maktx"] = Field(..., description="Grouping mode")
    demand_period_days: int = Field(default=365, ge=1, le=730)
    capacity_buffer_percent: float = Field(default=0.10, ge=0, le=0.5)
    constraints: Optional[OptimizationConstraints] = Field(default=None)
    solver_options: Optional[SolverOptions] = Field(default=None)


class ComparePoliciesRequest(BaseModel):
    """Request model for policy comparison"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    profile_id: str = Field(..., description="Profile identifier")
    optimization_job_id: str = Field(..., description="Job ID from optimization run")
    mode: Literal["matkl", "matnr", "maktx"] = Field(..., description="Grouping mode")
    output_format: Literal["summary", "async"] = Field(
        default="async", 
        description="Output format - summary for lightweight, async for full comparison"
    )
    cost_components: Optional[CostComponents] = Field(
        default=None, 
        description="Optional cost component toggles for comparison"
    )