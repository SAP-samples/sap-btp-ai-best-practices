"""Pydantic models for the Sales Forecast Dashboard."""

from typing import List, Optional
from pydantic import BaseModel


class DMA(BaseModel):
    """DMA (Designated Market Area) summary model."""

    dma: str
    lat: float
    lng: float
    store_count: int  # Number of comp stores in DMA

    # Combined AUV fields (Annualized Unit Volume - 52-week sum)
    total_auv_p50: Optional[float] = None  # Sum of store AUVs in DMA (B&M + WEB)
    total_auv_p90: Optional[float] = None
    avg_auv_p50: Optional[float] = None  # Average AUV per comp store

    # Combined YoY change based on AUV comparison
    yoy_auv_change_pct: Optional[float] = None  # Weighted average YoY AUV % for all stores in DMA
    yoy_status: Optional[str] = None  # "increase", "stable", "decrease", or None

    # B&M channel AUV aggregates
    bm_total_auv_p50: Optional[float] = None
    bm_total_auv_p90: Optional[float] = None
    bm_avg_auv_p50: Optional[float] = None
    bm_yoy_auv_change_pct: Optional[float] = None
    bm_yoy_status: Optional[str] = None

    # WEB channel AUV aggregates
    web_total_auv_p50: Optional[float] = None
    web_total_auv_p90: Optional[float] = None
    web_avg_auv_p50: Optional[float] = None
    web_yoy_auv_change_pct: Optional[float] = None
    web_yoy_status: Optional[str] = None

    # DEPRECATED: Kept for backward compatibility (weekly average).
    # Use channel-specific fields instead:
    # - bm_yoy_auv_change_pct for B&M channel
    # - web_yoy_auv_change_pct for WEB channel
    # - yoy_auv_change_pct for weighted combined
    total_pred_sales_p50: float
    total_pred_sales_p90: float
    yoy_sales_change_pct: Optional[float] = None  # DEPRECATED: Use bm_yoy_auv_change_pct or web_yoy_auv_change_pct


class Store(BaseModel):
    """Individual store model with predictions."""

    id: int
    name: Optional[str] = None
    dma: str
    lat: float
    lng: float
    city: Optional[str] = None
    state: Optional[str] = None

    # Combined AUV fields (Annualized Unit Volume - 52-week sum, B&M + WEB)
    auv_p50: Optional[float] = None  # Sum of pred_sales_p50 across 52 weeks
    auv_p90: Optional[float] = None  # Sum of pred_sales_p90 across 52 weeks
    auv_weeks_count: Optional[int] = None  # Number of weeks summed (should be 52)

    # YoY change based on AUV comparison (weighted average of B&M and WEB)
    # For channel-specific analysis, use bm_yoy_auv_change or web_yoy_auv_change
    yoy_auv_change: Optional[float] = None  # Weighted combined YoY AUV % (computed from channel values)

    # B&M channel AUV fields
    bm_auv_p50: Optional[float] = None
    bm_auv_p90: Optional[float] = None
    bm_auv_weeks_count: Optional[int] = None
    bm_yoy_auv_change: Optional[float] = None

    # WEB channel AUV fields
    web_auv_p50: Optional[float] = None
    web_auv_p90: Optional[float] = None
    web_auv_weeks_count: Optional[int] = None
    web_yoy_auv_change: Optional[float] = None

    # Store attributes
    is_outlet: bool = False
    is_new_store: bool = False
    is_comp_store: bool = False
    store_design_sf: Optional[float] = None
    merchandising_sf: Optional[float] = None
    bcg_category: Optional[str] = None

    # DEPRECATED: Kept for backward compatibility (weekly average).
    # Use channel-specific fields instead:
    # - bm_yoy_auv_change for B&M channel
    # - web_yoy_auv_change for WEB channel
    # - yoy_auv_change for weighted combined
    pred_sales_p50: float
    pred_sales_p90: float
    yoy_sales_change: Optional[float] = None  # DEPRECATED: Use bm_yoy_auv_change or web_yoy_auv_change


class ShapFeature(BaseModel):
    """SHAP feature importance model."""

    feature: str
    value: Optional[str] = None
    baseline_value: Optional[str] = None
    impact: float
    has_baseline: bool = True


class ExplanationDriver(BaseModel):
    """Single driver/offset in an explanation."""

    feature: str
    display_name: str
    direction: str  # "positive" or "negative"
    shap_impact: Optional[float] = None
    dollar_impact: Optional[float] = None
    description: str


class Explanation(BaseModel):
    """Structured text explanation for sales change."""

    summary: str
    drivers: List[ExplanationDriver] = []
    offsets: List[ExplanationDriver] = []
    other_factors_impact: Optional[float] = None
    is_estimated: bool = True
    no_baseline: bool = False
    generated_by: str = "static"


class TimeseriesPoint(BaseModel):
    """Single time point in timeseries."""

    date: str
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    fiscal_month: Optional[int] = None
    fiscal_week: Optional[int] = None
    pred_sales_p50: Optional[float] = None
    pred_sales_p90: Optional[float] = None
    baseline_sales_p50: Optional[float] = None
    yoy_change_pct: Optional[float] = None
    pred_aov_mean: Optional[float] = None
    pred_aov_p50: Optional[float] = None
    pred_aov_p90: Optional[float] = None
    pred_traffic_p10: Optional[float] = None
    pred_traffic_p50: Optional[float] = None
    pred_traffic_p90: Optional[float] = None
    shap_features: Optional[List[ShapFeature]] = None
    explanation: Optional[Explanation] = None


class StoreTimeseries(BaseModel):
    """Store timeseries response."""

    store_id: int
    channel: Optional[str] = None  # "B&M" or "WEB"
    timeseries: List[TimeseriesPoint]


class DMATimeseries(BaseModel):
    """DMA aggregated timeseries response."""

    dma: str
    timeseries: List[TimeseriesPoint]


class DMASummaryResponse(BaseModel):
    """Response for all DMAs summary."""

    dmas: List[DMA]
    total: int


class StoresResponse(BaseModel):
    """Response for stores list."""

    stores: List[Store]
    total: int
