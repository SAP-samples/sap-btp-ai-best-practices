"""
Pydantic models for SPA API endpoints

Request and Response models for:
- Quick Lookup API
- Agent Chat API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================
# QUICK LOOKUP MODELS
# ============================================================

class QuickLookupRequest(BaseModel):
    """Request for Quick Lookup API"""
    customer_id: str = Field(..., description="Customer ID to analyze")
    top_n_similar: int = Field(50, description="Number of similar customers to find", ge=1, le=100)
    min_similar_count: int = Field(2, description="Minimum similar customers that must have SPA for recommendation", ge=1, le=10)
    include_rfm: bool = Field(True, description="Include RFM boost in similarity scoring")
    include_price_group: bool = Field(True, description="Include Price Group boost in similarity scoring")


class SavingsAnalysis(BaseModel):
    """Customer savings analysis"""
    total_savings: Optional[float] = Field(None, description="Total potential savings")
    total_base_cost: Optional[float] = Field(None, description="Total baseline cost without SPAs")
    total_spa_cost: Optional[float] = Field(None, description="Total cost with SPA prices")
    material_count: Optional[int] = Field(None, description="Number of materials with savings potential")
    savings_percent: Optional[float] = Field(None, description="Savings percentage")
    covered_cogs: Optional[float] = Field(None, description="Actual customer COGS currently covered by assigned SPAs")
    uncovered_cogs: Optional[float] = Field(None, description="Customer COGS not currently covered by assigned SPAs")
    coverage_percent: Optional[float] = Field(None, description="Coverage of total COGS by assigned SPAs")
    total_material_count: Optional[int] = Field(None, description="Total number of purchased materials in the Q4 snapshot")
    current_pricing_sources: Dict[str, int] = Field(
        default_factory=dict,
        description="Current-state pricing source counts, e.g. A703 exact/netted and A704 multiplier-based"
    )
    pricing_source_note: Optional[str] = Field(
        None,
        description="Plain-English note explaining current pricing source assumptions"
    )


class CurrentSPADetail(BaseModel):
    """Current active SPA assignment detail"""
    sales_deal: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    agreement_type: Optional[str] = None
    grouping: Optional[str] = None
    assignment_scope: Optional[str] = None
    description_of_agreement: Optional[str] = None
    external_description: Optional[str] = None
    agreement_description: Optional[str] = None
    is_supplyforce: Optional[bool] = None
    pricing_sources: List[str] = Field(default_factory=list)
    pricing_source_counts: Dict[str, int] = Field(default_factory=dict)
    pricing_source_note: Optional[str] = None
    is_active: bool = True
    covered_cogs: Optional[float] = None
    current_savings: Optional[float] = None
    covered_materials: Optional[int] = None


class CustomerProfileSummary(BaseModel):
    """Customer profile summary"""
    customer_id: str
    customer_name: Optional[str] = None
    sales_office: Optional[str] = None
    pl_type: Optional[str] = None
    price_group: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    spa_count: int
    current_spas: List[str] = Field(default_factory=list, description="List of current SPAs customer has")
    current_spa_details: List[CurrentSPADetail] = Field(
        default_factory=list,
        description="Detailed list of current SPAs including validity dates"
    )
    current_spa_count_unique: Optional[int] = Field(
        None,
        description="Unique active SPA agreement count"
    )
    current_spa_row_count: Optional[int] = Field(
        None,
        description="Active A701 row count including plant-level duplicates"
    )
    current_spa_count_rule: Optional[str] = Field(
        None,
        description="Explanation of how SPA count is calculated"
    )
    snapshot_date: Optional[str] = Field(
        None,
        description="Snapshot date used for active SPA filtering"
    )
    rfm_segment: Optional[str] = None
    total_cogs: Optional[float] = None
    savings: Optional[SavingsAnalysis] = Field(None, description="Savings analysis data")


class SimilarCustomer(BaseModel):
    """Similar customer with similarity score"""
    customer_id: str
    customer_name: Optional[str] = None
    sales_office: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    similarity_score: float
    cogs_similarity: float
    rfm_boost: float
    pg_boost: float
    rfm_segment: Optional[str] = None


class MissingSPA(BaseModel):
    """Missing SPA recommendation"""
    sales_deal: str
    vendor: Optional[str] = None
    description: Optional[str] = None
    description_of_agreement: Optional[str] = None
    external_description: Optional[str] = None
    agreement_description: Optional[str] = None
    agreement_type: Optional[str] = None
    grouping: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    is_supplyforce: Optional[bool] = None
    customer_count: Optional[int] = None
    a700_vendor_scope: Optional[bool] = None
    opportunity_scope: Optional[str] = None
    is_out_of_scope: Optional[bool] = None
    count_in_similar: int
    percentage_in_similar: float
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    eligibility_status: Optional[str] = None
    eligibility_label: Optional[str] = None
    eligibility_reason: Optional[str] = None
    geo_relevance: Optional[str] = None
    is_addable_candidate: Optional[bool] = None
    is_reference_only: Optional[bool] = None
    is_out_of_area: Optional[bool] = None
    customer_area: Optional[str] = None
    candidate_areas: List[str] = Field(default_factory=list)
    candidate_sales_offices: List[str] = Field(default_factory=list)
    candidate_plants: List[str] = Field(default_factory=list)
    candidate_vendor_id: Optional[str] = None
    candidate_vendor_name: Optional[str] = None
    candidate_spa_type: Optional[str] = None
    candidate_primary_category: Optional[str] = None


class QuickLookupResponse(BaseModel):
    """Response for Quick Lookup API"""
    target_customer: CustomerProfileSummary
    similar_customers: List[SimilarCustomer]
    missing_spas: List[MissingSPA]
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )


# ============================================================
# AGENT CHAT MODELS
# ============================================================

class AgentChatRequest(BaseModel):
    """Request for Agent Chat API"""
    message: str = Field(..., min_length=1, description="User message/question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context persistence")
    customer_id: Optional[str] = Field(None, description="Optional customer context")
    exclude_unknown: Optional[bool] = Field(False, description="Exclude customers with 'Unknown' names")


class AgentAction(BaseModel):
    """Agent action/tool call"""
    tool: str
    input: Dict[str, Any]
    output: Optional[Any] = None


class AgentChatResponse(BaseModel):
    """Response for Agent Chat API"""
    message: str = Field(..., description="Agent's response message")
    conversation_id: str = Field(..., description="Conversation ID")
    actions: List[AgentAction] = Field(default_factory=list, description="Actions taken by agent")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured data including entities for clickable links")
    quick_actions: List[str] = Field(default_factory=list, description="Suggested quick action buttons")


# ============================================================
# SUMMARY VIEW MODELS
# ============================================================

class CustomerSummaryItem(BaseModel):
    """Customer summary item for Summary View"""
    customer_id: str
    customer_name: Optional[str] = "Unknown"
    sales_office: Optional[str] = "N/A"
    rfm_segment: Optional[str] = "N/A"
    pl_type: str
    price_group: Optional[str] = None
    total_cogs: float
    total_savings: float = Field(0.0, description="Current savings from existing SPAs")
    savings_percent: float = Field(0.0, description="Current savings percentage")
    coverage_percent: float = Field(0.0, description="Percentage of COGS covered by SPAs")
    savings_percent_normalized: float = Field(0.0, description="Normalized savings percentage (capped at 40%)")
    cogs_covered: float = Field(0.0, description="Amount of COGS covered by SPAs")
    cogs_not_covered: float = Field(0.0, description="Amount of COGS not covered by SPAs")
    missing_spas_count: int = Field(..., description="Number of missing SPAs identified")
    missing_spas_total_confidence: float = Field(..., description="Average confidence score of missing SPAs")
    potential_value_estimate: float = Field(..., description="Estimated potential value from missing SPAs")
    top_missing_spa: Optional[str] = Field(None, description="Highest confidence missing SPA")
    top_missing_spa_confidence: float = Field(0, description="Confidence score of top missing SPA")


class CustomerSummaryFilters(BaseModel):
    """Filters for customer summary"""
    rfm_segment: Optional[str] = Field(None, description="Filter by RFM segment (e.g., 'Champions')")
    sales_office: Optional[str] = Field(None, description="Filter by sales office (single or comma-separated)")
    min_missing_spas: Optional[int] = Field(None, description="Minimum missing SPAs count", ge=0)
    min_cogs: Optional[float] = Field(None, description="Minimum total COGS", ge=0)
    min_potential: Optional[float] = Field(None, description="Minimum potential savings value", ge=0)


class CustomerSummaryRequest(BaseModel):
    """Request for Customer Summary API"""
    filters: Optional[CustomerSummaryFilters] = None
    sort_by: str = Field("missing_spas_count", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    limit: Optional[int] = Field(None, description="Max number of results", ge=1, le=1000)


class CustomerSummaryResponse(BaseModel):
    """Response for Customer Summary API"""
    total_customers: int = Field(..., description="Total customers in database")
    filtered_customers: int = Field(..., description="Customers after applying filters")
    customers: List[CustomerSummaryItem]
    summary_stats: Dict[str, Any] = Field(
        ...,
        description="Aggregate statistics (total_missing_spas, avg_missing_spas, etc.)"
    )
    actions: List[AgentAction] = Field(default_factory=list, description="Actions taken by agent")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured data if applicable")
    quick_actions: List[str] = Field(default_factory=list, description="Suggested quick action buttons")


# ============================================================
# CUSTOMER SEARCH MODELS
# ============================================================

class CustomerSearchRequest(BaseModel):
    """Request for customer search"""
    query: Optional[str] = Field(None, description="Search query (customer ID or name)")
    sales_office: Optional[str] = None
    pl_type: Optional[str] = None
    price_group: Optional[str] = None
    state: Optional[str] = None
    rfm_segment: Optional[str] = None
    limit: int = Field(50, ge=1, le=500)


class CustomerSearchResult(BaseModel):
    """Customer search result"""
    customer_id: str
    customer_name: Optional[str] = None
    sales_office: Optional[str] = None
    pl_type: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class CustomerSearchResponse(BaseModel):
    """Response for customer search"""
    customers: List[CustomerSearchResult]
    total: int


# ============================================================
# GAP ANALYSIS MODELS
# ============================================================

class GapAnalysisRequest(BaseModel):
    """Request for detailed gap analysis"""
    customer_id: str
    spa_id: str
    top_n_similar: int = Field(10, ge=1, le=50)


class ConfidenceBreakdown(BaseModel):
    """Confidence score breakdown"""
    total_score: float
    confidence_level: str
    spa_type_score: float
    similar_count_score: float
    material_coverage_score: float
    rfm_quality_score: float


class MaterialCoverageSummary(BaseModel):
    """Material coverage summary"""
    total_materials: int
    covered_count: int
    coverage_percentage: float
    cogs_coverage_percentage: float


class GapAnalysisResponse(BaseModel):
    """Response for gap analysis"""
    customer_id: str
    spa_id: str
    confidence: ConfidenceBreakdown
    material_coverage: MaterialCoverageSummary
    similar_customers_count: int
    similar_customers_with_spa: int


# ============================================================
# MATERIAL HIERARCHY MODELS
# ============================================================

class MaterialCategoryNode(BaseModel):
    """Material hierarchy category node"""
    category_code: str = Field(..., description="Category code (e.g., '1A', '1A10', '1A1010')")
    category_name: str = Field(..., description="Category description")
    level: int = Field(..., description="Hierarchy level (1-4)", ge=1, le=4)
    total_cogs: float = Field(..., description="Total COGS for this category")
    percentage_of_total: float = Field(..., description="Percentage of customer's total COGS")
    transaction_count: int = Field(..., description="Number of transactions in this category")
    unique_materials: int = Field(..., description="Number of unique materials purchased")
    spa_coverage_cogs: float = Field(..., description="COGS amount covered by SPAs")
    spa_coverage_percentage: float = Field(..., description="Percentage of COGS covered by SPAs")
    spas_covering: List[str] = Field(default_factory=list, description="SPA IDs covering materials in this category")
    has_children: bool = Field(..., description="True if category can be drilled down further")


class MaterialHierarchySummaryResponse(BaseModel):
    """Response for material hierarchy summary"""
    customer_id: str
    customer_name: Optional[str] = None
    total_cogs: float = Field(..., description="Total COGS across all categories")
    total_transactions: int = Field(..., description="Total transaction count")
    overall_spa_coverage_percentage: float = Field(..., description="Overall SPA coverage percentage")
    categories: List[MaterialCategoryNode] = Field(default_factory=list, description="List of category nodes")


class MaterialDrilldownRequest(BaseModel):
    """Request for material hierarchy drill-down"""
    level: int = Field(..., description="Target hierarchy level (1-4)", ge=1, le=4)
    parent_category: Optional[str] = Field(None, description="Parent category code to filter by")


# ============================================================
# ERROR MODELS
# ============================================================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
