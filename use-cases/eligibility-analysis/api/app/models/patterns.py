"""
Pattern Analysis Data Models

Pydantic models for eligibility pattern detection results.
Patterns represent recurring rejection behaviors detected across
historical invoice eligibility data.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .eligibility import RULE_DESCRIPTIONS, RuleCode


@dataclass
class PatternFilters:
    """Bundle of filter parameters for pattern analysis queries."""

    seller_id: Optional[str] = None
    debtor_id: Optional[str] = None
    programa: Optional[str] = None
    insurer_id: Optional[str] = None
    lookback_days: int = 90
    min_invoices: int = 3


class PatternType(str, Enum):
    """Types of rejection patterns that can be detected."""

    CHRONIC_RULE_FAILURE = "chronic_rule_failure"
    TRENDING_INCREASE = "trending_increase"
    REPEAT_OFFENDER = "repeat_offender"
    RULE_CONCENTRATION = "rule_concentration"
    AMOUNT_AT_RISK = "amount_at_risk"


PATTERN_TYPE_LABELS = {
    PatternType.CHRONIC_RULE_FAILURE: "Chronic Rule Failure",
    PatternType.TRENDING_INCREASE: "Trending Increase",
    PatternType.REPEAT_OFFENDER: "Repeat Offender",
    PatternType.RULE_CONCENTRATION: "Rule Concentration",
    PatternType.AMOUNT_AT_RISK: "Amount at Risk",
}


class Severity(str, Enum):
    """Alert severity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Maps (rule_code) -> a recommendation string for chronic failures.
# These are actionable suggestions that the lender can relay to their clients.
RULE_RECOMMENDATIONS: Dict[str, str] = {
    "R1": (
        "Review payment terms with the client to increase the gap between "
        "purchase date and due date. Consider negotiating longer payment windows."
    ),
    "R2": (
        "Investigate the invoice submission process for this client. "
        "Duplicate invoices may indicate a system integration issue or "
        "manual re-submission errors."
    ),
    "R11": (
        "Verify the currency setup for this client. If invoices are consistently "
        "submitted in an ineligible currency, consider adding the currency to "
        "the allowed list or advising the client to invoice in an eligible currency."
    ),
    "R13": (
        "This client consistently submits overdue invoices. Consider "
        "discussing earlier submission timelines or adjusting the purchase "
        "date window to accommodate their payment cycle."
    ),
    "R16": (
        "The tenor (issuance to due date) consistently exceeds the maximum. "
        "Review whether the TEIH threshold is appropriate for this client's "
        "business, or advise the client to issue invoices closer to their due dates."
    ),
    "R17": (
        "Invoices are being issued too close to the purchase date. Advise "
        "the client to submit invoices earlier, or review the ISSPUR threshold."
    ),
}


class PatternAlert(BaseModel):
    """
    A single detected pattern with severity, description, and recommendation.

    Each alert identifies a recurring rejection behavior, assigns a severity,
    and provides an actionable recommendation.
    """

    pattern_type: PatternType = Field(..., description="Category of pattern")
    severity: Severity = Field(..., description="Alert severity")
    seller_id: str = Field(..., description="Seller involved in the pattern")
    seller_name: Optional[str] = Field(None, description="Seller display name")
    debtor_id: Optional[str] = Field(
        None, description="Debtor involved (if pattern is debtor-specific)"
    )
    debtor_name: Optional[str] = Field(None, description="Debtor display name")
    rule_code: Optional[str] = Field(
        None, description="Rule code involved (if pattern is rule-specific)"
    )
    rule_description: Optional[str] = Field(
        None, description="Human-readable rule description"
    )
    title: str = Field(..., description="Short headline for the alert")
    description: str = Field(
        ..., description="Detailed human-readable explanation"
    )
    recommendation: str = Field(..., description="Actionable advice")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting data: counts, rates, amounts, deltas",
    )
    time_window_days: Optional[int] = Field(
        None, description="Number of days the analysis window covers"
    )
    first_seen: Optional[datetime] = Field(
        None, description="Earliest invoice contributing to this pattern"
    )
    last_seen: Optional[datetime] = Field(
        None, description="Most recent invoice contributing to this pattern"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: str(v),
        }


class PatternSummary(BaseModel):
    """
    Aggregated result of all pattern detection algorithms.

    Contains the full list of alerts plus summary counts by severity.
    """

    total_patterns: int = Field(..., description="Total number of patterns detected")
    high_severity: int = Field(0, description="Count of HIGH severity patterns")
    medium_severity: int = Field(0, description="Count of MEDIUM severity patterns")
    low_severity: int = Field(0, description="Count of LOW severity patterns")
    patterns: List[PatternAlert] = Field(
        default_factory=list, description="All detected patterns sorted by severity"
    )
    analysis_window_start: Optional[datetime] = Field(
        None, description="Start of the analysis window"
    )
    analysis_window_end: Optional[datetime] = Field(
        None, description="End of the analysis window"
    )
    total_invoices_analyzed: int = Field(
        0, description="Total invoices in the analysis window"
    )
    overall_eligibility_rate: float = Field(
        0.0, description="Overall eligibility rate in the window"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class TrendDataPoint(BaseModel):
    """
    A single data point in a non-eligibility rate time-series.

    Used for charting non-eligibility trends over time at configurable granularity.
    """

    period_start: date = Field(..., description="Start of the period")
    period_end: date = Field(..., description="End of the period")
    total_invoices: int = Field(..., description="Total invoices in the period")
    rejected_invoices: int = Field(
        ..., description="Non-eligible invoices in the period"
    )
    rejection_rate: float = Field(
        ..., description="Non-eligibility rate (0.0 to 1.0)"
    )
    rejection_by_rule: Dict[str, int] = Field(
        default_factory=dict,
        description="Non-eligible count per rule code in this period",
    )

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat(),
        }


class DebtorRuleProfile(BaseModel):
    """
    Non-eligibility profile for a specific debtor within a seller's portfolio.

    Provides a holistic view of how a debtor performs across eligibility rules,
    including the dominant failure rule, amounts at risk, and batch-level statistics.
    """

    debtor_id: str = Field(..., description="Debtor identifier")
    debtor_name: Optional[str] = Field(None, description="Debtor display name")
    seller_id: str = Field(..., description="Seller identifier")
    seller_name: Optional[str] = Field(None, description="Seller display name")
    total_invoices: int = Field(
        ..., description="Total invoices processed for this debtor"
    )
    rejected_invoices: int = Field(
        ..., description="Number of non-eligible invoices"
    )
    rejection_rate: float = Field(
        ..., description="Non-eligibility rate (0.0 to 1.0)"
    )
    dominant_rule: Optional[str] = Field(
        None, description="Rule code that causes the most non-eligibility"
    )
    dominant_rule_description: Optional[str] = Field(
        None, description="Human-readable description of the dominant rule"
    )
    dominant_rule_rate: float = Field(
        0.0,
        description="Fraction of this debtor's non-eligible invoices caused by the dominant rule",
    )
    rejection_by_rule: Dict[str, int] = Field(
        default_factory=dict,
        description="Non-eligible count per rule code",
    )
    total_amount_rejected: Optional[Decimal] = Field(
        None, description="Total non-eligible invoice amount (original currency)"
    )
    batch_count: int = Field(
        0, description="Number of distinct processing batches this debtor appeared in"
    )
    batches_with_rejection: int = Field(
        0,
        description="Number of batches where at least one invoice was non-eligible",
    )
    trend: List[TrendDataPoint] = Field(
        default_factory=list,
        description="Time-series of non-eligibility rates for this debtor",
    )

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class BulkImportResult(BaseModel):
    """Result of importing multiple historical offer files."""

    success: bool = Field(..., description="Whether the import completed")
    files_processed: int = Field(0, description="Number of files processed")
    total_invoices: int = Field(
        0, description="Total invoices across all files"
    )
    total_eligible: int = Field(0, description="Total eligible invoices")
    total_rejected: int = Field(0, description="Total non-eligible invoices")
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages for files that failed to process",
    )
