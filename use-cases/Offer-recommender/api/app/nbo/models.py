"""Domain dataclasses for legacy recommendations and the new decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class FactSource(str, Enum):
    WORKBOOK = "workbook"
    DERIVED = "derived"
    CUSTOMER_ANSWER = "customer_answer"
    EXTERNAL = "external"
    AI_NORMALIZED = "ai_normalized"
    SYSTEM = "system"


class DecisionStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    NEEDS_INFO = "needs_info"
    MANUAL_REVIEW = "manual_review"


class WorkflowStage(str, Enum):
    PRIMARY_OFFER_FINAL = "primary_offer_final"
    PRIMARY_OFFER_WITH_FOLLOWUP = "primary_offer_with_followup"
    NEEDS_CORE_FACTS = "needs_core_facts"
    SYSTEM_BLOCKED = "system_blocked"


@dataclass
class FactValue:
    """A typed fact used by the decision engine."""

    fact_id: str
    value_type: str
    source: FactSource
    value: Any = None
    confidence: Confidence = Confidence.MEDIUM
    evidence: list[str] = field(default_factory=list)
    missing_reason: str | None = None

    @property
    def is_known(self) -> bool:
        return self.value is not None


@dataclass
class AnswerOptionData:
    """Serializable answer option for questions."""
    label: str
    value: Any
    description: str | None = None


@dataclass
class DecisionExplanation:
    """Structured explanation attached to an offer or follow-up decision.

    Inputs are deterministic decision facts, rule codes, blockers, and source
    documents. Outputs are user-facing summary/detail text plus the underlying
    audit evidence used to produce that text.
    """

    summary: str
    details: list[str] = field(default_factory=list)
    facts_used: list[str] = field(default_factory=list)
    rules_used: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    polish_status: str = "fallback"


@dataclass
class DecisionQuestion:
    """A follow-up question needed to confirm or reject offer candidates."""

    question_id: str
    prompt: str
    expected_fact: str
    candidate_programs: list[str] = field(default_factory=list)
    priority: int = 100
    source: str = "customer"
    answer_options: list[AnswerOptionData] = field(default_factory=list)
    explanation: DecisionExplanation | None = None


@dataclass
class OfferDecision:
    """Outcome of evaluating a single program against one account."""

    program_id: str
    display_name: str
    status: DecisionStatus
    rank: int | None = None
    confidence: Confidence = Confidence.MEDIUM
    reason_codes: list[str] = field(default_factory=list)
    blocking_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: DecisionExplanation | None = None


@dataclass
class RecommendationExplanation:
    """Structured explanation for a staged recommendation outcome."""

    summary: str
    facts_used: list[str] = field(default_factory=list)
    missing_core_facts: list[str] = field(default_factory=list)
    next_step: str | None = None


@dataclass
class DecisionResult:
    """Full decision record for one account."""

    billing_account: str
    customer_type: str
    final_offer: OfferDecision | None = None
    eligible_offers: list[OfferDecision] = field(default_factory=list)
    blocked_offers: list[OfferDecision] = field(default_factory=list)
    questions: list[DecisionQuestion] = field(default_factory=list)
    facts: dict[str, FactValue] = field(default_factory=dict)
    decision_trace: list[str] = field(default_factory=list)
    ai_trace: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    workflow_stage: WorkflowStage | None = None
    explanation: RecommendationExplanation | None = None
    source_documents: list[str] = field(default_factory=list)
    suppression_reasons: list[str] = field(default_factory=list)
    routing_stage: str | None = None
    current_enrollment_detected: bool = False


# ── Legacy v1 compatibility models ────────────────────────────────────────


@dataclass
class AccountFacts:
    """Legacy account facts used by the old compatibility wrapper."""

    billing_account: str
    customer_type: str | None = None
    current_status: str | None = None
    current_rate_plan: str | None = None
    segment_name: str | None = None
    current_program_codes: set[str] = field(default_factory=set)
    avg_on_peak_kwh_3m: float | None = None
    avg_off_peak_kwh_3m: float | None = None
    avg_total_usage_3m: float | None = None
    latest_bill_shortfall: float | None = None
    has_payment_distress_signal: bool = False
    has_high_usage_signal: bool = False
    bill_increase_yoy: float | None = None
    payments_on_time: bool | None = None
    prepay_advance_offers_this_month: int | None = None
    three_day_usage_cost_estimate: float | None = None
    persona_priority_hints: dict[str, str] = field(default_factory=dict)
    snapshot_read_date: datetime | None = None
    history_row_count: int = 0
    missing_facts: list[str] = field(default_factory=list)


@dataclass
class Offer:
    """Legacy single-offer record for the old `run_batch()` shape."""

    program_name: str
    rule_id: str
    priority: int
    confidence: Confidence = Confidence.MANUAL_REVIEW
    reason: str = ""
    requires_external: list[str] = field(default_factory=list)
    requires_followup: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Legacy compatibility record returned by `run_batch()`."""

    billing_account: str
    customer_type: str
    primary_offer: Offer | None = None
    secondary_offers: list[Offer] = field(default_factory=list)
    facts_summary: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    evaluation_trail: list[str] = field(default_factory=list)
