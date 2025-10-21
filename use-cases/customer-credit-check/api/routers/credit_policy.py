"""
Credit Policy API router.

Exposes endpoints to evaluate credit requests based on the credit policy
engine implemented in `credit_policy_engine.py` at the project root.

Endpoints:
- POST /api/credit/evaluate: Full evaluation across use cases (new/update/exception)
- POST /api/credit/scores: Compute historical payment scores (CAL, C3M, etc.)
- GET  /api/credit/health: Simple health endpoint
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

# Import the policy engine from project root. The app is started from project root
# (e.g. `uvicorn api.api_server:app`) so this import will work.
from routers.tools.credit_policy_engine import (
    Invoice as _Invoice,
    CustomerInput as _CustomerInput,
    DocsInput as _DocsInput,
    CreditRequest as _CreditRequest,
    Investigation as _Investigation,
    BehaviorInput as _BehaviorInput,
    RoleContext as _RoleContext,
    classify_group,
    compute_payment_scores,
    evaluate_credit,
)


# -------------------------------
# Pydantic request models
# -------------------------------


class Invoice(BaseModel):
    """Single invoice record used for payment history calculations."""

    invoice_id: str
    issue_date: datetime
    due_date: datetime
    paid_date: Optional[datetime] = None
    amount: float
    currency: str = Field(default="MXN", description="Invoice currency (MXN/USD/EUR)")


class CustomerInput(BaseModel):
    """Customer master information used by the credit policy engine."""

    customer_id: str
    legal_name: str
    persona: str  # "PF" or "PM"
    country: str
    entity_name: Optional[str] = None
    customer_group: Optional[str] = Field(
        default=None, description="Explicit group 'A' or 'B' (optional)"
    )
    cgv_signed_date: Optional[datetime] = None
    pagare_signed: bool = False
    guarantors: int = 0
    insurance_full_credit: bool = False


class DocsInput(BaseModel):
    """Documents presence and recency info."""

    kyc_date: Optional[datetime] = None
    seller_comments_present: bool = True
    address_proof_date: Optional[datetime] = None
    tax_cert_date: Optional[datetime] = None


class CreditRequest(BaseModel):
    """Requested credit line update/new/exception and parameters."""

    use_case: str  # "new" | "update" | "exception"
    requested_amount: float
    requested_currency: str = "MXN"
    requested_terms_days: int = 30
    last_update_date: Optional[datetime] = None
    current_credit_line: float = 0.0
    current_credit_currency: str = "MXN"


class Investigation(BaseModel):
    """Investigation results and flags."""

    mmr_amount: Optional[float] = None
    mmr_currency: str = "MXN"
    legal_risk: Optional[str] = None  # "low" | "medium" | "high"
    external_investigation_date: Optional[datetime] = None
    onsite_visit_done: bool = False


class BehaviorInput(BaseModel):
    """Observed payment behavior and exceptions count."""

    invoices: List[Invoice]
    has_overdue_invoices: bool = False
    advance_purchases_count: int = 0
    has_active_credit: bool = False
    exceptions_in_semester: int = 0


class RoleContext(BaseModel):
    """Current user's role for approval limits and terms authority."""

    role: str = Field(default="analyst", description="'analyst' or 'coordinator'")


class EvaluateRequest(BaseModel):
    """Full evaluation request input aggregating all submodels."""

    customer: CustomerInput
    docs: DocsInput
    request: CreditRequest
    investigation: Investigation
    behavior: BehaviorInput
    role: RoleContext
    as_of: Optional[datetime] = Field(
        default=None, description="Optional point-in-time for evaluation"
    )


class ScoresRequest(BaseModel):
    """Request to compute payment scores for a set of invoices.

    Provide either `group` directly ("A"/"B"), or pass `entity_name` and
    optional `explicit_group` to have the group derived via policy rules.
    """

    invoices: List[Invoice]
    group: Optional[str] = Field(default=None, description="Explicit group A/B")
    entity_name: Optional[str] = None
    explicit_group: Optional[str] = None
    as_of: Optional[datetime] = None


# -------------------------------
# Converters to engine dataclasses
# -------------------------------


def _to_engine_models(req: EvaluateRequest):
    """Convert pydantic models into the engine's dataclasses for evaluation."""

    invoices_dc = [
        _Invoice(
            invoice_id=i.invoice_id,
            issue_date=i.issue_date,
            due_date=i.due_date,
            paid_date=i.paid_date,
            amount=i.amount,
            currency=i.currency,
        )
        for i in req.behavior.invoices
    ]

    customer_dc = _CustomerInput(
        customer_id=req.customer.customer_id,
        legal_name=req.customer.legal_name,
        persona=req.customer.persona,  # type: ignore[arg-type]
        country=req.customer.country,
        entity_name=req.customer.entity_name,
        customer_group=req.customer.customer_group,  # type: ignore[arg-type]
        cgv_signed_date=req.customer.cgv_signed_date,
        pagare_signed=req.customer.pagare_signed,
        guarantors=req.customer.guarantors,
        insurance_full_credit=req.customer.insurance_full_credit,
    )

    docs_dc = _DocsInput(
        kyc_date=req.docs.kyc_date,
        seller_comments_present=req.docs.seller_comments_present,
        address_proof_date=req.docs.address_proof_date,
        tax_cert_date=req.docs.tax_cert_date,
    )

    credit_req_dc = _CreditRequest(
        use_case=req.request.use_case,  # type: ignore[arg-type]
        requested_amount=req.request.requested_amount,
        requested_currency=req.request.requested_currency,
        requested_terms_days=req.request.requested_terms_days,
        last_update_date=req.request.last_update_date,
        current_credit_line=req.request.current_credit_line,
        current_credit_currency=req.request.current_credit_currency,
    )

    investigation_dc = _Investigation(
        mmr_amount=req.investigation.mmr_amount,
        mmr_currency=req.investigation.mmr_currency,
        legal_risk=req.investigation.legal_risk,  # type: ignore[arg-type]
        external_investigation_date=req.investigation.external_investigation_date,
        onsite_visit_done=req.investigation.onsite_visit_done,
    )

    behavior_dc = _BehaviorInput(
        invoices=invoices_dc,
        has_overdue_invoices=req.behavior.has_overdue_invoices,
        advance_purchases_count=req.behavior.advance_purchases_count,
        has_active_credit=req.behavior.has_active_credit,
        exceptions_in_semester=req.behavior.exceptions_in_semester,
    )

    role_dc = _RoleContext(role=req.role.role)  # type: ignore[arg-type]

    return customer_dc, docs_dc, credit_req_dc, investigation_dc, behavior_dc, role_dc


# -------------------------------
# Router and endpoints
# -------------------------------


router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    """Simple health endpoint for the credit policy router."""

    return {"status": "ok", "component": "credit", "version": "0.1.0"}


@router.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    """Evaluate a credit request according to policy rules.

    Returns a structured JSON with computed scores (CAL, C3M), rule checks
    per use case (new/update/exception), and a decision hint flag indicating
    whether director approval is needed.
    """

    customer_dc, docs_dc, credit_req_dc, investigation_dc, behavior_dc, role_dc = _to_engine_models(
        req
    )
    result = evaluate_credit(
        customer=customer_dc,
        docs=docs_dc,
        request=credit_req_dc,
        investigation=investigation_dc,
        behavior=behavior_dc,
        role=role_dc,
        as_of=req.as_of,
    )
    return result


@router.post("/scores")
def scores(req: ScoresRequest) -> Dict[str, Any]:
    """Compute payment scores (CAL, C3M, CA_by_year) for a set of invoices.

    If `group` is not provided, it will be derived using `entity_name` and
    `explicit_group` via the policy's group classification helper.
    """

    # Convert invoice models to engine dataclasses
    invoices_dc = [
        _Invoice(
            invoice_id=i.invoice_id,
            issue_date=i.issue_date,
            due_date=i.due_date,
            paid_date=i.paid_date,
            amount=i.amount,
            currency=i.currency,
        )
        for i in req.invoices
    ]

    group = req.group
    if not group:
        group = classify_group(req.entity_name, req.explicit_group)

    return compute_payment_scores(invoices_dc, group, as_of=req.as_of)


