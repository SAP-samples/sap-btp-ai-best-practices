"""
Eligibility Data Models

Pydantic models for the invoice eligibility analysis system.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


class RuleCode(str, Enum):
    """
    Enumeration of eligibility rule codes.

    Each rule has a specific business logic check:
    - R1: Due Date must be at least NDDT days after Purchase Date
    - R2: Invoice must not be a duplicate (doc_number + fiscal_year + reference_number)
    - R11: Currency must be in the allowed list
    - R13: Invoice must not be overdue (Due Date > Purchase Date)
    - R16: Tenor must be less than TEIH days (Due Date - Issuance Date)
    - R17: Invoice must be issued at least ISSPUR days before Purchase Date
    """

    R1 = "R1"
    R2 = "R2"
    R11 = "R11"
    R13 = "R13"
    R16 = "R16"
    R17 = "R17"


RULE_DESCRIPTIONS = {
    RuleCode.R1: "Due date too close to purchase date",
    RuleCode.R2: "Duplicate invoice",
    RuleCode.R11: "Currency not eligible",
    RuleCode.R13: "Invoice is overdue",
    RuleCode.R16: "Tenor exceeds maximum allowed days",
    RuleCode.R17: "Invoice issued too recently",
}


class RuleDiagnostic(BaseModel):
    """Structured diagnostics for a rejected rule."""

    rule_code: RuleCode
    description: str
    bullets: List[str] = Field(default_factory=list)
    invoice_values: Dict[str, Any] = Field(default_factory=dict)
    settings_values: Dict[str, Any] = Field(default_factory=dict)
    computed_values: Dict[str, Any] = Field(default_factory=dict)


class RejectionReason(BaseModel):
    """Details about why an invoice was rejected."""

    rule_code: RuleCode
    description: str
    details: Optional[str] = None
    diagnostic: Optional[RuleDiagnostic] = None

    @classmethod
    def from_rule(
        cls,
        rule_code: RuleCode,
        details: Optional[str] = None,
        diagnostic: Optional[RuleDiagnostic] = None,
    ) -> "RejectionReason":
        """Create a rejection reason from a rule code."""
        return cls(
            rule_code=rule_code,
            description=RULE_DESCRIPTIONS.get(rule_code, "Unknown rule"),
            details=details,
            diagnostic=diagnostic,
        )


class OfferInvoice(BaseModel):
    """
    Input invoice from the offer Excel file.

    Represents a single row from the offer file with all required fields.
    """

    programa: str = Field(..., description="Program name")
    seller_id: str = Field(..., description="Unique seller identifier")
    seller_name: str = Field(..., description="Seller name")
    debtor_id: str = Field(..., description="Unique debtor identifier")
    debtor_name: str = Field(..., description="Debtor name")
    insurer_id: Optional[str] = Field(None, description="Insurer identifier")
    doc_number: Optional[str] = Field(None, description="Document number")
    fiscal_year: Optional[str] = Field(None, description="Fiscal year")
    reference_number: Optional[str] = Field(None, description="Reference number")
    invoice_ref: str = Field(..., description="Invoice reference number (fallback to reference number)")
    goods_services: Optional[str] = Field(None, description="Goods/Services description")
    item: Optional[str] = Field(None, description="Item description")
    total_invoice_amount_original: Optional[Decimal] = Field(
        None, description="Total invoice amount in original currency"
    )
    total_net_value_original: Optional[Decimal] = Field(
        None, description="Total net value in original currency"
    )
    discount_percentage: Optional[Decimal] = Field(
        None, description="Discount percentage applied to invoice"
    )
    original_currency: str = Field(..., description="Currency code (e.g., EUR, USD)")
    funding_currency: Optional[str] = Field(None, description="Funding currency code")
    exchange_rate: Optional[Decimal] = Field(None, description="Exchange rate applied")
    despatch_date: Optional[date] = Field(None, description="Despatch date")
    issuance_date: date = Field(..., description="Date the invoice was issued")
    due_date: date = Field(..., description="Invoice due date")
    margin: Optional[Decimal] = Field(None, description="Margin value")
    amount_original: Optional[Decimal] = Field(None, description="Amount in original currency")
    amount_eur: Optional[Decimal] = Field(None, description="Amount in EUR")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class EligibilityResult(BaseModel):
    """Result of eligibility check for a single invoice."""

    invoice_ref: str
    seller_id: str
    is_eligible: bool
    passed_rules: List[RuleCode] = Field(default_factory=list)
    failed_rules: List[RejectionReason] = Field(default_factory=list)

    @property
    def rejection_reasons_text(self) -> str:
        """Get a comma-separated string of rejection reasons."""
        if not self.failed_rules:
            return ""
        return "; ".join([f"{r.rule_code.value}: {r.description}" for r in self.failed_rules])


class FundedInvoice(BaseModel):
    """Invoice that passed all eligibility checks and can be funded."""

    programa: str
    seller_id: str
    seller_name: str
    debtor_id: str
    debtor_name: str
    invoice_ref: str
    original_currency: str
    issuance_date: date
    due_date: date
    amount_original: Optional[Decimal] = None
    amount_eur: Optional[Decimal] = None
    purchase_date: date
    days_to_due: int = Field(..., description="Days from purchase date to due date")
    tenor: int = Field(..., description="Days from issuance date to due date")

    @computed_field
    @property
    def invoice_number(self) -> str:
        """UI-compatible alias for invoice_ref."""
        return self.invoice_ref

    @computed_field
    @property
    def currency(self) -> str:
        """UI-compatible alias for original_currency."""
        return self.original_currency

    @computed_field
    @property
    def amount(self) -> Optional[Decimal]:
        """UI-compatible alias for amount_original."""
        return self.amount_original

    @classmethod
    def from_invoice(
        cls,
        invoice: OfferInvoice,
        purchase_date: date,
    ) -> "FundedInvoice":
        """Create a funded invoice from an offer invoice."""
        days_to_due = (invoice.due_date - purchase_date).days
        tenor = (invoice.due_date - invoice.issuance_date).days
        return cls(
            programa=invoice.programa,
            seller_id=invoice.seller_id,
            seller_name=invoice.seller_name,
            debtor_id=invoice.debtor_id,
            debtor_name=invoice.debtor_name,
            invoice_ref=invoice.invoice_ref,
            original_currency=invoice.original_currency,
            issuance_date=invoice.issuance_date,
            due_date=invoice.due_date,
            amount_original=invoice.amount_original,
            amount_eur=invoice.amount_eur,
            purchase_date=purchase_date,
            days_to_due=days_to_due,
            tenor=tenor,
        )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class NonFundedInvoice(BaseModel):
    """Invoice that failed one or more eligibility checks."""

    programa: str
    seller_id: str
    seller_name: str
    debtor_id: str
    debtor_name: str
    invoice_ref: str
    original_currency: str
    issuance_date: date
    due_date: date
    amount_original: Optional[Decimal] = None
    amount_eur: Optional[Decimal] = None
    purchase_date: date
    rejection_reason: str = Field(..., description="Combined rejection reasons")
    failed_rules: List[str] = Field(..., description="List of failed rule codes")
    rejection_diagnostics: List[RuleDiagnostic] = Field(
        default_factory=list,
        description="Structured diagnostics for each failed rule",
    )

    @computed_field
    @property
    def invoice_number(self) -> str:
        """UI-compatible alias for invoice_ref."""
        return self.invoice_ref

    @computed_field
    @property
    def currency(self) -> str:
        """UI-compatible alias for original_currency."""
        return self.original_currency

    @computed_field
    @property
    def amount(self) -> Optional[Decimal]:
        """UI-compatible alias for amount_original."""
        return self.amount_original

    @computed_field
    @property
    def rejection_reasons(self) -> List[str]:
        """UI-compatible alias for failed_rules."""
        return self.failed_rules

    @classmethod
    def from_invoice(
        cls,
        invoice: OfferInvoice,
        purchase_date: date,
        result: EligibilityResult,
    ) -> "NonFundedInvoice":
        """Create a non-funded invoice from an offer invoice and eligibility result."""
        return cls(
            programa=invoice.programa,
            seller_id=invoice.seller_id,
            seller_name=invoice.seller_name,
            debtor_id=invoice.debtor_id,
            debtor_name=invoice.debtor_name,
            invoice_ref=invoice.invoice_ref,
            original_currency=invoice.original_currency,
            issuance_date=invoice.issuance_date,
            due_date=invoice.due_date,
            amount_original=invoice.amount_original,
            amount_eur=invoice.amount_eur,
            purchase_date=purchase_date,
            rejection_reason=result.rejection_reasons_text,
            failed_rules=[r.rule_code.value for r in result.failed_rules],
            rejection_diagnostics=[r.diagnostic for r in result.failed_rules if r.diagnostic],
        )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class CustomerLogEntry(BaseModel):
    """A single entry in the customer log database."""

    id: Optional[int] = None
    programa: Optional[str] = None
    seller_id: str
    seller_name: str
    debtor_id: Optional[str] = None
    debtor_name: Optional[str] = None
    insurer_id: Optional[str] = None
    doc_number: Optional[str] = None
    fiscal_year: Optional[str] = None
    reference_number: Optional[str] = None
    invoice_ref: str
    goods_services: Optional[str] = None
    item: Optional[str] = None
    total_invoice_amount_original: Optional[Decimal] = None
    total_net_value_original: Optional[Decimal] = None
    discount_percentage: Optional[Decimal] = None
    original_currency: Optional[str] = None
    funding_currency: Optional[str] = None
    exchange_rate: Optional[Decimal] = None
    despatch_date: Optional[date] = None
    issuance_date: Optional[date] = None
    due_date: Optional[date] = None
    margin: Optional[Decimal] = None
    purchase_date: Optional[date] = None
    processed_date: datetime
    is_eligible: bool
    rejection_rules: Optional[List[str]] = None
    amount: Optional[Decimal] = None
    currency: Optional[str] = None

    @computed_field
    @property
    def invoice_number(self) -> str:
        """UI-compatible alias for invoice_ref."""
        return self.invoice_ref

    @computed_field
    @property
    def status(self) -> str:
        """UI-compatible status field based on is_eligible."""
        return "Funded" if self.is_eligible else "Non-Funded"

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class RuleRejectionStats(BaseModel):
    """Statistics for rejections by a specific rule."""

    rule_code: str
    rule_description: str
    count: int
    percentage: float = Field(..., description="Percentage of total rejections")


class CustomerLogSummary(BaseModel):
    """Summary of historical rejection patterns for a seller."""

    seller_id: str
    seller_name: Optional[str] = None
    total_invoices_processed: int
    total_eligible: int
    total_rejected: int
    eligibility_rate: float = Field(..., description="Percentage of eligible invoices")
    rejection_by_rule: List[RuleRejectionStats] = Field(
        default_factory=list,
        description="Breakdown of rejections by rule",
    )
    first_processed: Optional[datetime] = None
    last_processed: Optional[datetime] = None

    @computed_field
    @property
    def rejection_breakdown(self) -> Dict[str, int]:
        """UI-compatible dict of {rule_code: count} for rejection statistics."""
        return {stat.rule_code: stat.count for stat in self.rejection_by_rule}

    @computed_field
    @property
    def total_invoices(self) -> int:
        """UI-compatible alias for total_invoices_processed."""
        return self.total_invoices_processed

    @computed_field
    @property
    def funded_invoices(self) -> int:
        """UI-compatible alias for total_eligible."""
        return self.total_eligible

    @computed_field
    @property
    def non_funded_invoices(self) -> int:
        """UI-compatible alias for total_rejected."""
        return self.total_rejected

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class AnalysisRequest(BaseModel):
    """Request parameters for eligibility analysis (when not using query params)."""

    purchase_date: Optional[date] = Field(None, description="Purchase date (defaults to today)")
    nddt: Optional[int] = Field(None, description="Minimum days to due date (R1)")
    teih: Optional[int] = Field(None, description="Maximum tenor days (R16)")
    isspur: Optional[int] = Field(None, description="Minimum days since issuance (R17)")
    eligible_currencies: Optional[str] = Field(
        None,
        description="Comma-separated allowed currencies (R11)",
    )


class AnalysisResponse(BaseModel):
    """Response from the eligibility analysis endpoint."""

    success: bool
    total_invoices: int
    funded_count: int
    non_funded_count: int
    funded_invoices: List[FundedInvoice] = Field(default_factory=list)
    non_funded_invoices: List[NonFundedInvoice] = Field(default_factory=list)
    output_file: Optional[str] = None
    error: Optional[str] = None
    settings_used: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Decimal: lambda v: str(v),
            date: lambda v: v.isoformat(),
        }


class ConfigResponse(BaseModel):
    """Response showing current configuration."""

    nddt: int
    teih: int
    isspur: int
    eligible_currencies: List[str]


class SellerHistoryResponse(BaseModel):
    """Response containing paginated seller history records."""

    records: List[CustomerLogEntry] = Field(default_factory=list)
    total: int = Field(..., description="Total number of records returned")
