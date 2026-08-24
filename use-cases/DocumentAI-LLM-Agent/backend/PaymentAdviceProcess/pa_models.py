"""
pa_models.py
------------
Pydantic models for the Payment Advice extraction and posting pipeline.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PaymentAdviceLine(BaseModel):
    """One invoice/item being settled in the payment advice."""

    invoice_number: str = ""
    invoice_date: str = ""
    gross_amount: float = 0.0
    discount_amount: float = 0.0
    net_payment_amount: float = 0.0
    currency: str = ""
    payment_reference: str = ""     # maps to AssignmentReference in S4


class ExtractedPaymentAdvice(BaseModel):
    """Structured payment advice extracted from a PDF via SAP Document AI."""

    payer_name: str = ""            # company sending the payment
    payer_bp: str = ""              # resolved Business Partner (filled by validator)
    payment_date: str = ""          # ISO YYYY-MM-DD
    total_amount: float = 0.0
    currency: str = ""
    bank_reference: str = ""        # bank transaction / wire reference
    our_reference: str = ""         # recipient's internal reference
    payment_advice_note: str = ""   # free-text note / memo
    line_items: list[PaymentAdviceLine] = Field(default_factory=list)
    raw_sap_fields: dict = Field(default_factory=dict)


class PostPaymentAdviceRequest(BaseModel):
    """Request payload to post a Payment Advice to S/4HANA FI."""

    payer_name: str = Field(..., description="Payer name for BP auto-match")
    payer_bp: str = Field(default="", description="BP number override — auto-match if empty")
    payment_date: str = Field(..., description="Payment date ISO YYYY-MM-DD")
    total_amount: float = Field(..., description="Total payment amount")
    currency: str = Field(..., description="Document currency (e.g. EUR, USD)")
    bank_reference: str = Field(default="", description="Bank transaction reference")
    payment_advice_note: str = Field(
        default="Payment Advice from Document AI",
        description="Free-text note on the payment advice header",
    )
    line_items: list[PaymentAdviceLine] = Field(default_factory=list)


class PostPaymentAdviceResponse(BaseModel):
    """Response after posting a Payment Advice to S/4HANA FI."""

    success: bool
    payment_advice: str = Field(default="", description="Payment Advice document number in S4")
    company_code: str = ""
    business_partner_used: str = ""
    payer_name_matched: str = ""
    error: str = ""
