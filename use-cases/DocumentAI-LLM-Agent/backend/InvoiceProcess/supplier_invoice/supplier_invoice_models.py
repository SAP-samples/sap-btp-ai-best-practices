"""
supplier_invoice_models.py
--------------------------
Pydantic models for the FI Supplier Invoice posting endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PostInvoiceRequest(BaseModel):
    supplier_name: str = Field(..., description="Supplier name extracted from the document")
    invoice_number: str = Field(..., description="Invoice number extracted from the document")
    invoice_date: str = Field(..., description="Invoice date in ISO format YYYY-MM-DD")
    total_amount: float = Field(..., description="Gross invoice amount")
    currency: str = Field(..., description="Document currency (e.g. USD, EUR)")
    business_partner: str = Field(
        default="",
        description="BP number override — leave empty to auto-match by supplier_name",
    )
    gl_account: str = Field(
        default="",
        description="GL account override — leave empty to use the server default (FI_EXPENSE_GL_ACCOUNT)",
    )


class PostInvoiceResponse(BaseModel):
    success: bool
    fi_document: str = Field(default="", description="FI document number created in S/4HANA")
    company_code: str = ""
    fiscal_year: str = ""
    business_partner_used: str = Field(
        default="", description="Business Partner number that was used"
    )
    supplier_name_matched: str = Field(
        default="", description="Supplier name as found in S/4HANA"
    )
    error: str = ""
