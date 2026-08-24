"""
po_invoice_models.py
--------------------
Pydantic models for the PO-based Supplier Invoice posting endpoint.
Replicates MIRO (MM-IV) via API_SUPPLIERINVOICE_PROCESS_SRV deep-create
with to_SuplrInvcItemPurOrdRef node.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PurchaseOrderItem(BaseModel):
    purchase_order: str = Field(..., description="PO number (e.g. 4500020561)")
    purchase_order_item: str = Field(
        default="00010",
        description="PO line item (e.g. 00010)",
    )
    amount: Optional[float] = Field(
        default=None,
        description="Line item amount — defaults to invoice total if omitted",
    )
    tax_code: str = Field(
        default="",
        description="Tax code (e.g. V0) — falls back to FI_PO_TAX_CODE from .env",
    )


class PostPOInvoiceRequest(BaseModel):
    supplier_name: str = Field(..., description="Supplier name extracted from the document")
    invoice_number: str = Field(..., description="Invoice number extracted from the document")
    invoice_date: str = Field(..., description="Invoice date ISO YYYY-MM-DD")
    total_amount: float = Field(..., description="Gross invoice amount")
    currency: str = Field(..., description="Document currency (e.g. EUR, USD)")
    purchase_order: str = Field(..., description="PO number detected in the document")
    purchase_order_item: str = Field(
        default="00010",
        description="PO line item number",
    )
    tax_code: str = Field(
        default="",
        description="Tax code — falls back to FI_PO_TAX_CODE from .env if empty",
    )
    business_partner: str = Field(
        default="",
        description="BP number override — leave empty to auto-match by supplier_name",
    )


class PostPOInvoiceResponse(BaseModel):
    success: bool
    fi_document: str = Field(default="", description="FI document number created in S/4HANA")
    company_code: str = ""
    fiscal_year: str = ""
    business_partner_used: str = ""
    supplier_name_matched: str = ""
    purchase_order: str = ""
    purchase_order_item: str = ""
    error: str = ""
