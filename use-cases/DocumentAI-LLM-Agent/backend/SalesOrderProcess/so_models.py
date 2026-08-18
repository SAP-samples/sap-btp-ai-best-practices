"""
so_models.py
------------
Pydantic models for the Sales Order Process module.

Covers the full pipeline:
  extracted PO  →  validation result  →  create request  →  create response
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Line item — shared across extraction, validation and creation
# ---------------------------------------------------------------------------


class SalesOrderLineItem(BaseModel):
    """A single line item from the customer Purchase Order / Sales Order."""

    material_code: str = ""
    """Material code as extracted from the customer PO (may be customer-internal)."""

    sap_material: str = ""
    """SAP material number resolved during validation."""

    description: str = ""
    quantity: float = 1.0
    uom: str = "EA"
    unit_price: Optional[float] = None
    currency: Optional[str] = None


# ---------------------------------------------------------------------------
# Extracted Purchase Order — output of so_extractor
# ---------------------------------------------------------------------------


class ExtractedPurchaseOrder(BaseModel):
    """Structured representation of a customer Purchase Order extracted by SAP Document AI."""

    customer_name: str = ""
    customer_bp: str = ""
    """SAP Business Partner number — filled in after validation."""

    purchase_order_number: str = ""
    order_date: str = ""
    requested_delivery_date: str = ""
    currency: str = ""
    total_amount: Optional[float] = None
    line_items: list[SalesOrderLineItem] = Field(default_factory=list)
    special_instructions: Optional[str] = None
    raw_sap_fields: Optional[dict] = None
    """Raw headerFields dict from SAP Document AI (field name → value)."""


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


class SOValidationResult(BaseModel):
    """Result of validating an extracted Purchase Order against S/4HANA master data."""

    customer_resolved: bool = False
    customer_bp: str = ""
    customer_name_matched: str = ""
    customer_score: float = 0.0

    items_validation: list[dict] = Field(default_factory=list)
    """
    Each entry:
      {
        "material_code_extracted": str,
        "sap_material": str,
        "description": str,
        "matched": bool,
        "score": float,
      }
    """

    ready_to_create: bool = False
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Create request — sent by frontend after user confirms
# ---------------------------------------------------------------------------


class CreateSORequest(BaseModel):
    """Payload to create a Sales Order in S/4HANA."""

    customer_bp: str
    purchase_order_number: Optional[str] = None
    sales_organization: str = "200"
    distribution_channel: str = "10"
    division: str = "00"
    currency: str = "USD"
    items: list[SalesOrderLineItem]
    special_instructions: Optional[str] = None


# ---------------------------------------------------------------------------
# Create response
# ---------------------------------------------------------------------------


class CreateSOResponse(BaseModel):
    """Response after attempting to create a Sales Order in S/4HANA."""

    success: bool
    sales_order: str = ""
    customer: str = ""
    items_created: int = 0
    message: str = ""
    error: str = ""
