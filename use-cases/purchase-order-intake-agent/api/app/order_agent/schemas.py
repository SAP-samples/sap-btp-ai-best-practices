"""Pydantic schema of the ``submit_order`` tool (what the agent hands over for validation).

Every field is a ``FieldValue`` so the reviewer later sees, per field, the normalised value,
the text as printed, the agent's confidence, where it was found and any fix the agent applied.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class FieldValue(BaseModel):
    """One extracted or resolved value (rules: header-extraction and confidence-and-fixes skills)."""

    value: Optional[Union[str, float]] = Field(default=None, description="normalised value")
    raw: Optional[str] = Field(default=None, description="as printed")
    confidence: float = Field(ge=0, le=1)
    source: Optional[str] = Field(default=None, description="where found")
    note: Optional[str] = Field(default=None, description="fix applied")


HEADER_FIELD_HELP = (
    "Header fields as {field_name: FieldValue}. Allowed names: po_number, po_date, requested_delivery_date, "
    "sold_to (value = customer number, raw = buyer name as printed), sold_to_name, sold_to_address, "
    "buyer_tax_id, buyer_phone, ship_to (value = ship-to customer number), ship_to_name, ship_to_address, "
    "sales_org, currency (ISO 4217), net_amount (final total before tax), gross_amount, payment_terms_text, "
    "incoterms (code), incoterms_place, header_note. Omit fields that are not on the document."
)
LINE_FIELD_HELP = (
    "Line fields as {field_name: FieldValue}. Allowed names: customer_item_no, material (value = material "
    "number, raw = code as printed), customer_material, description (value = as printed; Vietnamese translated "
    "to English), quantity (value = quantity in uom after conversion, raw = as printed with unit), "
    "uom (value = SAP unit, raw = as printed), "
    "unit_price (as printed), line_amount (as printed), colour_code (RAL / NCS / tint code of a coating only)."
)


class MatchEvidence(BaseModel):
    """Why the chosen material is the product on the line (rules: master-data-matching skill)."""

    matched: list[str] = Field(default_factory=list, description=(
        "Attributes of the printed line that the chosen material shares, each 'attribute: value', e.g. "
        "'product line: 325U Pelkit disc', 'grit: P600', 'size: 150 mm', 'pack: 100 pieces', "
        "'manufacturer part: 43038', 'price: 49.00 EUR = customer price'."))
    conflicts: list[str] = Field(default_factory=list, description=(
        "Attributes that disagree or that the line states but the material does not confirm, e.g. "
        "'pack: line 4 L, material 1 L'. Empty when everything agrees."))
    alternatives: list[str] = Field(default_factory=list, description=(
        "Material numbers of other candidates that fit the line equally well (you could not decide)."))


class LineSubmission(BaseModel):
    """One order line in document order."""

    line_type: Literal["MATERIAL", "CHARGE", "TEXT"] = Field(
        default="MATERIAL", description="CHARGE for freight/surcharges/fees, TEXT for notes that must be kept.")
    fields: dict[str, FieldValue] = Field(default_factory=dict, description=LINE_FIELD_HELP)
    material_evidence: Optional[MatchEvidence] = Field(default=None, description=(
        "Required whenever material.value is set: how the chosen material matches the printed line."))


class ReviewFlag(BaseModel):
    """A point the reviewer must check (conflicts, inconsistencies you could not resolve)."""

    line: Optional[int] = Field(default=None, description="1-based line number in `lines`; omit for header-level points.")
    field: Optional[str] = Field(default=None, description="Field name, e.g. 'ship_to', 'quantity'.")
    message: str


class OrderSubmission(BaseModel):
    """Complete hand-over of one order."""

    classification: Literal["STANDARD", "CONTRACT", "NOT_AN_ORDER"]
    classification_reason: str = Field(description="One or two sentences citing the evidence.")
    multiple_orders_detected: bool = False
    header: dict[str, FieldValue] = Field(default_factory=dict, description=HEADER_FIELD_HELP)
    lines: list[LineSubmission] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)


HEADER_COLUMNS = {
    "po_number": "PO_NUMBER", "po_date": "PO_DATE", "requested_delivery_date": "REQ_DELIVERY_DATE",
    "sold_to": "SOLD_TO", "sold_to_name": "SOLD_TO_NAME", "sold_to_address": "SOLD_TO_ADDRESS",
    "ship_to": "SHIP_TO", "ship_to_name": "SHIP_TO_NAME", "ship_to_address": "SHIP_TO_ADDRESS",
    "sales_org": "SALES_ORG", "currency": "CURRENCY", "net_amount": "NET_AMOUNT", "gross_amount": "GROSS_AMOUNT",
    "payment_terms_text": "PAYMENT_TERMS_TEXT", "incoterms": "INCOTERMS", "incoterms_place": "INCOTERMS_PLACE",
    "header_note": "HEADER_NOTE",
}
# Identifiers kept only as field metadata (used for matching confidence, not stored as columns).
HEADER_META_ONLY = {"buyer_tax_id": "BUYER_TAX_ID", "buyer_phone": "BUYER_PHONE"}
LINE_COLUMNS = {
    "customer_item_no": "CUSTOMER_ITEM_NO", "material": "MATERIAL", "customer_material": "CUSTOMER_MATERIAL",
    "description": "DESCRIPTION", "quantity": "QUANTITY", "uom": "UOM", "unit_price": "UNIT_PRICE",
    "line_amount": "LINE_AMOUNT", "colour_code": "COLOUR_CODE",
}
