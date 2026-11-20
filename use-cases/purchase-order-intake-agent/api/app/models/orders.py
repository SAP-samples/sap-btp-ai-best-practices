"""API models of the purchase-order endpoints (JSON keys in snake_case).

Customer, material and sales-document numbers are returned in display format (no leading
zeros); the API converts them back to SAP internal format on input.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

OrderStatus = Literal["QUEUED", "PROCESSING", "NEEDS_REVIEW", "POSTED", "FAILED", "NOT_AN_ORDER"]


class UploadedOrder(BaseModel):
    """One order created by an upload."""

    order_id: str
    message_id: str
    status: OrderStatus = "QUEUED"
    source: str


class UploadResponse(BaseModel):
    """Response of POST /api/orders/upload."""

    orders: list[UploadedOrder]


class OrderSummary(BaseModel):
    """One row of the incoming purchase orders table."""

    order_id: str
    status: OrderStatus
    classification: Optional[str] = None
    status_message: Optional[str] = None
    po_number: Optional[str] = None
    po_date: Optional[date] = None
    sold_to: Optional[str] = None
    sold_to_name: Optional[str] = None
    ship_to: Optional[str] = None
    ship_to_name: Optional[str] = None
    sales_org: Optional[str] = None
    currency: Optional[str] = None
    net_amount: Optional[float] = None
    sales_doc: Optional[str] = None
    sender: Optional[str] = None
    subject: Optional[str] = None
    received_at: Optional[datetime] = None
    source_type: Optional[str] = None
    item_count: int = 0
    issue_count: int = 0
    min_confidence: Optional[float] = None
    updated_at: Optional[datetime] = None


class OrderListResponse(BaseModel):
    """Response of GET /api/orders."""

    items: list[OrderSummary]


class OrderHeader(BaseModel):
    """Editable and system header fields of an order."""

    order_id: str
    message_id: str
    status: OrderStatus
    classification: Optional[str] = None
    classification_reason: Optional[str] = None
    status_message: Optional[str] = None
    attempts: int = 0
    reviewer_note: Optional[str] = None
    po_number: Optional[str] = None
    po_date: Optional[date] = None
    req_delivery_date: Optional[date] = None
    sold_to: Optional[str] = None
    sold_to_name: Optional[str] = None
    sold_to_address: Optional[str] = None
    ship_to: Optional[str] = None
    ship_to_name: Optional[str] = None
    ship_to_address: Optional[str] = None
    sales_org: Optional[str] = None
    distr_channel: Optional[str] = None
    division: Optional[str] = None
    order_type: Optional[str] = None
    po_type: Optional[str] = None
    currency: Optional[str] = None
    net_amount: Optional[float] = None
    gross_amount: Optional[float] = None
    payment_terms_text: Optional[str] = None
    payment_terms: Optional[str] = None
    incoterms: Optional[str] = None
    incoterms_place: Optional[str] = None
    header_note: Optional[str] = None
    agent_summary: Optional[str] = None
    sales_doc: Optional[str] = None
    posting_payload: Optional[str] = None
    posted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class OrderItem(BaseModel):
    """One order line."""

    item_no: int
    line_type: Literal["MATERIAL", "CHARGE", "TEXT"] = "MATERIAL"
    customer_item_no: Optional[str] = None
    material: Optional[str] = None
    customer_material: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    uom: Optional[str] = None
    unit_price: Optional[float] = None
    line_amount: Optional[float] = None
    colour_code: Optional[str] = None
    plant: Optional[str] = None


class FieldState(BaseModel):
    """Extraction / validation metadata of one field."""

    raw_value: Optional[str] = None
    source_ref: Optional[str] = None
    extract_confidence: Optional[float] = None
    match_confidence: Optional[float] = None
    confidence: Optional[float] = None
    changed_by: str = "AGENT"
    fix_note: Optional[str] = None
    issue_severity: Optional[Literal["ERROR", "REVIEW", "WARNING", "INFO"]] = None
    issue_code: Optional[str] = None
    issue_text: Optional[str] = None
    evidence: Optional[dict[str, Any]] = Field(
        default=None, description="Material match evidence: matched, conflicts, alternatives, price_check")

    @field_validator("evidence", mode="before")
    @classmethod
    def _parse_evidence(cls, value: Any) -> Any:
        """Accept the JSON text stored in POAGENT_FIELD_STATE.EVIDENCE."""
        return json.loads(value) if isinstance(value, str) else value


class Issue(BaseModel):
    """A validation finding; ``item_no`` 0 = header, field ``_ORDER`` = whole order."""

    severity: Literal["ERROR", "REVIEW", "WARNING", "INFO"]
    item_no: int = 0
    field: str
    code: Optional[str] = None
    message: str


class MessageInfo(BaseModel):
    """The inbound e-mail / upload an order came from."""

    source_type: str
    sender: Optional[str] = None
    recipients: Optional[str] = None
    subject: Optional[str] = None
    sent_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    body_text: Optional[str] = None
    sales_org_hint: Optional[str] = None


class AttachmentInfo(BaseModel):
    """A stored file; ``viewer`` tells the UI how to display it."""

    attachment_id: str
    parent_id: Optional[str] = None
    file_name: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    role: str
    model_input: str
    model_input_reason: Optional[str] = None
    viewer: Literal["pdf", "image", "text", "download"]
    extracted_text: Optional[str] = None


class RunSummary(BaseModel):
    """Latest agent run of the order."""

    outcome: Optional[str] = None
    model: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    llm_calls: Optional[int] = None
    input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_text: Optional[str] = None


class OrderDetail(BaseModel):
    """Response of GET /api/orders/{order_id}."""

    header: OrderHeader
    items: list[OrderItem] = Field(default_factory=list)
    fields: dict[str, dict[str, FieldState]] = Field(
        default_factory=dict, description="item_no (as string, '0' = header) -> field name -> state"
    )
    issues: list[Issue] = Field(default_factory=list)
    message: MessageInfo
    attachments: list[AttachmentInfo] = Field(default_factory=list)
    run: Optional[RunSummary] = None


class ItemUpdate(BaseModel):
    """One line in a Save request; omitted fields keep their value, null clears it."""

    item_no: Optional[int] = Field(default=None, description="Existing item number; omit for a new line")
    line_type: Optional[Literal["MATERIAL", "CHARGE", "TEXT"]] = None
    customer_item_no: Optional[str] = None
    material: Optional[str] = None
    customer_material: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    uom: Optional[str] = None
    unit_price: Optional[float] = None
    line_amount: Optional[float] = None
    colour_code: Optional[str] = None
    plant: Optional[str] = None


class OrderUpdate(BaseModel):
    """Body of PUT /api/orders/{id} (and optional body of confirm)."""

    header: dict[str, Optional[str | float | int]] = Field(
        default_factory=dict, description="Changed header fields by API name, e.g. {'ship_to': '762431'}")
    items: Optional[list[ItemUpdate]] = Field(default=None, description="Complete new line list; omit to keep lines")
    classification: Optional[Literal["STANDARD", "CONTRACT", "NOT_AN_ORDER"]] = None


class ConfirmResponse(BaseModel):
    """Result of a successful confirmation."""

    order_id: str
    status: OrderStatus
    sales_doc: str


class ReprocessRequest(BaseModel):
    """Body of POST /api/orders/{id}/reprocess."""

    note: Optional[str] = Field(default=None, max_length=2000, description="Instruction for the agent")


class DeleteResponse(BaseModel):
    """Answer of a successful order deletion."""

    order_id: str
    deleted: bool = True


class StatusResponse(BaseModel):
    """Generic status answer."""

    order_id: str
    status: OrderStatus


class SalesOrderSummary(BaseModel):
    """One posted sales document (order history)."""

    sales_doc: str
    order_type: Optional[str] = None
    sales_org: Optional[str] = None
    sold_to: Optional[str] = None
    sold_to_name: Optional[str] = None
    ship_to: Optional[str] = None
    ship_to_name: Optional[str] = None
    po_number: Optional[str] = None
    po_date: Optional[date] = None
    currency: Optional[str] = None
    net_value: Optional[float] = None
    item_count: int = 0
    created_on: Optional[date] = None
    order_id: Optional[str] = None


class SalesOrderListResponse(BaseModel):
    """Response of GET /api/sales-orders."""

    items: list[SalesOrderSummary]
