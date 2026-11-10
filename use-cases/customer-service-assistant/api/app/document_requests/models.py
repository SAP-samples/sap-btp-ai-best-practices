from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ClientCountry = str
DocumentType = Literal[
    "invoice",
    "billing_document",
    "broker_file",
    "certificate",
    "packing_list",
    "delivery_note",
    "transportation_order",
    "outbound_delivery_status",
    "other",
]
PickingStatus = Literal["completed", "partially_completed", "not_started"]
PGIStatus = Literal["completed", "not_started"]
DraftStatus = Literal[
    "ready_for_review",
    "needs_human_review",
    "not_document_request",
    "failed",
    "approved",
    "rejected",
]


class RequestedDocument(BaseModel):
    """Document type and optional identifier codes extracted from one email.

    Attributes:
        document_type: Requested document family such as invoice or packing list.
        identifiers: Specific codes to find in raw document text.
        notes: Short extraction note preserved for review.
        confidence: Model confidence for this requested document.
    """

    document_type: DocumentType | None = None
    identifiers: list[str] = Field(default_factory=list)
    notes: str | None = None
    confidence: float | None = None


class DocumentRequestExtraction(BaseModel):
    """Normalized extraction result for an incoming Gmail message.

    Attributes:
        is_document_request: True when the sender asks for shipment documents or order status.
        reason: Short explanation of the classification.
        confidence: Classification confidence from 0 to 1.
        sales_order_id: Sales order number extracted from the email (primary identifier).
        outbound_delivery_id: Outbound delivery number if mentioned.
        line_item_number: Specific line item number if a line-level query.
        product_id: Product/material id when the sender asks about a product.
        batch_number: Batch/lot number when the sender asks about a batch.
        requested_information: Compact free-text summary of the facts requested.
        client_country: Client country inferred from email content.
        requested_documents: Specific requested document families and identifiers.
        send_all: True when the request is broad or unclear and all order documents should be attached.
    """

    is_document_request: bool
    reason: str
    confidence: float
    sales_order_id: str | None = None
    outbound_delivery_id: str | None = None
    line_item_number: str | None = None
    product_id: str | None = None
    batch_number: str | None = None
    requested_information: str | None = None
    client_country: ClientCountry | None = None
    requested_documents: list[RequestedDocument] = Field(default_factory=list)
    send_all: bool = False


class GmailEmail(BaseModel):
    """Normalized Gmail email stored before extraction.

    Attributes:
        gmail_message_id: Gmail API message id.
        thread_id: Gmail thread id for reply threading.
        rfc_message_id: RFC Message-ID header used for In-Reply-To.
        sender_name: Sender display name.
        sender_email: Sender email address.
        subject: Email subject.
        email_date: Parsed email Date header.
        body: Readable plain text email body.
        fetched_at: Timestamp when the API fetched the message.
        processed_at: Timestamp when processing finished.
        processing_status: Pipeline status for this email.
    """

    gmail_message_id: str
    thread_id: str | None = None
    rfc_message_id: str | None = None
    sender_name: str | None = None
    sender_email: str | None = None
    subject: str | None = None
    email_date: datetime | None = None
    body: str
    fetched_at: datetime | None = None
    processed_at: datetime | None = None
    processing_status: str = "fetched"


class DocumentAttachment(BaseModel):
    """Selected document attachment with metadata and binary content.

    Attributes:
        document_id: Stable HANA document id.
        file_name: Original file name to use in the outgoing email.
        document_type: Normalized document type.
        mime_type: Attachment MIME type.
        size_bytes: File size in bytes.
        content: Raw file bytes retrieved from HANA.
    """

    document_id: str
    file_name: str
    document_type: DocumentType
    mime_type: str
    size_bytes: int
    content: bytes

    def model_dump_for_ui(self) -> dict[str, Any]:
        """Serialize attachment metadata without exposing binary content.

        Returns:
            Dictionary suitable for JSON API responses.
        """
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }


class DraftEmailProposal(BaseModel):
    """Draft response email proposed by the agent and reviewed by a user.

    Attributes:
        status: Review status for the draft.
        gmail_message_id: Source Gmail message id.
        extraction_id: Stored extraction id.
        sales_order_id: Sales order used for document retrieval.
        client_country: Country bucket used for HANA document retrieval.
        to_email: Recipient email address.
        subject: Draft reply subject.
        body: Draft email body.
        selected_documents: Attachments selected for the response.
        confidence: Confidence propagated from extraction and matching.
        explanation: Human-readable summary of matching decisions.
        validation_errors: Blocking issues requiring human review.
    """

    status: DraftStatus
    gmail_message_id: str
    extraction_id: str
    sales_order_id: str | None = None
    client_country: ClientCountry | None = None
    to_email: str | None
    subject: str
    body: str
    selected_documents: list[DocumentAttachment] = Field(default_factory=list)
    confidence: float
    explanation: str
    validation_errors: list[str] = Field(default_factory=list)


class FetchSummary(BaseModel):
    """Summary returned by the Gmail fetch endpoint.

    Attributes:
        total: Number of emails found for this run (fetched + skipped_existing when finished).
        fetched: Number of new Gmail messages stored for processing.
        skipped_existing: Number of already-processed Gmail messages skipped.
        not_document_request: Number of messages classified outside the feature scope.
        extraction_failed: Number of messages whose extraction output could not be normalized.
        agent_failed: Number of messages whose draft resolution failed.
        drafts_created: Number of review rows created.
        ready_for_review: Number of drafts ready for approval.
        needs_human_review: Number of drafts needing manual inspection.
        created_draft_ids: Persisted draft ids created during this fetch run.
    """

    total: int = 0
    fetched: int = 0
    skipped_existing: int = 0
    not_document_request: int = 0
    extraction_failed: int = 0
    agent_failed: int = 0
    drafts_created: int = 0
    ready_for_review: int = 0
    needs_human_review: int = 0
    created_draft_ids: list[str] = Field(default_factory=list)


class FetchRunStatus(BaseModel):
    """State of a background email-processing run, as polled by the UI.

    Attributes:
        source: Run source, "gmail" (inbox fetch) or "scenarios" (sample emails).
        status: idle (never run), running, success, or failed.
        started_at: When the latest run started.
        finished_at: When the latest run finished (None while running).
        summary: Progress counters; updated after every processed email.
        error_message: Failure reason when status is failed.
    """

    source: str
    status: str = "idle"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary: FetchSummary = Field(default_factory=FetchSummary)
    error_message: str | None = None


JsonObject = dict[str, Any]


class OrderStatus(BaseModel):
    """Mock order status representing picking/packing/PGI state from S4.

    Attributes:
        sales_order_id: Sales order number.
        client_country: Country associated with this order.
        picking_status: Picking status at header or line-item level.
        packing_status: Packing status.
        pgi_status: Post Goods Issue (shipped out of stock) status.
        shipped: Whether the order has been fully shipped.
        shipped_date: Shipping date when PGI is completed.
        back_order: Whether there are unfulfilled line items.
        expected_completion: Expected completion/shipping date for pending items.
    """

    sales_order_id: str
    client_country: str | None = None
    picking_status: PickingStatus = "not_started"
    packing_status: PickingStatus = "not_started"
    pgi_status: PGIStatus = "not_started"
    shipped: bool = False
    shipped_date: str | None = None
    back_order: bool = False
    expected_completion: str | None = None
