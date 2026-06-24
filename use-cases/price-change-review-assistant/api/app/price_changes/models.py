from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


PriceMode = Literal["absolute", "relative_percent", "relative_amount"]
DraftStatus = Literal[
    "ready_for_review",
    "needs_human_review",
    "not_price_request",
    "failed",
    "approved",
    "rejected",
]


class RequestedPrice(BaseModel):
    mode: PriceMode | None = None
    value: str | None = None


class RawExtractionItem(BaseModel):
    """Raw model-extracted clues for one requested price-change line.

    Attributes:
        material_number: Primary material number selected by the extractor, if clear.
        material_numbers: All material-like numbers mentioned for this item, including
            incorrect numbers and correction candidates.
        material_description: Product description clue from the email.
        requested_price: Requested price change value and mode.
        currency: Requested or mentioned currency.
        valid_from_raw: Raw effective-from text from the email.
        valid_to_raw: Raw effective-to text from the email.
        uom: Unit of measure mentioned in the email.
        supplier_material_number: Supplier-side material number, if separate.
        quotation_number: Referenced quotation number, if present.
        notes: Extra item-level extraction notes.
        confidence: Extractor confidence for this item.
    """

    material_number: str | None = None
    material_numbers: list[str] = Field(default_factory=list)
    material_description: str | None = None
    requested_price: RequestedPrice | None = None
    currency: str | None = None
    valid_from_raw: str | None = None
    valid_to_raw: str | None = None
    uom: str | None = None
    supplier_material_number: str | None = None
    quotation_number: str | None = None
    notes: str | None = None
    confidence: float | None = None


class RawExtraction(BaseModel):
    is_price_request: bool
    reason: str
    confidence: float
    supplier_id: str | None = None
    supplier_email: str | None = None
    supplier_name: str | None = None
    items: list[RawExtractionItem] = Field(default_factory=list)


class GmailEmail(BaseModel):
    gmail_message_id: str
    thread_id: str | None = None
    sender_name: str | None = None
    sender_email: str | None = None
    subject: str | None = None
    email_date: datetime | None = None
    body: str
    fetched_at: datetime | None = None
    processed_at: datetime | None = None
    processing_status: str = "fetched"


class EmailAttachment(BaseModel):
    """Persistable email attachment content and extraction preview data.

    Attributes:
        attachment_id: Repository-assigned attachment id.
        gmail_message_id: Gmail or synthetic manual message id owning the attachment.
        source: Input route that provided the attachment.
        provider_attachment_id: Gmail attachment id when downloaded from Gmail.
        filename: Sanitized original filename.
        mime_type: Attachment MIME type.
        file_extension: Supported lowercase extension without a leading dot.
        size_bytes: Raw attachment byte size.
        content_base64: Base64-encoded raw attachment bytes for HANA NCLOB storage.
        text_preview: Bounded text extracted from CSV/XLSX attachments for model input.
    """

    attachment_id: str | None = None
    gmail_message_id: str | None = None
    source: Literal["gmail", "manual"]
    provider_attachment_id: str | None = None
    filename: str
    mime_type: str
    file_extension: Literal["pdf", "csv", "xlsx"]
    size_bytes: int
    content_base64: str
    text_preview: str | None = None


class EmailAttachmentSummary(BaseModel):
    """Attachment metadata exposed to the UI without raw content.

    Attributes:
        attachment_id: Repository-assigned attachment id used for downloads.
        filename: Sanitized original filename.
        mime_type: Attachment MIME type.
        file_extension: Supported lowercase extension without a leading dot.
        size_bytes: Raw attachment byte size.
    """

    attachment_id: str
    filename: str
    mime_type: str
    file_extension: str
    size_bytes: int


class ProcessingRunCreateRequest(BaseModel):
    """Request body for creating a processing progress run before work starts.

    Attributes:
        processing_run_id: Browser-generated run id used for polling.
        source_type: Processing source flow.
    """

    processing_run_id: str = Field(min_length=1)
    source_type: Literal["manual", "gmail_fetch"] = "manual"

    @field_validator("processing_run_id")
    @classmethod
    def strip_processing_run_id(cls, value: str) -> str:
        """Trim and require a processing run id.

        Args:
            value: Raw run id from the UI.

        Returns:
            Trimmed run id.

        Raises:
            ValueError: If the run id is blank.
        """
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("processing_run_id must not be blank.")
        return trimmed


class ManualEmailRequest(BaseModel):
    """Manual email payload submitted from the UI for testing the processing flow.

    Attributes:
        sender_email: Email address to use as the synthetic sender.
        subject: Subject line to pass to extraction and audit views.
        body: Plain-text email body to pass to extraction and audit views.
    """

    sender_email: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    processing_run_id: str | None = None

    @field_validator("sender_email", "subject", "body")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        """Trim manual email text fields and reject blank values.

        Args:
            value: Raw request value for a required manual email field.

        Returns:
            Trimmed text value.

        Raises:
            ValueError: If the trimmed text is empty.
        """
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Manual email fields must not be blank.")
        return trimmed

    @field_validator("processing_run_id")
    @classmethod
    def strip_optional_processing_run_id(cls, value: str | None) -> str | None:
        """Trim an optional browser-generated processing run id.

        Args:
            value: Raw processing run id from the UI.

        Returns:
            Trimmed run id, or None when omitted or blank.
        """
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class SupplierCandidate(BaseModel):
    supplier_id: str
    supplier_name: str
    supplier_email: str | None = None
    score: float


class MaterialCandidate(BaseModel):
    material_number: str
    material_description: str
    supplier_id: str | None = None
    current_price: str | None = None
    currency: str | None = None
    score: float


class AgentProposal(BaseModel):
    status: DraftStatus
    supplier_id: str | None = None
    supplier_name: str | None = None
    supplier_email: str | None = None
    material_number: str | None = None
    material_description: str | None = None
    original_price: str | None = None
    requested_new_price: str | None = None
    currency: str | None = None
    uom: str | None = None
    price_change_mode: PriceMode | None = None
    price_change_value: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    email_date: str
    gmail_message_id: str
    confidence: float
    explanation: str
    candidate_materials: list[MaterialCandidate] = Field(default_factory=list)
    candidate_suppliers: list[SupplierCandidate] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class DraftPatchRequest(BaseModel):
    supplier_id: str | None = None
    supplier_name: str | None = None
    supplier_email: str | None = None
    material_number: str | None = None
    material_description: str | None = None
    original_price: str | None = None
    requested_new_price: str | None = None
    currency: str | None = None
    uom: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    status: DraftStatus | None = None
    explanation: str | None = None


class TokenUsageRecord(BaseModel):
    """Token usage for one LLM call made during email processing.

    Attributes:
        stage: Logical processing stage that made the LLM call.
        model_name: GenAI Hub model name used for the call.
        gmail_message_id: Gmail message processed by the call, when known.
        extraction_id: Persisted extraction id for agent calls, when known.
        item_index: Extracted item index for agent calls, when known.
        call_index: One-based sequence number within this API fetch run.
        input_tokens: Prompt/input token count.
        output_tokens: Completion/output token count.
        total_tokens: Provider total token count, or input plus output when omitted.
        usage_available: True when the provider exposed token metadata.
    """

    stage: str
    model_name: str
    gmail_message_id: str | None = None
    extraction_id: str | None = None
    item_index: int | None = None
    call_index: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = False


class TokenUsageGroup(BaseModel):
    """Aggregate token totals for a set of LLM calls.

    Attributes:
        calls: Number of model calls in this group.
        input_tokens: Sum of prompt/input tokens.
        output_tokens: Sum of completion/output tokens.
        total_tokens: Sum of total tokens.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TokenUsageReport(BaseModel):
    """Per-run LLM token usage report returned only when requested.

    Attributes:
        records: Per-call usage records.
        totals: Aggregate totals across all recorded calls.
        by_stage: Aggregate totals keyed by processing stage.
        by_model: Aggregate totals keyed by model name.
    """

    records: list[TokenUsageRecord] = Field(default_factory=list)
    totals: TokenUsageGroup = Field(default_factory=TokenUsageGroup)
    by_stage: dict[str, TokenUsageGroup] = Field(default_factory=dict)
    by_model: dict[str, TokenUsageGroup] = Field(default_factory=dict)


class FetchSummary(BaseModel):
    """Summary returned by the Gmail fetch endpoint.

    Attributes:
        fetched: Number of new Gmail messages fetched.
        skipped_existing: Number of Gmail messages already present in HANA.
        not_price_request: Number of messages classified as non-price requests.
        extraction_failed: Number of extraction failures.
        agent_failed: Number of agent failures.
        drafts_created: Number of draft proposals created.
        needs_human_review: Number of drafts requiring manual review.
        ready_for_review: Number of drafts ready for review.
        attachments_downloaded: Number of supported attachments persisted.
        attachments_skipped: Number of unsupported or over-limit attachments skipped.
        attachments_failed: Number of supported Gmail attachments that could not be downloaded.
        created_draft_ids: Draft ids created by this fetch run.
        token_usage: Optional per-call LLM token report for local debugging.
    """

    fetched: int = 0
    skipped_existing: int = 0
    not_price_request: int = 0
    extraction_failed: int = 0
    agent_failed: int = 0
    drafts_created: int = 0
    needs_human_review: int = 0
    ready_for_review: int = 0
    attachments_downloaded: int = 0
    attachments_skipped: int = 0
    attachments_failed: int = 0
    created_draft_ids: list[str] = Field(default_factory=list)
    token_usage: TokenUsageReport | None = None


JsonObject = dict[str, Any]
