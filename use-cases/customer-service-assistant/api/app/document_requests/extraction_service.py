from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import create_response_text, default_responses_client
from .models import DocumentRequestExtraction, GmailEmail
from .settings import DocumentRequestSettings

EXTRACTION_SYSTEM_PROMPT = """You classify customer emails and extract document/order-status requests.

Return exactly one JSON object and nothing else.

Top-level fields:
- is_document_request: boolean — true when the sender is asking for shipment documents OR asking about order status
- reason: string — short explanation of why you classified this way
- confidence: number from 0 to 1
- sales_order_id: string or null — the sales order number (e.g. "6090004185")
- outbound_delivery_id: string or null — outbound delivery number if explicitly mentioned
- line_item_number: string or null — specific line item number if the sender asks about a particular line
- product_id: string or null — product/material id if explicitly mentioned
- batch_number: string or null — batch/lot number if explicitly mentioned
- requested_information: string or null — compact phrase naming the facts requested (e.g. "GI date and shipper box")
- client_country: one of "RU", "GB", "ES", "AT", "NO", "UA", "DE", "PL", or null — only if clearly identifiable
- send_all: boolean — true when the request is broad/unclear and all available documents should be sent
- requested_documents: array

Each requested_documents item:
- document_type: one of "invoice", "billing_document", "broker_file", "certificate", "packing_list", "delivery_note", "transportation_order", "outbound_delivery_status", "other", or null
- identifiers: array of strings (document numbers, reference codes)
- notes: string or null
- confidence: number from 0 to 1 or null

Rules:
- The primary identifier is the sales_order_id. Extract it from patterns like "order 6090004185", "SO 6090004185", "sales order #6090004185", or any standalone 10-digit order-like number.
- If the sender asks about a specific line item (e.g. "line 20", "item 20", "position 20"), extract line_item_number.
- If the sender asks about a specific product/material or batch/lot, extract product_id and/or batch_number.
- Summarize requested facts in requested_information when the sender asks for status details beyond documents.
- If the sender asks about an outbound delivery (e.g. "delivery 90000785", "OBD 90000785"), extract outbound_delivery_id.
- If the email asks for order status, shipment status, or delivery status, set is_document_request to true (these are handled by the same workflow).
- If the email is not asking for documents or order status, set is_document_request to false, all IDs to null, send_all to false, and requested_documents to [].
- If the sender asks generally for documents, paperwork, or attachments without specifying types, set send_all to true.
- If the sender asks for a specific document type, add it to requested_documents with send_all false.
- Do not invent identifiers, order numbers, or countries that are not in the email.
- Country is secondary information — extract it if obvious from the email context but do not guess.
"""


@dataclass
class ExtractionResult:
    """Extraction service return value.

    Attributes:
        extraction: Normalized document-request extraction.
        raw_model_output: Raw text emitted by the model.
        validation_errors: Normalization or validation errors.
    """

    extraction: DocumentRequestExtraction
    raw_model_output: str
    validation_errors: list[str]


def strip_code_fences(raw_text: str) -> str:
    """Remove Markdown code fences around model JSON.

    Args:
        raw_text: Raw model output text.

    Returns:
        Text with a single surrounding Markdown code fence removed.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_raw_extraction_payload(payload: dict[str, Any]) -> DocumentRequestExtraction:
    """Normalize and validate raw extractor JSON.

    Args:
        payload: Raw JSON object returned by the extractor model.

    Returns:
        Validated DocumentRequestExtraction model.
    """
    normalized = dict(payload)
    if normalized.get("requested_documents") is None:
        normalized["requested_documents"] = []
    if "send_all" not in normalized:
        normalized["send_all"] = bool(normalized.get("is_document_request")) and not normalized["requested_documents"]
    return DocumentRequestExtraction.model_validate(normalized)


def build_extraction_prompt(email: GmailEmail) -> str:
    """Build the user prompt for email classification and extraction.

    Args:
        email: Normalized Gmail email.

    Returns:
        Prompt containing sender, subject, date, and body fields.
    """
    email_date = email.email_date.isoformat() if email.email_date else "unknown"
    return (
        f"Sender name: {email.sender_name or ''}\n"
        f"Sender email: {email.sender_email or ''}\n"
        f"Email date: {email_date}\n"
        f"Subject: {email.subject or ''}\n"
        "Body:\n"
        f"{email.body}\n"
    )


class ExtractionService:
    """Call SAP GenAI Hub to classify and extract document-request fields."""

    def __init__(self, settings: DocumentRequestSettings, responses_client: Any | None = None) -> None:
        """Initialize the extractor.

        Args:
            settings: Runtime settings containing the extractor model name.
            responses_client: Optional Responses API client (test fake); defaults to Gen AI Hub.
        """
        self.settings = settings
        self.responses_client = responses_client or default_responses_client()

    def extract(self, email: GmailEmail) -> ExtractionResult:
        """Extract normalized document-request fields from one email.

        Args:
            email: Normalized Gmail email.

        Returns:
            ExtractionResult with normalized data or a non-request fallback on errors.
        """
        raw_text = create_response_text(
            self.responses_client,
            model=self.settings.extractor_model,
            instructions=EXTRACTION_SYSTEM_PROMPT,
            user_input=build_extraction_prompt(email),
        )
        try:
            payload = json.loads(strip_code_fences(raw_text))
            extraction = normalize_raw_extraction_payload(payload)
            return ExtractionResult(extraction=extraction, raw_model_output=raw_text, validation_errors=[])
        except Exception as exc:
            fallback = DocumentRequestExtraction(
                is_document_request=False,
                reason=f"Extraction failed: {exc}",
                confidence=0.0,
                client_country=None,
                requested_documents=[],
                send_all=False,
            )
            return ExtractionResult(extraction=fallback, raw_model_output=raw_text, validation_errors=[str(exc)])
