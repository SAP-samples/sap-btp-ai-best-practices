from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .attachments import attachments_to_responses_content_blocks
from .llm_usage_logging import LlmUsageContext, emit_llm_usage_for_response
from .models import EmailAttachment, GmailEmail, RawExtraction, RawExtractionItem
from .model_clients import make_genai_hub_chat_model
from .settings import PriceChangeSettings
from .token_usage import TokenUsageTracker

EXTRACTION_SYSTEM_PROMPT = """You classify supplier emails and extract raw price-change clues.

Return exactly one JSON object and nothing else.
Required top-level fields:
- is_price_request: boolean
- reason: string
- confidence: number from 0 to 1

Optional nullable top-level fields:
- supplier_id
- supplier_email
- supplier_name
- items

Each item may contain nullable fields:
- material_number
- material_numbers: array of every material-like code mentioned for the item, including typos and corrected codes
- material_description
- requested_price: object with mode absolute, relative_percent, relative_amount, or null; and value string or null
- currency
- valid_from_raw
- valid_to_raw
- uom
- supplier_material_number
- quotation_number
- notes
- confidence

Do not invent missing fields. Preserve relative changes as relative changes.
If the email says one material number is wrong and another is correct, include both in material_numbers.
Example: "855544X (should be 8555440)" means material_numbers must include ["855544X", "8555440"].
If the email is not a supplier price-change request, set is_price_request to false and use an empty items array.
"""


MATERIAL_IDENTIFIER_PATTERN = re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{5,}\b", re.IGNORECASE)
SUPPLIER_ID_PATTERN = re.compile(r"\bSUP[0-9A-Z_-]+\b", re.IGNORECASE)


@dataclass
class ExtractionResult:
    extraction: RawExtraction
    raw_model_output: str
    validation_errors: list[str]
    model_name: str | None = None


@dataclass
class ExtractionModelOutput:
    """Raw provider response returned by an extraction model adapter.

    Attributes:
        raw_text: Text content emitted by the model.
        response: Native or LangChain provider response used for token telemetry.
    """

    raw_text: str
    response: Any
    model_name: str | None = None


class OpenAIChatCompletionExtractionClient:
    """OpenAI-compatible chat-completions adapter for extraction tests and legacy callers."""

    def __init__(self, chat_client: Any) -> None:
        """Initialize the adapter.

        Args:
            chat_client: Gen AI Hub OpenAI-compatible chat client.
        """
        self.chat_client = chat_client

    def extract(self, model_name: str, system_prompt: str, user_prompt: str) -> ExtractionModelOutput:
        """Run one OpenAI-compatible chat-completions extraction call.

        Args:
            model_name: Configured extraction model.
            system_prompt: Extraction system instruction.
            user_prompt: Email context prompt.

        Returns:
            Raw text and native provider response.
        """
        response = self.chat_client.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1200,
        )
        content = response.choices[0].message.content
        raw_text = content if isinstance(content, str) else str(content)
        return ExtractionModelOutput(raw_text=raw_text, response=response, model_name=model_name)


class LangChainExtractionClient:
    """Provider-aware LangChain adapter for SAP Gen AI Hub extraction calls."""

    def __init__(self, llm: Any) -> None:
        """Initialize the adapter.

        Args:
            llm: LangChain-compatible chat model.
        """
        self.llm = llm

    def extract(self, model_name: str, system_prompt: str, user_prompt: str) -> ExtractionModelOutput:
        """Run one provider-aware extraction call through a LangChain chat model.

        Args:
            model_name: Configured extraction model. Included for a stable adapter interface.
            system_prompt: Extraction system instruction.
            user_prompt: Email context prompt.

        Returns:
            Raw text and LangChain response object.
        """
        _ = model_name
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        raw_text = content if isinstance(content, str) else str(content)
        return ExtractionModelOutput(raw_text=raw_text, response=response, model_name=model_name)


def load_openai_responses_client() -> Any:
    """Load the SAP Gen AI Hub native OpenAI Responses client lazily.

    Returns:
        Global Responses client routed through SAP Gen AI Hub.
    """
    from gen_ai_hub.proxy.native.openai import responses

    return responses


class OpenAIResponsesAttachmentExtractionClient:
    """Attachment-aware extraction adapter using OpenAI Responses content blocks."""

    def __init__(self, responses_client: Any | None = None) -> None:
        """Initialize the attachment extraction adapter.

        Args:
            responses_client: Optional fake or real Responses client.
        """
        self._responses_client = responses_client

    @property
    def responses_client(self) -> Any:
        """Return the configured Responses client, loading it lazily when needed."""
        if self._responses_client is None:
            self._responses_client = load_openai_responses_client()
        return self._responses_client

    def extract(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        attachments: list[EmailAttachment],
        reasoning_effort: str,
    ) -> ExtractionModelOutput:
        """Run one attachment-aware extraction call through Responses.

        Args:
            model_name: PDF-capable OpenAI model deployment name.
            system_prompt: Extraction system instruction.
            user_prompt: Email metadata and body prompt.
            attachments: Supported attachments to include as typed content blocks.
            reasoning_effort: Responses API reasoning-effort setting.

        Returns:
            Raw text and native provider response.
        """
        content: list[dict[str, str]] = [{"type": "input_text", "text": user_prompt}]
        content.extend(attachments_to_responses_content_blocks(attachments))
        response = self.responses_client.create(
            model=model_name,
            instructions=system_prompt,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": reasoning_effort},
        )
        raw_text = getattr(response, "output_text", "") or ""
        return ExtractionModelOutput(raw_text=raw_text, response=response, model_name=model_name)


def strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_raw_extraction_payload(payload: dict[str, Any]) -> RawExtraction:
    """Normalize and validate the raw extractor JSON payload.

    Args:
        payload: Parsed JSON object returned by the extractor model.

    Returns:
        RawExtraction with empty lists filled and per-item material numbers deduplicated.
    """
    normalized = dict(payload)
    if normalized.get("items") is None:
        normalized["items"] = []
    normalized["items"] = [
        normalize_raw_extraction_item_payload(item)
        for item in normalized["items"]
    ]
    return RawExtraction.model_validate(normalized)


def append_unique_material_number(values: list[str], value: str | None) -> None:
    """Append one normalized material number candidate if it has not been seen.

    Args:
        values: Ordered material number list updated in place.
        value: Raw extracted or regex-matched value.

    Returns:
        None. The values list is mutated when value is useful.
    """
    if not value:
        return
    normalized = value.strip().upper()
    if not normalized or SUPPLIER_ID_PATTERN.fullmatch(normalized) or normalized in values:
        return
    values.append(normalized)


def extract_material_numbers_from_text(text: str | None) -> list[str]:
    """Extract material-like identifiers from free text.

    Args:
        text: Email, subject, or item text to scan.

    Returns:
        Ordered, deduplicated material-like identifiers excluding supplier ids.
    """
    values: list[str] = []
    if not text:
        return values
    for match in MATERIAL_IDENTIFIER_PATTERN.finditer(text):
        append_unique_material_number(values, match.group(0))
    return values


def normalize_raw_extraction_item_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw extraction item before Pydantic validation.

    Args:
        item: Raw item dictionary from the extractor model.

    Returns:
        Item dictionary with a deduplicated material_numbers list.
    """
    normalized = dict(item)
    material_numbers: list[str] = []
    raw_material_numbers = normalized.get("material_numbers") or []
    if isinstance(raw_material_numbers, str):
        raw_material_numbers = [raw_material_numbers]
    for value in raw_material_numbers:
        append_unique_material_number(material_numbers, str(value) if value is not None else None)
    append_unique_material_number(material_numbers, normalized.get("material_number"))
    append_unique_material_number(material_numbers, normalized.get("supplier_material_number"))
    for field_name in ("material_description", "notes"):
        for value in extract_material_numbers_from_text(normalized.get(field_name)):
            append_unique_material_number(material_numbers, value)
    normalized["material_numbers"] = material_numbers
    return normalized


def supplement_extraction_material_numbers(extraction: RawExtraction, email: GmailEmail) -> RawExtraction:
    """Backfill material-like codes from the email only when there is one item.

    Args:
        extraction: Validated model extraction.
        email: Original Gmail message used as a deterministic backstop for material codes.

    Returns:
        RawExtraction with single-item corrections recovered from email text. Multi-item
        emails keep item-scoped material numbers to avoid cross-line contamination.
    """
    if not extraction.is_price_request:
        return extraction
    items = extraction.items or []
    if len(items) > 1:
        return extraction
    email_numbers: list[str] = []
    for text in (email.subject, email.body):
        for value in extract_material_numbers_from_text(text):
            append_unique_material_number(email_numbers, value)
    if not email_numbers:
        return extraction
    if not items:
        items = [RawExtractionItem()]
    updated_items = []
    for item in items:
        material_numbers = list(item.material_numbers)
        for value in email_numbers:
            append_unique_material_number(material_numbers, value)
        updated_items.append(item.model_copy(update={"material_numbers": material_numbers}))
    return extraction.model_copy(update={"items": updated_items})


def build_extraction_prompt(email: GmailEmail) -> str:
    """Build the user prompt passed to the extraction model.

    Args:
        email: Gmail message to classify and extract.

    Returns:
        Plain text prompt containing sender metadata and full body.
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
    def __init__(
        self,
        settings: PriceChangeSettings,
        chat_client: Any | None = None,
        model_client: Any | None = None,
        attachment_responses_client: Any | None = None,
        attachment_model_client: Any | None = None,
    ) -> None:
        """Initialize the extraction service.

        Args:
            settings: Price-change runtime settings.
            chat_client: Optional GenAI Hub native OpenAI-compatible chat client for tests.
            model_client: Optional extraction adapter for tests.
            attachment_responses_client: Optional Responses client for attachment extraction tests.
            attachment_model_client: Optional attachment extraction adapter for tests.
        """
        self.settings = settings
        if model_client is not None:
            self.model_client = model_client
        elif chat_client is not None:
            self.model_client = OpenAIChatCompletionExtractionClient(chat_client)
        else:
            self.model_client = LangChainExtractionClient(
                make_genai_hub_chat_model(settings.extractor_model),
            )
        self.attachment_model_client = (
            attachment_model_client
            if attachment_model_client is not None
            else OpenAIResponsesAttachmentExtractionClient(attachment_responses_client)
        )

    def extract(
        self,
        email: GmailEmail,
        attachments: list[EmailAttachment] | None = None,
        token_usage_tracker: TokenUsageTracker | None = None,
        llm_usage_context: LlmUsageContext | None = None,
    ) -> ExtractionResult:
        """Classify one email and extract raw price-change clues.

        Args:
            email: Gmail message to process.
            attachments: Supported attachments persisted with the email.
            token_usage_tracker: Optional per-run token usage tracker.
            llm_usage_context: Optional Cloud Logging request context.

        Returns:
            ExtractionResult with normalized extraction, raw model output, and validation errors.
        """
        attachment_list = attachments or []
        model_name = self.settings.attachment_extractor_model if attachment_list else self.settings.extractor_model
        llm_endpoint = "responses" if attachment_list else "chat.completions"
        started_at = time.perf_counter()
        model_output: ExtractionModelOutput | None = None
        try:
            if attachment_list:
                model_output = self.attachment_model_client.extract(
                    model_name=self.settings.attachment_extractor_model,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=build_extraction_prompt(email),
                    attachments=attachment_list,
                    reasoning_effort=self.settings.agent_reasoning_effort,
                )
            else:
                model_output = self.model_client.extract(
                    model_name=self.settings.extractor_model,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=build_extraction_prompt(email),
                )
            model_name = model_output.model_name or model_name
            emit_llm_usage_for_response(
                context=llm_usage_context,
                model=model_name,
                llm_endpoint=llm_endpoint,
                response=model_output.response,
                outcome="success",
                started_at=started_at,
            )
        except Exception:
            emit_llm_usage_for_response(
                context=llm_usage_context,
                model=model_name,
                llm_endpoint=llm_endpoint,
                response=model_output.response if model_output is not None else None,
                outcome="error",
                started_at=started_at,
            )
            raise
        if token_usage_tracker is not None:
            token_usage_tracker.record_response(
                stage="extractor",
                model_name=model_name,
                response=model_output.response,
                gmail_message_id=email.gmail_message_id,
            )
        raw_text = model_output.raw_text
        try:
            payload = json.loads(strip_code_fences(raw_text))
            extraction = supplement_extraction_material_numbers(
                normalize_raw_extraction_payload(payload),
                email,
            )
            return ExtractionResult(
                extraction=extraction,
                raw_model_output=raw_text,
                validation_errors=[],
                model_name=model_name,
            )
        except Exception as exc:
            fallback = RawExtraction(
                is_price_request=False,
                reason=f"Extraction failed: {exc}",
                confidence=0.0,
                items=[],
            )
            return ExtractionResult(
                extraction=fallback,
                raw_model_output=raw_text,
                validation_errors=[str(exc)],
                model_name=model_name,
            )
