from __future__ import annotations

import json
from typing import Any

from .llm import create_response_text, default_responses_client
from .models import DocumentRequestExtraction, GmailEmail
from .settings import DocumentRequestSettings
from .tools import build_reply_body

RESPONSE_COMPOSER_SYSTEM_PROMPT = """You write customer email replies for shipment document requests.

Use only the JSON context provided by the application. Do not infer, invent, or rely on outside knowledge.
If a requested fact is not present in the context, say that it is not available in the current order data.
Return only the plain-text email body.
"""


class DocumentResponseComposer:
    """Compose grounded draft responses from retrieved HANA context."""

    def __init__(self, settings: DocumentRequestSettings, responses_client: Any | None = None) -> None:
        """Initialize the grounded response composer.

        Args:
            settings: Runtime settings containing the agent model name.
            responses_client: Optional Responses API client (test fake); defaults to Gen AI Hub.
        """
        self.settings = settings
        self.responses_client = responses_client or default_responses_client()

    def compose(self, email: GmailEmail, extraction: DocumentRequestExtraction, context: dict[str, Any]) -> str:
        """Write a plain-text reply grounded in retrieved HANA rows.

        Args:
            email: Source customer email.
            extraction: Normalized extraction result.
            context: HANA rows and selected attachment metadata.

        Returns:
            Plain-text email body.
        """
        body = create_response_text(
            self.responses_client,
            model=self.settings.agent_model,
            instructions=RESPONSE_COMPOSER_SYSTEM_PROMPT,
            user_input=build_response_prompt(email, extraction, context),
        )
        return body.strip()


class TemplateResponseComposer:
    """Fallback composer used by pure unit tests and direct helper calls."""

    def compose(self, email: GmailEmail, extraction: DocumentRequestExtraction, context: dict[str, Any]) -> str:
        """Build a deterministic reply body from context.

        Args:
            email: Source customer email.
            extraction: Normalized extraction result.
            context: HANA rows and selected attachment metadata.

        Returns:
            Plain-text email body.
        """
        return build_reply_body(
            sales_order_id=extraction.sales_order_id,
            attachments=context.get("attachments", []),
            order_status=context.get("order_status"),
            client_country=context.get("client_country"),
        )


def build_response_prompt(email: GmailEmail, extraction: DocumentRequestExtraction, context: dict[str, Any]) -> str:
    """Build the grounded response prompt for the LLM composer.

    Args:
        email: Source customer email.
        extraction: Normalized extraction result.
        context: Retrieved HANA context and selected documents.

    Returns:
        User prompt for the response composer.
    """
    payload = {
        "customer_email": {
            "from": email.sender_email,
            "subject": email.subject,
            "body": email.body,
        },
        "extraction": extraction.model_dump(mode="json"),
        "hana_context": json_safe_context(context),
    }
    return (
        "Write a concise response email body for this customer request. "
        "Include order status, requested line/product details, and attachment availability when present. "
        "Use only this JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def json_safe_context(context: dict[str, Any]) -> dict[str, Any]:
    """Convert composer context into JSON-safe values.

    Args:
        context: Raw context containing model objects and attachments.

    Returns:
        JSON-serializable context.
    """
    safe = dict(context)
    order_status = safe.get("order_status")
    if hasattr(order_status, "model_dump"):
        safe["order_status"] = order_status.model_dump(mode="json")
    attachments = safe.get("attachments") or []
    safe["attachments"] = [
        item.model_dump_for_ui() if hasattr(item, "model_dump_for_ui") else item
        for item in attachments
    ]
    return safe
