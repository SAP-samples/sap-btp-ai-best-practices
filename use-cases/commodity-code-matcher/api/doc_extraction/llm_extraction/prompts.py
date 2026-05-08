"""
Prompt templates used when interacting with GPT-4.1 through SAP Gen AI Hub.

Separating the prompt generation logic makes it easier to iterate on the
format and reuse the same system instructions across the text and vision
pipelines.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schemas import EXTRACTION_JSON_SCHEMA


# System prompts emphasise strict JSON output and the exact field names the
# pipeline expects. The LLM is asked to infer missing values only when they are
# explicitly present in the document; otherwise it must return null.
TEXT_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert invoice analyst that extracts structured data from raw "
    "text. Respond with a single JSON object that conforms to the provided "
    "schema. Do not include natural language explanations or markdown."
)

VISION_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert invoice analyst that extracts structured data from "
    "images of documents. Respond with a single JSON object that conforms to "
    "the provided schema. Use null for any field that is not visible within "
    "the current page image."
)

# Guidance describing how to populate the newly introduced usageSummary field.
USAGE_SUMMARY_INSTRUCTION = (
    "For each line item, populate 'usageSummary' with a couple of sentences that "
    "explain the item's typical business purpose in semantically rich, domain-neutral "
    "language. Avoid internal catalog or system labels (for example, 'SAP ARIBA CATALOG')."
)

# Guidance for recognizing line items in various formats (not just tables).
LINE_ITEM_FORMAT_INSTRUCTION = (
    "IMPORTANT: Line items can appear in MANY different formats throughout the document:\n"
    "- Traditional tables with rows and columns\n"
    "- Bulleted or numbered lists with descriptions and prices/rates\n"
    "- Cost breakdowns or rate cards (e.g., 'Service name: XXXkr/h' or 'Labor category - XXX€/hour')\n"
    "- Simple text listings with associated amounts\n"
    "- Service descriptions followed by hourly rates, unit prices, or total amounts\n"
    "- Labor categories, service types, or product names with associated pricing\n\n"
    "Examples of valid line items:\n"
    "• 'Project management: 945kr/h' → description: 'Project management', unitPrice: 945\n"
    "• '• Programming PLC and Robot: 940kr/h' → description: 'Programming PLC and Robot', unitPrice: 940\n"
    "• Row in a table [Item | Qty | Price] → extract each row as a separate line item\n"
    "• 'Mechanical design 850kr/h' → description: 'Mechanical design', unitPrice: 850\n\n"
    "Extract ALL items that represent distinct products, services, cost components, or line entries. "
    "DO NOT skip items just because they are not in a traditional table format. "
    "Look for line items on ALL pages of the document, not just the first page."
)


def _schema_reminder() -> str:
    """
    Return a compact reminder of the JSON schema as a string literal.

    Embedding the schema within the prompt helps the model comply with the
    expected structure without sending verbose instructions in every call.
    """

    return json.dumps(EXTRACTION_JSON_SCHEMA, separators=(",", ":"))


def build_text_messages(document_text: str) -> List[Dict[str, Any]]:
    """
    Build the message list for text-only extraction.

    The entire PDF text is provided in the user message. When the text exceeds
    the model context window the caller must chunk it before invoking this
    helper.
    """

    schema = _schema_reminder()
    user_prompt = (
        "Extract the header and line item information from the following "
        "document text. Return JSON that validates against this schema:\n"
        f"{schema}\n\n"
        f"{LINE_ITEM_FORMAT_INSTRUCTION}\n\n"
        f"{USAGE_SUMMARY_INSTRUCTION}\n\n"
        "Document text:\n"
        "```text\n"
        f"{document_text.strip()}\n"
        "```"
    )
    return [
        {"role": "system", "content": TEXT_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_vision_messages(
    *,
    page_index: int,
    base64_image: str,
    mime_type: str,
    known_header: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the message list for page-level image extraction.

    ``known_header`` may contain header fields obtained from previous pages.
    Supplying this context helps the model avoid contradicting already
    extracted values while still allowing it to populate missing fields.
    """

    schema = _schema_reminder()
    header_hint = (
        "Known header values so far:\n"
        f"{json.dumps(known_header, ensure_ascii=False)}\n\n"
        if known_header
        else "No header values have been confirmed yet.\n\n"
    )
    user_text = (
        f"You are looking at page {page_index + 1} of a multi-page document.\n"
        f"{header_hint}"
        "Only report fields visible on this page. For fields that are not "
        "present, return null. Respond with JSON that validates against this "
        f"schema:\n{schema}\n\n"
        f"{LINE_ITEM_FORMAT_INSTRUCTION}\n\n"
        f"{USAGE_SUMMARY_INSTRUCTION}"
    )

    return [
        {"role": "system", "content": VISION_EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def build_text_with_image_prompt(
    *,
    document_text: str,
    base64_image: str,
    mime_type: str = "image/png",
) -> List[Dict[str, Any]]:
    """
    Build the message list for combined text + first-page image extraction.

    This approach sends both the full document text AND the first page as an
    image in a single LLM call. The image is particularly useful for extracting
    vendor information from logos or letterheads that may not be captured in
    the text extraction.

    This combines the best of both approaches:
    - Full document text provides complete content for line items and details
    - First page image enables visual extraction of logos, headers, and formatting
    """

    schema = _schema_reminder()
    user_text = (
        "You are analyzing a document using both its full text content AND an "
        "image of the first page.\n\n"
        "The first page image is provided to help you identify visual elements "
        "such as company logos, letterheads, and formatted headers that may not "
        "be fully captured in the text extraction. Pay special attention to the "
        "vendor/company name which often appears as a logo or in the letterhead.\n\n"
        "Extract the header and line item information from the document. "
        "Return JSON that validates against this schema:\n"
        f"{schema}\n\n"
        f"{LINE_ITEM_FORMAT_INSTRUCTION}\n\n"
        f"{USAGE_SUMMARY_INSTRUCTION}\n\n"
        "Full document text:\n"
        "```text\n"
        f"{document_text.strip()}\n"
        "```\n\n"
        "First page image is attached below for visual reference."
    )

    return [
        {"role": "system", "content": VISION_EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                },
            ],
        },
    ]

