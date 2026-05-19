"""
Type definitions and JSON schemas used by the LLM-based PDF extraction module.

The extraction pipeline exchanges dictionary payloads between components and
with the LLM responses. Having a single source of truth for the expected keys
and types helps us validate the responses and apply consistent normalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

try:
    # Python < 3.11
    from typing_extensions import NotRequired  # type: ignore
except ImportError:  # pragma: no cover - fallback for newer runtimes
    from typing import NotRequired  # type: ignore


class DocumentHeader(TypedDict, total=False):
    """
    Header-level fields that describe the entire document.

    All fields are optional in the raw LLM response, but downstream processing
    should attempt to populate each key with either a value or None.
    """

    documentDate: str | None
    deliveryDate: str | None
    senderAddress: str | None
    vendorName: str | None
    receiverID: str | None
    shipToName: str | None
    shipToAddress: str | None
    currencyCode: str | None
    netAmount: float | None


class LineItem(TypedDict, total=False):
    """
    Fields describing a single line item within the document.

    Each value may be absent or None if the information is not available in the
    PDF or cannot be inferred by the LLM.
    """

    description: str | None
    netAmount: float | None
    quantity: float | None
    unitPrice: float | None
    materialNumber: str | None
    itemNumber: str | None
    usageSummary: str | None  # Semantic summary describing how the item is generally used.


class PageExtraction(TypedDict, total=False):
    """
    Intermediate payload returned for a single PDF page when we are processing
    an image document. Header fields that are repeated across pages can be
    merged later into a single DocumentHeader instance.
    """

    header: DocumentHeader
    lineItems: List[LineItem]


class ExtractionResult(TypedDict, total=False):
    """
    Final normalized extraction payload returned to callers.

    The optional ``table`` key is reserved for a tabular representation, for
    example a Pandas DataFrame serialized into records.
    """

    header: DocumentHeader
    lineItems: List[LineItem]
    docType: Literal["text", "image"]
    table: NotRequired[Any]


# JSON schema mirroring the ExtractionResult shape. The schema is used to
# instruct the LLM to return a predictable structure and to validate responses.
EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PDFExtractionResult",
    "type": "object",
    "required": ["header", "lineItems"],
    "properties": {
        "header": {
            "type": "object",
            "required": [
                "documentDate",
                "deliveryDate",
                "senderAddress",
                "vendorName",
                "receiverID",
                "shipToName",
                "shipToAddress",
                "currencyCode",
                "netAmount",
            ],
            "properties": {
                "documentDate": {"type": ["string", "null"]},
                "deliveryDate": {"type": ["string", "null"]},
                "senderAddress": {"type": ["string", "null"]},
                "vendorName": {"type": ["string", "null"]},
                "receiverID": {"type": ["string", "null"]},
                "shipToName": {"type": ["string", "null"]},
                "shipToAddress": {"type": ["string", "null"]},
                "currencyCode": {"type": ["string", "null"]},
                "netAmount": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
        "lineItems": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "description",
                    "netAmount",
                    "quantity",
                    "unitPrice",
                    "materialNumber",
                    "itemNumber",
                    "usageSummary",
                ],
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "netAmount": {"type": ["number", "null"]},
                    "quantity": {"type": ["number", "null"]},
                    "unitPrice": {"type": ["number", "null"]},
                    "materialNumber": {"type": ["string", "null"]},
                    "itemNumber": {"type": ["string", "null"]},
                    "usageSummary": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "docType": {"enum": ["text", "image"]},
        "table": {},
    },
    "additionalProperties": False,
}


# List of header and line item fields. Keeping them in arrays simplifies
# normalization and later DataFrame column ordering logic.
HEADER_FIELDS: List[str] = [
    "documentDate",
    "deliveryDate",
    "senderAddress",
    "vendorName",
    "receiverID",
    "shipToName",
    "shipToAddress",
    "currencyCode",
    "netAmount",
]

LINE_ITEM_FIELDS: List[str] = [
    "description",
    "netAmount",
    "quantity",
    "unitPrice",
    "materialNumber",
    "itemNumber",
    "usageSummary",
]
