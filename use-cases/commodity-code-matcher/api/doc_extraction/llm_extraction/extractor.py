"""
Public API of the LLM-based PDF extraction module.

The :func:`extract` entrypoint analyses a single PDF, selects the appropriate
processing strategy (text vs. image) and returns a dictionary containing the
header information, line items, document type and optionally a tabular view.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional for Cloud Foundry
    pd = None  # type: ignore[assignment]

from .llm_client import LLMClient
from .merger import merge_page_extractions, normalise_header, normalise_line_item
from .prompts import build_text_messages, build_text_with_image_prompt, build_vision_messages
from .schemas import (
    ExtractionResult,
    HEADER_FIELDS,
    LINE_ITEM_FIELDS,
    DocumentHeader,
    LineItem,
    PageExtraction,
)
from .text_image import extract_first_page_image, extract_full_text, iter_page_images


LOGGER = logging.getLogger(__name__)

TEXT_LENGTH_THRESHOLD = 100
MIN_ALPHA_RATIO_FOR_TEXT = 0.18
TABLE_LINE_PREFIX = "item_"


def classify_document(text: str) -> str:
    """
    Determine whether the PDF should be processed via text or image mode.

    Some suppliers send PDFs where the table text is converted to vector
    outlines. In those cases ``fitz`` extracts only bullets or numbers,
    which previously caused us to incorrectly treat the document as text.
    We now look at the overall density of alphabetic characters to detect
    those situations and fall back to the image-based flow.
    """

    content = (text or "").strip()
    if not content:
        LOGGER.debug("Document classification → image (no text content)")
        return "image"

    length = len(content)
    non_whitespace = sum(1 for ch in content if not ch.isspace())
    alpha_count = sum(1 for ch in content if ch.isalpha())
    bullet_count = content.count("•") + content.count("·")
    alpha_ratio = alpha_count / max(non_whitespace, 1)

    doc_type = "text"
    reasons: list[str] = []

    if length <= TEXT_LENGTH_THRESHOLD:
        doc_type = "image"
        reasons.append("short_text")
    elif alpha_ratio < MIN_ALPHA_RATIO_FOR_TEXT:
        doc_type = "image"
        reasons.append("low_alpha_ratio")
    elif bullet_count >= 3 and alpha_ratio < 0.3:
        # Heavy bullet usage with little alphabetic content usually means
        # the readable text is embedded in the image layer.
        doc_type = "image"
        reasons.append("bullet_heavy")

    LOGGER.debug(
        "Document classification → %s (len=%s, non_ws=%s, alpha_ratio=%.3f, bullets=%s, reasons=%s)",
        doc_type,
        length,
        non_whitespace,
        alpha_ratio,
        bullet_count,
        reasons or None,
    )
    return doc_type


def _call_llm_text(client: LLMClient, document_text: str) -> tuple[DocumentHeader, List[LineItem]]:
    """Invoke the text-only prompt and normalise the response."""

    messages = build_text_messages(document_text)
    payload, response = client.complete_json(messages)
    LOGGER.debug("Text extraction LLM tokens: %s", getattr(response, "usage", None))
    header = normalise_header(payload.get("header"))
    line_items = [normalise_line_item(item) for item in payload.get("lineItems", [])]
    return header, line_items


def _call_llm_image(client: LLMClient, pdf_path: Path) -> tuple[DocumentHeader, List[LineItem]]:
    """Process an image-based document page by page."""

    page_payloads: List[PageExtraction] = []
    current_header: Dict[str, Any] = {}

    for page in iter_page_images(pdf_path):
        base64_image = base64.b64encode(page.bytes).decode("utf-8")
        known_header = {k: v for k, v in current_header.items() if v}
        messages = build_vision_messages(
            page_index=page.index,
            base64_image=base64_image,
            mime_type="image/png",
            known_header=known_header or None,
        )
        payload, response = client.complete_json(messages)
        LOGGER.debug(
            "Image extraction page %s LLM tokens: %s",
            page.index + 1,
            getattr(response, "usage", None),
        )
        page_payloads.append(payload)

        # Update header context for subsequent pages.
        page_header = payload.get("header") or {}
        for key, value in page_header.items():
            if current_header.get(key) in (None, "", 0) and value not in (None, ""):
                current_header[key] = value

    return merge_page_extractions(page_payloads)


def _call_llm_text_with_first_page_image(
    client: LLMClient,
    pdf_path: Path,
    document_text: str,
) -> tuple[DocumentHeader, List[LineItem]]:
    """
    Process a text document with first page image for enhanced extraction.

    This method combines the full document text with the first page rendered
    as an image. The image helps the LLM extract vendor information from logos
    or letterheads that may not be captured in plain text extraction.

    This is particularly useful for:
    - Extracting vendor/company names from logos
    - Identifying formatted headers and letterheads
    - Capturing visual elements that complement the text
    """

    # Extract first page as image
    first_page = extract_first_page_image(pdf_path)
    base64_image = base64.b64encode(first_page.bytes).decode("utf-8")

    # Build combined prompt with text + image
    messages = build_text_with_image_prompt(
        document_text=document_text,
        base64_image=base64_image,
        mime_type="image/png",
    )

    # Call LLM with combined prompt
    payload, response = client.complete_json(messages)
    LOGGER.debug(
        "Text+Image extraction LLM tokens: %s",
        getattr(response, "usage", None),
    )

    # Normalize and return results
    header = normalise_header(payload.get("header"))
    line_items = [normalise_line_item(item) for item in payload.get("lineItems", [])]
    return header, line_items


def extract(
    pdf_path: str | Path,
    *,
    client: Optional[LLMClient] = None,
    return_dataframe: bool = True,
) -> ExtractionResult:
    """
    Extract structured information from a PDF document.

    Parameters
    ----------
    pdf_path:
        Path to the PDF that should be analysed.
    client:
        Optional :class:`LLMClient` instance. A default client is created when
        None is supplied.
    return_dataframe:
        When True the result includes a ``table`` key containing either a
        Pandas DataFrame or, if Pandas is not installed, a list of dictionaries.
    """

    pdf_path = Path(pdf_path)
    client = client or LLMClient()

    document_text = extract_full_text(pdf_path)
    doc_type = classify_document(document_text)

    if doc_type == "text":
        # Use combined text + first page image extraction for better vendor detection
        header, line_items = _call_llm_text_with_first_page_image(client, pdf_path, document_text)
    else:
        header, line_items = _call_llm_image(client, pdf_path)

    result: ExtractionResult = {
        "header": header,
        "lineItems": line_items,
        "docType": doc_type,
    }

    if return_dataframe:
        result["table"] = to_table(result)

    return result


def _table_rows(header: DocumentHeader, line_items: Iterable[LineItem]) -> List[Dict[str, Any]]:
    """
    Build table rows combining header and line item data.

    Each row describes one line item. Header values are repeated for every row
    to simplify downstream analytics (e.g. exporting to CSV).
    """

    header_values = {field: header.get(field) for field in HEADER_FIELDS}
    rows: List[Dict[str, Any]] = []
    items = list(line_items)
    if not items:
        rows.append(header_values.copy())
        return rows
    for item in items:
        row = header_values.copy()
        for field in LINE_ITEM_FIELDS:
            row[f"{TABLE_LINE_PREFIX}{field}"] = item.get(field)
        rows.append(row)
    return rows


def to_table(result: ExtractionResult) -> Any:
    """
    Convert the extraction result into a tabular structure.

    - If Pandas is installed, a DataFrame is returned.
    - Otherwise, a list of dictionaries is returned, which can still be written
      to CSV using the standard library.
    """

    rows = _table_rows(result["header"], result.get("lineItems", []))
    if pd is not None:
        return pd.DataFrame(rows)  # type: ignore[no-any-return]
    return rows
