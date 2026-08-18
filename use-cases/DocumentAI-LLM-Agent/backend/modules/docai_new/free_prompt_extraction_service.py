"""
free_prompt_extraction_service.py
----------------------------------
DOC AI NEW — Free Prompt Extraction Service.

Always uses Free Prompt extraction (never predefined prompts).
Sends the complete PDF to the LLM and extracts all invoice fields.

Supports both searchable and scanned PDFs via multimodal LLM.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm_multimodal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free Prompt — always used, never predefined
# ---------------------------------------------------------------------------

FREE_PROMPT_INVOICE = """Extract all invoice information from this document.

Return ONLY valid JSON.

Extract:

customer_name
customer_address
customer_tax_id
invoice_number
invoice_date
due_date
subtotal
tax_amount
total_amount

line_items:
- description
- quantity
- unit_price
- line_total

Also return page number and coordinates for each extracted field when available.

Return the result in this exact JSON structure:

{
  "customer_name": "<string or null>",
  "customer_address": "<string or null>",
  "customer_tax_id": "<string or null>",
  "invoice_number": "<string or null>",
  "invoice_date": "<YYYY-MM-DD or null>",
  "due_date": "<YYYY-MM-DD or null>",
  "subtotal": <number or null>,
  "tax_amount": <number or null>,
  "total_amount": <number or null>,
  "currency": "<ISO 4217 code or null>",
  "line_items": [
    {
      "description": "<string or null>",
      "quantity": <number or null>,
      "unit_price": <number or null>,
      "line_total": <number or null>
    }
  ],
  "field_coordinates": {
    "customer_name": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "customer_address": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "customer_tax_id": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "invoice_number": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "invoice_date": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "due_date": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "subtotal": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "tax_amount": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0},
    "total_amount": {"page": 1, "x": 0, "y": 0, "width": 0, "height": 0}
  },
  "confidence": {
    "customer_name": <0.0-1.0>,
    "customer_address": <0.0-1.0>,
    "customer_tax_id": <0.0-1.0>,
    "invoice_number": <0.0-1.0>,
    "invoice_date": <0.0-1.0>,
    "due_date": <0.0-1.0>,
    "subtotal": <0.0-1.0>,
    "tax_amount": <0.0-1.0>,
    "total_amount": <0.0-1.0>
  }
}

Rules:
- Monetary values: plain numbers without currency symbols
- Dates: YYYY-MM-DD format only
- For field_coordinates: use actual bounding box values if visible in the document, otherwise use zeros
- confidence: 1.0=certain, 0.8=very likely, 0.6=likely, 0.4=uncertain, 0.0=not found
- Return ONLY the JSON object, no markdown, no explanations"""


class FreePromptExtractionService:
    """
    Extracts invoice data using Free Prompt technique exclusively.

    Sends the complete PDF to the LLM multimodal and extracts all fields.
    Works for both searchable and scanned PDFs.

    Usage:
        service = FreePromptExtractionService()
        result = service.extract(pdf_path)
    """

    def extract(self, pdf_path: Path) -> dict[str, Any]:
        """
        Extract all invoice fields using Free Prompt.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary with extracted fields, coordinates, and confidence scores.

        Raises:
            LLMClientError: If the LLM does not respond.
            ValueError: If the response is not valid JSON.
        """
        logger.info(
            "DOC AI NEW — Free Prompt Extraction: processing '%s'...",
            pdf_path.name,
        )

        raw_response = ask_llm_multimodal(
            prompt=FREE_PROMPT_INVOICE,
            pdf_path=pdf_path,
        )

        logger.debug(
            "LLM response (%d chars): %s...", len(raw_response), raw_response[:200]
        )

        result = self._parse_json_response(raw_response)

        # Count non-null fields
        core_fields = [
            "customer_name", "customer_address", "customer_tax_id",
            "invoice_number", "invoice_date", "due_date",
            "subtotal", "tax_amount", "total_amount",
        ]
        found = sum(1 for f in core_fields if result.get(f) is not None)
        line_items = result.get("line_items") or []

        logger.info(
            "Free Prompt Extraction completed. Fields: %d/%d | Line items: %d",
            found, len(core_fields), len(line_items),
        )

        return result

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response with robust fallbacks."""
        # Direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Search for JSON block in markdown or free text
        for pattern in [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not parse JSON from LLM response.\n"
            f"Response: {raw[:500]}"
        )

    def extract_customer_name(self, extraction_result: dict[str, Any]) -> str:
        """
        Extract and normalize the customer name from extraction result.

        Rules:
        - Use customer_name field
        - Normalize: remove invalid filesystem characters
        - Trim spaces

        Returns:
            Normalized customer name string.
        """
        raw_name = extraction_result.get("customer_name") or ""
        if not raw_name:
            return "Unknown_Customer"

        # Remove invalid filesystem characters
        invalid_chars = r'<>:"/\\|?*'
        normalized = raw_name
        for ch in invalid_chars:
            normalized = normalized.replace(ch, "")

        # Trim and collapse spaces
        normalized = " ".join(normalized.split())

        return normalized or "Unknown_Customer"


def extract_with_free_prompt(pdf_path: Path) -> dict[str, Any]:
    """Convenience function for Free Prompt extraction."""
    return FreePromptExtractionService().extract(pdf_path)