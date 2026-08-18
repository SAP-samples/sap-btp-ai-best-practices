"""
po_multimodal_structured.py
---------------------------
LLM extraction specifically for Customer Purchase Orders.

Uses a PO-specific prompt that correctly identifies:
  - buyerName: the COMPANY sending the PO (the customer ordering goods)
  - vendorName: the company receiving the PO (AI4U — the seller)
  - purchaseOrderNumber, orderDate, requestedDeliveryDate, currency
  - lineItems with description, quantity, unitOfMeasure, unitPrice

This is separate from the invoice extractor (multimodal_structured.py)
which uses invoice-specific field names and would hallucinate on PO docs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm_multimodal

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# PO-specific prompt
# ---------------------------------------------------------------------------

PO_STRUCTURED_PROMPT = """You are a precise Purchase Order data extraction API with vision capabilities.
You are viewing a CUSTOMER PURCHASE ORDER document — NOT an invoice.

In a Purchase Order:
- The BUYER / CUSTOMER is the company that SENDS the PO (they are ordering goods)
- The VENDOR / SELLER is the company that RECEIVES the PO (they will deliver the goods)

YOUR ONLY OUTPUT MUST BE A SINGLE VALID JSON OBJECT.
NO text before or after the JSON. NO markdown. NO explanations. ONLY raw JSON.

STRICT RULES:
1. Output ONLY valid JSON — nothing else
2. buyerName: the company SENDING this purchase order (the customer/buyer)
3. vendorName: the company RECEIVING this purchase order (the seller/vendor)
4. Monetary values: plain numbers only (e.g. 128.50 not "EUR 128.50")
5. Dates: YYYY-MM-DD format (e.g. "2026-07-30")
6. Missing fields: use null
7. lineItems: one object per ordered item

REQUIRED JSON STRUCTURE:
{
  "buyerName": null,
  "buyerAddress": null,
  "vendorName": null,
  "purchaseOrderNumber": null,
  "orderDate": null,
  "requestedDeliveryDate": null,
  "currency": null,
  "totalAmount": null,
  "specialInstructions": null,
  "lineItems": [
    {
      "description": "exact product description from the document",
      "quantity": 1.0,
      "unitOfMeasure": "PC",
      "unitPrice": 0.0
    }
  ]
}

IMPORTANT:
- buyerName is the FROM / Auftraggeber / Buyer — the one placing the order
- vendorName is the TO / Lieferant / Vendor — the one fulfilling the order
- Do NOT confuse them — look for explicit labels like "BUYER:", "FROM:", "Auftraggeber"
- purchaseOrderNumber: the PO reference number (e.g. PO-2026-10011)
- lineItems descriptions: copy the exact text from the document

Output ONLY the completed JSON:"""

RETRY_PROMPT = """Your previous response was not valid JSON.
Error: {error}

Look at the Purchase Order document again and respond with ONLY a valid JSON object.
No text, no markdown — ONLY the raw JSON starting with {{ and ending with }}.

Required fields: buyerName, buyerAddress, vendorName, purchaseOrderNumber,
orderDate, requestedDeliveryDate, currency, totalAmount, specialInstructions, lineItems.

JSON:"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class POMultimodalStructuredExtractor:
    """LLM extractor for Customer Purchase Orders with PO-specific prompt."""

    def extract(self, pdf_path: Path) -> dict[str, Any]:
        logger.info("PO LLM extraction: %s", pdf_path.name)
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            prompt = PO_STRUCTURED_PROMPT if attempt == 1 else RETRY_PROMPT.format(error=str(last_error))
            if attempt > 1:
                logger.warning("PO LLM retry %d/%d: %s", attempt, MAX_RETRIES, last_error)

            try:
                raw = ask_llm_multimodal(prompt=prompt, pdf_path=pdf_path)
                logger.debug("PO LLM response (attempt %d, %d chars): %s...", attempt, len(raw), raw[:200])
                result = self._parse(raw)
                result = self._normalize(result)
                logger.info(
                    "PO LLM extracted | buyer=%r | po=%r | items=%d",
                    result.get("buyerName"), result.get("purchaseOrderNumber"),
                    len(result.get("lineItems") or []),
                )
                return result
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning("PO LLM invalid JSON attempt %d: %s", attempt, exc)
            except LLMClientError as exc:
                logger.error("PO LLM client error: %s", exc)
                return {}

        logger.error("PO LLM failed after %d attempts", MAX_RETRIES)
        return {}

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        for pat in [r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```", r"(\{[\s\S]+\})"]:
            m = re.search(pat, raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1).strip())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"No valid JSON in PO LLM response: {raw[:300]}")

    @staticmethod
    def _normalize(data: dict[str, Any]) -> dict[str, Any]:
        for field in ("totalAmount", "unitPrice"):
            val = data.get(field)
            if val is not None and not isinstance(val, (int, float)):
                try:
                    data[field] = float(re.sub(r"[^\d.]", "", str(val)))
                except (ValueError, TypeError):
                    data[field] = None

        if not isinstance(data.get("lineItems"), list):
            data["lineItems"] = []

        for item in data.get("lineItems", []):
            for f in ("quantity", "unitPrice"):
                v = item.get(f)
                if v is not None and not isinstance(v, (int, float)):
                    try:
                        item[f] = float(re.sub(r"[^\d.]", "", str(v)))
                    except (ValueError, TypeError):
                        item[f] = None
        return data


def extract_po_structured(pdf_path: Path) -> dict[str, Any]:
    """Convenience function — LLM extraction for Customer Purchase Orders."""
    return POMultimodalStructuredExtractor().extract(pdf_path)
