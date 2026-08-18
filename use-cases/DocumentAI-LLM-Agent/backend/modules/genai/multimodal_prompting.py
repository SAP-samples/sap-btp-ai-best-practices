"""
multimodal_prompting.py
-----------------------
TECNICA 1: Extraccion multimodal via prompt engineering libre.

Envia el PDF completo (como imagenes renderizadas) al LLM
y usa un prompt detallado para extraer todos los campos de la factura.
El LLM lee el contenido visual directamente — sin OCR previo.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm_multimodal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

MULTIMODAL_PROMPTING_PROMPT = """You are an expert invoice data extraction system with vision capabilities.
You are looking at the COMPLETE invoice document (all pages shown as images).

Carefully read ALL visible content including:
- Header and footer information
- Sender and receiver details
- Invoice numbers, dates, PO numbers
- All monetary amounts (net, gross, tax)
- Tax rates and tax IDs
- Bank account information
- ALL line items with descriptions, quantities, unit prices
- Addresses (street, city, postal code, country)
- Contact information

Extract ALL available data and return ONLY a valid JSON object with this structure:

{
  "taxAmount": <number or null>,
  "senderAddress": <full address string or null>,
  "senderBankAccount": <string or null>,
  "grossAmount": <number or null>,
  "receiverName": <string or null>,
  "purchaseOrderNumber": <string or null>,
  "senderName": <string or null>,
  "currencyCode": <ISO 4217 code e.g. "USD","EUR","ARS" or null>,
  "documentNumber": <string or null>,
  "documentDate": <"YYYY-MM-DD" or null>,
  "receiverAddress": <full address string or null>,
  "taxId": <string or null>,
  "netAmount": <number or null>,
  "deliveryDate": <"YYYY-MM-DD" or null>,
  "receiverContact": <string or null>,
  "taxRate": <percentage as number e.g. 21.0 or null>,
  "senderCity": <string or null>,
  "senderCountryCode": <ISO 3166-1 alpha-2 e.g. "US" or null>,
  "senderHouseNumber": <string or null>,
  "senderStreet": <string or null>,
  "senderPostalCode": <string or null>,
  "receiverCity": <string or null>,
  "receiverCountryCode": <ISO 3166-1 alpha-2 or null>,
  "receiverHouseNumber": <string or null>,
  "receiverStreet": <string or null>,
  "lineItems": [
    {
      "description": <EXACT text of the line item description/product name — string or null>,
      "quantity": <numeric quantity ordered — number or null>,
      "unitPrice": <price per unit — number or null>,
      "netAmount": <total net amount for this line = quantity * unitPrice — number or null>
    }
  ],
  "fieldConfidence": {
    "taxAmount": <0.0-1.0 — your confidence this value is correct>,
    "senderAddress": <0.0-1.0>,
    "senderBankAccount": <0.0-1.0>,
    "grossAmount": <0.0-1.0>,
    "receiverName": <0.0-1.0>,
    "purchaseOrderNumber": <0.0-1.0>,
    "senderName": <0.0-1.0>,
    "currencyCode": <0.0-1.0>,
    "documentNumber": <0.0-1.0>,
    "documentDate": <0.0-1.0>,
    "receiverAddress": <0.0-1.0>,
    "taxId": <0.0-1.0>,
    "netAmount": <0.0-1.0>,
    "deliveryDate": <0.0-1.0>,
    "receiverContact": <0.0-1.0>,
    "taxRate": <0.0-1.0>,
    "senderCity": <0.0-1.0>,
    "senderCountryCode": <0.0-1.0>,
    "senderHouseNumber": <0.0-1.0>,
    "senderStreet": <0.0-1.0>,
    "senderPostalCode": <0.0-1.0>,
    "receiverCity": <0.0-1.0>,
    "receiverCountryCode": <0.0-1.0>,
    "receiverHouseNumber": <0.0-1.0>,
    "receiverStreet": <0.0-1.0>
  }
}

RULES:
- Monetary values: plain numbers without currency symbols (e.g. 1500.00)
- Dates: YYYY-MM-DD format only
- Country codes: 2-letter ISO codes only
- fieldConfidence: assign a score for EVERY field (even null ones):
    1.0 = found and certain | 0.8 = very likely correct | 0.6 = likely
    0.4 = uncertain | 0.2 = guessed | 0.0 = not found in document
- lineItems: use EXACTLY these keys: "description", "quantity", "unitPrice", "netAmount"
  DO NOT use: "item", "product", "desc", "qty", "price", "amount" or any other variant
- Return ONLY the JSON object, no markdown, no explanations"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MultimodalPromptingExtractor:
    """
    Extrae datos de facturas enviando el PDF completo al LLM (multimodal).

    Tecnica 1: prompt engineering libre con vision multimodal.
    El LLM recibe las imagenes del PDF y extrae los datos directamente.
    """

    def extract(self, pdf_path: Path) -> dict[str, Any]:
        """
        Extrae datos de la factura enviando el PDF completo al LLM.

        Args:
            pdf_path: Ruta al archivo PDF original.

        Returns:
            Diccionario con campos extraidos y confidence scores.

        Raises:
            LLMClientError: Si el LLM no responde.
            ValueError: Si la respuesta no es JSON valido.
        """
        logger.info("Technique 1 (Multimodal Prompting): processing '%s'...", pdf_path.name)

        raw_response = ask_llm_multimodal(
            prompt=MULTIMODAL_PROMPTING_PROMPT,
            pdf_path=pdf_path,
        )

        logger.debug("LLM response (%d chars): %s...", len(raw_response), raw_response[:200])

        result = self._parse_json_response(raw_response)

        non_null = sum(
            1 for k, v in result.items()
            if k not in ("fieldConfidence", "lineItems") and v is not None
        )
        line_items = result.get("lineItems") or []
        logger.info(
            "Technique 1 completed. Fields: %d | Line items: %d",
            non_null, len(line_items),
        )
        return result

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Parsea JSON de la respuesta del LLM con fallbacks robustos."""
        # Intento directo
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Buscar bloque JSON en markdown o texto libre
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


def extract_multimodal_prompting(pdf_path: Path) -> dict[str, Any]:
    """Funcion de conveniencia para extraccion multimodal por prompting."""
    return MultimodalPromptingExtractor().extract(pdf_path)