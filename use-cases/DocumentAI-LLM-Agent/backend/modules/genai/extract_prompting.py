"""
extract_prompting.py
--------------------
TECNICA 1: Extraccion libre via prompt engineering.

Envia el texto del PDF al LLM con un prompt detallado y
parsea la respuesta JSON resultante.
"""

import json
import logging
import re
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert invoice data extraction system.
Analyze the following invoice text and extract ALL available information.

Return ONLY a valid JSON object with this exact structure (use null for missing fields):

{{
  "taxAmount": <number or null>,
  "senderAddress": <string or null>,
  "senderBankAccount": <string or null>,
  "grossAmount": <number or null>,
  "receiverName": <string or null>,
  "purchaseOrderNumber": <string or null>,
  "senderName": <string or null>,
  "currencyCode": <string or null>,
  "documentNumber": <string or null>,
  "documentDate": <string YYYY-MM-DD or null>,
  "receiverAddress": <string or null>,
  "taxId": <string or null>,
  "netAmount": <number or null>,
  "deliveryDate": <string YYYY-MM-DD or null>,
  "receiverContact": <string or null>,
  "taxRate": <number or null>,
  "senderCity": <string or null>,
  "senderCountryCode": <string or null>,
  "senderHouseNumber": <string or null>,
  "senderStreet": <string or null>,
  "senderPostalCode": <string or null>,
  "receiverCity": <string or null>,
  "receiverCountryCode": <string or null>,
  "receiverHouseNumber": <string or null>,
  "receiverStreet": <string or null>,
  "lineItems": [
    {{
      "description": <string or null>,
      "quantity": <number or null>,
      "unitPrice": <number or null>,
      "netAmount": <number or null>
    }}
  ],
  "confidence": {{
    "overall": <0.0-1.0>,
    "fields": {{
      "taxAmount": <0.0-1.0 or null>,
      "grossAmount": <0.0-1.0 or null>,
      "netAmount": <0.0-1.0 or null>,
      "senderName": <0.0-1.0 or null>,
      "receiverName": <0.0-1.0 or null>,
      "documentNumber": <0.0-1.0 or null>,
      "documentDate": <0.0-1.0 or null>
    }}
  }}
}}

Rules:
- Extract ALL monetary amounts as plain numbers (no currency symbols)
- Dates must be in YYYY-MM-DD format
- Country codes must be ISO 3166-1 alpha-2 (e.g. "US", "DE", "AR")
- confidence values: 1.0=certain, 0.7=likely, 0.4=uncertain, 0.0=not found
- Return ONLY the JSON, no explanations, no markdown

INVOICE TEXT:
---
{invoice_text}
---

JSON:"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PromptingExtractor:
    """
    Extrae datos de facturas usando prompt engineering libre.

    Tecnica 1: el LLM recibe el texto completo y un prompt detallado,
    y devuelve un JSON con todos los campos extraidos.
    """

    def extract(self, invoice_text: str) -> dict[str, Any]:
        """
        Extrae datos de la factura usando prompting libre.

        Args:
            invoice_text: Texto extraido del PDF.

        Returns:
            Diccionario con los campos extraidos y scores de confianza.

        Raises:
            LLMClientError: Si el LLM no responde.
            ValueError: Si la respuesta no es JSON valido.
        """
        if not invoice_text or not invoice_text.strip():
            raise ValueError("El texto de la factura esta vacio.")

        # Truncar si es muy largo (limite de contexto)
        text = invoice_text[:12000]

        prompt = EXTRACTION_PROMPT.format(invoice_text=text)

        logger.info("Tecnica 1 (Prompting): invocando LLM...")
        raw_response = ask_llm(prompt)
        logger.debug("Respuesta raw LLM (%d chars): %s...", len(raw_response), raw_response[:200])

        result = self._parse_json_response(raw_response)
        logger.info(
            "Tecnica 1 completada. Campos extraidos: %d",
            self._count_non_null(result),
        )
        return result

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """
        Parsea la respuesta del LLM extrayendo el JSON.
        Maneja casos donde el LLM incluye texto extra o markdown.
        """
        # Intentar parsear directamente
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Buscar bloque JSON entre ```json ... ``` o ``` ... ```
        patterns = [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"No se pudo parsear JSON de la respuesta del LLM.\n"
            f"Respuesta recibida: {raw[:500]}"
        )

    @staticmethod
    def _count_non_null(data: dict) -> int:
        """Cuenta campos de primer nivel que no son null/None."""
        skip = {"confidence", "lineItems"}
        return sum(
            1 for k, v in data.items()
            if k not in skip and v is not None
        )


def extract_with_prompting(invoice_text: str) -> dict[str, Any]:
    """Funcion de conveniencia para extraccion por prompting."""
    return PromptingExtractor().extract(invoice_text)