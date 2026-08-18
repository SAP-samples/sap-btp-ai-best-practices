"""
multimodal_structured.py
------------------------
TECNICA 2: Extraccion multimodal con JSON schema estricto.

Envia el PDF completo al LLM y fuerza una respuesta JSON valida
siguiendo un schema predefinido. Valida y reintenta automaticamente.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm_multimodal

logger = logging.getLogger(__name__)

MAX_JSON_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Prompt estricto
# ---------------------------------------------------------------------------

STRUCTURED_PROMPT = """You are a precise invoice data extraction API with vision capabilities.
You are viewing the COMPLETE invoice document.

YOUR ONLY OUTPUT MUST BE A SINGLE VALID JSON OBJECT.
NO text before or after the JSON.
NO markdown code blocks.
NO explanations.
ONLY the raw JSON.

STRICT RULES:
1. Output ONLY valid JSON — nothing else
2. Monetary values: plain numbers, no symbols (e.g. 1500.00 not "$1,500.00")
3. Dates: YYYY-MM-DD format only (e.g. "2024-01-15")
4. Country codes: ISO 3166-1 alpha-2 only (e.g. "US", "DE", "AR", "CN")
5. Missing or unclear fields: use null (not empty string, not 0)
6. lineItems: one object per line item found in the invoice table
   MANDATORY keys per item (use EXACTLY these names):
     "description" = full text description of the product/service
     "quantity"    = numeric quantity (e.g. 1.0, 2.0, 10.0)
     "unitPrice"   = price per unit as number
     "netAmount"   = total net amount for this line as number
   DO NOT use any other key names for line items
7. fieldConfidence: 0.0=not found, 0.5=uncertain, 0.8=likely, 1.0=certain

REQUIRED JSON STRUCTURE:
{
  "taxAmount": null,
  "senderAddress": null,
  "senderBankAccount": null,
  "grossAmount": null,
  "receiverName": null,
  "purchaseOrderNumber": null,
  "senderName": null,
  "currencyCode": null,
  "documentNumber": null,
  "documentDate": null,
  "receiverAddress": null,
  "taxId": null,
  "netAmount": null,
  "deliveryDate": null,
  "receiverContact": null,
  "taxRate": null,
  "senderCity": null,
  "senderCountryCode": null,
  "senderHouseNumber": null,
  "senderStreet": null,
  "senderPostalCode": null,
  "receiverCity": null,
  "receiverCountryCode": null,
  "receiverHouseNumber": null,
  "receiverStreet": null,
  "lineItems": [
    {
      "description": "EXACT product/service description text from the invoice",
      "quantity": 1.0,
      "unitPrice": 0.0,
      "netAmount": 0.0
    }
  ],
  "fieldConfidence": {
    "taxAmount": 0.0,
    "senderAddress": 0.0,
    "senderBankAccount": 0.0,
    "grossAmount": 0.0,
    "receiverName": 0.0,
    "purchaseOrderNumber": 0.0,
    "senderName": 0.0,
    "currencyCode": 0.0,
    "documentNumber": 0.0,
    "documentDate": 0.0,
    "receiverAddress": 0.0,
    "taxId": 0.0,
    "netAmount": 0.0,
    "deliveryDate": 0.0,
    "receiverContact": 0.0,
    "taxRate": 0.0,
    "senderCity": 0.0,
    "senderCountryCode": 0.0,
    "senderHouseNumber": 0.0,
    "senderStreet": 0.0,
    "senderPostalCode": 0.0,
    "receiverCity": 0.0,
    "receiverCountryCode": 0.0,
    "receiverHouseNumber": 0.0,
    "receiverStreet": 0.0
  }
}

CONFIDENCE SCORING (replace 0.0 with your actual confidence for each field):
  1.0 = found and certain | 0.8 = very likely | 0.6 = likely
  0.4 = uncertain | 0.2 = guessed | 0.0 = not found in document

Fill in ALL fields you can identify from the invoice document.
Replace null with actual values where found.
Replace 0.0 confidence with your actual confidence score for each field.
Output ONLY the completed JSON:"""

# Prompt de reintento cuando el JSON es invalido
RETRY_PROMPT_TEMPLATE = """Your previous response was not valid JSON.
Error: {error}

Look at the invoice document again and respond with ONLY a valid JSON object.
No text, no markdown, no explanations — ONLY the raw JSON starting with {{ and ending with }}.

Required fields: taxAmount, senderAddress, senderBankAccount, grossAmount, receiverName,
purchaseOrderNumber, senderName, currencyCode, documentNumber, documentDate, receiverAddress,
taxId, netAmount, deliveryDate, receiverContact, taxRate, senderCity, senderCountryCode,
senderHouseNumber, senderStreet, senderPostalCode, receiverCity, receiverCountryCode,
receiverHouseNumber, receiverStreet, lineItems (array), fieldConfidence (object).

JSON:"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MultimodalStructuredExtractor:
    """
    Extrae datos de facturas con JSON schema estricto via multimodal LLM.

    Tecnica 2: el LLM recibe el PDF completo y debe responder SOLO JSON.
    Valida automaticamente y reintenta si el JSON es invalido.
    """

    def extract(self, pdf_path: Path) -> dict[str, Any]:
        """
        Extrae datos de la factura con extraccion estructurada estricta.

        Args:
            pdf_path: Ruta al archivo PDF original.

        Returns:
            Diccionario validado con los campos extraidos.

        Raises:
            LLMClientError: Si el LLM no responde.
            ValueError: Si no se obtiene JSON valido tras los reintentos.
        """
        logger.info(
            "Technique 2 (Multimodal Structured): processing '%s'...", pdf_path.name
        )

        last_error = None

        for attempt in range(1, MAX_JSON_RETRIES + 1):
            if attempt == 1:
                prompt = STRUCTURED_PROMPT
            else:
                prompt = RETRY_PROMPT_TEMPLATE.format(error=str(last_error))
                logger.warning(
                    "Retry %d/%d due to invalid JSON: %s",
                    attempt, MAX_JSON_RETRIES, last_error,
                )

            raw_response = ask_llm_multimodal(
                prompt=prompt,
                pdf_path=pdf_path,
            )

            logger.debug(
                "LLM response attempt %d (%d chars): %s...",
                attempt, len(raw_response), raw_response[:200],
            )

            try:
                result = self._parse_and_validate(raw_response)
                result = self._normalize(result)

                non_null = sum(
                    1 for k, v in result.items()
                    if k not in ("fieldConfidence", "lineItems") and v is not None
                )
                line_items = result.get("lineItems") or []
                logger.info(
                    "Technique 2 completed (attempt %d). Fields: %d | Line items: %d",
                    attempt, non_null, len(line_items),
                )
                return result

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning("Invalid JSON on attempt %d: %s", attempt, exc)

        raise ValueError(
            f"No valid JSON obtained after {MAX_JSON_RETRIES} attempts.\n"
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_and_validate(raw: str) -> dict[str, Any]:
        """Parsea y valida el JSON de la respuesta."""
        # Intento directo
        try:
            data = json.loads(raw.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Buscar JSON en la respuesta
        for pattern in [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"No valid JSON found in response: {raw[:300]}")

    @staticmethod
    def _normalize(data: dict[str, Any]) -> dict[str, Any]:
        """
        Normaliza tipos de datos del resultado.
        Convierte strings numericos a float, valida formatos.
        """
        numeric_fields = {
            "taxAmount", "grossAmount", "netAmount", "taxRate",
        }
        for field in numeric_fields:
            val = data.get(field)
            if val is not None and not isinstance(val, (int, float)):
                try:
                    # Limpiar simbolos de moneda y separadores
                    clean = str(val).replace(",", "").replace("$", "").strip()
                    data[field] = float(clean)
                except (ValueError, TypeError):
                    data[field] = None

        # Asegurar que lineItems sea lista
        if not isinstance(data.get("lineItems"), list):
            data["lineItems"] = []

        # Normalizar line items
        for item in data.get("lineItems", []):
            for num_field in ("quantity", "unitPrice", "netAmount"):
                val = item.get(num_field)
                if val is not None and not isinstance(val, (int, float)):
                    try:
                        clean = str(val).replace(",", "").replace("$", "").strip()
                        item[num_field] = float(clean)
                    except (ValueError, TypeError):
                        item[num_field] = None

        # Asegurar fieldConfidence
        if not isinstance(data.get("fieldConfidence"), dict):
            data["fieldConfidence"] = {}

        return data


def extract_multimodal_structured(pdf_path: Path) -> dict[str, Any]:
    """Funcion de conveniencia para extraccion multimodal estructurada."""
    return MultimodalStructuredExtractor().extract(pdf_path)