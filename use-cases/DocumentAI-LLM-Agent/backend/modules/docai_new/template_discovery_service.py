"""
template_discovery_service.py
------------------------------
DOC AI NEW — Template Discovery Service.

Searches existing SAP Document AI templates by customer name.
Uses exact and fuzzy matching to find the best template match.

Template name normalization rules (SAP hardening):
- Unicode transliteration (ü→u, é→e, ñ→n)
- Remove invalid characters (keep alphanumeric, space, hyphen, dot, underscore)
- Collapse duplicate spaces
- Strip leading/trailing special chars: space . & _ - ~ $ # ,
- Truncate to 80 characters (SAP template name limit)
- Fallback to "Unknown_Customer" if empty after normalization
"""

import logging
import re
import unicodedata
from typing import Any

from modules.templates.get_templates import DocumentAIError, get_templates


class TemplatesNotAvailableError(Exception):
    """Raised when the SAP service plan does not support the Templates API (403/E115)."""
    pass

logger = logging.getLogger(__name__)


class TemplateDiscoveryService:
    """
    Discovers existing SAP Document AI templates by customer name.

    Usage:
        service = TemplateDiscoveryService()
        template = service.find_template_by_customer("Flickr GmbH")
    """

    def find_template_by_customer(
        self,
        customer_name: str,
        client_id: str = "default",
    ) -> dict[str, Any] | None:
        """
        Search for an existing template matching the customer name.

        Matching strategy:
        1. Exact match: template.name == customer_name
        2. Case-insensitive match
        3. Partial match (customer_name in template.name or vice versa)

        Args:
            customer_name: Normalized customer name to search for.
            client_id: SAP Document AI client ID.

        Returns:
            Template dict if found, None otherwise.
        """
        if not customer_name:
            return None

        logger.info(
            "Searching for template matching customer: '%s'", customer_name
        )

        try:
            templates_response = get_templates(client_id=client_id)
        except DocumentAIError as exc:
            msg = str(exc)
            if "403" in msg or "E115" in msg or "Service plan is invalid" in msg:
                raise TemplatesNotAvailableError(
                    "Templates API returned 403 — service plan does not support templates. "
                    "Falling back to free_prompt_only."
                ) from exc
            logger.warning("Could not load templates: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Could not load templates: %s", exc)
            return None

        templates = (
            templates_response.get("templates")
            or templates_response.get("results")
            or []
        )

        if not templates:
            logger.info("No templates available in SAP Document AI.")
            return None

        customer_lower = customer_name.lower().strip()

        # 1. Exact match
        for t in templates:
            if (t.get("name") or "").strip() == customer_name:
                logger.info(
                    "Exact template match found: '%s' (id=%s)",
                    t.get("name"),
                    t.get("id"),
                )
                return t

        # 2. Case-insensitive match
        for t in templates:
            if (t.get("name") or "").lower().strip() == customer_lower:
                logger.info(
                    "Case-insensitive template match: '%s' (id=%s)",
                    t.get("name"),
                    t.get("id"),
                )
                return t

        # 3. Partial match
        for t in templates:
            t_name_lower = (t.get("name") or "").lower().strip()
            if customer_lower in t_name_lower or t_name_lower in customer_lower:
                logger.info(
                    "Partial template match: '%s' (id=%s)",
                    t.get("name"),
                    t.get("id"),
                )
                return t

        logger.info(
            "No template found for customer '%s'. Template creation required.",
            customer_name,
        )
        return None

    def list_all_templates(self, client_id: str = "default") -> list[dict[str, Any]]:
        """
        Return all available templates.

        Args:
            client_id: SAP Document AI client ID.

        Returns:
            List of template dicts.
        """
        try:
            response = get_templates(client_id=client_id)
            return (
                response.get("templates")
                or response.get("results")
                or []
            )
        except Exception as exc:
            logger.warning("Could not load templates: %s", exc)
            return []

    def normalize_template_name(self, name: str) -> str:
        """
        Normalize a customer name for use as a SAP Document AI template name.

        Rules (applied in order):
        1. Unicode NFKD decomposition → strip combining marks → ASCII
           e.g. ü→u, é→e, ñ→n, ö→o
        2. Replace characters not in [A-Za-z0-9 ._-] with space
        3. Collapse duplicate spaces into single space
        4. Strip leading/trailing special chars: space . & _ - ~ $ # ,
        5. Truncate to 80 characters (SAP template name limit)
        6. Strip again after truncation
        7. Fallback to "Unknown_Customer" if empty

        Examples:
            "Berühmter Influencer" → "Beruhmter Influencer"
            "ACME GmbH / Germany"  → "ACME GmbH Germany"
            "***???"               → "Unknown_Customer"
            "- ACME GmbH"         → "ACME GmbH"
            "ACME GmbH -"         → "ACME GmbH"
            ". ACME"              → "ACME"

        Args:
            name: Raw customer name from LLM extraction.

        Returns:
            SAP-safe template name (max 80 chars, no special chars).
        """
        if not name:
            return "Unknown_Customer"

        # Step 1: Unicode NFKD decomposition → strip combining marks
        nfkd = unicodedata.normalize("NFKD", name)
        normalized = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Step 2: Replace characters not in [A-Za-z0-9 ._-] with space
        normalized = re.sub(r"[^A-Za-z0-9 .\-_]", " ", normalized)

        # Step 3: Collapse duplicate spaces
        normalized = re.sub(r"\s+", " ", normalized)

        # Step 4: Strip leading/trailing special chars
        normalized = normalized.strip(" .&_-~$#,")

        # Step 5: Truncate to 80 characters (SAP limit)
        normalized = normalized[:80]

        # Step 6: Strip again after truncation (may end with space/special)
        normalized = normalized.strip(" .&_-~$#,")

        # Step 7: Fallback if empty after normalization
        if not normalized:
            normalized = "Unknown_Customer"

        logger.info("SAP SAFE TEMPLATE NAME: '%s'", normalized)
        return normalized