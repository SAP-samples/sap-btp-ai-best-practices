"""
Business Partners Mapper

Maps raw S/4HANA API_BUSINESS_PARTNER response to BusinessPartner model.

Name field priority:
  1. OrganizationBPName1   (org name, most common for companies)
  2. BusinessPartnerFullName (full name, available for persons)
  3. BusinessPartnerName   (alternative name field)
  4. SearchTerm1           (search term, often contains abbreviated name)
  5. BusinessPartner       (fallback: use the code itself)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from S4.business_partners.business_partners_models import BusinessPartner

logger = logging.getLogger(__name__)

_first_record_logged = False


def map_business_partner(raw: dict[str, Any]) -> BusinessPartner | None:
    """Map a single raw S/4HANA business partner record to BusinessPartner model."""
    global _first_record_logged

    try:
        bp_code = (raw.get("BusinessPartner") or "").strip()
        if not bp_code:
            return None

        # Log the first record to inspect available SAP fields
        if not _first_record_logged:
            _first_record_logged = True
            logger.info(
                "First BP record from SAP (field inspection):\n%s",
                json.dumps(
                    {k: v for k, v in raw.items() if not isinstance(v, dict)},
                    indent=2,
                    default=str,
                ),
            )

        # Name: try all available name fields in priority order
        name = (
            (raw.get("OrganizationBPName1") or "").strip()
            or (raw.get("BusinessPartnerFullName") or "").strip()
            or (raw.get("BusinessPartnerName") or "").strip()
            or (raw.get("SearchTerm1") or "").strip()
            or bp_code
        )

        # Build full_description: name + city/state if available
        # Address fields may be present if expanded
        city = (raw.get("CityName") or raw.get("City") or "").strip()
        region = (raw.get("Region") or raw.get("RegionName") or "").strip()
        postal = (raw.get("PostalCode") or "").strip()

        if city and (region or postal):
            location = f"{city} {region} {postal}".strip()
            full_description = f"{name} / {location}"
        elif city:
            full_description = f"{name} / {city}"
        else:
            full_description = name

        return BusinessPartner(
            business_partner=bp_code,
            business_partner_name=name,
            full_description=full_description,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to map business partner: %s | raw=%s", exc, raw)
        return None


def map_raw_list(raw_list: list[dict[str, Any]]) -> list[BusinessPartner]:
    """Map a list of raw records, skipping any that fail."""
    results = []
    for raw in raw_list:
        mapped = map_business_partner(raw)
        if mapped:
            results.append(mapped)
    return results