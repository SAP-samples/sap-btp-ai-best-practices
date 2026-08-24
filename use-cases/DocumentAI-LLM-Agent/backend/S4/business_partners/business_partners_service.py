"""
Business Partners Service – READ-ONLY.

Fetches ALL Business Partners from SAP API_BUSINESS_PARTNER with automatic pagination.
No hardcoded limit. Uses $top/$skip loop until all records are retrieved.
"""

from __future__ import annotations

import asyncio
import logging
import time

import requests

from S4.s4_client import BASE_URL, sess
from S4.business_partners.business_partners_mapper import map_raw_list
from S4.business_partners.business_partners_models import BusinessPartner
from config import settings

logger = logging.getLogger(__name__)

API_BP: str = f"{BASE_URL}/sap/opu/odata/sap/API_BUSINESS_PARTNER"
PAGE_SIZE = 1000

logger.info("API_BP=%s", API_BP)


def _fetch_business_partners_sync() -> list[BusinessPartner]:
    """
    Fetch ALL business partners from SAP API_BUSINESS_PARTNER with automatic pagination.

    Uses $top/$skip to loop through all records — no hardcoded limit.
    """
    endpoint = f"{API_BP}/A_BusinessPartner"
    all_partners: list[BusinessPartner] = []
    skip = 0
    start_time = time.time()

    logger.info("Fetching ALL business partners from S/4HANA | endpoint=%s", endpoint)

    session = sess()

    while True:
        params = {
            "sap-client": settings.S4_CLIENT,
            "$format": "json",
            "$top": str(PAGE_SIZE),
            "$skip": str(skip),
            "$select": (
                "BusinessPartner,"
                "OrganizationBPName1,"
                "BusinessPartnerFullName,"
                "BusinessPartnerName,"
                "SearchTerm1,"
                "BusinessPartnerType"
            ),
        }

        logger.info("Business partners page | skip=%d | top=%d", skip, PAGE_SIZE)

        try:
            response = session.get(endpoint, params=params, timeout=60)
            response.raise_for_status()

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            if status_code == 401:
                raise RuntimeError(
                    "Authentication failed – check S4_USERNAME and S4_PASSWORD"
                ) from exc
            elif status_code == 403:
                raise RuntimeError(
                    "Access denied – user lacks authorization for API_BUSINESS_PARTNER"
                ) from exc
            elif status_code == 404:
                raise RuntimeError(
                    "Business Partner API endpoint not found – check S4_BASE_URL"
                ) from exc
            else:
                raise RuntimeError(
                    f"HTTP {status_code} error from S/4HANA Business Partner API"
                ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "S/4HANA Business Partner API timed out after 60 s"
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach S/4HANA server at {API_BP}"
            ) from exc

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Unexpected error fetching Business Partners: {type(exc).__name__}"
            ) from exc

        try:
            data = response.json()
            raw_list: list[dict] = data.get("d", {}).get("results", [])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to parse S/4HANA response as JSON: {exc}"
            ) from exc

        if not raw_list:
            break

        page_partners = map_raw_list(raw_list)
        all_partners.extend(page_partners)

        logger.info(
            "Business partners page done | skip=%d | page=%d | total_so_far=%d",
            skip,
            len(raw_list),
            len(all_partners),
        )

        if len(raw_list) < PAGE_SIZE:
            break

        skip += PAGE_SIZE

    elapsed = time.time() - start_time
    logger.info(
        "Business partners fetch complete | total=%d | elapsed=%.2fs",
        len(all_partners),
        elapsed,
    )
    return all_partners


async def get_business_partners() -> list[BusinessPartner]:
    """
    Fetch ALL business partners from S/4HANA asynchronously.

    Runs the blocking HTTP call in a thread pool via asyncio.to_thread
    so the FastAPI event loop is never blocked.
    """
    logger.info("get_business_partners() called — fetching ALL partners (no limit)")
    return await asyncio.to_thread(_fetch_business_partners_sync)