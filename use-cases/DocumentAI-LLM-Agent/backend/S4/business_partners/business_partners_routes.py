"""
Business Partners FastAPI router.

Endpoint
--------
GET /api/business-partners
    Returns ALL Business Partners from S/4HANA API_BUSINESS_PARTNER / A_BusinessPartner.
    No limit. Automatic pagination handled server-side.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from S4.business_partners.business_partners_models import BusinessPartnersResponse
from S4.business_partners.business_partners_service import get_business_partners

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Business Partners"])


@router.get(
    "/business-partners",
    response_model=BusinessPartnersResponse,
    summary="Retrieve ALL Business Partners from S/4HANA",
    description=(
        "Fetches ALL Business Partners from the S/4HANA OData API "
        "(API_BUSINESS_PARTNER / A_BusinessPartner) using automatic pagination. "
        "No hardcoded limit. This is a READ-ONLY endpoint."
    ),
)
async def list_business_partners() -> BusinessPartnersResponse:
    """Return ALL Business Partners from S/4HANA."""
    logger.info("GET /api/business-partners — fetching all partners")

    try:
        partners = await get_business_partners()
    except RuntimeError as exc:
        logger.error("Failed to retrieve Business Partners | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in GET /api/business-partners")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(exc).__name__}",
        ) from exc

    logger.info("GET /api/business-partners | returned=%d partners", len(partners))
    return BusinessPartnersResponse(
        success=True,
        count=len(partners),
        limit=len(partners),
        business_partners=partners,
    )