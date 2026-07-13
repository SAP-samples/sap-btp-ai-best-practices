"""
Potential Breakdown API Router

Provides detailed material-level breakdown for potential savings
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import logging

from app.models.common import ErrorResponse
from app.services.potential_breakdown_service import (
    get_potential_breakdown,
    get_spa_materials
)
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spa", tags=["potential-breakdown"])


@router.get(
    "/customer/{customer_id}/potential-breakdown",
    dependencies=[Depends(require_api_key)],
    summary="Get Material-Level Potential Breakdown",
    description="Detailed breakdown of potential savings by material and SPA"
)
async def get_customer_potential_breakdown(customer_id: str):
    """
    Get detailed material-level breakdown of potential savings

    Returns:
    - Coverage statistics (materials with SPA pricing)
    - Top materials by potential savings
    - SPA breakdown (potential per missing SPA)
    """
    try:
        result = get_potential_breakdown(customer_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting potential breakdown for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/spa/{spa_id}/materials",
    dependencies=[Depends(require_api_key)],
    summary="Get Materials Covered by SPA",
    description="Get list of materials covered by a specific SPA"
)
async def get_spa_materials_endpoint(
    spa_id: str,
    customer_id: Optional[str] = Query(None, description="Optional customer ID to filter to customer's purchases")
):
    """
    Get materials covered by a specific SPA

    If customer_id provided:
    - Filters to materials customer actually purchases
    - Includes customer COGS and potential savings
    """
    try:
        result = get_spa_materials(spa_id, customer_id)
        return result
    except Exception as e:
        logger.error(f"Error getting SPA materials for {spa_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
