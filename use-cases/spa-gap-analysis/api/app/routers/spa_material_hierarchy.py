"""
Material Hierarchy API Router

API endpoints for material hierarchy aggregation and drill-down.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from app.models.spa_models import (
    MaterialHierarchySummaryResponse,
    MaterialDrilldownRequest
)
from app.services.material_aggregator import aggregate_material_hierarchy
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/customer/{customer_id}/material-summary",
    response_model=MaterialHierarchySummaryResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get Material Hierarchy Summary",
    description="Get aggregated material purchases by hierarchy Level 1 with SPA coverage"
)
async def get_material_summary(customer_id: str) -> MaterialHierarchySummaryResponse:
    """
    Get material hierarchy summary for a customer at Level 1.

    Returns top-level product categories with COGS, transaction counts,
    and SPA coverage percentages.

    Args:
        customer_id: Customer ID to analyze

    Returns:
        MaterialHierarchySummaryResponse with Level 1 categories
    """
    logger.info(f"Material summary requested for customer: {customer_id}")

    try:
        result = aggregate_material_hierarchy(customer_id=customer_id, level=1)
        return MaterialHierarchySummaryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating material summary for {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate material summary: {str(e)}"
        )


@router.post(
    "/customer/{customer_id}/material-drilldown",
    response_model=MaterialHierarchySummaryResponse,
    dependencies=[Depends(require_api_key)],
    summary="Drill Down Material Hierarchy",
    description="Get child categories at a deeper hierarchy level"
)
async def get_material_drilldown(
    customer_id: str,
    request: MaterialDrilldownRequest
) -> MaterialHierarchySummaryResponse:
    """
    Drill down to a deeper hierarchy level.

    Enables progressive exploration of material hierarchy from Level 1 → 2 → 3 → 4.

    Args:
        customer_id: Customer ID to analyze
        request: Drill-down request with level and optional parent_category

    Returns:
        MaterialHierarchySummaryResponse with child categories at requested level
    """
    logger.info(
        f"Material drilldown requested for customer: {customer_id}, "
        f"level: {request.level}, parent: {request.parent_category}"
    )

    try:
        result = aggregate_material_hierarchy(
            customer_id=customer_id,
            level=request.level,
            parent_category=request.parent_category
        )
        return MaterialHierarchySummaryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in material drilldown for {customer_id} "
            f"(level={request.level}, parent={request.parent_category}): {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to drill down material hierarchy: {str(e)}"
        )
