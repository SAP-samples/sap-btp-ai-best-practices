"""
Margin API Router

Endpoints for savings/margin calculations using BASE_COST comparison
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from ..services.margin_calculator import (
    calculate_customer_savings,
    calculate_agreement_savings,
    get_savings_leaderboard,
    get_material_savings_by_group,
    get_savings_summary,
    get_savings_distribution
)

router = APIRouter(prefix="/api/margin", tags=["margin"])
logger = logging.getLogger(__name__)


@router.get("/customer/{customer_id}")
async def get_customer_margin(customer_id: str):
    """
    Get savings/margin for specific customer

    Returns:
    - total_savings: Total savings amount ($)
    - total_base_cost: Total base cost ($)
    - total_spa_cost: Total SPA cost ($)
    - savings_percent: Savings percentage
    - material_count: Number of materials with savings
    - top_savings_materials: Top 5 materials by savings
    """
    try:
        result = calculate_customer_savings(customer_id)
        return result
    except Exception as e:
        logger.error(f"Error getting customer savings for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agreement/{agreement_id}")
async def get_agreement_margin(agreement_id: str):
    """
    Get savings/margin for specific SPA agreement

    Returns:
    - agreement_id: SPA agreement ID
    - total_savings: Total savings for this agreement
    - average_savings_percent: Average savings %
    - material_count: Number of materials in agreement
    - customer_count: Number of customers using agreement
    - top_materials: Top 10 materials by savings
    """
    try:
        result = calculate_agreement_savings(agreement_id)
        return result
    except Exception as e:
        logger.error(f"Error getting agreement savings for {agreement_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_margin_leaderboard(
    top_n: int = Query(50, ge=1, le=500),
    sort_by: str = Query("total_savings", pattern="^(total_savings|savings_percent)$")
):
    """
    Get top customers by savings/margin

    Parameters:
    - top_n: Number of top customers to return (1-500, default 50)
    - sort_by: Sort by 'total_savings' or 'savings_percent' (default: total_savings)

    Returns:
    - List of customers with savings data
    """
    try:
        result = get_savings_leaderboard(top_n=top_n, sort_by=sort_by)
        return {
            "count": len(result),
            "sort_by": sort_by,
            "customers": result
        }
    except Exception as e:
        logger.error(f"Error getting savings leaderboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/material-group/{material_group}")
async def get_material_group_margin(material_group: str):
    """
    Get savings/margin for specific material group

    Returns:
    - material_group: Material group code
    - total_savings: Total savings for group
    - average_savings_percent: Average savings %
    - material_count: Number of materials in group
    """
    try:
        result = get_material_savings_by_group(material_group)
        return result
    except Exception as e:
        logger.error(f"Error getting material group savings for {material_group}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_margin_summary():
    """
    Get overall savings/margin summary

    Returns:
    - total_savings: Total savings across all customers
    - total_base_cost: Total base cost
    - total_spa_cost: Total SPA cost
    - average_savings_percent: Average savings %
    - customer_count: Number of customers with savings
    - material_count: Number of materials with savings
    """
    try:
        result = get_savings_summary()
        return result
    except Exception as e:
        logger.error(f"Error getting savings summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution")
async def get_margin_distribution():
    """
    Get distribution of savings across customer segments

    Returns distribution by savings percentage ranges:
    - 0-5%
    - 5-10%
    - 10-15%
    - 15-20%
    - 20-25%
    - 25%+

    Each range includes:
    - count: Number of customers
    - percentage: Percentage of total customers
    """
    try:
        result = get_savings_distribution()
        return {
            "distribution": result,
            "description": "Customer distribution by savings percentage"
        }
    except Exception as e:
        logger.error(f"Error getting savings distribution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
