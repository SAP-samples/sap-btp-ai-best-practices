"""
RFM Analysis API Router

Endpoints for RFM (Recency, Frequency, Monetary) analysis with quarterly trends
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from ..services.rfm_analyzer import (
    get_rfm_segment,
    get_rfm_distribution,
    get_quarterly_trend,
    get_rfm_by_customer_group,
    get_rfm_segment_with_quarterly,
    get_customer_groups_summary
)

router = APIRouter(prefix="/api/rfm", tags=["rfm"])
logger = logging.getLogger(__name__)


@router.get("/customer/{customer_id}")
async def get_customer_rfm(customer_id: str):
    """
    Get RFM segment for specific customer

    Returns:
    - customer_id
    - rfm_segment: Champions, Loyal, At Risk, Promising, or Need Attention
    - recency_score: 1-100 (higher = more recent)
    - frequency_score: 1-100 (higher = more frequent)
    - monetary_score: 1-100 (higher = more spend)
    - recency_days: Days since last order
    - frequency_count: Number of orders
    - monetary_total: Total COGS
    """
    try:
        result = get_rfm_segment(customer_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RFM for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quarterly/{customer_id}")
async def get_customer_quarterly_trend(customer_id: str):
    """
    Get quarterly COGS trend (Q1 → Q2 → Q3 → Q4)

    Returns:
    - customer_id
    - q1_cogs, q2_cogs, q3_cogs, q4_cogs: Quarterly COGS values
    - total_cogs: Sum of all quarters
    - growth_trend: 'up', 'down', or 'stable' (Q4 vs Q1)
    - active_quarters: Number of quarters with activity (1-4)
    - quarterly_average: Average COGS per quarter
    """
    try:
        result = get_quarterly_trend(customer_id)
        return result
    except Exception as e:
        logger.error(f"Error getting quarterly trend for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customer/{customer_id}/full")
async def get_customer_rfm_full(customer_id: str):
    """
    Get combined RFM + quarterly data for customer

    Returns all RFM metrics + quarterly breakdown in one call
    """
    try:
        result = get_rfm_segment_with_quarterly(customer_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting full RFM for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-group/{customer_group}")
async def get_rfm_for_customer_group(
    customer_group: str,
    limit: int = Query(100, ge=1, le=500)
):
    """
    Get RFM distribution for specific customer group

    Parameters:
    - customer_group: Customer group code (AC, ME, CB, CW, AB, etc. - 41 groups)
    - limit: Max customers to return (1-500, default 100)

    Returns:
    - customer_group: Group code
    - customer_count: Total customers in group
    - rfm_distribution: Segment breakdown (Champions, Loyal, At Risk, etc.)
    - avg_total_cogs: Average COGS per customer
    - top_customers: Top customers by COGS (limited by limit parameter)
    """
    try:
        result = get_rfm_by_customer_group(customer_group, limit=limit)
        return result
    except Exception as e:
        logger.error(f"Error getting RFM for group {customer_group}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/groups/summary")
async def get_all_customer_groups_summary():
    """
    Get summary of all customer groups (41 groups) with RFM stats

    Returns list of all customer groups with:
    - customer_group: Group code
    - customer_count: Number of customers in group
    - avg_total_cogs: Average COGS
    - rfm_distribution: RFM segment breakdown

    Sorted by customer_count descending
    """
    try:
        result = get_customer_groups_summary()
        return {
            "count": len(result),
            "groups": result
        }
    except Exception as e:
        logger.error(f"Error getting customer groups summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution")
async def get_overall_rfm_distribution():
    """
    Get overall RFM segment distribution across all customers

    Returns:
    - segment_counts: Count of customers in each segment
    - segment_percentages: Percentage of customers in each segment
    - total_customers: Total number of customers analyzed
    """
    try:
        result = get_rfm_distribution()
        return result
    except Exception as e:
        logger.error(f"Error getting RFM distribution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
