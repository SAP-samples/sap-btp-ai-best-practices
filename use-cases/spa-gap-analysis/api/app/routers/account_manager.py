"""
Account Manager API Router

Endpoints for Account Manager dashboards and metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from ..services.account_manager_service import (
    get_am_dashboard,
    get_am_leaderboard,
    get_am_customer_list,
    get_am_summary
)

router = APIRouter(prefix="/api/am", tags=["account-manager"])
logger = logging.getLogger(__name__)


@router.get("/dashboard/{am_id}")
async def get_account_manager_dashboard(am_id: str):
    """
    Get Account Manager dashboard with full metrics

    Returns:
    - account_manager_id: AM ID
    - account_manager_name: AM name (from Ai4U template if available)
    - customer_count: Number of customers managed
    - total_cogs: Total COGS across all customers
    - total_savings: Total savings across all customers
    - savings_percent: Average savings %
    - rfm_distribution: RFM segment breakdown (Champions, Loyal, At Risk, etc.)
    - top_customers: Top 10 customers by COGS
    - gap_opportunities: Number of gap opportunities (future)
    """
    try:
        result = get_am_dashboard(am_id)
        return result
    except Exception as e:
        logger.error(f"Error getting AM dashboard for {am_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_account_manager_leaderboard(
    top_n: int = Query(50, ge=1, le=200),
    sort_by: str = Query(
        "total_savings",
        pattern="^(total_savings|total_cogs|customer_count|savings_percent)$"
    )
):
    """
    Get Account Manager leaderboard

    Parameters:
    - top_n: Number of top AMs to return (1-200, default 50)
    - sort_by: Sort by 'total_savings', 'total_cogs', 'customer_count', or 'savings_percent'

    Returns:
    - List of AMs with metrics:
      - account_manager_id
      - customer_count
      - total_cogs
      - total_savings
      - savings_percent
    """
    try:
        result = get_am_leaderboard(top_n=top_n, sort_by=sort_by)
        return {
            "count": len(result),
            "sort_by": sort_by,
            "account_managers": result
        }
    except Exception as e:
        logger.error(f"Error getting AM leaderboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers/{am_id}")
async def get_account_manager_customers(
    am_id: str,
    include_rfm: bool = Query(True),
    include_savings: bool = Query(True)
):
    """
    Get full customer list for Account Manager

    Parameters:
    - am_id: Account Manager ID
    - include_rfm: Include RFM data for each customer (default: true)
    - include_savings: Include savings data for each customer (default: true)

    Returns:
    - List of customers with:
      - customer_id
      - customer_group
      - sales_office
      - rfm_segment (if include_rfm=true)
      - total_cogs (if include_rfm=true)
      - growth_trend (if include_rfm=true)
      - total_savings (if include_savings=true)
      - savings_percent (if include_savings=true)
    """
    try:
        result = get_am_customer_list(
            am_id,
            include_rfm=include_rfm,
            include_savings=include_savings
        )
        return {
            "account_manager_id": am_id,
            "customer_count": len(result),
            "customers": result
        }
    except Exception as e:
        logger.error(f"Error getting customer list for AM {am_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_account_manager_summary():
    """
    Get overall Account Manager statistics

    Returns:
    - total_ams: Total number of Account Managers
    - avg_customers_per_am: Average customers per AM
    - avg_cogs_per_am: Average COGS per AM
    - avg_savings_per_am: Average savings per AM
    - top_am_by_savings: Top AM by total savings
    - top_am_by_customers: Top AM by customer count
    """
    try:
        result = get_am_summary()
        return result
    except Exception as e:
        logger.error(f"Error getting AM summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
