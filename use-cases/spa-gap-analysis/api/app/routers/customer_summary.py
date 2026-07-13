"""
Customer Summary API Router

Provides combined customer data for UI summary view
"""

from fastapi import APIRouter, HTTPException, Body, Depends
from typing import Optional, Dict
import logging

from ..services.customer_summary_service import (
    get_customer_summary,
    get_customer_summary_stats
)
from ..security import require_api_key

router = APIRouter(prefix="/api/customer", tags=["customer-summary"])
logger = logging.getLogger(__name__)


@router.post("/summary", dependencies=[Depends(require_api_key)])
async def get_summary(
    filters: Optional[Dict] = Body(None),
    sort_by: str = Body("total_cogs"),
    sort_order: str = Body("desc"),
    limit: int = Body(100),
    exclude_unknown: bool = Body(False)
):
    """
    Get customer summary for UI table view

    Request body:
    {
        "filters": {
            "rfm_segment": "Champions",  // optional
            "sales_office": "1",          // optional
            "min_cogs": 10000             // optional
        },
        "sort_by": "total_cogs",          // field to sort by
        "sort_order": "desc",             // asc or desc
        "limit": 100,                     // max customers to return
        "exclude_unknown": false          // exclude customers with name="Unknown"
    }

    Returns:
    {
        "total_customers": 17041,
        "filtered_customers": 1403,
        "summary_stats": {
            "total_savings": 28954378.42,
            "total_cogs": 1082859023.39,
            "customers_with_savings": 2713,
            "champions_count": 1403
        },
        "customers": [
            {
                "customer_id": "16850",
                "rfm_segment": "Champions",
                "total_cogs": 20171935.62,
                "total_savings": 123456.78,
                "savings_percent": 45.2,
                ...
            }
        ]
    }
    """
    try:
        result = get_customer_summary(
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            exclude_unknown=exclude_unknown
        )
        return result
    except Exception as e:
        logger.error(f"Error getting customer summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/stats", dependencies=[Depends(require_api_key)])
async def get_stats():
    """
    Get overall customer summary statistics

    Returns:
    {
        "total_customers": 17041,
        "total_cogs": 1082859023.39,
        "total_savings": 28954378.42,
        "customers_with_savings": 2713,
        "rfm_distribution": {
            "Champions": 1403,
            "Loyal": 1015,
            ...
        }
    }
    """
    try:
        result = get_customer_summary_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting summary stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
