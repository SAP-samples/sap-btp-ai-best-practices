"""
Customer Summary API Router

Provides high-level overview of SPA gaps across all customers.
Used for Summary View (Tab 3) to quickly identify opportunities.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import logging

from app.models.spa_models import (
    CustomerSummaryRequest,
    CustomerSummaryResponse,
    CustomerSummaryFilters,
    ErrorResponse
)
from app.services.summary_aggregator import (
    aggregate_customer_summaries,
    generate_and_cache_summary
)
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spa", tags=["SPA Summary"])


@router.post(
    "/customer-summary",
    response_model=CustomerSummaryResponse,
    dependencies=[Depends(require_api_key)],
    summary="Customer Summary with Missing SPAs",
    description="Get aggregated view of all customers with missing SPAs count and potential value"
)
async def customer_summary(request: CustomerSummaryRequest) -> CustomerSummaryResponse:
    """
    Get customer summary with missing SPAs analysis.

    This endpoint provides a high-level overview of SPA gaps across all customers.
    Results are cached for performance - use /regenerate-summary to refresh cache.

    **Use Cases:**
    - Summary View (Tab 3): Display all customers with missing SPAs
    - Identify top opportunities by missing SPAs count or potential value
    - Filter by RFM segment, sales office, minimum COGS
    - Sort by various metrics

    **Performance:**
    - First call (no cache): 2-5 minutes (calculates missing SPAs for all customers)
    - Subsequent calls (cached): <2 seconds

    **Caching Strategy:**
    - Cache generated during ETL run
    - Manual refresh via /regenerate-summary endpoint
    - Cache valid until next ETL run

    Args:
        request: CustomerSummaryRequest with filters, sort, limit

    Returns:
        CustomerSummaryResponse with aggregated customer data

    Raises:
        HTTPException 500: If summary generation fails
    """
    try:
        logger.info(f"Customer summary request: filters={request.filters}, sort={request.sort_by}")

        # Convert filters to dict
        filters_dict = {}
        if request.filters:
            if request.filters.rfm_segment:
                filters_dict['rfm_segment'] = request.filters.rfm_segment
            if request.filters.sales_office:
                # Support comma-separated offices
                offices = request.filters.sales_office.split(',')
                filters_dict['sales_office'] = [o.strip() for o in offices]
            if request.filters.min_missing_spas is not None:
                filters_dict['min_missing_spas'] = request.filters.min_missing_spas
            if request.filters.min_cogs is not None:
                filters_dict['min_cogs'] = request.filters.min_cogs
            if request.filters.min_potential is not None:
                filters_dict['min_potential'] = request.filters.min_potential

        # Get aggregated summary
        result = aggregate_customer_summaries(
            filters=filters_dict if filters_dict else None,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            limit=request.limit,
            use_cache=True
        )

        logger.info(
            f"Summary complete: {result['filtered_customers']}/{result['total_customers']} customers, "
            f"{len(result['customers'])} returned"
        )

        return CustomerSummaryResponse(**result)

    except Exception as e:
        logger.error(f"Error in customer summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate customer summary: {str(e)}"
        )


@router.post(
    "/regenerate-summary",
    dependencies=[Depends(require_api_key)],
    summary="Regenerate Customer Summary Cache",
    description="Force regeneration of customer summary cache (admin endpoint)"
)
async def regenerate_summary():
    """
    Regenerate and cache customer summary.

    **WARNING:** This operation takes 2-5 minutes and calculates missing SPAs for ALL customers.
    Use sparingly - typically only needed after data changes or ETL runs.

    **Recommended Usage:**
    - Call this endpoint as part of ETL pipeline
    - Schedule daily/weekly regeneration
    - Use manual trigger only when needed

    Returns:
        Dict with generation results

    Raises:
        HTTPException 500: If regeneration fails
    """
    try:
        logger.info("Starting customer summary regeneration (admin request)")

        result = generate_and_cache_summary()

        if result['success']:
            logger.info(
                f"Summary regeneration complete: {result['customers_processed']} customers, "
                f"{result['cache_size_mb']} MB cached"
            )
            return {
                "status": "success",
                "message": "Customer summary cache regenerated successfully",
                "details": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save summary cache"
            )

    except Exception as e:
        logger.error(f"Error regenerating summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate summary: {str(e)}"
        )


@router.get(
    "/summary-stats",
    dependencies=[Depends(require_api_key)],
    summary="Get Summary Statistics",
    description="Get high-level statistics about customer SPA gaps"
)
async def summary_stats(
    rfm_segment: Optional[str] = Query(None, description="Filter by RFM segment"),
    sales_office: Optional[str] = Query(None, description="Filter by sales office")
):
    """
    Get high-level summary statistics.

    Faster than /customer-summary when you only need aggregates.

    Args:
        rfm_segment: Optional RFM segment filter
        sales_office: Optional sales office filter

    Returns:
        Dict with summary statistics:
            - total_customers
            - total_missing_spas
            - avg_missing_spas
            - total_potential_value
            - high_confidence_opportunities
            - rfm_distribution
            - office_distribution
    """
    try:
        filters_dict = {}
        if rfm_segment:
            filters_dict['rfm_segment'] = rfm_segment
        if sales_office:
            offices = sales_office.split(',')
            filters_dict['sales_office'] = [o.strip() for o in offices]

        result = aggregate_customer_summaries(
            filters=filters_dict if filters_dict else None,
            limit=0  # No customer records, just stats
        )

        return {
            "total_customers": result['total_customers'],
            "filtered_customers": result['filtered_customers'],
            "summary_stats": result['summary_stats']
        }

    except Exception as e:
        logger.error(f"Error getting summary stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get summary stats: {str(e)}"
        )
