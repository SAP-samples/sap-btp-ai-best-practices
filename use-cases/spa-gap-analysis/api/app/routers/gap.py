"""
Gap Analysis API Router

Endpoints for SPA gap detection with coverage and savings analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from ..services.gap_detector import (
    detect_spa_gaps,
    detect_spa_gaps_with_coverage,
    get_gap_recommendations,
    get_gap_recommendations_with_savings,
    compare_customer_spas
)

router = APIRouter(prefix="/api/gap", tags=["gap-analysis"])
logger = logging.getLogger(__name__)


@router.get("/customer/{customer_id}")
async def detect_customer_gaps(
    customer_id: str,
    top_n_similar: int = Query(10, ge=1, le=50),
    min_similar_count: int = Query(2, ge=1, le=10)
):
    """
    Detect SPA gaps for customer (basic version)

    Parameters:
    - customer_id: Customer ID to analyze
    - top_n_similar: Number of similar customers to compare (1-50, default 10)
    - min_similar_count: Minimum similar customers that must have SPA (1-10, default 2)

    Returns:
    - target_customer_id
    - target_spas: List of SPAs customer currently has
    - target_spa_count: Number of SPAs
    - similar_customers_count: Number of similar customers found
    - missing_spas: List of gap SPAs with details:
      - sales_deal: SPA ID
      - vendor: Vendor name
      - description: SPA description
      - count_in_similar: How many similar customers have this
      - percentage_in_similar: Percentage of similar customers
      - confidence_score: Recommendation confidence (0-100)
    - missing_spa_count: Number of gaps found
    """
    try:
        result = detect_spa_gaps(
            customer_id,
            top_n_similar=top_n_similar,
            min_similar_count=min_similar_count
        )
        return result
    except Exception as e:
        logger.error(f"Error detecting gaps for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customer/{customer_id}/with-coverage")
async def detect_customer_gaps_with_coverage(
    customer_id: str,
    top_n_similar: int = Query(10, ge=1, le=50),
    min_similar_count: int = Query(2, ge=1, le=10),
    include_savings: bool = Query(True)
):
    """
    Detect SPA gaps with material coverage and potential savings (enhanced version)

    Parameters:
    - customer_id: Customer ID to analyze
    - top_n_similar: Number of similar customers (1-50, default 10)
    - min_similar_count: Minimum similar customers that must have SPA (1-10, default 2)
    - include_savings: Calculate potential savings (default: true)

    Returns same as basic version plus:
    - agreement_grouping: Agreement tier (D, A, B, C, E, K)
    - coverage_level: 'material' (A703), 'group' (A704), or 'none'
    - materials_count: Number of materials in A703
    - groups_count: Number of material groups in A704
    - total_coverage_count: Total materials/groups covered
    - potential_savings: Estimated savings if customer gets this SPA ($)
    - total_potential_savings: Sum of all gap savings
    - coverage_distribution: Breakdown by coverage level
    """
    try:
        result = detect_spa_gaps_with_coverage(
            customer_id,
            top_n_similar=top_n_similar,
            min_similar_count=min_similar_count,
            include_potential_savings=include_savings
        )
        return result
    except Exception as e:
        logger.error(f"Error detecting gaps with coverage for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{customer_id}")
async def get_gap_recommendations_basic(
    customer_id: str,
    top_n_similar: int = Query(10, ge=1, le=50),
    min_similar_count: int = Query(2, ge=1, le=10),
    top_n_recommendations: int = Query(5, ge=1, le=20)
):
    """
    Get top N SPA recommendations (basic version)

    Parameters:
    - customer_id: Customer ID
    - top_n_similar: Number of similar customers (1-50, default 10)
    - min_similar_count: Min similar customers with SPA (1-10, default 2)
    - top_n_recommendations: Number of recommendations to return (1-20, default 5)

    Returns:
    - List of top recommended SPAs (simplified, sorted by confidence)
    """
    try:
        result = get_gap_recommendations(
            customer_id,
            top_n_similar=top_n_similar,
            min_similar_count=min_similar_count,
            top_n_recommendations=top_n_recommendations
        )
        return {
            "customer_id": customer_id,
            "recommendation_count": len(result),
            "recommendations": result
        }
    except Exception as e:
        logger.error(f"Error getting recommendations for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{customer_id}/with-savings")
async def get_gap_recommendations_enhanced(
    customer_id: str,
    top_n_similar: int = Query(10, ge=1, le=50),
    min_similar_count: int = Query(2, ge=1, le=10),
    top_n_recommendations: int = Query(5, ge=1, le=20),
    sort_by: str = Query(
        "potential_savings",
        pattern="^(potential_savings|count_in_similar|confidence_score)$"
    )
):
    """
    Get top N SPA recommendations with savings (enhanced version)

    Parameters:
    - customer_id: Customer ID
    - top_n_similar: Number of similar customers (1-50, default 10)
    - min_similar_count: Min similar customers with SPA (1-10, default 2)
    - top_n_recommendations: Number of recommendations (1-20, default 5)
    - sort_by: 'potential_savings', 'count_in_similar', or 'confidence_score'

    Returns:
    - List of top recommended SPAs with:
      - All basic fields
      - coverage_level, materials_count, groups_count
      - potential_savings
      - agreement_grouping (tier)

    Sorted by selected metric
    """
    try:
        result = get_gap_recommendations_with_savings(
            customer_id,
            top_n_similar=top_n_similar,
            min_similar_count=min_similar_count,
            top_n_recommendations=top_n_recommendations,
            sort_by=sort_by
        )
        return {
            "customer_id": customer_id,
            "recommendation_count": len(result),
            "sort_by": sort_by,
            "recommendations": result
        }
    except Exception as e:
        logger.error(f"Error getting recommendations with savings for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/{customer1_id}/{customer2_id}")
async def compare_customers(customer1_id: str, customer2_id: str):
    """
    Compare SPAs between two customers

    Returns:
    - customer1: {customer_id, customer_name, spa_count, spas}
    - customer2: {customer_id, customer_name, spa_count, spas}
    - comparison:
      - both_have_count: SPAs both customers have
      - both_have: List of shared SPAs
      - only_customer1_count: SPAs only customer1 has
      - only_customer1: List
      - only_customer2_count: SPAs only customer2 has
      - only_customer2: List
      - overlap_percentage: Percentage of SPAs in common
    """
    try:
        result = compare_customer_spas(customer1_id, customer2_id)
        return result
    except Exception as e:
        logger.error(f"Error comparing {customer1_id} and {customer2_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
