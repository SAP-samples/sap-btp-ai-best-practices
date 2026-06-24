"""
LangChain tool wrappers for SPA Gap Analysis.

This module wraps existing services as LangChain tools for LLM agent use.
"""
import logging
import numpy as np
import json
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any

from app.services import (
    get_customer_profile,
    search_customers,
    find_similar_customers,
    get_rfm_distribution,
    load_from_parquet
)
from app.services.onboarding_service import (
    research_customer_with_sonar,
    find_similar_customers_by_profile
)
from app.utils.fuzzy_search import fuzzy_search_customers

logger = logging.getLogger(__name__)


def _profile_rfm_segment(profile: Dict[str, Any]) -> str:
    """Read RFM segment from the canonical customer profile shape."""
    rfm = profile.get("rfm") or {}
    return rfm.get("segment") or profile.get("rfm_segment") or "N/A"


def _profile_current_spas(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read current SPAs from canonical profile details."""
    details = profile.get("current_spa_details") or []
    if details:
        return details
    return [{"sales_deal": spa} for spa in profile.get("spas", [])]


def _summary_missing_spas(customer_id: str) -> Dict[str, Any]:
    """Read canonical missing SPA count/details from the summary cache."""
    try:
        cache = load_from_parquet("customer_summary_cache.parquet")
        row = cache[cache["customer_id"].astype(str) == str(customer_id)]
        if row.empty:
            return {"missing_spas": [], "missing_spa_count": 0}

        record = row.iloc[0]
        details_raw = record.get("missing_spas_details") or "[]"
        try:
            details = json.loads(details_raw) if isinstance(details_raw, str) else []
        except Exception:
            details = []

        return {
            "missing_spas": details,
            "missing_spa_count": int(record.get("missing_spas_count") or len(details)),
        }
    except Exception as exc:
        logger.warning(f"Could not load canonical summary cache for customer {customer_id}: {exc}")
        return {"missing_spas": [], "missing_spa_count": 0}


def clean_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@tool
def analyze_customer_tool(customer_id: str) -> dict:
    """
    Analyze a customer for SPA gaps and missing opportunities.

    Use this tool when the user asks to analyze a specific customer,
    find missing SPAs, or understand what agreements they should have.

    Args:
        customer_id: Customer ID to analyze (e.g., "999001", "375567")

    Returns:
        Dict with customer profile, missing SPAs, confidence scores, and similar customer analysis.
        Includes: customer_id, customer_name, sales_office, customer_type, current_spas,
        missing_spas (with confidence scores), similar_customers_count
    """
    try:
        # Get customer profile
        profile = get_customer_profile(customer_id)
        if not profile:
            return {"error": f"Customer {customer_id} not found in database"}

        current_spas = _profile_current_spas(profile)
        summary_gap = _summary_missing_spas(customer_id)
        similar_customers = find_similar_customers(customer_id, top_n=50)

        result = {
            "customer_id": customer_id,
            "customer_name": profile.get("customer_name", "Unknown"),
            "sales_office": profile.get("sales_office"),
            "customer_type": profile.get("pl_type", "N/A"),
            "city": profile.get("city"),
            "state": profile.get("state"),
            "rfm_segment": _profile_rfm_segment(profile),
            "current_spas": current_spas,
            "current_spa_count": int(
                profile.get("current_spa_count_unique")
                or profile.get("spa_count")
                or len(current_spas)
            ),
            "missing_spas": summary_gap["missing_spas"],
            "missing_spa_count": summary_gap["missing_spa_count"],
            "similar_customers_count": len(similar_customers),
            "analysis_complete": True
        }

        # Clean numpy types for JSON serialization
        return clean_numpy_types(result)

    except Exception as e:
        logger.error(f"Error analyzing customer {customer_id}: {e}", exc_info=True)
        return {"error": f"Failed to analyze customer: {str(e)}"}


@tool
def search_customers_tool(
    query: Optional[str] = None,
    state: Optional[str] = None,
    rfm_segment: Optional[str] = None,
    sales_office: Optional[str] = None,
    limit: int = 10
) -> dict:
    """
    Search for customers by name, location, or attributes.

    Use this tool when the user wants to find customers by name (supports fuzzy matching),
    location (state), RFM segment, or sales office. Good for queries like "Find customers in Utah"
    or "Find ABC Corp" or "Show me Champions customers".

    Args:
        query: Customer name or partial name (supports fuzzy matching like "TEST" finds "TEST ELECTRIC CO")
        state: US state code (e.g., "UT" for Utah, "AZ" for Arizona, "CA" for California)
        rfm_segment: RFM segment filter (e.g., "Champions", "Loyal", "At Risk", "Promising", "Need Attention")
        sales_office: Sales office code (e.g., "402", "401")
        limit: Maximum results to return (default 10)

    Returns:
        Dict with customers list and count. Each customer includes: customer_id, customer_name,
        city, state, rfm_segment, sales_office
    """
    try:
        # Use fuzzy search if query provided
        if query:
            customers = fuzzy_search_customers(query, limit=limit, threshold=60)

            # Apply additional filters if provided
            if state:
                customers = [c for c in customers if c.get('state', '').upper() == state.upper()]
            if rfm_segment:
                from app.services import get_customer_profile
                filtered = []
                for c in customers:
                    profile = get_customer_profile(c['customer_id'])
                    if profile and _profile_rfm_segment(profile) == rfm_segment:
                        c['rfm_segment'] = rfm_segment
                        filtered.append(c)
                customers = filtered[:limit]

        else:
            # Use standard search
            customers = search_customers(
                query=query,
                state=state,
                rfm_segment=rfm_segment,
                sales_office=sales_office,
                limit=limit
            )

        result = {
            "customers": customers,
            "count": len(customers),
            "search_params": {
                "query": query,
                "state": state,
                "rfm_segment": rfm_segment,
                "sales_office": sales_office,
                "limit": limit
            }
        }

        # Clean numpy types for JSON serialization
        return clean_numpy_types(result)

    except Exception as e:
        logger.error(f"Error searching customers: {e}", exc_info=True)
        return {"error": f"Failed to search customers: {str(e)}"}


@tool
def find_similar_customers_tool(customer_id: str, top_n: int = 10, exclude_unknown: bool = False) -> dict:
    """
    Find customers similar to the target customer based on sales office, type, and spending patterns.

    Use this tool when the user wants to compare a customer with peers, find similar customers,
    or understand who else has similar characteristics.

    Args:
        customer_id: Target customer ID to find similar customers for
        top_n: Number of similar customers to return (default 10, max 50)
        exclude_unknown: If True, exclude customers with "Unknown" names (default False)

    Returns:
        Dict with similar_customers list and count. Each similar customer includes:
        customer_id, customer_name, similarity_score, sales_office, customer_type, shared_spas
    """
    try:
        # Limit top_n to reasonable range
        top_n = min(max(1, top_n), 50)

        # Find similar customers
        similar = find_similar_customers(customer_id, top_n=top_n, exclude_unknown=exclude_unknown)

        if not similar:
            return {
                "similar_customers": [],
                "count": 0,
                "message": f"No similar customers found for customer {customer_id}"
            }

        result = {
            "similar_customers": similar,
            "count": len(similar),
            "target_customer_id": customer_id,
            "top_n": top_n
        }

        # Clean numpy types for JSON serialization
        return clean_numpy_types(result)

    except Exception as e:
        logger.error(f"Error finding similar customers for {customer_id}: {e}", exc_info=True)
        return {"error": f"Failed to find similar customers: {str(e)}"}


@tool
def get_rfm_distribution_tool() -> dict:
    """
    Get RFM (Recency, Frequency, Monetary) segment distribution across all customers.

    Use this tool when the user asks about customer segments, RFM distribution,
    how many Champions/Loyal/At Risk customers exist, or wants overall customer breakdown.

    Returns:
        Dict with segment counts and percentages. Includes:
        - Champions: High value, frequent buyers
        - Loyal: Regular customers
        - At Risk: Used to buy frequently, declining
        - Promising: Recent customers with potential
        - Need Attention: Below average
        - Lost: Haven't purchased recently
        - Total customer count
    """
    try:
        distribution = get_rfm_distribution()

        # Calculate total customers from segment data (skip 'total_customers' key if present)
        total = 0
        for key, value in distribution.items():
            if key != 'total_customers' and isinstance(value, dict) and 'count' in value:
                total += value['count']

        result = {
            "rfm_distribution": distribution,
            "total_customers": total or distribution.get('total_customers', 0),
            "segments": [k for k in distribution.keys() if k != 'total_customers']
        }

        # Clean numpy types for JSON serialization
        return clean_numpy_types(result)

    except Exception as e:
        logger.error(f"Error getting RFM distribution: {e}", exc_info=True)
        return {"error": f"Failed to get RFM distribution: {str(e)}"}


@tool
def research_customer_onboarding_tool(
    customer_name: str,
    location: Optional[str] = None,
    exclude_unknown: bool = True,
) -> dict:
    """
    Research a new customer for onboarding using Sonar-Pro web intelligence.
    Find public information about the company and identify similar existing customers.

    Use this tool when the user wants to onboard a new customer, research a prospect,
    or find information about a company that is not yet in the database.

    Context: Our customers are local electrical contractors and installers serving
    schools, hospitals, data centers, commercial buildings, and industrial facilities.

    Args:
        customer_name: Name of the new customer/prospect to research (e.g., "ACME Electric LLC")
        location: Optional location for the customer (city, state) (e.g., "Demo City, AZ" or "Utah")
        exclude_unknown: Accepted for Agent Chat tool-call compatibility. Similar-customer
            search already prioritizes named customers where source data allows it.

    Returns:
        Dict with research results including:
        - profile: Customer business profile from Sonar
        - business_type: Type of business (school_contractor, hospital_contractor, etc.)
        - confidence: Confidence score (0-100)
        - materials_likely_needed: List of material category codes
        - similar_customers: List of existing customers with similar profiles
    """
    try:
        # Step 1: Research customer with Sonar-Pro
        logger.info(
            f"Researching customer onboarding: {customer_name}, location: {location}, "
            f"exclude_unknown={exclude_unknown}"
        )
        research_result = research_customer_with_sonar(customer_name, location)

        if not research_result.get('success'):
            return {"error": f"Failed to research customer: {research_result.get('profile', 'Unknown error')}"}

        # Step 2: Find similar existing customers if confidence is reasonable
        confidence = research_result.get('confidence', 0)
        similar_customers = []

        if confidence >= 30:  # Minimum threshold to search for similar customers
            business_type = research_result.get('business_type', 'general_contractor')
            material_categories = research_result.get('materials_likely_needed', [])

            logger.info(f"Searching for similar customers: business_type={business_type}, categories={material_categories}")
            similar_customers = find_similar_customers_by_profile(
                business_type=business_type,
                material_categories=material_categories,
                top_n=5
            )

        result = {
            "customer_name": customer_name,
            "location": location,
            "profile": research_result.get('profile', ''),
            "business_type": research_result.get('business_type', 'unknown'),
            "confidence": confidence,
            "materials_likely_needed": research_result.get('materials_likely_needed', []),
            "sources": research_result.get('sources', []),  # Add sources
            "similar_customers": similar_customers,
            "similar_customers_count": len(similar_customers)
        }

        # Clean numpy types for JSON serialization
        return clean_numpy_types(result)

    except Exception as e:
        logger.error(f"Error in customer onboarding research: {e}", exc_info=True)
        return {"error": f"Failed to research customer onboarding: {str(e)}"}


# Export all tools
__all__ = [
    'analyze_customer_tool',
    'search_customers_tool',
    'find_similar_customers_tool',
    'get_rfm_distribution_tool',
    'research_customer_onboarding_tool'
]
