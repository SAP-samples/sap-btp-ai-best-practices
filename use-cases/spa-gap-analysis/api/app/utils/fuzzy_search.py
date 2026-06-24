"""
Fuzzy search utility for customer matching using Levenshtein distance.
"""
from rapidfuzz import fuzz, process
from typing import List, Dict
from app.services.data_loader import load_customer_master


def fuzzy_search_customers(query: str, limit: int = 10, threshold: int = 60) -> List[Dict]:
    """
    Fuzzy search customers by name using Levenshtein distance.

    Args:
        query: Search query (partial customer name)
        limit: Maximum results to return
        threshold: Minimum similarity score (0-100)

    Returns:
        List of customers with similarity scores

    Examples:
        >>> fuzzy_search_customers("TEST ELECTRIC", limit=5)
        [{"customer_id": "999001", "customer_name": "TEST DATA - TEST ELECTRIC CO", "similarity_score": 95, ...}]
    """
    # Load customer master data
    customers_df = load_customer_master()

    for optional_column in ["city", "state"]:
        if optional_column not in customers_df.columns:
            customers_df[optional_column] = None

    # Extract customer records with necessary fields
    customer_records = customers_df[['customer_id', 'customer_name', 'city', 'state']].to_dict('records')

    # Extract customer names for fuzzy matching
    customer_names = [c['customer_name'] for c in customer_records]

    # Perform fuzzy matching using WRatio scorer (handles partial matches well)
    matches = process.extract(
        query,
        customer_names,
        scorer=fuzz.WRatio,
        limit=limit * 2  # Get more candidates to filter
    )

    # Filter by threshold and prepare results
    results = []
    for match_text, score, idx in matches:
        if score >= threshold:
            customer = customer_records[idx].copy()
            customer['similarity_score'] = round(score, 1)
            results.append(customer)

    # Return top matches
    return results[:limit]


def fuzzy_match_best_customer(query: str, threshold: int = 60) -> Dict | None:
    """
    Find the best matching customer using fuzzy search.

    Args:
        query: Search query (partial customer name or ID)
        threshold: Minimum similarity score (0-100)

    Returns:
        Best matching customer dict or None if no match above threshold

    Examples:
        >>> fuzzy_match_best_customer("TEST ELECTRIC")
        {"customer_id": "999001", "customer_name": "TEST DATA - TEST ELECTRIC CO", "similarity_score": 95, ...}
    """
    results = fuzzy_search_customers(query, limit=1, threshold=threshold)
    return results[0] if results else None
