"""
Similarity Engine: Find similar customers

Algorithm:
1. Exact match: SOff (Sales Office) - MANDATORY
2. Exact match: PLType (Price List Type) - MANDATORY
3. Rank by: COGS similarity (volume-based)
4. Boost by: RFM score (prefer active customers)
5. Boost by: Same Price Group (optional)

MVP Phase 1: Simple, explainable, fast
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from .data_loader import (
    load_customer_master,
    load_customer_cogs,
    load_rfm_scores
)

logger = logging.getLogger(__name__)


def find_similar_customers(
    target_customer_id: str,
    top_n: int = 10,
    include_rfm: bool = True,
    include_price_group: bool = True,
    exclude_unknown: bool = False
) -> List[Dict]:
    """
    Find customers similar to target customer

    Similarity Algorithm (MVP Phase 1):
    - Filter: Same SOff (Sales Office) - MANDATORY
    - Filter: Same PLType (Price List Type) - MANDATORY
    - Rank: COGS proximity (similar volume)
    - Boost: RFM score (prefer Champions/Loyal)
    - Boost: Same Price Group (+weight if match)

    Args:
        target_customer_id: Customer ID to find similar customers for
        top_n: Number of similar customers to return
        include_rfm: Include RFM boost in similarity score
        include_price_group: Include Price Group boost
        exclude_unknown: Exclude customers with "Unknown" names

    Returns:
        List of similar customer dicts with similarity scores
    """
    logger.info(f"Finding similar customers for {target_customer_id}")

    # Load data
    customer_master = load_customer_master()
    customer_cogs = load_customer_cogs()

    # Get target customer profile
    target = customer_master[
        customer_master['customer_id'] == target_customer_id
    ]

    if target.empty:
        logger.error(f"Customer {target_customer_id} not found")
        return []

    target = target.iloc[0]

    # STEP 1: MANDATORY FILTERS
    # Filter by same SOff (Sales Office)
    candidates = customer_master[
        customer_master['sales_office'] == target['sales_office']
    ].copy()

    logger.info(f"After SOff filter: {len(candidates)} candidates")

    # Filter by same PLType (Price List Type)
    candidates = candidates[
        candidates['pl_type'] == target['pl_type']
    ]

    logger.info(f"After PLType filter: {len(candidates)} candidates")

    # Exclude target customer itself
    candidates = candidates[
        candidates['customer_id'] != target_customer_id
    ]

    # Exclude "Unknown" customers if requested
    if exclude_unknown:
        candidates = candidates[
            (candidates['customer_name'].notna()) &
            (candidates['customer_name'] != 'Unknown') &
            (candidates['customer_name'] != '')
        ]
        logger.info(f"After excluding Unknown: {len(candidates)} candidates")

    if candidates.empty:
        logger.warning("No similar customers found after filters")
        return []

    # STEP 2: COGS SIMILARITY
    # Merge with customer COGS data
    candidates = candidates.merge(
        customer_cogs[['customer_id', 'total_cogs']],
        on='customer_id',
        how='left'
    )

    # Get target COGS
    target_cogs_row = customer_cogs[
        customer_cogs['customer_id'] == target_customer_id
    ]

    if target_cogs_row.empty:
        target_cogs = 0
        logger.warning(f"No COGS data for target customer {target_customer_id}")
    else:
        target_cogs = target_cogs_row.iloc[0]['total_cogs']

    # Calculate COGS similarity score (0-100)
    # Using log scale to handle large ranges
    candidates['cogs_similarity'] = candidates['total_cogs'].apply(
        lambda x: _calculate_cogs_similarity(x, target_cogs)
    )

    # STEP 3: RFM BOOST (optional)
    if include_rfm:
        rfm_scores = load_rfm_scores()

        candidates = candidates.merge(
            rfm_scores[['customer_id', 'rfm_segment', 'recency_score', 'frequency_score', 'monetary_score']],
            on='customer_id',
            how='left'
        )

        # Calculate RFM boost (0-30 points)
        # Prefer Champions and Loyal customers
        candidates['rfm_boost'] = candidates.apply(
            lambda row: _calculate_rfm_boost(row),
            axis=1
        )
    else:
        candidates['rfm_boost'] = 0

    # STEP 4: PRICE GROUP BOOST (optional)
    if include_price_group and pd.notna(target.get('price_group')):
        candidates['pg_boost'] = candidates['price_group'].apply(
            lambda x: 10 if x == target['price_group'] else 0
        )
    else:
        candidates['pg_boost'] = 0

    # STEP 5: CALCULATE FINAL SIMILARITY SCORE
    # Base: COGS similarity (0-100)
    # + RFM boost (0-30)
    # + Price Group boost (0-10)
    # = Total: 0-140
    candidates['similarity_score'] = (
        candidates['cogs_similarity'] +
        candidates['rfm_boost'] +
        candidates['pg_boost']
    )

    # Sort by similarity score
    candidates = candidates.sort_values('similarity_score', ascending=False)

    # Take top N
    top_similar = candidates.head(top_n)

    # Format results
    results = []
    for _, row in top_similar.iterrows():
        result = {
            'customer_id': row['customer_id'],
            'customer_name': row.get('customer_name', ''),
            'sales_office': str(row['sales_office']) if pd.notna(row['sales_office']) else None,
            'pl_type': row['pl_type'],
            'price_group': row.get('price_group'),
            'city': row.get('city'),
            'state': row.get('state'),
            'total_cogs': float(row['total_cogs']) if pd.notna(row['total_cogs']) else 0,
            'similarity_score': float(row['similarity_score']),
            'cogs_similarity': float(row['cogs_similarity']),
            'rfm_boost': float(row.get('rfm_boost', 0)),
            'pg_boost': float(row.get('pg_boost', 0)),
            'rfm_segment': row.get('rfm_segment')
        }
        results.append(result)

    logger.info(f"Found {len(results)} similar customers")

    return results


def calculate_similarity_score(
    customer1_id: str,
    customer2_id: str
) -> float:
    """
    Calculate similarity score between two specific customers

    Args:
        customer1_id: First customer ID
        customer2_id: Second customer ID

    Returns:
        Similarity score (0-140)
    """
    logger.info(f"Calculating similarity between {customer1_id} and {customer2_id}")

    customer_master = load_customer_master()

    c1 = customer_master[customer_master['customer_id'] == customer1_id]
    c2 = customer_master[customer_master['customer_id'] == customer2_id]

    if c1.empty or c2.empty:
        logger.error("One or both customers not found")
        return 0.0

    c1 = c1.iloc[0]
    c2 = c2.iloc[0]

    # Check mandatory matches
    if c1['sales_office'] != c2['sales_office']:
        return 0.0  # Different SOff = not similar

    if c1['pl_type'] != c2['pl_type']:
        return 0.0  # Different PLType = not similar

    # Calculate COGS similarity
    customer_cogs = load_customer_cogs()

    c1_cogs_row = customer_cogs[customer_cogs['customer_id'] == customer1_id]
    c2_cogs_row = customer_cogs[customer_cogs['customer_id'] == customer2_id]

    c1_cogs = c1_cogs_row.iloc[0]['total_cogs'] if not c1_cogs_row.empty else 0
    c2_cogs = c2_cogs_row.iloc[0]['total_cogs'] if not c2_cogs_row.empty else 0

    cogs_sim = _calculate_cogs_similarity(c1_cogs, c2_cogs)

    # RFM boost for customer 2
    rfm_scores = load_rfm_scores()
    c2_rfm = rfm_scores[rfm_scores['customer_id'] == customer2_id]

    if not c2_rfm.empty:
        rfm_boost = _calculate_rfm_boost(c2_rfm.iloc[0])
    else:
        rfm_boost = 0

    # Price Group boost
    pg_boost = 10 if c1.get('price_group') == c2.get('price_group') and pd.notna(c1.get('price_group')) else 0

    # Total score
    total_score = cogs_sim + rfm_boost + pg_boost

    return float(total_score)


def _calculate_cogs_similarity(cogs1: float, cogs2: float) -> float:
    """
    Calculate COGS proximity score (0-100)

    Uses log scale to handle wide COGS ranges (e.g., $100 to $1M)

    Formula:
    - If both are 0: score = 0
    - Otherwise: score = 100 - (log distance / max_log_range) * 100

    Args:
        cogs1: First COGS value
        cogs2: Second COGS value

    Returns:
        Similarity score (0-100)
    """
    if cogs1 == 0 and cogs2 == 0:
        return 0.0

    if cogs1 == 0 or cogs2 == 0:
        # One has COGS, other doesn't - low similarity
        return 20.0

    # Use log scale
    log1 = np.log10(max(cogs1, 1))
    log2 = np.log10(max(cogs2, 1))

    # Distance on log scale
    log_distance = abs(log1 - log2)

    # Max log range: 10^1 ($10) to 10^7 ($10M) = 6 orders of magnitude
    max_log_range = 6.0

    # Normalize to 0-100
    normalized_distance = min(log_distance / max_log_range, 1.0)
    similarity = 100.0 * (1.0 - normalized_distance)

    return similarity


def _calculate_rfm_boost(customer_row: pd.Series) -> float:
    """
    Calculate RFM boost score (0-30 points)

    Segment weights:
    - Champions: +30
    - Loyal: +25
    - Promising: +15
    - At Risk: +10
    - Need Attention: +5

    Args:
        customer_row: Customer row with RFM data

    Returns:
        RFM boost score (0-30)
    """
    segment = customer_row.get('rfm_segment', '')

    boost_map = {
        'Champions': 30,
        'Loyal': 25,
        'Promising': 15,
        'At Risk': 10,
        'Need Attention': 5
    }

    return boost_map.get(segment, 0)
