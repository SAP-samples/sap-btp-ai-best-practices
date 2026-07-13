"""
Confidence Scorer: Calculate recommendation confidence scores

Confidence score factors:
1. SPA Type (Blanket vs Targeted) - 0-30 points
2. Similar Customer Count - 0-25 points
3. Material Coverage % - 0-25 points
4. RFM Segment of similar customers - 0-20 points
"""

import pandas as pd
from typing import Dict, List
import logging

from .data_loader import (
    load_qualifications,
    load_header_data
)
from .material_matcher import check_material_coverage

logger = logging.getLogger(__name__)


def calculate_simple_confidence(
    similar_count: int,
    total_similar: int
) -> float:
    """
    Simplified confidence calculation based only on similar customer count

    Used for fast summary calculations where we don't have full customer/SPA details

    Args:
        similar_count: Number of similar customers with this SPA
        total_similar: Total number of similar customers analyzed

    Returns:
        Confidence score (0-100)
    """
    if total_similar == 0:
        return 0.0

    percentage = (similar_count / total_similar) * 100

    # Score based on percentage (simplified from full algorithm)
    if percentage >= 80:
        return 85.0
    elif percentage >= 60:
        return 70.0
    elif percentage >= 40:
        return 55.0
    elif percentage >= 20:
        return 35.0
    else:
        return 15.0


def calculate_confidence_score(
    target_customer_id: str,
    recommended_spa: str,
    similar_customers: List[Dict]
) -> Dict:
    """
    Calculate confidence score for SPA recommendation

    Confidence Factors:
    1. SPA Type: Blanket (30pts) vs Targeted (20pts)
    2. Similar Count: 100% have it (25pts) → 20% have it (5pts)
    3. Material Coverage: 80%+ (25pts) → 20%- (5pts)
    4. RFM Quality: Champions/Loyal (20pts) → Need Attention (5pts)

    Total: 0-100 points

    Args:
        target_customer_id: Customer ID receiving recommendation
        recommended_spa: SPA being recommended
        similar_customers: List of similar customer dicts

    Returns:
        Dict with confidence score and breakdown
    """
    logger.info(f"Calculating confidence for SPA {recommended_spa} → Customer {target_customer_id}")

    # Initialize scores
    spa_type_score = 0
    similar_count_score = 0
    material_coverage_score = 0
    rfm_quality_score = 0

    # FACTOR 1: SPA Type (Blanket vs Targeted)
    qualifications = load_qualifications()

    # Convert recommended_spa to int if it's a string
    spa_id_int = int(recommended_spa) if isinstance(recommended_spa, str) else recommended_spa

    spa_quals = qualifications[qualifications['sales_deal'] == spa_id_int]

    if not spa_quals.empty:
        # Check if any qualification has expansion_type = 'price_group_blanket'
        is_blanket = (spa_quals['expansion_type'] == 'price_group_blanket').any()

        if is_blanket:
            spa_type_score = 30  # Blanket SPAs are high confidence
            spa_type = 'Blanket'
        else:
            spa_type_score = 20  # Targeted SPAs are medium confidence
            spa_type = 'Targeted'
    else:
        spa_type_score = 15
        spa_type = 'Unknown'

    # FACTOR 2: Similar Customer Count
    # How many of similar customers have this SPA?
    similar_with_spa = 0
    for similar in similar_customers:
        similar_id = similar['customer_id']

        # Convert to sold_to format (with .0)
        similar_sold_to = f"{similar_id}.0"

        similar_spas = qualifications[
            qualifications['sold_to'] == similar_sold_to
        ]['sales_deal'].unique()

        if spa_id_int in similar_spas:
            similar_with_spa += 1

    total_similar = len(similar_customers)
    pct_similar_with_spa = (similar_with_spa / total_similar * 100) if total_similar > 0 else 0

    # Score based on percentage
    if pct_similar_with_spa >= 80:
        similar_count_score = 25
    elif pct_similar_with_spa >= 60:
        similar_count_score = 20
    elif pct_similar_with_spa >= 40:
        similar_count_score = 15
    elif pct_similar_with_spa >= 20:
        similar_count_score = 10
    else:
        similar_count_score = 5

    # FACTOR 3: Material Coverage
    # What % of customer's materials would be covered by this SPA?
    coverage = check_material_coverage(target_customer_id, recommended_spa)
    cogs_coverage = coverage['cogs_coverage_percentage']

    if cogs_coverage >= 80:
        material_coverage_score = 25
    elif cogs_coverage >= 60:
        material_coverage_score = 20
    elif cogs_coverage >= 40:
        material_coverage_score = 15
    elif cogs_coverage >= 20:
        material_coverage_score = 10
    else:
        material_coverage_score = 5

    # FACTOR 4: RFM Quality of Similar Customers
    # Are similar customers high-value (Champions/Loyal)?
    rfm_segments = [s.get('rfm_segment') for s in similar_customers if s.get('rfm_segment')]

    segment_weights = {
        'Champions': 20,
        'Loyal': 18,
        'Promising': 12,
        'At Risk': 8,
        'Need Attention': 5
    }

    if rfm_segments:
        # Average weight of similar customers
        avg_weight = sum(segment_weights.get(seg, 0) for seg in rfm_segments) / len(rfm_segments)
        rfm_quality_score = round(avg_weight)
    else:
        rfm_quality_score = 10  # Default if no RFM data

    # TOTAL CONFIDENCE SCORE
    total_score = (
        spa_type_score +
        similar_count_score +
        material_coverage_score +
        rfm_quality_score
    )

    # Confidence Level
    if total_score >= 80:
        confidence_level = 'High'
    elif total_score >= 60:
        confidence_level = 'Medium'
    else:
        confidence_level = 'Low'

    result = {
        'total_score': total_score,
        'confidence_level': confidence_level,
        'breakdown': {
            'spa_type': {
                'score': spa_type_score,
                'max_score': 30,
                'type': spa_type
            },
            'similar_customer_count': {
                'score': similar_count_score,
                'max_score': 25,
                'similar_with_spa': similar_with_spa,
                'total_similar': total_similar,
                'percentage': round(pct_similar_with_spa, 1)
            },
            'material_coverage': {
                'score': material_coverage_score,
                'max_score': 25,
                'cogs_coverage_pct': round(cogs_coverage, 1)
            },
            'rfm_quality': {
                'score': rfm_quality_score,
                'max_score': 20,
                'avg_segment_weight': round(rfm_quality_score)
            }
        }
    }

    return result


def get_recommendation_details(
    target_customer_id: str,
    recommended_spa: str,
    top_n_similar: int = 10
) -> Dict:
    """
    Get full recommendation details with confidence score

    Combines:
    - Gap detection
    - Confidence scoring
    - Material coverage
    - Similar customer info

    Args:
        target_customer_id: Customer ID
        recommended_spa: SPA to recommend
        top_n_similar: Number of similar customers to analyze

    Returns:
        Dict with full recommendation details
    """
    # Lazy import to avoid circular dependency
    from .gap_detector import detect_spa_gaps

    logger.info(f"Getting recommendation details for SPA {recommended_spa} → Customer {target_customer_id}")

    # Get gap analysis (includes similar customers)
    gaps = detect_spa_gaps(
        target_customer_id,
        top_n_similar=top_n_similar
    )

    similar_customers = gaps['similar_customers']

    # Calculate confidence score
    confidence = calculate_confidence_score(
        target_customer_id,
        recommended_spa,
        similar_customers
    )

    # Get material coverage
    coverage = check_material_coverage(
        target_customer_id,
        recommended_spa
    )

    # Get SPA details
    header_data = load_header_data()
    spa_info = header_data[header_data['sales_deal'] == recommended_spa]

    if not spa_info.empty:
        spa_info = spa_info.iloc[0]
        spa_detail = {
            'sales_deal': recommended_spa,
            'vendor': spa_info.get('vendor'),
            'description': spa_info.get('description'),
            'external_description': spa_info.get('external_description')
        }
    else:
        spa_detail = {
            'sales_deal': recommended_spa,
            'vendor': None,
            'description': None,
            'external_description': None
        }

    result = {
        'target_customer_id': target_customer_id,
        'recommended_spa': spa_detail,
        'confidence': confidence,
        'material_coverage': {
            'total_materials': coverage['total_materials_purchased'],
            'covered_count': coverage['covered_materials_count'],
            'coverage_percentage': coverage['coverage_percentage'],
            'cogs_coverage_percentage': coverage['cogs_coverage_percentage']
        },
        'similar_customers_count': len(similar_customers),
        'similar_customers_with_spa': confidence['breakdown']['similar_customer_count']['similar_with_spa']
    }

    return result
