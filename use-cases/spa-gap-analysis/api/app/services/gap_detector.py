"""
Gap Detector: Identify missing SPAs for a customer

Compares target customer SPAs vs similar customers' SPAs
Returns SPAs that similar customers have but target doesn't
"""

import pandas as pd
from typing import List, Dict, Optional, Set
import logging

from .data_loader import (
    load_qualifications,
    load_header_data,
    load_customer_master
)
from .similarity_engine import find_similar_customers
from .confidence_scorer import calculate_simple_confidence

logger = logging.getLogger(__name__)


def detect_spa_gaps(
    target_customer_id: str,
    top_n_similar: int = 50,
    min_similar_count: int = 2
) -> Dict:
    """
    Detect SPA gaps for target customer

    Process:
    1. Find similar customers
    2. Get SPAs for target customer
    3. Get SPAs for similar customers
    4. Identify gaps (SPAs similar have but target doesn't)
    5. Rank gaps by frequency among similar customers

    Args:
        target_customer_id: Customer ID to analyze
        top_n_similar: Number of similar customers to compare against
        min_similar_count: Minimum number of similar customers that must have SPA for it to be recommended

    Returns:
        Dict with gap analysis results
    """
    logger.info(f"Detecting SPA gaps for {target_customer_id}")

    # STEP 1: Find similar customers
    similar_customers = find_similar_customers(
        target_customer_id,
        top_n=top_n_similar
    )

    if not similar_customers:
        logger.warning(f"No similar customers found for {target_customer_id}")
        return {
            'target_customer_id': target_customer_id,
            'target_spas': [],
            'similar_customers': [],
            'missing_spas': [],
            'message': 'No similar customers found'
        }

    similar_customer_ids = [c['customer_id'] for c in similar_customers]

    # STEP 2: Get target customer SPAs
    qualifications = load_qualifications()

    # Convert customer_id to sold_to format (with .0)
    target_sold_to = f"{target_customer_id}.0"

    target_spas = qualifications[
        qualifications['sold_to'] == target_sold_to
    ]['sales_deal'].unique().tolist()

    logger.info(f"Target customer has {len(target_spas)} SPAs")

    # STEP 3: Get SPAs for each similar customer
    similar_spas_map = {}

    for similar_id in similar_customer_ids:
        # Convert customer_id to sold_to format (with .0)
        similar_sold_to = f"{similar_id}.0"

        similar_cust_spas = qualifications[
            qualifications['sold_to'] == similar_sold_to
        ]['sales_deal'].unique().tolist()

        similar_spas_map[similar_id] = similar_cust_spas

    # STEP 4: Identify gap SPAs
    # SPAs that similar customers have but target doesn't
    target_spa_set = set(target_spas)
    gap_spa_counts = {}

    for similar_id, spa_list in similar_spas_map.items():
        for spa in spa_list:
            if spa not in target_spa_set:
                # This is a gap SPA
                gap_spa_counts[spa] = gap_spa_counts.get(spa, 0) + 1

    # Filter by min_similar_count
    gap_spas_filtered = {
        spa: count
        for spa, count in gap_spa_counts.items()
        if count >= min_similar_count
    }

    logger.info(f"Found {len(gap_spas_filtered)} gap SPAs (min count: {min_similar_count})")

    # STEP 5: Enrich gap SPAs with details
    header_data = load_header_data()

    missing_spas = []

    for spa, count in sorted(gap_spas_filtered.items(), key=lambda x: x[1], reverse=True):
        # Get SPA details from HEADER
        spa_info = header_data[header_data['sales_deal'] == spa]

        if not spa_info.empty:
            spa_info = spa_info.iloc[0]

            # Calculate confidence score (simplified for performance)
            confidence = calculate_simple_confidence(
                similar_count=count,
                total_similar=len(similar_customer_ids)
            )

            spa_detail = {
                'sales_deal': spa,
                'vendor': spa_info.get('vendor'),
                'description': spa_info.get('description'),
                'external_description': spa_info.get('external_description'),
                'grouping': spa_info.get('grouping'),
                'count_in_similar': int(count),
                'percentage_in_similar': round(count / len(similar_customer_ids) * 100, 1),
                'confidence_score': confidence
            }
        else:
            # SPA not in HEADER (shouldn't happen, but handle gracefully)
            confidence = calculate_simple_confidence(
                similar_count=count,
                total_similar=len(similar_customer_ids)
            )

            spa_detail = {
                'sales_deal': spa,
                'vendor': None,
                'description': f'SPA {spa}',
                'external_description': None,
                'grouping': None,
                'count_in_similar': int(count),
                'percentage_in_similar': round(count / len(similar_customer_ids) * 100, 1),
                'confidence_score': confidence
            }

        missing_spas.append(spa_detail)

    # STEP 6: Compile results
    result = {
        'target_customer_id': target_customer_id,
        'target_spas': target_spas,
        'target_spa_count': len(target_spas),
        'similar_customers_count': len(similar_customer_ids),
        'missing_spas': missing_spas,
        'missing_spa_count': len(missing_spas),
        'similar_customers': similar_customers
    }

    return result


def get_gap_recommendations(
    target_customer_id: str,
    top_n_similar: int = 10,
    min_similar_count: int = 2,
    top_n_recommendations: int = 5
) -> List[Dict]:
    """
    Get top N SPA recommendations for customer

    Simplified version of detect_spa_gaps that returns only top recommendations

    Args:
        target_customer_id: Customer ID to analyze
        top_n_similar: Number of similar customers to analyze
        min_similar_count: Minimum similar customers that must have SPA
        top_n_recommendations: Number of top SPAs to recommend

    Returns:
        List of top SPA recommendations
    """
    logger.info(f"Getting SPA recommendations for {target_customer_id}")

    gaps = detect_spa_gaps(
        target_customer_id,
        top_n_similar=top_n_similar,
        min_similar_count=min_similar_count
    )

    # Return top N missing SPAs
    recommendations = gaps['missing_spas'][:top_n_recommendations]

    return recommendations


def compare_customer_spas(
    customer1_id: str,
    customer2_id: str
) -> Dict:
    """
    Compare SPAs between two customers

    Shows:
    - SPAs both have (intersection)
    - SPAs only customer1 has
    - SPAs only customer2 has

    Args:
        customer1_id: First customer ID
        customer2_id: Second customer ID

    Returns:
        Dict with comparison results
    """
    logger.info(f"Comparing SPAs between {customer1_id} and {customer2_id}")

    qualifications = load_qualifications()

    # Get SPAs for each customer
    c1_spas = set(
        qualifications[qualifications['sold_to'] == customer1_id]['sales_deal'].unique()
    )

    c2_spas = set(
        qualifications[qualifications['sold_to'] == customer2_id]['sales_deal'].unique()
    )

    # Calculate intersections and differences
    both_have = c1_spas & c2_spas
    only_c1 = c1_spas - c2_spas
    only_c2 = c2_spas - c1_spas

    # Get customer names
    customer_master = load_customer_master()

    c1_name = customer_master[customer_master['customer_id'] == customer1_id]['customer_name'].iloc[0] if not customer_master[customer_master['customer_id'] == customer1_id].empty else customer1_id

    c2_name = customer_master[customer_master['customer_id'] == customer2_id]['customer_name'].iloc[0] if not customer_master[customer_master['customer_id'] == customer2_id].empty else customer2_id

    result = {
        'customer1': {
            'customer_id': customer1_id,
            'customer_name': c1_name,
            'spa_count': len(c1_spas),
            'spas': list(c1_spas)
        },
        'customer2': {
            'customer_id': customer2_id,
            'customer_name': c2_name,
            'spa_count': len(c2_spas),
            'spas': list(c2_spas)
        },
        'comparison': {
            'both_have_count': len(both_have),
            'both_have': list(both_have),
            'only_customer1_count': len(only_c1),
            'only_customer1': list(only_c1),
            'only_customer2_count': len(only_c2),
            'only_customer2': list(only_c2),
            'overlap_percentage': round(len(both_have) / len(c1_spas | c2_spas) * 100, 1) if (c1_spas | c2_spas) else 0
        }
    }

    return result


# ============================================================================
# NEW: Enhanced Gap Detection with A704 Material Group Fallback
# ============================================================================

def detect_spa_gaps_with_coverage(
    target_customer_id: str,
    top_n_similar: int = 10,
    min_similar_count: int = 2,
    include_potential_savings: bool = True
) -> Dict:
    """
    Enhanced gap detection using A703 + A704 material coverage

    NEW FEATURES:
    - Shows material coverage level for each gap (A703 specific or A704 group)
    - Calculates potential savings for each gap SPA
    - Includes agreement_grouping tier (D, A, B, C, E, K)

    Args:
        target_customer_id: Customer ID to analyze
        top_n_similar: Number of similar customers to compare against
        min_similar_count: Minimum similar customers that must have SPA
        include_potential_savings: Calculate potential savings for gaps

    Returns:
        Dict with enhanced gap analysis including:
        - coverage_level (material/group/none)
        - potential_savings
        - agreement_grouping
    """
    logger.info(f"Detecting SPA gaps with coverage for {target_customer_id}")

    # Run standard gap detection first
    gaps = detect_spa_gaps(
        target_customer_id,
        top_n_similar=top_n_similar,
        min_similar_count=min_similar_count
    )

    if not gaps['missing_spas']:
        return gaps

    # Load new data sources
    try:
        from .data_loader import load_from_parquet
        material_coverage = load_from_parquet('material_coverage_map.parquet')
        spa_header = load_from_parquet('spa_header.parquet')

        if include_potential_savings:
            material_savings = load_from_parquet('material_savings.parquet')

        # Enrich each gap SPA with coverage and savings data
        enriched_gaps = []

        for gap in gaps['missing_spas']:
            spa_id = gap['sales_deal']

            # Get agreement grouping from SPA header
            spa_info = spa_header[spa_header['agreement_id'] == spa_id]
            agreement_grouping = None
            if not spa_info.empty:
                agreement_grouping = spa_info.iloc[0].get('agreement_grouping')

            # Calculate material coverage for this SPA
            # Get materials in this SPA from A703/A704
            try:
                # Load A703 and A704 to see which materials are in this SPA
                a703 = load_from_parquet('a703.parquet')
                a704 = load_from_parquet('a704.parquet')

                # Materials in A703 for this SPA
                a703_materials = a703[a703['sales_deal'] == spa_id]['material'].unique()

                # Material groups in A704 for this SPA
                a704_groups = a704[a704['agreement_id'] == spa_id]['material_group'].unique()

                # Count coverage
                materials_count = len(a703_materials)
                groups_count = len(a704_groups)

                # Get total materials covered (via material_coverage)
                if materials_count > 0:
                    coverage_level = 'material'
                    coverage_count = materials_count
                elif groups_count > 0:
                    coverage_level = 'group'
                    # Count materials in these groups
                    coverage_count = len(
                        material_coverage[
                            material_coverage['material_group'].isin(a704_groups)
                        ]
                    )
                else:
                    coverage_level = 'none'
                    coverage_count = 0

            except Exception as e:
                logger.warning(f"Could not calculate coverage for SPA {spa_id}: {e}")
                coverage_level = 'unknown'
                coverage_count = 0
                materials_count = 0
                groups_count = 0

            # Calculate potential savings
            potential_savings = 0.0
            if include_potential_savings:
                try:
                    spa_savings = material_savings[material_savings['sales_deal'] == spa_id]
                    if not spa_savings.empty:
                        potential_savings = float(spa_savings['savings'].sum())
                except Exception as e:
                    logger.warning(f"Could not calculate savings for SPA {spa_id}: {e}")

            # Enrich gap record
            enriched_gap = {
                **gap,  # Original gap fields
                'agreement_grouping': agreement_grouping,
                'coverage_level': coverage_level,
                'materials_count': int(materials_count),
                'groups_count': int(groups_count),
                'total_coverage_count': int(coverage_count),
                'potential_savings': potential_savings
            }

            enriched_gaps.append(enriched_gap)

        # Replace missing_spas with enriched version
        gaps['missing_spas'] = enriched_gaps

        # Add summary stats
        gaps['total_potential_savings'] = sum(g['potential_savings'] for g in enriched_gaps)
        gaps['coverage_distribution'] = {
            level: sum(1 for g in enriched_gaps if g['coverage_level'] == level)
            for level in ['material', 'group', 'none', 'unknown']
        }

        logger.info(f"Enhanced {len(enriched_gaps)} gap SPAs with coverage and savings data")
        logger.info(f"Total potential savings: ${gaps['total_potential_savings']:,.2f}")

    except Exception as e:
        logger.error(f"Error enriching gaps with coverage: {e}", exc_info=True)
        # Return original gaps if enrichment fails
        pass

    return gaps


def get_gap_recommendations_with_savings(
    target_customer_id: str,
    top_n_similar: int = 10,
    min_similar_count: int = 2,
    top_n_recommendations: int = 5,
    sort_by: str = 'potential_savings'
) -> List[Dict]:
    """
    Get top N SPA recommendations sorted by potential savings

    Args:
        target_customer_id: Customer ID to analyze
        top_n_similar: Number of similar customers to analyze
        min_similar_count: Minimum similar customers that must have SPA
        top_n_recommendations: Number of top SPAs to recommend
        sort_by: 'potential_savings', 'count_in_similar', 'confidence_score'

    Returns:
        List of top SPA recommendations with savings potential
    """
    logger.info(f"Getting SPA recommendations with savings for {target_customer_id}")

    gaps = detect_spa_gaps_with_coverage(
        target_customer_id,
        top_n_similar=top_n_similar,
        min_similar_count=min_similar_count,
        include_potential_savings=True
    )

    # Sort by requested metric
    if sort_by == 'potential_savings':
        sorted_gaps = sorted(
            gaps['missing_spas'],
            key=lambda x: x.get('potential_savings', 0),
            reverse=True
        )
    elif sort_by == 'count_in_similar':
        sorted_gaps = sorted(
            gaps['missing_spas'],
            key=lambda x: x.get('count_in_similar', 0),
            reverse=True
        )
    elif sort_by == 'confidence_score':
        sorted_gaps = sorted(
            gaps['missing_spas'],
            key=lambda x: x.get('confidence_score', 0),
            reverse=True
        )
    else:
        sorted_gaps = gaps['missing_spas']

    # Return top N
    recommendations = sorted_gaps[:top_n_recommendations]

    return recommendations

