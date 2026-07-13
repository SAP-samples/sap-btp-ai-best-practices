"""
RFM Analyzer: RFM segmentation and customer value analysis
"""

import pandas as pd
from typing import List, Dict, Optional
import logging

from .data_loader import (
    load_rfm_scores,
    load_customer_master,
    load_customer_cogs
)

logger = logging.getLogger(__name__)


def load_quarterly_rfm() -> pd.DataFrame:
    """Load quarterly_rfm.parquet with Q1-Q4 data"""
    from .data_loader import load_from_parquet
    return load_from_parquet('quarterly_rfm.parquet')


def load_customer_master_enriched() -> pd.DataFrame:
    """Load customer_master_enriched.parquet with customer groups"""
    from .data_loader import load_from_parquet
    return load_from_parquet('customer_master_enriched.parquet')


def get_rfm_segment(customer_id: str) -> Optional[Dict]:
    """
    Get RFM segment for customer

    Args:
        customer_id: Customer ID

    Returns:
        RFM segment dict or None if not found
    """
    logger.info(f"Getting RFM segment for {customer_id}")

    rfm_scores = load_rfm_scores()

    rfm = rfm_scores[rfm_scores['customer_id'] == customer_id]

    if rfm.empty:
        logger.warning(f"No RFM data for customer {customer_id}")
        return None

    rfm = rfm.iloc[0]

    result = {
        'customer_id': customer_id,
        'rfm_segment': rfm.get('rfm_segment'),
        'recency_score': int(rfm.get('recency_score', 0)),
        'frequency_score': int(rfm.get('frequency_score', 0)),
        'monetary_score': int(rfm.get('monetary_score', 0)),
        'recency_days': int(rfm.get('recency_days', 0)),
        'frequency_count': int(rfm.get('frequency_count', 0)),
        'monetary_total': float(rfm.get('monetary_total', 0)),
        'last_order_date': str(rfm.get('last_order_date')) if pd.notna(rfm.get('last_order_date')) else None
    }

    return result


def get_segment_customers(
    segment: str,
    limit: int = 100
) -> List[Dict]:
    """
    Get all customers in a specific RFM segment

    Args:
        segment: RFM segment name (Champions, Loyal, At Risk, Promising, Need Attention)
        limit: Maximum number of customers to return

    Returns:
        List of customer dicts with RFM data
    """
    logger.info(f"Getting customers in segment: {segment}")

    rfm_scores = load_rfm_scores()

    segment_customers = rfm_scores[
        rfm_scores['rfm_segment'] == segment
    ].sort_values('monetary_total', ascending=False).head(limit)

    # Enrich with customer names
    customer_master = load_customer_master()

    results = []
    for _, row in segment_customers.iterrows():
        customer_id = row['customer_id']

        # Get customer name
        customer_info = customer_master[
            customer_master['customer_id'] == customer_id
        ]

        if not customer_info.empty:
            customer_name = customer_info.iloc[0].get('customer_name')
        else:
            customer_name = None

        results.append({
            'customer_id': customer_id,
            'customer_name': customer_name,
            'rfm_segment': row.get('rfm_segment'),
            'recency_score': int(row.get('recency_score', 0)),
            'frequency_score': int(row.get('frequency_score', 0)),
            'monetary_score': int(row.get('monetary_score', 0)),
            'monetary_total': float(row.get('monetary_total', 0)),
            'frequency_count': int(row.get('frequency_count', 0)),
            'recency_days': int(row.get('recency_days', 0))
        })

    logger.info(f"Found {len(results)} customers in segment {segment}")

    return results


def calculate_customer_value(customer_id: str) -> Dict:
    """
    Calculate customer lifetime value and metrics

    Args:
        customer_id: Customer ID

    Returns:
        Dict with value metrics
    """
    logger.info(f"Calculating value for customer {customer_id}")

    # Get COGS data
    customer_cogs = load_customer_cogs()

    cogs = customer_cogs[customer_cogs['customer_id'] == customer_id]

    if cogs.empty:
        return {
            'customer_id': customer_id,
            'lifetime_value': 0,
            'order_count': 0,
            'average_order_value': 0
        }

    cogs = cogs.iloc[0]

    total_cogs = float(cogs.get('total_cogs', 0))
    order_count = int(cogs.get('order_count', 0))
    avg_order = total_cogs / order_count if order_count > 0 else 0

    # Get RFM for recency
    rfm_scores = load_rfm_scores()
    rfm = rfm_scores[rfm_scores['customer_id'] == customer_id]

    if not rfm.empty:
        rfm = rfm.iloc[0]
        recency_days = int(rfm.get('recency_days', 0))
        last_order = str(rfm.get('last_order_date')) if pd.notna(rfm.get('last_order_date')) else None
    else:
        recency_days = None
        last_order = None

    result = {
        'customer_id': customer_id,
        'lifetime_value': total_cogs,
        'order_count': order_count,
        'average_order_value': round(avg_order, 2),
        'recency_days': recency_days,
        'last_order_date': last_order,
        'total_quantity': float(cogs.get('total_quantity', 0))
    }

    return result


def get_rfm_distribution() -> Dict:
    """
    Get distribution of customers across RFM segments

    Returns:
        Dict with segment counts and percentages
    """
    logger.info("Getting RFM distribution")

    rfm_scores = load_rfm_scores()

    total_customers = len(rfm_scores)

    segment_counts = rfm_scores['rfm_segment'].value_counts().to_dict()

    distribution = {}
    for segment, count in segment_counts.items():
        distribution[segment] = {
            'count': int(count),
            'percentage': round(count / total_customers * 100, 1)
        }

    distribution['total_customers'] = total_customers

    return distribution


# ============================================================================
# NEW: Quarterly RFM Functions
# ============================================================================

def get_quarterly_trend(customer_id: str) -> Dict:
    """
    Get quarterly COGS trend (Q1 → Q2 → Q3 → Q4)

    Returns:
        {
            'customer_id': str,
            'q1_cogs': float,
            'q2_cogs': float,
            'q3_cogs': float,
            'q4_cogs': float,
            'total_cogs': float,
            'growth_trend': 'up'/'down'/'stable',
            'active_quarters': int,
            'quarterly_average': float
        }
    """
    logger.info(f"Getting quarterly trend for customer {customer_id}")

    # Load quarterly RFM
    quarterly_rfm = load_quarterly_rfm()

    # Get customer data
    customer_data = quarterly_rfm[quarterly_rfm['customer_id'] == customer_id]

    if customer_data.empty:
        logger.warning(f"No quarterly data found for customer {customer_id}")
        return {
            'customer_id': customer_id,
            'q1_cogs': 0.0,
            'q2_cogs': 0.0,
            'q3_cogs': 0.0,
            'q4_cogs': 0.0,
            'total_cogs': 0.0,
            'growth_trend': 'stable',
            'active_quarters': 0,
            'quarterly_average': 0.0
        }

    row = customer_data.iloc[0]

    result = {
        'customer_id': customer_id,
        'q1_cogs': float(row['q1_cogs']),
        'q2_cogs': float(row['q2_cogs']),
        'q3_cogs': float(row['q3_cogs']),
        'q4_cogs': float(row['q4_cogs']),
        'total_cogs': float(row['total_cogs']),
        'growth_trend': row['growth_trend'],
        'active_quarters': int(row['active_quarters']),
        'quarterly_average': float(row['total_cogs']) / 4 if row['total_cogs'] > 0 else 0.0
    }

    return result


def get_rfm_by_customer_group(customer_group: str, limit: int = 100) -> Dict:
    """
    Get RFM distribution for specific customer group (AC, ME, CB, etc.)

    Args:
        customer_group: Customer group code (AC, ME, CB, CW, AB, etc. - 41 groups)
        limit: Max customers to return

    Returns:
        {
            'customer_group': str,
            'customer_count': int,
            'rfm_distribution': Dict with segment counts,
            'avg_total_cogs': float,
            'top_customers': List[Dict]
        }
    """
    logger.info(f"Getting RFM for customer group: {customer_group}")

    # Load data
    customer_master = load_customer_master_enriched()
    quarterly_rfm = load_quarterly_rfm()

    # Get customers in this group
    group_customers = customer_master[
        customer_master['customer_group'] == customer_group
    ]['customer_id'].tolist()

    if not group_customers:
        logger.warning(f"No customers found in group {customer_group}")
        return {
            'customer_group': customer_group,
            'customer_count': 0,
            'rfm_distribution': {},
            'avg_total_cogs': 0.0,
            'top_customers': []
        }

    # Get RFM data for these customers
    group_rfm = quarterly_rfm[quarterly_rfm['customer_id'].isin(group_customers)]

    # Calculate distribution
    rfm_distribution = {}
    if not group_rfm.empty:
        rfm_counts = group_rfm['rfm_segment'].value_counts().to_dict()
        total = len(group_rfm)
        rfm_distribution = {
            segment: {
                'count': int(count),
                'percentage': round(count / total * 100, 1)
            }
            for segment, count in rfm_counts.items()
        }

    # Calculate average COGS
    avg_cogs = float(group_rfm['total_cogs'].mean()) if not group_rfm.empty else 0.0

    # Get top customers
    top_customers = []
    if not group_rfm.empty:
        top_rfm = group_rfm.nlargest(limit, 'total_cogs')

        for _, row in top_rfm.iterrows():
            top_customers.append({
                'customer_id': row['customer_id'],
                'rfm_segment': row['rfm_segment'],
                'total_cogs': float(row['total_cogs']),
                'growth_trend': row['growth_trend'],
                'active_quarters': int(row['active_quarters'])
            })

    result = {
        'customer_group': customer_group,
        'customer_count': len(group_customers),
        'rfm_distribution': rfm_distribution,
        'avg_total_cogs': avg_cogs,
        'top_customers': top_customers
    }

    logger.info(f"Group {customer_group}: {len(group_customers)} customers, avg COGS ${avg_cogs:,.2f}")

    return result


def get_rfm_segment_with_quarterly(customer_id: str) -> Optional[Dict]:
    """
    Get RFM segment with quarterly breakdown (enhanced version)

    Returns standard RFM data + quarterly trends

    Returns:
        Combined dict with RFM segment + quarterly trend data
    """
    logger.info(f"Getting RFM with quarterly data for {customer_id}")

    # Get standard RFM
    rfm_data = get_rfm_segment(customer_id)

    if not rfm_data:
        return None

    # Get quarterly trend
    quarterly_data = get_quarterly_trend(customer_id)

    # Combine
    result = {
        **rfm_data,
        **quarterly_data
    }

    return result


def get_customer_groups_summary() -> List[Dict]:
    """
    Get summary of all customer groups with RFM stats

    Returns list of all 41 customer groups with:
    - customer_count
    - avg_cogs
    - rfm_distribution
    """
    logger.info("Getting customer groups summary")

    # Load data
    customer_master = load_customer_master_enriched()

    # Get unique customer groups
    unique_groups = customer_master['customer_group'].dropna().unique()

    logger.info(f"Found {len(unique_groups)} unique customer groups")

    # Get stats for each group
    groups_summary = []

    for group in unique_groups:
        group_stats = get_rfm_by_customer_group(group, limit=0)  # No top customers needed
        groups_summary.append({
            'customer_group': group,
            'customer_count': group_stats['customer_count'],
            'avg_total_cogs': group_stats['avg_total_cogs'],
            'rfm_distribution': group_stats['rfm_distribution']
        })

    # Sort by customer count
    groups_summary = sorted(groups_summary, key=lambda x: x['customer_count'], reverse=True)

    return groups_summary

