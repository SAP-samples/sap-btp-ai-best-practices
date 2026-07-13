"""
Margin Calculator Service

Calculate savings/margin by comparing SPA prices vs BASE_COST
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

from .data_loader import load_from_parquet

logger = logging.getLogger(__name__)


def load_material_savings() -> pd.DataFrame:
    """Load material_savings.parquet"""
    return load_from_parquet('material_savings.parquet')


def load_customer_savings() -> pd.DataFrame:
    """Load customer_savings.parquet"""
    return load_from_parquet('customer_savings.parquet')


def calculate_customer_savings(customer_id: str) -> Dict:
    """
    Calculate total savings for customer across all materials

    Returns:
        {
            'customer_id': str,
            'total_savings': float,
            'total_base_cost': float,
            'total_spa_cost': float,
            'savings_percent': float,
            'material_count': int,
            'top_savings_materials': List[Dict]
        }
    """
    logger.info(f"Calculating savings for customer {customer_id}")

    # Load customer savings summary
    customer_savings = load_customer_savings()

    # Get customer record
    customer_data = customer_savings[customer_savings['customer_id'] == customer_id]

    if customer_data.empty:
        logger.warning(f"No savings data found for customer {customer_id}")
        return {
            'customer_id': customer_id,
            'total_savings': 0.0,
            'total_base_cost': 0.0,
            'total_spa_cost': 0.0,
            'savings_percent': 0.0,
            'material_count': 0,
            'top_savings_materials': []
        }

    customer_row = customer_data.iloc[0]

    # Load material-level savings to get top materials
    material_savings = load_material_savings()

    # Get all material savings for this customer (via quarterly sales and A701)
    # We need to join material_savings with customer's materials
    # For now, get top materials by savings overall (we can refine this)
    top_materials = material_savings.nlargest(5, 'savings')[[
        'material',
        'sales_deal',
        'spa_price',
        'base_cost',
        'savings',
        'savings_percent'
    ]].to_dict('records')

    result = {
        'customer_id': customer_id,
        'total_savings': float(customer_row['total_savings']),
        'total_base_cost': float(customer_row['total_base_cost']),
        'total_spa_cost': float(customer_row['total_spa_cost']),
        'savings_percent': float(customer_row['savings_percent']),
        'material_count': int(customer_row['material_count']),
        'top_savings_materials': top_materials
    }

    return result


def calculate_agreement_savings(agreement_id: str) -> Dict:
    """
    Calculate savings for specific SPA agreement

    Returns:
        {
            'agreement_id': str,
            'total_savings': float,
            'average_savings_percent': float,
            'material_count': int,
            'customer_count': int,
            'top_materials': List[Dict]
        }
    """
    logger.info(f"Calculating savings for agreement {agreement_id}")

    # Load material savings
    material_savings = load_material_savings()

    # Filter by agreement
    agreement_savings = material_savings[
        material_savings['sales_deal'] == agreement_id
    ]

    if agreement_savings.empty:
        logger.warning(f"No savings data found for agreement {agreement_id}")
        return {
            'agreement_id': agreement_id,
            'total_savings': 0.0,
            'average_savings_percent': 0.0,
            'material_count': 0,
            'customer_count': 0,
            'top_materials': []
        }

    # Calculate aggregate metrics
    total_savings = agreement_savings['savings'].sum()
    avg_savings_pct = agreement_savings['savings_percent'].mean()
    material_count = agreement_savings['material'].nunique()

    # Get customer count (requires A701 join - for now estimate)
    customer_count = 0  # TODO: Join with A701 to get actual customer count

    # Top materials by savings
    top_materials = agreement_savings.nlargest(10, 'savings')[[
        'material',
        'spa_price',
        'base_cost',
        'savings',
        'savings_percent'
    ]].to_dict('records')

    result = {
        'agreement_id': agreement_id,
        'total_savings': float(total_savings),
        'average_savings_percent': float(avg_savings_pct),
        'material_count': int(material_count),
        'customer_count': customer_count,
        'top_materials': top_materials
    }

    return result


def get_savings_leaderboard(
    top_n: int = 50,
    sort_by: str = 'total_savings'
) -> List[Dict]:
    """
    Get top customers by total savings

    Args:
        top_n: Number of top customers to return
        sort_by: 'total_savings' or 'savings_percent'

    Returns:
        List of customer dicts with savings data
    """
    logger.info(f"Getting savings leaderboard (top {top_n}, sort by {sort_by})")

    # Load customer savings
    customer_savings = load_customer_savings()

    # Sort and get top N
    leaderboard = customer_savings.nlargest(top_n, sort_by)

    # Convert to list of dicts
    results = []
    for _, row in leaderboard.iterrows():
        results.append({
            'customer_id': row['customer_id'],
            'total_savings': float(row['total_savings']),
            'total_base_cost': float(row['total_base_cost']),
            'total_spa_cost': float(row['total_spa_cost']),
            'savings_percent': float(row['savings_percent']),
            'material_count': int(row['material_count'])
        })

    logger.info(f"Found {len(results)} customers in leaderboard")

    return results


def get_material_savings_by_group(material_group: str) -> Dict:
    """
    Calculate total savings for a material group

    Args:
        material_group: Material group code

    Returns:
        Dict with aggregate savings for material group
    """
    logger.info(f"Calculating savings for material group {material_group}")

    # Load material savings
    material_savings = load_material_savings()

    # Load material coverage map to get material → material_group mapping
    try:
        coverage_map = load_from_parquet('material_coverage_map.parquet')

        # Filter materials in this group
        group_materials = coverage_map[
            coverage_map['material_group'] == material_group
        ]['material'].tolist()

        # Filter savings
        group_savings = material_savings[
            material_savings['material'].isin(group_materials)
        ]

        if group_savings.empty:
            logger.warning(f"No savings data found for material group {material_group}")
            return {
                'material_group': material_group,
                'total_savings': 0.0,
                'average_savings_percent': 0.0,
                'material_count': 0
            }

        # Aggregate
        total_savings = group_savings['savings'].sum()
        avg_savings_pct = group_savings['savings_percent'].mean()
        material_count = group_savings['material'].nunique()

        result = {
            'material_group': material_group,
            'total_savings': float(total_savings),
            'average_savings_percent': float(avg_savings_pct),
            'material_count': int(material_count)
        }

        return result

    except Exception as e:
        logger.error(f"Error calculating material group savings: {e}")
        return {
            'material_group': material_group,
            'total_savings': 0.0,
            'average_savings_percent': 0.0,
            'material_count': 0,
            'error': str(e)
        }


def get_savings_summary() -> Dict:
    """
    Get overall savings summary

    Returns:
        {
            'total_savings': float,
            'total_base_cost': float,
            'total_spa_cost': float,
            'average_savings_percent': float,
            'customer_count': int,
            'material_count': int
        }
    """
    logger.info("Getting overall savings summary")

    # Load customer savings
    customer_savings = load_customer_savings()

    # Load material savings
    material_savings = load_material_savings()

    summary = {
        'total_savings': float(customer_savings['total_savings'].sum()),
        'total_base_cost': float(customer_savings['total_base_cost'].sum()),
        'total_spa_cost': float(customer_savings['total_spa_cost'].sum()),
        'average_savings_percent': float(customer_savings['savings_percent'].mean()),
        'customer_count': len(customer_savings),
        'material_count': int(material_savings['material'].nunique())
    }

    logger.info(f"Savings summary: ${summary['total_savings']:,.2f} across {summary['customer_count']:,} customers")

    return summary


def get_savings_distribution() -> Dict:
    """
    Get distribution of savings across customer segments

    Returns:
        Dict with savings distribution by savings_percent ranges
    """
    logger.info("Getting savings distribution")

    # Load customer savings
    customer_savings = load_customer_savings()

    # Create savings percent bins
    bins = [0, 5, 10, 15, 20, 25, 100]
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25%+']

    customer_savings['savings_bin'] = pd.cut(
        customer_savings['savings_percent'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Count customers in each bin
    distribution = customer_savings['savings_bin'].value_counts().sort_index().to_dict()

    # Convert to percentages
    total_customers = len(customer_savings)
    distribution_pct = {
        str(k): {
            'count': int(v),
            'percentage': round(v / total_customers * 100, 1)
        }
        for k, v in distribution.items()
    }

    return distribution_pct
