"""
Material Matcher: Check material coverage between customer purchases and SPAs
"""

import pandas as pd
from typing import List, Dict, Set
import logging

from .data_loader import (
    load_transactions,
    load_materials,
    load_sap_master
)

logger = logging.getLogger(__name__)


def check_material_coverage(
    customer_id: str,
    spa_id: str
) -> Dict:
    """
    Check what % of customer's purchased materials are covered by SPA

    Process:
    1. Get materials customer has purchased
    2. Get materials covered by SPA (from A901)
    3. Calculate coverage: intersection / customer materials

    Args:
        customer_id: Customer ID
        spa_id: SPA ID (Sales Deal)

    Returns:
        Dict with coverage analysis
    """
    logger.info(f"Checking material coverage for customer {customer_id}, SPA {spa_id}")

    transactions = load_transactions()
    materials_a901 = load_materials()

    # STEP 1: Get customer's purchased materials
    customer_txns = transactions[transactions['customer_id'] == customer_id]
    customer_materials = set(customer_txns['material'].unique())

    logger.info(f"Customer has purchased {len(customer_materials)} unique materials")

    # STEP 2: Get materials covered by SPA
    # Note: A901_Z005 doesn't have sales_deal column directly
    # We need to infer SPA from material presence in A901
    # For MVP Phase 1: assume all materials in A901 are potentially under some SPA
    # Phase 2: need proper material→SPA mapping

    spa_materials = set(materials_a901['material'].unique())

    logger.info(f"SPA-covered materials in catalog: {len(spa_materials)}")

    # STEP 3: Calculate coverage
    covered_materials = customer_materials & spa_materials
    uncovered_materials = customer_materials - spa_materials

    coverage_count = len(covered_materials)
    total_count = len(customer_materials)
    coverage_pct = (coverage_count / total_count * 100) if total_count > 0 else 0

    # Get COGS coverage
    covered_cogs = customer_txns[
        customer_txns['material'].isin(covered_materials)
    ]['cogs'].sum()

    total_cogs = customer_txns['cogs'].sum()
    cogs_coverage_pct = (covered_cogs / total_cogs * 100) if total_cogs > 0 else 0

    result = {
        'customer_id': customer_id,
        'spa_id': spa_id,
        'total_materials_purchased': total_count,
        'covered_materials_count': coverage_count,
        'uncovered_materials_count': len(uncovered_materials),
        'coverage_percentage': round(coverage_pct, 1),
        'total_cogs': float(total_cogs),
        'covered_cogs': float(covered_cogs),
        'cogs_coverage_percentage': round(cogs_coverage_pct, 1),
        'covered_materials': list(covered_materials)[:100],  # Limit to 100
        'uncovered_materials': list(uncovered_materials)[:100]  # Limit to 100
    }

    return result


def get_uncovered_materials(
    customer_id: str,
    spa_id: str,
    top_n: int = 20
) -> List[Dict]:
    """
    Get top uncovered materials by COGS

    Shows which materials customer buys but are NOT covered by SPA

    Args:
        customer_id: Customer ID
        spa_id: SPA ID
        top_n: Number of top materials to return

    Returns:
        List of uncovered material dicts with COGS
    """
    logger.info(f"Getting uncovered materials for customer {customer_id}, SPA {spa_id}")

    coverage = check_material_coverage(customer_id, spa_id)

    uncovered_materials = set(coverage['uncovered_materials'])

    if not uncovered_materials:
        return []

    # Get COGS for each uncovered material
    transactions = load_transactions()
    customer_txns = transactions[
        (transactions['customer_id'] == customer_id) &
        (transactions['material'].isin(uncovered_materials))
    ]

    # Aggregate by material
    material_cogs = customer_txns.groupby('material').agg({
        'cogs': 'sum',
        'quantity': 'sum'
    }).reset_index()

    # Sort by COGS
    material_cogs = material_cogs.sort_values('cogs', ascending=False)

    # Take top N
    top_uncovered = material_cogs.head(top_n)

    # Enrich with SAP Master data
    sap_master = load_sap_master()

    results = []
    for _, row in top_uncovered.iterrows():
        material_id = row['material']

        # Get material details from SAP Master
        material_info = sap_master[sap_master['material'] == material_id]

        if not material_info.empty:
            material_info = material_info.iloc[0]
            material_detail = {
                'material': material_id,
                'material_group': material_info.get('material_group'),
                'manufacturer_name': material_info.get('manufacturer_name'),
                'product_hierarchy': material_info.get('product_hierarchy')
            }
        else:
            material_detail = {
                'material': material_id,
                'material_group': None,
                'manufacturer_name': None,
                'product_hierarchy': None
            }

        material_detail.update({
            'total_cogs': float(row['cogs']),
            'total_quantity': float(row['quantity'])
        })

        results.append(material_detail)

    return results


def get_spa_material_list(spa_id: str) -> List[str]:
    """
    Get list of materials covered by SPA

    Args:
        spa_id: SPA ID

    Returns:
        List of material IDs
    """
    logger.info(f"Getting material list for SPA {spa_id}")

    materials_a901 = load_materials()

    # For MVP Phase 1: return all materials in A901
    # Phase 2: filter by actual SPA assignment
    spa_materials = materials_a901['material'].unique().tolist()

    logger.info(f"SPA {spa_id} covers {len(spa_materials)} materials")

    return spa_materials
