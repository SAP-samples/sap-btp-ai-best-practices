"""
Material Hierarchy Aggregation Service

Aggregates customer transaction data by product hierarchy levels
and calculates SPA coverage percentages.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .data_loader import (
    load_transactions,
    load_sap_master,
    load_product_hierarchy,
    load_a703_nets,
    load_customer_master
)

logger = logging.getLogger(__name__)


def aggregate_material_hierarchy(
    customer_id: str,
    level: int = 1,
    parent_category: Optional[str] = None
) -> Dict:
    """
    Aggregate material purchases by hierarchy level with SPA coverage.

    Args:
        customer_id: Customer ID to analyze
        level: Target hierarchy level (1-4)
        parent_category: Optional parent category code to filter by

    Returns:
        Dict with customer info, totals, and category breakdown
    """
    logger.info(f"Aggregating material hierarchy for customer {customer_id}, level {level}")

    # STEP 1: Load customer transactions
    transactions = load_transactions()
    customer_tx = transactions[transactions['customer_id'] == customer_id].copy()

    if len(customer_tx) == 0:
        logger.warning(f"No transactions found for customer {customer_id}")
        return _empty_response(customer_id)

    # Get customer name
    customer_master = load_customer_master()
    customer_row = customer_master[customer_master['customer_id'] == customer_id]
    customer_name = customer_row['customer_name'].iloc[0] if len(customer_row) > 0 else None

    # STEP 2: Join to SAP Master + Product Hierarchy
    sap_master = load_sap_master()
    hierarchy = load_product_hierarchy()

    # Convert material column to int64 for join
    customer_tx['material_int'] = pd.to_numeric(customer_tx['material'], errors='coerce')
    customer_tx = customer_tx.dropna(subset=['material_int'])
    customer_tx['material_int'] = customer_tx['material_int'].astype('int64')

    # Join transactions → sap_master
    customer_tx = customer_tx.merge(
        sap_master[['material', 'product_hierarchy']],
        left_on='material_int',
        right_on='material',
        how='left',
        suffixes=('', '_sap')
    )

    # Join to product hierarchy to get level descriptions
    customer_tx = customer_tx.merge(
        hierarchy[[
            'product_hierarchy',
            'level_1', 'level_2', 'level_3', 'level_4',
            'level_1_description', 'level_2_description',
            'level_3_description', 'level_4_description',
            'leaf_node'
        ]],
        on='product_hierarchy',
        how='left'
    )

    # Calculate total COGS and transactions
    total_cogs = customer_tx['cogs'].sum()
    total_transactions = len(customer_tx)

    # STEP 3: Aggregate by target level
    level_column = f'level_{level}'
    level_desc_column = f'level_{level}_description'

    # Filter by parent category if specified
    if parent_category:
        parent_level = _get_level_from_code(parent_category)
        if parent_level < level:
            parent_column = f'level_{parent_level}'
            customer_tx = customer_tx[customer_tx[parent_column] == parent_category]

    # Group by level
    grouped = customer_tx.groupby(level_column, dropna=False).agg({
        'cogs': 'sum',
        'material_int': ['count', 'nunique'],
        level_desc_column: 'first'
    }).reset_index()

    grouped.columns = [
        'category_code',
        'total_cogs',
        'transaction_count',
        'unique_materials',
        'category_name'
    ]

    # Remove NaN categories (transactions without hierarchy)
    grouped = grouped[grouped['category_code'].notna()]

    # Calculate percentages
    grouped['percentage_of_total'] = (grouped['total_cogs'] / total_cogs * 100) if total_cogs > 0 else 0

    # STEP 4: Calculate SPA coverage
    spa_materials = load_a703_nets()
    spa_material_ids = set(spa_materials['material'].unique())

    categories = []
    for _, row in grouped.iterrows():
        category_code = str(row['category_code'])

        # Get materials in this category
        category_mask = customer_tx[level_column] == row['category_code']
        category_materials = customer_tx[category_mask]['material_int'].unique()

        # Check which materials are covered by SPAs
        covered_materials = set(category_materials) & spa_material_ids

        # Calculate COGS covered by SPAs
        covered_mask = category_mask & customer_tx['material_int'].isin(covered_materials)
        spa_coverage_cogs = customer_tx[covered_mask]['cogs'].sum()
        spa_coverage_percentage = (spa_coverage_cogs / row['total_cogs'] * 100) if row['total_cogs'] > 0 else 0.0

        # Get SPA IDs covering these materials
        spas_covering = spa_materials[spa_materials['material'].isin(covered_materials)]['agreement'].unique()
        spas_covering_list = [str(int(spa)) for spa in spas_covering]

        # STEP 5: Determine has_children flag
        has_children = _can_drilldown(level, category_code, customer_tx, hierarchy)

        categories.append({
            'category_code': category_code,
            'category_name': str(row['category_name']) if pd.notna(row['category_name']) else f"Category {category_code}",
            'level': level,
            'total_cogs': float(row['total_cogs']),
            'percentage_of_total': float(row['percentage_of_total']),
            'transaction_count': int(row['transaction_count']),
            'unique_materials': int(row['unique_materials']),
            'spa_coverage_cogs': float(spa_coverage_cogs),
            'spa_coverage_percentage': float(spa_coverage_percentage),
            'spas_covering': spas_covering_list,
            'has_children': has_children
        })

    # Sort by COGS descending
    categories = sorted(categories, key=lambda x: x['total_cogs'], reverse=True)

    # Calculate overall SPA coverage
    overall_spa_coverage_cogs = sum(cat['spa_coverage_cogs'] for cat in categories)
    overall_spa_coverage_percentage = (overall_spa_coverage_cogs / total_cogs * 100) if total_cogs > 0 else 0.0

    return {
        'customer_id': customer_id,
        'customer_name': str(customer_name) if customer_name and pd.notna(customer_name) else None,
        'total_cogs': float(total_cogs),
        'total_transactions': int(total_transactions),
        'overall_spa_coverage_percentage': float(overall_spa_coverage_percentage),
        'categories': categories
    }


def _empty_response(customer_id: str) -> Dict:
    """Return empty response for customer with no transactions"""
    return {
        'customer_id': customer_id,
        'customer_name': None,
        'total_cogs': 0.0,
        'total_transactions': 0,
        'overall_spa_coverage_percentage': 0.0,
        'categories': []
    }


def _get_level_from_code(category_code: str) -> int:
    """Determine hierarchy level from category code length"""
    code_len = len(str(category_code))
    if code_len <= 2:
        return 1
    elif code_len <= 4:
        return 2
    elif code_len <= 6:
        return 3
    else:
        return 4


def _can_drilldown(current_level: int, category_code: str, transactions_df: pd.DataFrame, hierarchy_df: pd.DataFrame) -> bool:
    """
    Check if a category can be drilled down further.

    A category can drill down if:
    - Current level < 4 (max level)
    - There exist child nodes in hierarchy
    """
    if current_level >= 4:
        return False

    # Check if any transactions in this category have deeper hierarchy levels
    level_column = f'level_{current_level}'
    category_mask = transactions_df[level_column] == category_code

    # Check next level
    next_level = current_level + 1
    next_level_column = f'level_{next_level}'

    # If any transaction has a non-null next level, can drill down
    has_children = transactions_df[category_mask][next_level_column].notna().any()

    return bool(has_children)
