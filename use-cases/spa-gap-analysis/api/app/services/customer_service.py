"""
Customer Service: Customer profiling and search operations
"""

import pandas as pd
from typing import List, Dict, Optional
import logging

from .data_loader import (
    load_customer_master,
    load_qualifications,
    load_customer_spa_assignments,
    load_customer_current_metrics,
    load_customer_material_current_pricing,
    load_rfm_scores,
    load_customer_cogs,
    load_transactions,
    load_from_parquet
)

logger = logging.getLogger(__name__)


def _format_date_value(value) -> Optional[str]:
    """Return ISO date string for timestamps, otherwise None."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    try:
        return pd.to_datetime(value).date().isoformat()
    except Exception:
        return str(value)


def _clean_optional(value):
    """Convert pandas missing values to None for API serialization."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _pricing_source_note(source_counts: Dict[str, int]) -> Optional[str]:
    """Explain current-state pricing source assumptions."""
    if not source_counts:
        return None

    sources = {str(key).upper(): int(value) for key, value in source_counts.items() if key}
    if "A704" in sources:
        return (
            "Current-state coverage may include A704 multiplier-based pricing. "
            "A703 is treated as exact/netted/rebated cost; A704 exact monetary opportunity is Phase 2."
        )
    if "A703" in sources:
        return "Current-state pricing uses A703 exact/netted/rebated cost."
    return None


def _get_customer_current_assignments(customer_id: str) -> List[Dict]:
    """
    Get current active SPA assignments for a customer from canonical parquet.

    Falls back to legacy qualifications if canonical data is not available.
    """
    try:
        assignments = load_customer_spa_assignments()
        current = assignments[
            (assignments['customer_id'] == customer_id) &
            (assignments['is_active'])
        ].copy()

        if current.empty:
            return []

        current = current.sort_values(['agreement_id', 'assignment_valid_from'])

        try:
            header = load_from_parquet('spa_header_enhanced.parquet')
            header = header[[
                'agreement_id',
                'description_of_agreement',
                'external_description',
                'agreement_description',
                'is_supplyforce',
            ]].drop_duplicates('agreement_id')
            current = current.merge(header, on='agreement_id', how='left')
        except Exception as header_exc:
            logger.warning(f"Could not enrich current assignments with SPA header metadata: {header_exc}")
            for col in [
                'description_of_agreement',
                'external_description',
                'agreement_description',
                'is_supplyforce',
            ]:
                current[col] = None

        return current[[
            'agreement_id',
            'assignment_valid_from',
            'assignment_valid_to',
            'agreement_type',
            'agreement_grouping',
            'assignment_scope',
            'description_of_agreement',
            'external_description',
            'agreement_description',
            'is_supplyforce',
            'is_active'
        ]].drop_duplicates('agreement_id').to_dict('records')
    except Exception as exc:
        logger.warning(f"Falling back to legacy qualifications for customer {customer_id}: {exc}")

    qualifications = load_qualifications()
    customer_quals = qualifications[
        (qualifications['sold_to'] == customer_id) |
        (qualifications['sold_to'] == f"{customer_id}.0")
    ].copy()

    if customer_quals.empty:
        return []

    legacy_rows = []
    for sales_deal in customer_quals['sales_deal'].dropna().astype(str).unique():
        spa_info = customer_quals[customer_quals['sales_deal'].astype(str) == sales_deal].iloc[0]
        legacy_rows.append({
            'agreement_id': sales_deal,
            'assignment_valid_from': spa_info.get('valid_from'),
            'assignment_valid_to': spa_info.get('valid_to'),
            'agreement_type': None,
            'agreement_grouping': None,
            'assignment_scope': spa_info.get('expansion_type', 'legacy'),
            'description_of_agreement': None,
            'external_description': None,
            'agreement_description': None,
            'is_supplyforce': None,
            'is_active': True
        })

    return legacy_rows


def _get_customer_spa_metadata(customer_id: str) -> Dict:
    """Get canonical SPA count semantics and snapshot metadata."""
    metadata = {
        'current_spa_count_unique': 0,
        'current_spa_row_count': 0,
        'current_spa_count_rule': (
            'spa_count reflects unique active agreements; '
            'row count reflects active A701 rows including plant-level duplicates'
        ),
        'snapshot_date': None
    }

    try:
        assignments = load_customer_spa_assignments()
        active_rows = assignments[
            (assignments['customer_id'] == customer_id) &
            (assignments['is_active'])
        ].copy()

        if active_rows.empty:
            return metadata

        metadata['current_spa_row_count'] = int(len(active_rows))
        metadata['current_spa_count_unique'] = int(active_rows['agreement_id'].astype(str).nunique())

        snapshot_series = active_rows.get('snapshot_date')
        if snapshot_series is not None and len(snapshot_series.dropna()) > 0:
            metadata['snapshot_date'] = _format_date_value(snapshot_series.dropna().iloc[0])

        return metadata
    except Exception as exc:
        logger.warning(f"Could not load canonical SPA metadata for {customer_id}: {exc}")

    assignments = _get_customer_current_assignments(customer_id)
    metadata['current_spa_count_unique'] = len(assignments)
    metadata['current_spa_row_count'] = len(assignments)
    return metadata


def _get_customer_current_pricing_summary(customer_id: str) -> Dict:
    """Get current-state coverage counts and per-SPA covered COGS/savings."""
    summary = {
        'total_material_count': 0,
        'covered_material_count': 0,
        'covered_cogs': 0.0,
        'uncovered_cogs': 0.0,
        'coverage_percent': 0.0,
        'pricing_source_counts': {},
        'pricing_source_note': None,
        'per_spa': {}
    }

    try:
        pricing = load_customer_material_current_pricing()
        customer_pricing = pricing[pricing['customer_id'] == customer_id].copy()
        if customer_pricing.empty:
            return summary

        customer_pricing['cogs_12m'] = pd.to_numeric(customer_pricing['cogs_12m'], errors='coerce').fillna(0.0)
        customer_pricing['current_material_savings_value'] = pd.to_numeric(
            customer_pricing['current_material_savings_value'], errors='coerce'
        ).fillna(0.0)
        covered = customer_pricing[customer_pricing['is_currently_covered'].fillna(False)].copy()
        covered['current_best_assigned_spa'] = covered['current_best_assigned_spa'].astype(str)

        total_material_count = int(customer_pricing['material'].nunique())
        covered_material_count = int(covered['material'].nunique())
        total_cogs = float(customer_pricing['cogs_12m'].sum())
        covered_cogs = float(covered['cogs_12m'].sum())

        summary.update({
            'total_material_count': total_material_count,
            'covered_material_count': covered_material_count,
            'covered_cogs': covered_cogs,
            'uncovered_cogs': max(total_cogs - covered_cogs, 0.0),
            'coverage_percent': (covered_cogs / total_cogs * 100.0) if total_cogs > 0 else 0.0,
        })

        if covered.empty:
            return summary

        if 'current_pricing_source' in covered.columns:
            source_counts = (
                covered['current_pricing_source']
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )
            summary['pricing_source_counts'] = {str(k): int(v) for k, v in source_counts.items()}
            summary['pricing_source_note'] = _pricing_source_note(summary['pricing_source_counts'])

        grouped = covered.groupby('current_best_assigned_spa', dropna=True).agg({
            'cogs_12m': 'sum',
            'current_material_savings_value': 'sum',
            'material': 'nunique',
        }).reset_index()

        source_counts_by_spa = {}
        if 'current_pricing_source' in covered.columns:
            source_counts_by_spa = {
                str(spa_id): group['current_pricing_source'].dropna().astype(str).value_counts().to_dict()
                for spa_id, group in covered.groupby('current_best_assigned_spa', dropna=True)
            }

        summary['per_spa'] = {
            str(row['current_best_assigned_spa']): {
                'covered_cogs': float(row['cogs_12m']),
                'current_savings': float(row['current_material_savings_value']),
                'covered_materials': int(row['material']),
                'pricing_source_counts': {
                    str(k): int(v)
                    for k, v in source_counts_by_spa.get(str(row['current_best_assigned_spa']), {}).items()
                },
                'pricing_sources': sorted(source_counts_by_spa.get(str(row['current_best_assigned_spa']), {}).keys()),
                'pricing_source_note': _pricing_source_note(
                    source_counts_by_spa.get(str(row['current_best_assigned_spa']), {})
                ),
            }
            for _, row in grouped.iterrows()
        }
        return summary
    except Exception as exc:
        logger.warning(f"Could not load customer current pricing summary for {customer_id}: {exc}")
        return summary


def _get_customer_total_cogs(customer_id: str) -> float:
    """Get canonical Q4 rolling-12M total COGS, with legacy fallback."""
    try:
        current_metrics = load_customer_current_metrics()
        metrics_row = current_metrics[current_metrics['customer_id'] == customer_id]
        if not metrics_row.empty:
            return float(metrics_row.iloc[0].get('total_cogs_q4', 0.0))
    except Exception as exc:
        logger.warning(f"Could not load customer_current_metrics for {customer_id}: {exc}")

    try:
        quarterly_sales = load_from_parquet('quarterly_sales_raw.parquet')
        customer_id_int = int(customer_id)
        customer_sales = quarterly_sales[
            (quarterly_sales['sold_to'] == customer_id_int) &
            (quarterly_sales['quarter'] == 'Q4')
        ]
        return float(customer_sales['cogs_12m'].sum()) if len(customer_sales) > 0 else 0.0
    except Exception as exc:
        logger.warning(f"Could not load quarterly_sales_raw for COGS: {exc}")

    try:
        quarterly_rfm = load_from_parquet('quarterly_rfm.parquet')
        rfm_full = quarterly_rfm[quarterly_rfm['customer_id'] == customer_id]
        return float(rfm_full.iloc[0].get('total_cogs', 0.0)) if not rfm_full.empty else 0.0
    except Exception:
        return 0.0


def get_customer_profile(customer_id: str) -> Optional[Dict]:
    """
    Get complete customer profile

    Includes:
    - Basic info (name, office, location)
    - SPAs assigned
    - RFM segment
    - Spending summary

    Args:
        customer_id: Customer ID to retrieve

    Returns:
        Customer profile dict or None if not found
    """
    logger.info(f"Getting profile for customer {customer_id}")

    customer_master = load_customer_master()

    customer = customer_master[customer_master['customer_id'] == customer_id]

    if customer.empty:
        logger.warning(f"Customer {customer_id} not found")
        return None

    customer = customer.iloc[0]

    # Get current SPAs from canonical assignment model
    customer_assignments = _get_customer_current_assignments(customer_id)
    customer_spas = [str(row['agreement_id']) for row in customer_assignments]
    spa_metadata = _get_customer_spa_metadata(customer_id)
    pricing_summary = _get_customer_current_pricing_summary(customer_id)

    customer_spa_details = []
    for assignment in customer_assignments:
        spa_id = str(assignment['agreement_id'])
        spa_perf = pricing_summary['per_spa'].get(spa_id, {})
        customer_spa_details.append({
            'sales_deal': spa_id,
            'valid_from': _format_date_value(assignment.get('assignment_valid_from')),
            'valid_to': _format_date_value(assignment.get('assignment_valid_to')),
            'agreement_type': _clean_optional(assignment.get('agreement_type')),
            'grouping': _clean_optional(assignment.get('agreement_grouping')),
            'assignment_scope': _clean_optional(assignment.get('assignment_scope')),
            'description_of_agreement': _clean_optional(assignment.get('description_of_agreement')),
            'external_description': _clean_optional(assignment.get('external_description')),
            'agreement_description': _clean_optional(assignment.get('agreement_description')),
            'is_supplyforce': (
                bool(assignment.get('is_supplyforce'))
                if _clean_optional(assignment.get('is_supplyforce')) is not None
                else None
            ),
            'pricing_sources': spa_perf.get('pricing_sources', []),
            'pricing_source_counts': spa_perf.get('pricing_source_counts', {}),
            'pricing_source_note': spa_perf.get('pricing_source_note'),
            'is_active': bool(assignment.get('is_active', True)),
            'covered_cogs': float(spa_perf.get('covered_cogs', 0.0)),
            'current_savings': float(spa_perf.get('current_savings', 0.0)),
            'covered_materials': int(spa_perf.get('covered_materials', 0)),
        })

    customer_spa_details = sorted(
        customer_spa_details,
        key=lambda row: (
            -float(row.get('covered_cogs', 0.0) or 0.0),
            -float(row.get('current_savings', 0.0) or 0.0),
            str(row.get('sales_deal', ''))
        )
    )

    # Get RFM data (includes segment info)
    rfm_scores = load_rfm_scores()
    rfm = rfm_scores[rfm_scores['customer_id'] == customer_id]

    total_cogs = _get_customer_total_cogs(customer_id)

    if not rfm.empty:
        rfm = rfm.iloc[0]
        rfm_data = {
            'segment': rfm.get('rfm_segment'),
            'recency_score': int(rfm.get('recency_score', 0)),
            'frequency_score': int(rfm.get('frequency_score', 0)),
            'monetary_score': int(rfm.get('monetary_score', 0)),
            'last_order_date': str(rfm.get('last_order_date')) if pd.notna(rfm.get('last_order_date')) else None
        }

        # COGS already calculated above from quarterly_sales_raw
        cogs_data = {
            'total_cogs': total_cogs,
            'total_quantity': 0.0,  # Not available in quarterly_sales_raw
            'order_count': 0,  # Not available in quarterly_sales_raw
            'first_order_date': None,
            'last_order_date': rfm_data.get('last_order_date')
        }
    else:
        rfm_data = None
        cogs_data = None

    # Compile profile
    profile = {
        'customer_id': customer_id,
        'customer_name': customer.get('customer_name'),
        'sales_office': str(customer.get('sales_office')) if pd.notna(customer.get('sales_office')) else None,
        'pl_type': customer.get('pl_type'),
        'price_group': customer.get('price_group'),
        'city': customer.get('city'),
        'state': customer.get('state'),
        'account_manager': customer.get('account_manager'),
        'spas': customer_spas,
        'spa_count': len(customer_spas),
        'current_spa_details': customer_spa_details,
        'current_spa_count_unique': spa_metadata['current_spa_count_unique'],
        'current_spa_row_count': spa_metadata['current_spa_row_count'],
        'current_spa_count_rule': spa_metadata['current_spa_count_rule'],
        'snapshot_date': spa_metadata['snapshot_date'],
        'current_pricing_summary': pricing_summary,
        'rfm': rfm_data,
        'spending': cogs_data
    }

    return profile


def search_customers(
    query: Optional[str] = None,
    sales_office: Optional[str] = None,
    pl_type: Optional[str] = None,
    price_group: Optional[str] = None,
    state: Optional[str] = None,
    rfm_segment: Optional[str] = None,
    limit: int = 50
) -> List[Dict]:
    """
    Search customers by various criteria

    Args:
        query: Search query (matches customer ID or name)
        sales_office: Filter by sales office
        pl_type: Filter by PL type
        price_group: Filter by price group
        state: Filter by state
        rfm_segment: Filter by RFM segment
        limit: Maximum number of results

    Returns:
        List of customer dicts
    """
    logger.info(f"Searching customers with filters: query={query}, soff={sales_office}, pltype={pl_type}")

    customer_master = load_customer_master()

    # Apply filters
    filtered = customer_master.copy()

    if query:
        # Search in customer_id or customer_name
        query_lower = query.lower()
        filtered = filtered[
            filtered['customer_id'].str.lower().str.contains(query_lower, na=False) |
            filtered['customer_name'].str.lower().str.contains(query_lower, na=False)
        ]

    if sales_office:
        filtered = filtered[filtered['sales_office'] == sales_office]

    if pl_type:
        filtered = filtered[filtered['pl_type'] == pl_type]

    if price_group:
        filtered = filtered[filtered['price_group'] == price_group]

    if state:
        filtered = filtered[filtered['state'] == state]

    # RFM filter requires joining
    if rfm_segment:
        rfm_scores = load_rfm_scores()
        rfm_filtered = rfm_scores[rfm_scores['rfm_segment'] == rfm_segment]
        filtered = filtered[filtered['customer_id'].isin(rfm_filtered['customer_id'])]

    # Limit results
    filtered = filtered.head(limit)

    # Convert to list of dicts
    results = []
    for _, row in filtered.iterrows():
        results.append({
            'customer_id': row['customer_id'],
            'customer_name': row.get('customer_name'),
            'sales_office': row['sales_office'],
            'pl_type': row['pl_type'],
            'price_group': row.get('price_group'),
            'city': row.get('city'),
            'state': row.get('state')
        })

    logger.info(f"Found {len(results)} customers")

    return results


def get_customer_spas(customer_id: str) -> List[Dict]:
    """
    Get all SPAs for a customer with details

    Args:
        customer_id: Customer ID

    Returns:
        List of SPA dicts with details
    """
    logger.info(f"Getting SPAs for customer {customer_id}")

    assignments = _get_customer_current_assignments(customer_id)

    spas = []
    for assignment in assignments:
        spas.append({
            'sales_deal': str(assignment['agreement_id']),
            'valid_from': _format_date_value(assignment.get('assignment_valid_from')),
            'valid_to': _format_date_value(assignment.get('assignment_valid_to')),
            'expansion_type': assignment.get('assignment_scope', 'sold_to'),
            'agreement_type': assignment.get('agreement_type'),
            'grouping': assignment.get('agreement_grouping'),
            'is_active': bool(assignment.get('is_active', True))
        })

    return spas


def get_customer_materials(
    customer_id: str,
    limit: int = 100
) -> List[Dict]:
    """
    Get materials purchased by customer

    Args:
        customer_id: Customer ID
        limit: Maximum number of materials to return

    Returns:
        List of material dicts with purchase summary
    """
    logger.info(f"Getting materials for customer {customer_id}")

    transactions = load_transactions()

    customer_txns = transactions[transactions['customer_id'] == customer_id]

    # Aggregate by material
    material_summary = customer_txns.groupby('material').agg({
        'quantity': 'sum',
        'cogs': 'sum',
        'sales_order_date': ['min', 'max', 'count']
    }).reset_index()

    # Flatten column names
    material_summary.columns = [
        'material',
        'total_quantity',
        'total_cogs',
        'first_purchase',
        'last_purchase',
        'purchase_count'
    ]

    # Sort by total COGS
    material_summary = material_summary.sort_values('total_cogs', ascending=False)

    # Limit
    material_summary = material_summary.head(limit)

    # Convert to list
    results = []
    for _, row in material_summary.iterrows():
        results.append({
            'material': row['material'],
            'total_quantity': float(row['total_quantity']),
            'total_cogs': float(row['total_cogs']),
            'purchase_count': int(row['purchase_count']),
            'first_purchase': str(row['first_purchase']),
            'last_purchase': str(row['last_purchase'])
        })

    return results
