"""
Analyze infeasibility issues in procurement optimization.
This script checks for capacity constraints and demand-supply mismatches.
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
import argparse

# Import settings with fallback
try:
    from config import settings
    DEMAND_PERIOD_DAYS = getattr(settings, 'DEMAND_PERIOD_DAYS', 365)
except ImportError:
    DEMAND_PERIOD_DAYS = 365

# Configuration from optimize_procurement.py
CAPACITY_PEAK_PERIOD = 'M'  # Monthly peak
CAPACITY_BUFFER_PERCENT = 0.10  # 10% buffer

# Exchange rates for USD conversion
EXCHANGE_RATES = {
    'EUR': 1.14, 'MXN': 0.051, 'USD': 1.00
}

def load_data(file_path, usecols=None, dtype=None):
    """Loads a CSV file with basic error handling."""
    try:
        print(f"Loading {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path, low_memory=False, usecols=usecols, dtype=dtype, encoding='utf-8-sig')
        print(f"Loaded {os.path.basename(file_path)} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def convert_to_usd(row, value_col='NETWR', currency_col='WAERS'):
    """Convert value to USD using exchange rates."""
    value = row[value_col]
    currency = row[currency_col]
    rate = EXCHANGE_RATES.get(currency, 1.0)
    if pd.isna(value) or pd.isna(currency):
        return 0.0
    return value * rate

def analyze_infeasibility(ranking_file_path, tables_dir, table_map_path, column_map_path, mode='maktx'):
    """Main analysis function."""
    
    # Load mappings
    try:
        with open(table_map_path, 'r') as f:
            table_map = json.load(f)
        with open(column_map_path, 'r') as f:
            column_map = json.load(f)
    except Exception as e:
        print(f"Error loading mapping files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get mapped column names
    col_po_num = column_map.get('PO_Number') or 'EBELN'
    col_mat_num = column_map.get('Material_ID') or 'MATNR'
    col_mat_group = column_map.get('Material_Group_Code') or 'MATKL'
    col_item_qty = column_map.get('Order_Quantity') or 'MENGE'
    col_order_unit = column_map.get('Order_Unit') or 'MEINS'
    col_net_value = column_map.get('Item_Net_Order_Value') or 'NETWR'
    col_supplier_id = column_map.get('Supplier_ID') or 'LIFNR'
    col_po_date = column_map.get('PO_Creation_Date') or 'BEDAT'
    col_currency = column_map.get('PO_Currency_Code') or 'WAERS'
    col_mat_desc = column_map.get('Material_Description') or 'MAKTX'
    
    # Determine grouping key based on mode
    if mode == 'matkl':
        grouping_key_mapped = col_mat_group
    elif mode == 'matnr':
        grouping_key_mapped = col_mat_num
    elif mode == 'maktx':
        grouping_key_mapped = col_mat_desc
    
    print(f"\n=== OPTIMIZATION INFEASIBILITY ANALYSIS ===")
    print(f"Mode: {mode}")
    print(f"Grouping by: {grouping_key_mapped}")
    print(f"Demand period: {DEMAND_PERIOD_DAYS} days")
    print(f"Capacity calculation: Peak {CAPACITY_PEAK_PERIOD} with {CAPACITY_BUFFER_PERCENT*100}% buffer\n")
    
    # 1. Load vendor ranking data
    print("1. Loading vendor ranking data...")
    df_rank = load_data(ranking_file_path)
    df_rank['LIFNR'] = df_rank['LIFNR'].astype(str)
    df_rank[grouping_key_mapped] = df_rank[grouping_key_mapped].astype(str)
    
    # Get unique vendors and items
    unique_vendors = df_rank['LIFNR'].nunique()
    unique_items = df_rank[grouping_key_mapped].nunique()
    print(f"   - Unique vendors: {unique_vendors}")
    print(f"   - Unique {grouping_key_mapped} items: {unique_items}")
    print(f"   - Total vendor-item combinations: {len(df_rank)}")
    
    # 2. Load historical data
    print("\n2. Loading historical data...")
    po_items_filename = table_map.get('SAP_VLY_IL_PO_ITEMS.csv') or 'SAP_VLY_IL_PO_ITEMS.csv'
    po_header_filename = table_map.get('SAP_VLY_IL_PO_HEADER.csv') or 'SAP_VLY_IL_PO_HEADER.csv'
    material_filename = table_map.get('SAP_VLY_IL_MATERIAL.csv') or 'SAP_VLY_IL_MATERIAL.csv'
    
    # PO_ITEMS doesn't have MAKTX, so we need to load it separately
    hist_items_cols = [col_po_num, col_mat_group, col_mat_num, col_net_value, col_item_qty, col_order_unit]
    hist_header_cols = [col_po_num, col_supplier_id, col_po_date, col_currency]
    material_cols = [col_mat_num, col_mat_desc]
    
    df_items_hist = load_data(os.path.join(tables_dir, po_items_filename), usecols=hist_items_cols)
    df_header_hist = load_data(os.path.join(tables_dir, po_header_filename), usecols=hist_header_cols)
    df_material = load_data(os.path.join(tables_dir, material_filename), usecols=material_cols)
    
    # Convert data types
    for col in [col_po_num, col_supplier_id, col_mat_group, col_mat_num]:
        if col in df_items_hist.columns:
            df_items_hist[col] = df_items_hist[col].astype(str)
        if col in df_header_hist.columns:
            df_header_hist[col] = df_header_hist[col].astype(str)
    
    # Convert material data types
    if col_mat_num in df_material.columns:
        df_material[col_mat_num] = df_material[col_mat_num].astype(str)
    if col_mat_desc in df_material.columns:
        df_material[col_mat_desc] = df_material[col_mat_desc].astype(str)
    
    df_items_hist[col_net_value] = pd.to_numeric(df_items_hist[col_net_value], errors='coerce')
    df_items_hist[col_item_qty] = pd.to_numeric(df_items_hist[col_item_qty], errors='coerce')
    df_header_hist[col_po_date] = pd.to_datetime(df_header_hist[col_po_date], errors='coerce')
    
    # First merge items with material to get MAKTX
    df_items_with_desc = pd.merge(
        df_items_hist,
        df_material.drop_duplicates(subset=[col_mat_num]),
        on=col_mat_num,
        how='left'
    )
    
    # Fill missing MAKTX
    if col_mat_desc in df_items_with_desc.columns:
        df_items_with_desc[col_mat_desc] = df_items_with_desc[col_mat_desc].fillna('Unknown Material')
    
    # Then merge with header
    df_hist = pd.merge(
        df_items_with_desc.dropna(subset=[col_item_qty]),
        df_header_hist.dropna(subset=[col_supplier_id, col_po_date]),
        on=col_po_num, how='inner'
    )
    
    # Only filter by grouping_key if we have valid data
    if grouping_key_mapped in df_hist.columns:
        df_hist = df_hist.dropna(subset=[grouping_key_mapped])
    
    # 3. Calculate demand
    print("\n3. Calculating demand...")
    demand_end_date = df_hist[col_po_date].max()
    demand_start_date = demand_end_date - timedelta(days=DEMAND_PERIOD_DAYS)
    df_demand_period = df_hist[
        (df_hist[col_po_date] >= demand_start_date) & 
        (df_hist[col_po_date] <= demand_end_date)
    ].copy()
    
    # MEINS consistency check
    print("   - Checking MEINS consistency...")
    meins_consistency = df_demand_period.groupby(grouping_key_mapped)[col_order_unit].nunique()
    inconsistent_groups = meins_consistency[meins_consistency > 1].index
    if not inconsistent_groups.empty:
        print(f"   - WARNING: {len(inconsistent_groups)} items have inconsistent MEINS")
        df_demand_consistent = df_demand_period[~df_demand_period[grouping_key_mapped].isin(inconsistent_groups)].copy()
    else:
        df_demand_consistent = df_demand_period.copy()
    
    # Calculate demand quantities
    demand_by_item = df_demand_consistent.groupby(grouping_key_mapped)[col_item_qty].sum()
    demand_by_item = demand_by_item[demand_by_item > 0]
    
    print(f"   - Items with demand: {len(demand_by_item)}")
    print(f"   - Total demand quantity: {demand_by_item.sum():,.0f}")
    
    # 4. Calculate vendor capacities
    print("\n4. Calculating vendor capacities...")
    df_hist['NETWR_USD'] = df_hist.apply(convert_to_usd, axis=1, value_col=col_net_value, currency_col=col_currency)
    df_hist['CapacityPeriod'] = df_hist[col_po_date].dt.to_period(CAPACITY_PEAK_PERIOD)
    
    periodic_volume = df_hist.groupby([col_supplier_id, 'CapacityPeriod'])['NETWR_USD'].sum()
    peak_volume = periodic_volume.groupby(col_supplier_id).max()
    
    periods_in_demand = DEMAND_PERIOD_DAYS / (365.25 / 12) if CAPACITY_PEAK_PERIOD == 'M' else DEMAND_PERIOD_DAYS / (365.25 / 4)
    vendor_capacity_usd = peak_volume * periods_in_demand * (1 + CAPACITY_BUFFER_PERCENT)
    
    print(f"   - Vendors with capacity data: {len(vendor_capacity_usd)}")
    print(f"   - Total capacity (USD): ${vendor_capacity_usd.sum():,.2f}")
    
    # 5. Analyze potential infeasibility issues
    print("\n5. INFEASIBILITY ANALYSIS RESULTS:")
    
    # 5.1 Items without any vendors in ranking
    items_in_demand = set(demand_by_item.index)
    items_in_ranking = set(df_rank[grouping_key_mapped].unique())
    missing_items = items_in_demand - items_in_ranking
    
    if missing_items:
        print(f"\n   5.1. CRITICAL: {len(missing_items)} items have demand but NO vendors in ranking:")
        missing_demand = demand_by_item[demand_by_item.index.isin(missing_items)]
        for item in list(missing_items)[:10]:  # Show first 10
            print(f"       - {item}: demand = {demand_by_item.get(item, 0):,.2f}")
        if len(missing_items) > 10:
            print(f"       ... and {len(missing_items) - 10} more")
        print(f"       Total missing demand: {missing_demand.sum():,.2f}")
    else:
        print("\n   5.1. OK: All items with demand have at least one vendor")
    
    # 5.2 Vendors with zero or very low capacity
    print("\n   5.2. Vendor capacity analysis:")
    vendors_in_ranking = df_rank['LIFNR'].unique()
    zero_capacity_vendors = []
    low_capacity_vendors = []
    
    for vendor in vendors_in_ranking:
        capacity = vendor_capacity_usd.get(vendor, 0)
        if capacity == 0:
            zero_capacity_vendors.append(vendor)
        elif capacity < 1000:  # Threshold for "low" capacity
            low_capacity_vendors.append((vendor, capacity))
    
    if zero_capacity_vendors:
        print(f"       - CRITICAL: {len(zero_capacity_vendors)} vendors have ZERO capacity")
        # Show which items these vendors are assigned to
        affected_items = df_rank[df_rank['LIFNR'].isin(zero_capacity_vendors)][grouping_key_mapped].unique()
        print(f"       - These vendors appear in ranking for {len(affected_items)} items")
    
    if low_capacity_vendors:
        print(f"       - WARNING: {len(low_capacity_vendors)} vendors have capacity < $1,000")
    
    # 5.3 Check if total capacity can meet demand per item
    print("\n   5.3. Per-item capacity vs demand analysis:")
    
    infeasible_items = []
    constrained_items = []
    
    for item in items_in_ranking:
        if item not in demand_by_item:
            continue
            
        item_demand_qty = demand_by_item.get(item, 0)
        
        # Get vendors for this item from ranking
        item_vendors = df_rank[df_rank[grouping_key_mapped] == item]
        
        # Calculate total available capacity for this item
        total_capacity_usd = 0
        total_capacity_qty = 0
        
        for _, vendor_row in item_vendors.iterrows():
            vendor_id = vendor_row['LIFNR']
            vendor_capacity = vendor_capacity_usd.get(vendor_id, 0)
            avg_price = vendor_row.get('AvgUnitPriceUSD_raw', 0)
            
            if avg_price > 0:
                vendor_qty_capacity = vendor_capacity / avg_price
                total_capacity_qty += vendor_qty_capacity
                total_capacity_usd += vendor_capacity
        
        # Check if capacity meets demand
        if total_capacity_qty < item_demand_qty:
            capacity_ratio = total_capacity_qty / item_demand_qty if item_demand_qty > 0 else 0
            if capacity_ratio < 0.5:  # Less than 50% of demand can be met
                infeasible_items.append({
                    'item': item,
                    'demand': item_demand_qty,
                    'capacity': total_capacity_qty,
                    'ratio': capacity_ratio,
                    'vendors': len(item_vendors)
                })
            elif capacity_ratio < 1.0:  # Between 50% and 100%
                constrained_items.append({
                    'item': item,
                    'demand': item_demand_qty,
                    'capacity': total_capacity_qty,
                    'ratio': capacity_ratio,
                    'vendors': len(item_vendors)
                })
    
    if infeasible_items:
        print(f"\n       - CRITICAL: {len(infeasible_items)} items have capacity < 50% of demand:")
        infeasible_items_sorted = sorted(infeasible_items, key=lambda x: x['ratio'])
        for item_info in infeasible_items_sorted[:10]:
            print(f"         {item_info['item'][:50]}: demand={item_info['demand']:,.0f}, " +
                  f"capacity={item_info['capacity']:,.0f} ({item_info['ratio']:.1%}), " +
                  f"vendors={item_info['vendors']}")
        if len(infeasible_items) > 10:
            print(f"         ... and {len(infeasible_items) - 10} more")
    
    if constrained_items:
        print(f"\n       - WARNING: {len(constrained_items)} items have capacity 50-100% of demand")
    
    # 5.4 Summary statistics
    print("\n   5.4. SUMMARY:")
    print(f"       - Total items with demand: {len(demand_by_item)}")
    print(f"       - Items missing from ranking: {len(missing_items)}")
    print(f"       - Items with severe capacity constraints (<50%): {len(infeasible_items)}")
    print(f"       - Items with moderate constraints (50-100%): {len(constrained_items)}")
    print(f"       - Vendors with zero capacity: {len(zero_capacity_vendors)}")
    
    # 6. Generate detailed report
    print("\n6. Generating detailed analysis report...")
    
    report_data = {
        'summary': {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode,
            'grouping_key': grouping_key_mapped,
            'demand_period_days': DEMAND_PERIOD_DAYS,
            'total_items_with_demand': len(demand_by_item),
            'total_demand_quantity': float(demand_by_item.sum()),
            'items_missing_vendors': len(missing_items),
            'items_severely_constrained': len(infeasible_items),
            'items_moderately_constrained': len(constrained_items),
            'vendors_zero_capacity': len(zero_capacity_vendors),
            'vendors_low_capacity': len(low_capacity_vendors)
        },
        'missing_items': list(missing_items)[:100],  # First 100
        'infeasible_items': infeasible_items[:100],  # First 100
        'constrained_items': constrained_items[:100],  # First 100
        'zero_capacity_vendors': zero_capacity_vendors[:100],  # First 100
    }
    
    # Save JSON report
    report_filename = f'infeasibility_analysis_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"   - Saved JSON report: {report_filename}")
    
    # Save detailed CSV reports
    if infeasible_items or constrained_items:
        capacity_issues_df = pd.DataFrame(infeasible_items + constrained_items)
        capacity_issues_df['severity'] = capacity_issues_df['ratio'].apply(
            lambda x: 'CRITICAL' if x < 0.5 else 'WARNING'
        )
        capacity_filename = f'capacity_constraints_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        capacity_issues_df.to_csv(capacity_filename, index=False)
        print(f"   - Saved capacity constraints CSV: {capacity_filename}")
    
    # Save vendor capacity analysis
    vendor_analysis = []
    for vendor in vendors_in_ranking:
        vendor_name = df_rank[df_rank['LIFNR'] == vendor]['NAME1'].iloc[0] if len(df_rank[df_rank['LIFNR'] == vendor]) > 0 else 'Unknown'
        capacity = vendor_capacity_usd.get(vendor, 0)
        items_count = df_rank[df_rank['LIFNR'] == vendor][grouping_key_mapped].nunique()
        vendor_analysis.append({
            'LIFNR': vendor,
            'NAME1': vendor_name,
            'capacity_usd': capacity,
            'items_in_ranking': items_count,
            'status': 'ZERO_CAPACITY' if capacity == 0 else ('LOW_CAPACITY' if capacity < 1000 else 'OK')
        })
    
    vendor_df = pd.DataFrame(vendor_analysis)
    vendor_filename = f'vendor_capacity_analysis_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    vendor_df.to_csv(vendor_filename, index=False)
    print(f"   - Saved vendor capacity analysis CSV: {vendor_filename}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    
    return report_data

def main():
    parser = argparse.ArgumentParser(description="Analyze infeasibility issues in procurement optimization")
    parser.add_argument('--ranking-results-path', required=True, 
                        help='Full path to the ranking results CSV file')
    parser.add_argument('--tables-dir', required=True, 
                        help='Directory containing the mapped table CSV files')
    parser.add_argument('--table-map', required=True, 
                        help='Path to the JSON file mapping table concepts to filenames')
    parser.add_argument('--column-map', required=True, 
                        help='Path to the JSON file mapping column concepts to actual column names')
    parser.add_argument('--mode', required=True, choices=['matkl', 'matnr', 'maktx'], 
                        help="Grouping mode: 'matkl', 'matnr', or 'maktx'")
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_infeasibility(
        ranking_file_path=args.ranking_results_path,
        tables_dir=args.tables_dir,
        table_map_path=args.table_map,
        column_map_path=args.column_map,
        mode=args.mode
    )

if __name__ == "__main__":
    main()