import pandas as pd
import numpy as np
import pulp
import argparse
import os
import sys
import json # Added for loading mappings
from datetime import datetime, timedelta
# Removed duplicate sys and datetime imports

# --- Configuration ---
# Removed hardcoded paths

# Import demand period from settings with fallback
try:
    from config import settings
    DEMAND_PERIOD_DAYS = getattr(settings, 'DEMAND_PERIOD_DAYS', 365)
except ImportError:
    DEMAND_PERIOD_DAYS = 365  # Fallback if settings import fails
CAPACITY_PEAK_PERIOD = 'M' # Use 'M' for Month, 'Q' for Quarter to find peak rate
CAPACITY_BUFFER_PERCENT = 0.10 # Optional: Allow 10% over historical peak (0.0 for no buffer)

# --- Helper Functions ---

def load_data(file_path, usecols=None, dtype=None):
    """Loads a CSV file with basic error handling."""
    try:
        print(f"Loading {os.path.basename(file_path)}...")
        # Use utf-8-sig to handle potential BOM
        df = pd.read_csv(file_path, low_memory=False, usecols=usecols, dtype=dtype, encoding='utf-8-sig')
        print(f"Loaded {os.path.basename(file_path)} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

EXCHANGE_RATES = {
    'EUR': 1.14, 'MXN': 0.051, 'USD': 1.00
}
def convert_to_usd(row, value_col='NETWR', currency_col='WAERS'):
    """Applies exchange rate to convert value to USD."""
    value = row[value_col]
    currency = row[currency_col]
    rate = EXCHANGE_RATES.get(currency, 1.0)
    if pd.isna(value) or pd.isna(currency):
        return 0.0
    return value * rate


def sanitize_name(name):
    """Replaces characters not suitable for PuLP names with underscores."""
    import re
    name = str(name) # Ensure it's a string
    # Replace non-alphanumeric (excluding hyphen) with underscore
    name = re.sub(r'[^\w\-]+', '_', name)
    # Collapse multiple underscores to a single underscore
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')

    # If name is empty after sanitization, provide a default
    if not name:
        name = "s_default" # Short for sanitized_default

    # Add a prefix if the name starts with a digit (PuLP might not like it)
    if name and name[0].isdigit():
        name = 'N_' + name
    
    # Limit length to avoid excessively long names (PuLP has limits, often around 255)
    # Applying limit after potential prefixing. Max length for CBC names is typically 255.
    # Let's keep 100 for now as it was, but ensure it's applied last.
    return name[:100]

# --- Main Optimization Logic ---
def main():
    parser = argparse.ArgumentParser(description="Optimize procurement allocation to minimize total effective cost based on target quantities.") # Updated description
    # Updated arguments
    parser.add_argument('--ranking-results-path', required=True, help='Full path to the ranking results CSV file (output of evaluate_vendor_material.py).')
    parser.add_argument('--tables-dir', required=True, help='Directory containing the mapped table CSV files.')
    parser.add_argument('--optimization-output-path', required=True, help='Full path for the optimization results CSV file.')
    parser.add_argument('--table-map', required=True, help='Path to the JSON file mapping table concepts to filenames.')
    parser.add_argument('--column-map', required=True, help='Path to the JSON file mapping column concepts to actual column names.')
    parser.add_argument('--mode', required=True, choices=['matkl', 'matnr', 'maktx'], help="Grouping mode used for ranking and optimization: 'matkl', 'matnr', or 'maktx'.")

    args = parser.parse_args()
    mode = args.mode
    tables_dir = args.tables_dir
    optimization_output_path = args.optimization_output_path
    ranking_file_path = args.ranking_results_path # Full path now
    ranking_file_name = os.path.splitext(os.path.basename(ranking_file_path))[0]  # Extract filename without extension for logging
    table_map_path = args.table_map # Use argument path
    column_map_path = args.column_map # Use argument path

    # Load mappings
    try:
        with open(table_map_path, 'r') as f:
            table_map = json.load(f)
        with open(column_map_path, 'r') as f:
            column_map = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Mapping file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from mapping file: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(optimization_output_path), exist_ok=True)

    # --- Get mapped column names ---
    col_po_num = column_map.get('PO_Number') or 'EBELN'
    col_mat_num = column_map.get('Material_ID') or 'MATNR'
    col_mat_group = column_map.get('Material_Group_Code') or 'MATKL'
    col_item_qty = column_map.get('Order_Quantity') or 'MENGE' # Conceptual name for PO Item Quantity
    col_order_unit = column_map.get('Order_Unit') or 'MEINS' # For MEINS check
    col_net_value = column_map.get('Item_Net_Order_Value') or 'NETWR' # Conceptual name for PO Item Net Value
    col_supplier_id = column_map.get('Supplier_ID') or 'LIFNR'
    col_po_date = column_map.get('PO_Creation_Date') or 'BEDAT'
    col_currency = column_map.get('PO_Currency_Code') or 'WAERS'
    col_supplier_name = column_map.get('Supplier_Name_1') or 'NAME1'
    col_mat_group_desc = column_map.get('Material_Group_Description') or 'MATKL_DESC'
    col_mat_desc = column_map.get('Material_Description') or 'MAKTX'

    # Columns from ranking file (need cost and avg price)
    col_effective_cost = 'EffectiveCostPerUnit_USD' # From evaluate_vendor_material.py output
    col_avg_unit_price_raw = 'AvgUnitPriceUSD_raw' # From evaluate_vendor_material.py output
    # Keep score for potential tie-breaking or filtering if needed later
    col_final_score = 'FinalScore' # Still present in ranking file


    # Determine grouping key and description column based on mode and mappings
    if mode == 'matkl':
        grouping_key_mapped = col_mat_group
        desc_key_mapped = col_mat_group
        desc_col_mapped = col_mat_group_desc
        desc_concept = 'Material Group'
    elif mode == 'matnr':
        grouping_key_mapped = col_mat_num
        desc_key_mapped = col_mat_num
        desc_col_mapped = col_mat_desc
        desc_concept = 'Material'
    elif mode == 'maktx':
        grouping_key_mapped = col_mat_desc
        desc_key_mapped = col_mat_num # Key for merging descriptions is MATNR
        desc_col_mapped = col_mat_desc # Description column itself
        desc_concept = 'Material'
    else:
        sys.exit(1) # Should not happen

    print(f"Running optimization in mode: '{mode}', grouping by mapped key: '{grouping_key_mapped}'")

    # --- 1. Load Ranking Data ---
    # Load necessary columns: keys, effective cost, avg raw price, and MATNR for linking
    rank_cols_to_load = [
        col_supplier_id, grouping_key_mapped, col_mat_num, # Ensure MATNR is always loaded
        col_effective_cost, col_avg_unit_price_raw,
        col_final_score # Keep score for reference/potential use
    ]
    # Add description columns if they are different from the grouping key / MATNR
    # Note: Supplier Name is often needed for final output
    if col_supplier_name != col_supplier_id: rank_cols_to_load.append(col_supplier_name)
    # Add the specific description column if it's different from the grouping key AND different from MATNR
    if desc_col_mapped != grouping_key_mapped and desc_col_mapped != col_mat_num:
         rank_cols_to_load.append(desc_col_mapped)

    # Remove duplicates and any None/empty strings
    rank_cols_to_load = sorted(list(set(filter(None, rank_cols_to_load))))
    print(f"Loading columns from ranking file: {rank_cols_to_load}")

    try:
        # Check header first
        header_df_rank = pd.read_csv(ranking_file_path, nrows=0, low_memory=False, encoding='utf-8-sig')
        missing_rank_cols = [col for col in rank_cols_to_load if col not in header_df_rank.columns]
        if missing_rank_cols:
             print(f"Error: Required columns missing in ranking file '{ranking_file_path}': {missing_rank_cols}", file=sys.stderr)
             sys.exit(1)
        df_rank = load_data(ranking_file_path, usecols=rank_cols_to_load)
        # Convert cost and price to numeric, coercing errors
        df_rank[col_effective_cost] = pd.to_numeric(df_rank[col_effective_cost], errors='coerce')
        df_rank[col_avg_unit_price_raw] = pd.to_numeric(df_rank[col_avg_unit_price_raw], errors='coerce')
        # Drop rows where essential cost/price info is missing
        df_rank.dropna(subset=[col_effective_cost, col_avg_unit_price_raw], inplace=True)
        
        # Convert key columns to string to match historical data format
        key_cols_in_rank = [col_supplier_id, grouping_key_mapped, col_mat_num]
        for col in key_cols_in_rank:
            if col in df_rank.columns:
                if df_rank[col].dtype == 'float64':
                    try: 
                        df_rank[col] = df_rank[col].astype(pd.Int64Dtype()).astype(str)
                    except: 
                        df_rank[col] = df_rank[col].astype(str)
                else: 
                    df_rank[col] = df_rank[col].astype(str)

    except Exception as e:
        print(f"Error loading or checking ranking file {ranking_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional: Set non-positive effective cost to 0 or a small positive number to avoid issues in minimization if needed
    df_rank.loc[df_rank[col_effective_cost] <= 0, col_effective_cost] = 0.0001 # Example: Set to small positive value

    if df_rank.empty:
        print(f"Error: No valid vendor/{grouping_key_mapped} combinations with cost data found in the ranking file.", file=sys.stderr)
        sys.exit(1)


    # --- 2. Load Historical Data for Demand (Quantity) and Capacity (Value) ---
    # Load required files using table_map
    po_items_filename = table_map.get('SAP_VLY_IL_PO_ITEMS.csv') or 'SAP_VLY_IL_PO_ITEMS.csv'
    po_header_filename = table_map.get('SAP_VLY_IL_PO_HEADER.csv') or 'SAP_VLY_IL_PO_HEADER.csv'
    # Determine description file based on mode
    if desc_concept == 'Material Group':
        desc_filename = table_map.get('SAP_VLY_IL_MATERIAL_GROUP.csv') or 'SAP_VLY_IL_MATERIAL_GROUP.csv'
    else: # Material or Material Description mode
        desc_filename = table_map.get('SAP_VLY_IL_MATERIAL.csv') or 'SAP_VLY_IL_MATERIAL.csv'
    supplier_filename = table_map.get('SAP_VLY_IL_SUPPLIER.csv') or 'SAP_VLY_IL_SUPPLIER.csv'

    # Verify required files exist
    required_files = {
        "PO Items": po_items_filename,
        "PO Header": po_header_filename,
        "Description": desc_filename,
        "Supplier": supplier_filename
    }
    missing_files = [name for name, fname in required_files.items() if not os.path.isfile(os.path.join(tables_dir, fname))]
    if missing_files:
        print(f"Error: Required table files not found in tables directory '{tables_dir}': {missing_files}", file=sys.stderr)
        sys.exit(1)

    # Define required columns for historical data
    hist_items_cols_to_load = [col_po_num, col_mat_group, col_mat_num, col_net_value, col_item_qty, col_order_unit] # Added qty and unit
    hist_header_cols_to_load = [col_po_num, col_supplier_id, col_po_date, col_currency]
    desc_file_cols_to_load = [desc_key_mapped, desc_col_mapped] # e.g., ['MATNR', 'MAKTX'] or ['MATKL', 'MATKL_DESC']
    supplier_cols_to_load = [col_supplier_id, col_supplier_name]

    # Remove duplicates/None
    hist_items_cols_to_load = sorted(list(set(filter(None, hist_items_cols_to_load))))
    hist_header_cols_to_load = sorted(list(set(filter(None, hist_header_cols_to_load))))
    desc_file_cols_to_load = sorted(list(set(filter(None, desc_file_cols_to_load))))
    supplier_cols_to_load = sorted(list(set(filter(None, supplier_cols_to_load))))

    try:
        df_items_hist = load_data(os.path.join(tables_dir, po_items_filename), usecols=hist_items_cols_to_load)
        df_header_hist = load_data(os.path.join(tables_dir, po_header_filename), usecols=hist_header_cols_to_load)
        df_desc_hist = load_data(os.path.join(tables_dir, desc_filename), usecols=desc_file_cols_to_load).drop_duplicates(subset=[desc_key_mapped])
        df_supplier_desc = load_data(os.path.join(tables_dir, supplier_filename), usecols=supplier_cols_to_load).drop_duplicates(subset=[col_supplier_id])
    except Exception as e:
        print(f"Error loading historical data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Prepare Historical Data & Calculate Demand/Capacity ---
    print("Preparing historical data...")
    # Convert keys to string
    key_cols_to_convert = [col_po_num, col_supplier_id, col_mat_group, col_mat_num, desc_key_mapped, col_order_unit]
    dfs_to_clean = [df_items_hist, df_header_hist, df_desc_hist, df_supplier_desc]
    for df in dfs_to_clean:
        for col in key_cols_to_convert:
            if col in df.columns:
                # Handle potential float64 if IDs were numeric but read as float due to NaNs elsewhere
                if df[col].dtype == 'float64':
                    try: df[col] = df[col].astype(pd.Int64Dtype()).astype(str)
                    except: df[col] = df[col].astype(str)
                else: df[col] = df[col].astype(str)

    # Convert numeric and date columns
    df_items_hist[col_net_value] = pd.to_numeric(df_items_hist[col_net_value], errors='coerce')
    df_items_hist[col_item_qty] = pd.to_numeric(df_items_hist[col_item_qty], errors='coerce')
    df_header_hist[col_po_date] = pd.to_datetime(df_header_hist[col_po_date], errors='coerce')

    # Merge historical items and header
    df_hist_merged_base = pd.merge(
        df_items_hist.dropna(subset=[desc_key_mapped, col_item_qty, col_order_unit]), # Ensure keys and qty/unit are not NaN
        df_header_hist.dropna(subset=[col_supplier_id, col_po_date, col_currency]),
        on=col_po_num, how='inner'
    )

    # Merge description data (using desc_key_mapped, e.g., MATNR or MATKL)
    df_hist_merged = pd.merge(df_hist_merged_base, df_desc_hist, on=desc_key_mapped, how='left')
    # Fill description based on mode AFTER ensuring grouping_key_mapped exists
    # This handles cases where the merge didn't find a match or the description itself is NaN
    if desc_col_mapped in df_hist_merged.columns:
         df_hist_merged[desc_col_mapped] = df_hist_merged[desc_col_mapped].fillna(f'Unknown {mode.capitalize()}')
    elif grouping_key_mapped == desc_col_mapped: # e.g., maktx mode
         df_hist_merged[grouping_key_mapped] = df_hist_merged[grouping_key_mapped].fillna(f'Unknown {mode.capitalize()}')


    # Ensure the actual grouping_key_mapped column exists and drop NaNs
    if grouping_key_mapped not in df_hist_merged.columns:
         print(f"Error: Mapped grouping key '{grouping_key_mapped}' not found in historical merged data after merging descriptions.", file=sys.stderr)
         sys.exit(1)
    df_hist_merged.dropna(subset=[grouping_key_mapped], inplace=True)

    if df_hist_merged.empty: sys.exit("Error: No valid historical data found after merging and cleaning...")

    # Filter for Demand Period
    print(f"Filtering historical data for demand period: last {DEMAND_PERIOD_DAYS} days...")
    demand_end_date = df_hist_merged[col_po_date].max()
    if pd.isna(demand_end_date):
         print("Error: Cannot determine date range for demand/capacity calculation.", file=sys.stderr)
         sys.exit(1)
    demand_start_date = demand_end_date - timedelta(days=DEMAND_PERIOD_DAYS)
    df_hist_demand_period = df_hist_merged[
        (df_hist_merged[col_po_date] >= demand_start_date) & (df_hist_merged[col_po_date] <= demand_end_date)
    ].copy()

    if df_hist_demand_period.empty: sys.exit("Error: No historical data found within the specified demand period.")

    # MEINS Consistency Check for Both Demand and Supply Data
    print(f"Performing MEINS consistency check for both demand calculation and ranking data...")
    inconsistent_groups = []
    if col_order_unit in df_hist_demand_period.columns and grouping_key_mapped in df_hist_demand_period.columns:
        # Drop rows where grouping key or order unit is NaN before checking consistency
        df_check = df_hist_demand_period.dropna(subset=[grouping_key_mapped, col_order_unit])
        if not df_check.empty:
            meins_consistency = df_check.groupby(grouping_key_mapped)[col_order_unit].nunique()
            inconsistent_groups = meins_consistency[meins_consistency > 1].index.tolist()
            if inconsistent_groups:
                print(f"Warning: Found {len(inconsistent_groups)} groups with inconsistent MEINS values")
                print(f"Inconsistent groups: {inconsistent_groups}")
                print("These will be excluded from both demand calculation and ranking data for consistency")
                df_hist_demand_period_consistent = df_hist_demand_period[~df_hist_demand_period[grouping_key_mapped].isin(inconsistent_groups)].copy()
            else:
                df_hist_demand_period_consistent = df_hist_demand_period.copy()
                print("MEINS consistency check passed - no inconsistent groups found.")
        else:
             print("Warning: No valid data for MEINS consistency check after dropping NaNs.")
             df_hist_demand_period_consistent = df_hist_demand_period.copy() # Proceed cautiously
    else:
        print(f"Warning: Cannot perform MEINS check. Column '{col_order_unit}' or '{grouping_key_mapped}' missing.")
        df_hist_demand_period_consistent = df_hist_demand_period.copy() # Proceed without filtering if check fails

    # COMMENTED OUT: LeadTimeDays == 0 filtering to restore original behavior
    # Filter out transactions that would have LeadTimeDays == 0 for consistency with evaluation script
    # print("Applying LeadTimeDays consistency filter to historical demand data...")
    # 
    # # Load goods receipt data to calculate lead times for historical filtering
    # hist_gr_filename = table_map.get('SAP_VLY_IL_PO_HISTORY.csv') or 'SAP_VLY_IL_PO_HISTORY.csv'
    # col_hist_event_type = column_map.get('PO_History_Transaction_Type') or 'VGABE'
    # col_hist_post_date = column_map.get('History_Posting_Date') or 'BUDAT'
    # col_po_item = column_map.get('PO_Item_Number') or 'EBELP'
    # 
    # hist_gr_cols = [col_po_num, col_po_item, col_hist_event_type, col_hist_post_date]
    # 
    # try:
    #     df_hist_gr = load_data(os.path.join(tables_dir, hist_gr_filename), usecols=hist_gr_cols)
    #     # Convert keys to string and dates
    #     for col in [col_po_num, col_po_item]:
    #         if col in df_hist_gr.columns:
    #             if df_hist_gr[col].dtype == 'float64':
    #                 try: df_hist_gr[col] = df_hist_gr[col].astype(pd.Int64Dtype()).astype(str)
    #                 except: df_hist_gr[col] = df_hist_gr[col].astype(str)
    #             else: df_hist_gr[col] = df_hist_gr[col].astype(str)
    #     
    #     df_hist_gr[col_hist_post_date] = pd.to_datetime(df_hist_gr[col_hist_post_date], errors='coerce')
    #     
    #     # Filter for goods receipts and aggregate to get first GR date
    #     df_gr_filtered = df_hist_gr[df_hist_gr[col_hist_event_type].astype(str) == '1'].copy()
    #     if not df_gr_filtered.empty:
    #         df_gr_agg = df_gr_filtered.groupby([col_po_num, col_po_item])[col_hist_post_date].min().reset_index()
    #         df_gr_agg.rename(columns={col_hist_post_date: 'FirstGRDate'}, inplace=True)
    #         
    #         # Merge with historical demand data to calculate lead times
    #         df_hist_demand_period_consistent = pd.merge(
    #             df_hist_demand_period_consistent, 
    #             df_gr_agg, 
    #             left_on=[col_po_num], 
    #             right_on=[col_po_num], 
    #             how='left'
    #         )
    #         
    #         # Calculate lead time and filter out 0-day lead times
    #         df_hist_demand_period_consistent['LeadTimeDays'] = np.nan
    #         valid_lt_idx = (df_hist_demand_period_consistent['FirstGRDate'].notna() & 
    #                        df_hist_demand_period_consistent[col_po_date].notna())
    #         
    #         df_hist_demand_period_consistent.loc[valid_lt_idx, 'LeadTimeDays'] = (
    #             df_hist_demand_period_consistent.loc[valid_lt_idx, 'FirstGRDate'] - 
    #             df_hist_demand_period_consistent.loc[valid_lt_idx, col_po_date]
    #         ).dt.days
    #         
    #         # Filter out same-day deliveries for consistency
    #         before_leadtime_demand_filter = len(df_hist_demand_period_consistent)
    #         df_hist_demand_period_consistent = df_hist_demand_period_consistent[
    #             (df_hist_demand_period_consistent['LeadTimeDays'].isna()) | 
    #             (df_hist_demand_period_consistent['LeadTimeDays'] != 0)
    #         ]
    #         after_leadtime_demand_filter = len(df_hist_demand_period_consistent)
    #         print(f"Historical demand data after LeadTimeDays == 0 filter: {after_leadtime_demand_filter:,} rows (removed {before_leadtime_demand_filter - after_leadtime_demand_filter:,})")
    #     else:
    #         print("Warning: No goods receipt data found for lead time calculation. Proceeding without lead time filtering.")
    # except Exception as e:
    #     print(f"Warning: Could not load goods receipt data for lead time filtering: {e}. Proceeding without lead time filtering.")
    
    # Calculate Target Quantity Demand
    print(f"Calculating Target Quantity Demand by {grouping_key_mapped}...")
    TargetQuantity_per_grouping_key = df_hist_demand_period_consistent.groupby(grouping_key_mapped)[col_item_qty].sum()
    TargetQuantity_per_grouping_key = TargetQuantity_per_grouping_key[TargetQuantity_per_grouping_key > 0] # Keep only items with positive demand
    print(f"Calculated target quantity demand for {len(TargetQuantity_per_grouping_key)} {grouping_key_mapped} items.")
    if TargetQuantity_per_grouping_key.empty:
        sys.exit("Error: No target quantity demand found after filtering and MEINS check.")

    # Calculate Supplier Capacity (Value-based) - using the full historical merge before demand period filter
    print(f"Calculating Max Capacity Proxy (Value-based) based on peak '{CAPACITY_PEAK_PERIOD}' rate...")
    # Ensure NETWR_USD is calculated on the full df_hist_merged for capacity calculation
    if 'NETWR_USD' not in df_hist_merged.columns:
        df_hist_merged['NETWR_USD'] = df_hist_merged.apply(convert_to_usd, axis=1, value_col=col_net_value, currency_col=col_currency)
    # Ensure CapacityPeriod is calculated on the full df_hist_merged
    if 'CapacityPeriod' not in df_hist_merged.columns:
         if df_hist_merged[col_po_date].notna().any():
              df_hist_merged['CapacityPeriod'] = df_hist_merged[col_po_date].dt.to_period(CAPACITY_PEAK_PERIOD)
         else:
              sys.exit(f"Error: Cannot calculate capacity period due to missing or invalid '{col_po_date}' data.")

    periodic_volume_v = df_hist_merged.groupby([col_supplier_id, 'CapacityPeriod'])['NETWR_USD'].sum()
    peak_periodic_volume_v = periodic_volume_v.groupby(col_supplier_id).max()

    if CAPACITY_PEAK_PERIOD == 'M': periods_in_demand_range = DEMAND_PERIOD_DAYS / (365.25 / 12)
    elif CAPACITY_PEAK_PERIOD == 'Q': periods_in_demand_range = DEMAND_PERIOD_DAYS / (365.25 / 4)
    else: periods_in_demand_range = DEMAND_PERIOD_DAYS / (365.25 / 12) # Default monthly

    max_capacity_v_USD = peak_periodic_volume_v * periods_in_demand_range * (1 + CAPACITY_BUFFER_PERCENT)
    max_capacity_v_USD = max_capacity_v_USD.clip(lower=0)
    print(f"Calculated value-based capacity proxy for {len(max_capacity_v_USD)} vendors.")


    # --- 4. Prepare Data for PuLP ---
    print("Preparing data for PuLP...")
    
    # Apply the same MEINS consistency filter to ranking data that was applied to demand calculation
    # This ensures consistency between available supply (ranking) and demand constraints
    if inconsistent_groups:
        print(f"Applying MEINS consistency filter to ranking data: excluding {len(inconsistent_groups)} groups")
        ranking_before_meins = len(df_rank)
        # Filter ranking data to exclude MEINS-inconsistent groups
        df_rank = df_rank[~df_rank[grouping_key_mapped].isin(inconsistent_groups)].copy()
        print(f"Ranking data after MEINS filter: {len(df_rank):,} rows (removed {ranking_before_meins - len(df_rank):,} rows)")
    else:
        print("No MEINS filtering applied to ranking data - all groups are consistent")
    
    # Filter to only include materials with multiple suppliers (optimization focus)
    print("Filtering ranking data to only include materials with multiple suppliers...")
    suppliers_per_material_rank = df_rank.groupby(grouping_key_mapped)[col_supplier_id].nunique()
    multi_supplier_materials_rank = suppliers_per_material_rank[suppliers_per_material_rank > 1].index
    
    ranking_before_multi = len(df_rank)
    df_rank = df_rank[df_rank[grouping_key_mapped].isin(multi_supplier_materials_rank)].copy()
    print(f"Materials with multiple suppliers in ranking: {len(multi_supplier_materials_rank):,}")
    print(f"Ranking data after multi-supplier filter: {len(df_rank):,} rows (removed {ranking_before_multi - len(df_rank):,} rows)")
    
    # Filter ranking data to include only items with calculated target quantity demand
    df_rank_filtered = df_rank[df_rank[grouping_key_mapped].isin(TargetQuantity_per_grouping_key.index)].copy()

    if df_rank_filtered.empty:
         print("\n--- OPTIMIZATION FAILURE DIAGNOSTICS ---", file=sys.stderr)
         mats_in_rank_after_filters = df_rank[grouping_key_mapped].unique()
         mats_with_demand = TargetQuantity_per_grouping_key.index
         
         print(f"Error: No feasible (Vendor, {grouping_key_mapped}) pairs remain after filtering.", file=sys.stderr)
         print(f"This means the set of materials with recent demand and the set of materials available for optimization are disjoint.", file=sys.stderr)
         
         print(f"\n[1] Materials with calculated demand ({len(mats_with_demand)}):", file=sys.stderr)
         print(f"    {mats_with_demand.tolist()}", file=sys.stderr)
         
         print(f"\n[2] Materials in ranking file after MEINS and multi-supplier filters ({len(mats_in_rank_after_filters)}):", file=sys.stderr)
         print(f"    {mats_in_rank_after_filters.tolist()}", file=sys.stderr)
         
         disjoint_demand_mats = set(mats_with_demand) - set(mats_in_rank_after_filters)
         if disjoint_demand_mats:
             print(f"\n[3] Analysis: {len(disjoint_demand_mats)} materials with demand are NOT available in the filtered ranking data.", file=sys.stderr)
             print(f"    {sorted(list(disjoint_demand_mats))}", file=sys.stderr)

         print("\nRecommendation: Ensure the time period for demand calculation in `optimize_procurement.py` aligns with the data used to generate the ranking file.", file=sys.stderr)
         print("--- END DIAGNOSTICS ---\n", file=sys.stderr)
         sys.exit(1)

    # Ensure keys are string type for tuple creation
    df_rank_filtered[col_supplier_id] = df_rank_filtered[col_supplier_id].astype(str)
    df_rank_filtered[grouping_key_mapped] = df_rank_filtered[grouping_key_mapped].astype(str)
    
    # Aggregate ranking data at (LIFNR, grouping_key) level to avoid duplicates
    # Use weighted average based on some criteria (using AvgUnitPriceUSD_raw as weight)
    print(f"Aggregating ranking data from {len(df_rank_filtered)} rows to (LIFNR, {grouping_key_mapped}) level...")
    
    # Build aggregation dict dynamically to avoid conflicts when grouping_key_mapped == col_mat_num
    agg_dict = {
        col_effective_cost: 'mean',  # Average effective cost per unit
        col_avg_unit_price_raw: 'mean',  # Average unit price
        col_supplier_name: 'first'  # Take first supplier name (should be same)
    }
    
    # Only aggregate MATNR if it's not the same as grouping_key_mapped
    if col_mat_num != grouping_key_mapped:
        agg_dict[col_mat_num] = 'first'  # Take first MATNR (for reference only)
    
    df_rank_aggregated = df_rank_filtered.groupby([col_supplier_id, grouping_key_mapped]).agg(agg_dict).reset_index()

    print(f"Aggregated to {len(df_rank_aggregated)} unique (LIFNR, {grouping_key_mapped}) combinations.")

    # Create a mapping from simple integer indices to variable data
    var_data_map = {}
    for i, row_data in enumerate(df_rank_aggregated.to_dict('records')):
        var_data_map[i] = {
            'lifnr': row_data[col_supplier_id],
            'g_key': row_data[grouping_key_mapped],
            'matnr': row_data.get(col_mat_num, row_data[grouping_key_mapped] if col_mat_num == grouping_key_mapped else None), # Handle case where MATNR is the grouping key
            'eff_cost': row_data[col_effective_cost],
            'avg_price': row_data[col_avg_unit_price_raw],
            'supplier_name': row_data[col_supplier_name] # Keep supplier name if available
        }
    
    variable_indices = list(var_data_map.keys())
    print(f"Created {len(variable_indices)} unique variable definitions with simple indices.")

    # --- 5. Formulate Optimization Problem ---
    print("Formulating the optimization problem...")
    prob = pulp.LpProblem("Procurement_Cost_Minimization", pulp.LpMinimize) # Minimize cost

    # Decision Variables: Allocated Quantity (using simple integer indices)
    AllocatedQuantity_vars = pulp.LpVariable.dicts(
        "AllocatedQty", variable_indices, lowBound=0, cat='Continuous'
    )
    
    # Objective Function: Minimize Total Effective Cost
    prob += pulp.lpSum(var_data_map[idx]['eff_cost'] * AllocatedQuantity_vars[idx]
                       for idx in variable_indices), "Minimize_Total_Effective_Procurement_Cost"

    # --- 6. Define Constraints ---
    print("Adding constraints...")
    # Demand Fulfillment Constraint (Quantity)
    print("  - Adding demand fulfillment constraints...")
    grouping_keys_in_scope = TargetQuantity_per_grouping_key.index
    constraints_added = 0
    constraints_skipped = 0
    skipped_materials = []
    
    for i, g_key_val in enumerate(grouping_keys_in_scope): # Renamed g_key to g_key_val to avoid clash
        target_qty = TargetQuantity_per_grouping_key[g_key_val]
        
        relevant_vars_for_g_key_indices = [
            idx for idx in variable_indices if var_data_map[idx]['g_key'] == g_key_val
        ]
        
        if relevant_vars_for_g_key_indices: # Only add constraint if there are suppliers for this key
            safe_g_key_name = sanitize_name(g_key_val)
            constraint_name = f"MeetDemand_{mode}_{safe_g_key_name}_{i}" # Appended index for uniqueness
            # Hard constraint: Must meet exact demand (now feasible with multi-supplier filter)
            prob += pulp.lpSum(AllocatedQuantity_vars[idx] for idx in relevant_vars_for_g_key_indices) == target_qty, constraint_name
            constraints_added += 1
        else:
            constraints_skipped += 1
            skipped_materials.append(g_key_val)
            if constraints_skipped <= 5:  # Only show first 5 warnings
                print(f"Warning: No feasible suppliers found for grouping key '{g_key_val}' after filtering. Demand constraint skipped.")
    
    print(f"  - Added {constraints_added} demand constraints")
    
    # Calculate total demand (always needed for summary)
    total_demand = TargetQuantity_per_grouping_key.sum()
    
    if constraints_skipped > 0:
        skipped_demand = TargetQuantity_per_grouping_key[skipped_materials].sum()
        print(f"  - Skipped {constraints_skipped} materials with no feasible suppliers")
        print(f"  - Skipped demand: {skipped_demand:,.0f} ({skipped_demand/total_demand:.1%} of total demand)")
        if constraints_skipped > 5:
            print(f"  - (Showing only first 5 warnings, {constraints_skipped - 5} more materials skipped)")
    
    # Print optimization scope summary
    optimized_demand = TargetQuantity_per_grouping_key[TargetQuantity_per_grouping_key.index.isin([var_data_map[idx]['g_key'] for idx in variable_indices])].sum()
    print(f"  - Optimization scope: {constraints_added} materials with {optimized_demand:,.0f} total demand ({optimized_demand/total_demand:.1%} of total)")

    # Supplier Capacity Constraint (Value)
    print("  - Adding supplier capacity constraints...")
    all_lifnrs_in_scope = df_rank_aggregated[col_supplier_id].unique()
    for lifnr_val in all_lifnrs_in_scope: # Renamed lifnr to lifnr_val
        capacity_usd = max_capacity_v_USD.get(lifnr_val, 0)
        
        relevant_vars_for_lifnr_indices = [
            idx for idx in variable_indices if var_data_map[idx]['lifnr'] == lifnr_val
        ]
        
        if relevant_vars_for_lifnr_indices: # Only add constraint if supplier has feasible items
            # Attempt to get supplier name for clearer constraint names, fallback to LIFNR
            # This assumes var_data_map[idx]['supplier_name'] exists and is consistent for a given lifnr_val
            # Taking the first one should be fine.
            supplier_name_part = var_data_map[relevant_vars_for_lifnr_indices[0]].get('supplier_name', '')
            # Construct a name that includes the unique lifnr_val before sanitization
            # Ensure lifnr_val is part of the string to be sanitized for uniqueness
            raw_constraint_name_part = f"{supplier_name_part if pd.notna(supplier_name_part) else ''}_{lifnr_val}"
            safe_vendor_specific_name = sanitize_name(raw_constraint_name_part)
            constraint_name_cap = f"Capacity_Vendor_{safe_vendor_specific_name}"

            value_sum_expr = pulp.lpSum(
                AllocatedQuantity_vars[idx] * var_data_map[idx]['avg_price']
                for idx in relevant_vars_for_lifnr_indices
                if pd.notna(var_data_map[idx]['avg_price']) # Ensure avg_price is not NaN
            )
            
            # Only add constraint if capacity > 0, otherwise allocation should be 0 anyway
            if capacity_usd > 0:
                 prob += value_sum_expr <= capacity_usd, constraint_name_cap
            else:
                 # If capacity is 0, constrain the value sum to 0
                 prob += value_sum_expr == 0, constraint_name_cap
        # else: No need for constraint if supplier has no feasible items


    # --- 7. Solve Optimization Problem ---
    print("Solving the optimization problem...")
    # Write the problem to an LP file for debugging
    try:
        lp_file_path = "Procurement_Optimization_Debug.lp"
        prob.writeLP(lp_file_path)
        print(f"LP problem written to: {os.path.abspath(lp_file_path)}")
    except Exception as e:
        print(f"Warning: Could not write LP file: {e}", file=sys.stderr)
        
    prob.solve()
    print(f"Optimization Status: {pulp.LpStatus[prob.status]}")

    # --- 8. Process and Save Results ---
    if pulp.LpStatus[prob.status] == 'Optimal':
        opt_objective_value = pulp.value(prob.objective)
        print(f"Optimal Objective Value (Total Effective Cost): {opt_objective_value:,.2f}")

        results = []
        calculated_opt_total_value = 0 # For debugging/comparison

        for var_idx, var_obj in AllocatedQuantity_vars.items(): # var_idx is the simple integer index
            allocated_qty = var_obj.varValue
            if allocated_qty is not None and allocated_qty > 1e-6: # Use a small tolerance
                data = var_data_map[var_idx] # Retrieve all data using the simple index
                
                lifnr_val = data['lifnr']
                g_key_val = data['g_key']
                matnr_val = data['matnr'] # Directly from var_data_map
                eff_cost_val = data['eff_cost']
                avg_price_val = data['avg_price']
                # supplier_name_val = data['supplier_name'] # Available if needed for output directly

                result_entry = {
                    col_supplier_id: lifnr_val,
                    grouping_key_mapped: g_key_val,
                    col_mat_num: matnr_val, 
                    'Allocated_Quantity': allocated_qty,
                    col_effective_cost: eff_cost_val,
                    'Optimized_Total_Effective_Cost_for_Combo': allocated_qty * eff_cost_val if pd.notna(eff_cost_val) else 0,
                    col_avg_unit_price_raw: avg_price_val 
                }
                results.append(result_entry)
                calculated_opt_total_value += result_entry['Optimized_Total_Effective_Cost_for_Combo']

        print(f"DEBUG: Sum of Optimized Combo Costs from Solver Results: {calculated_opt_total_value:,.2f}")

        if not results:
             print("Warning: Optimization was optimal, but no allocation quantities > 1e-6 were found.", file=sys.stderr)
             # Still save an empty file or header? Decide based on desired behavior.
             # For now, let's save header only if no results, including MATNR
             df_results = pd.DataFrame(columns=[col_supplier_id, grouping_key_mapped, col_mat_num, 'Allocated_Quantity', col_effective_cost, 'Optimized_Total_Effective_Cost_for_Combo', col_avg_unit_price_raw])

        else:
            df_results = pd.DataFrame(results)

        # --- Add descriptions and save ---
        print("Adding descriptions to results...")
        # df_supplier_desc and df_desc_hist already loaded and keys converted

        # Ensure key types match for merging
        df_results[col_supplier_id] = df_results[col_supplier_id].astype(str)
        # grouping_key_mapped is already a column in df_results from its creation
        df_results[grouping_key_mapped] = df_results[grouping_key_mapped].astype(str)
        df_results[col_mat_num] = df_results[col_mat_num].astype(str)

        # Supplier descriptions
        df_results = pd.merge(df_results, df_supplier_desc, on=col_supplier_id, how='left')
        df_results[col_supplier_name] = df_results[col_supplier_name].fillna('Unknown Vendor')

        # Item/Group descriptions merge
        # temp_desc_df will be df_desc_hist, selecting only necessary columns for the merge
        temp_desc_df = df_desc_hist[[desc_key_mapped, desc_col_mapped]].copy()

        df_results = pd.merge(
            df_results,
            temp_desc_df,
            left_on=col_mat_num,
            right_on=desc_key_mapped, # This is MATNR for maktx mode
            how='left',
            suffixes=('_keep', '_new_desc') # _keep for left df_results, _new_desc for right temp_desc_df
        )

        # Restore grouping_key_mapped if it was suffixed (e.g. 'MAKTX_keep' -> 'MAKTX')
        # This happens if grouping_key_mapped had the same name as a column in temp_desc_df (e.g. desc_col_mapped)
        # and was not part of the join keys.
        if grouping_key_mapped + '_keep' in df_results.columns:
            df_results.rename(columns={grouping_key_mapped + '_keep': grouping_key_mapped}, inplace=True)
        
        # Identify the name of the description column that came from temp_desc_df
        # It will be desc_col_mapped + '_new_desc' if desc_col_mapped collided with a col in df_results (LHS)
        # or just desc_col_mapped if no collision.
        actual_new_desc_col_name = desc_col_mapped
        if desc_col_mapped + '_new_desc' in df_results.columns:
            actual_new_desc_col_name = desc_col_mapped + '_new_desc'
        
        # Now, handle filling NaNs and renaming based on mode and column names

        if grouping_key_mapped == desc_col_mapped:
            # Case: 'maktx' mode. grouping_key_mapped is 'MAKTX', desc_col_mapped is also 'MAKTX'.
            # The original 'MAKTX' from df_results (optimization key) is now in column `grouping_key_mapped`.
            # The description text from df_desc_hist is in `actual_new_desc_col_name` (e.g., 'MAKTX_new_desc').
            if actual_new_desc_col_name in df_results.columns and grouping_key_mapped in df_results.columns:
                # Use description from hist to fill NaNs in the main grouping key column
                df_results[grouping_key_mapped] = df_results[grouping_key_mapped].fillna(df_results[actual_new_desc_col_name])
            
            # Fill any remaining NaNs in the main grouping key column
            if grouping_key_mapped in df_results.columns:
                df_results[grouping_key_mapped] = df_results[grouping_key_mapped].fillna(f'Unknown {mode.capitalize()}')
            
            # Drop the (now redundant) suffixed description column from hist
            if actual_new_desc_col_name in df_results.columns and actual_new_desc_col_name != grouping_key_mapped :
                df_results.drop(columns=[actual_new_desc_col_name], inplace=True)
        else:
            # Case: 'matnr' or 'matkl' mode. grouping_key_mapped (e.g. 'MATNR') is different from desc_col_mapped (e.g. 'MAKTX').
            # The description column from hist is `actual_new_desc_col_name`. Rename it to `desc_col_mapped`.
            if actual_new_desc_col_name in df_results.columns:
                if actual_new_desc_col_name != desc_col_mapped: # If it was suffixed
                    df_results.rename(columns={actual_new_desc_col_name: desc_col_mapped}, inplace=True)
                # Fill NaNs in this (now correctly named) desc_col_mapped column
                df_results[desc_col_mapped] = df_results[desc_col_mapped].fillna(f'Unknown {mode.capitalize()}')
            elif desc_col_mapped not in df_results.columns :
                 # If it wasn't brought in for some reason, create and fill
                 df_results[desc_col_mapped] = f'Unknown {mode.capitalize()}'

        # Drop the redundant join key from the right side if it was brought in and suffixed
        # desc_key_mapped is the join key from right (e.g. 'MATNR')
        # If it was suffixed (e.g. 'MATNR_new_desc') and it's not the same as col_mat_num (original left key), drop it.
        # However, since it's a join key, pandas usually handles it well by not creating a suffixed version if names match.
        # If desc_key_mapped (from right) was different from col_mat_num (from left) AND it got suffixed, then drop.
        # But here, desc_key_mapped is used as right_on=desc_key_mapped, so it should be fine.
        # The only concern is if desc_key_mapped itself was also a non-join column name that collided.
        # Given temp_desc_df only has [desc_key_mapped, desc_col_mapped], this is less likely for desc_key_mapped.
        if desc_key_mapped + '_new_desc' in df_results.columns and desc_key_mapped != col_mat_num : # Check if it's not the left key
            df_results.drop(columns=[desc_key_mapped + '_new_desc'], inplace=True)


        # Define final columns dynamically
        final_cols = [col_supplier_id, col_supplier_name, col_mat_num, grouping_key_mapped] # Add MATNR
        # Add specific description only if it's different from grouping key and MATNR
        if desc_col_mapped != grouping_key_mapped and desc_col_mapped != col_mat_num and desc_col_mapped in df_results.columns:
             final_cols.append(desc_col_mapped)
        final_cols.extend([
            'Allocated_Quantity', col_effective_cost, 'Optimized_Total_Effective_Cost_for_Combo',
            col_avg_unit_price_raw # Include avg price for reference
        ])

        # Ensure only existing columns are selected and order them
        final_cols_present = [col for col in final_cols if col in df_results.columns]
        # Remove duplicates while preserving order
        final_cols_present = sorted(list(set(final_cols_present)), key=final_cols.index)
        df_results = df_results[final_cols_present]


        df_results.sort_values(by=[grouping_key_mapped, col_supplier_id], inplace=True)

        # Use the full output path argument directly
        output_filename = optimization_output_path
        try:
            df_results.to_csv(output_filename, index=False, float_format='%.4f') # Use more precision
            print(f"\nOptimized allocation saved successfully to: {output_filename}")
        except Exception as e:
            print(f"Error saving optimization results to {output_filename}: {e}", file=sys.stderr)

    elif pulp.LpStatus[prob.status] == 'Infeasible':
        print("Error: The optimization problem is infeasible. Check constraints:")
        print(f" - Is total target quantity for some {grouping_key_mapped} items higher than the sum of quantities suppliers can provide (considering value capacity)?")
        print(f" - Are there conflicting constraints? Check target quantities vs supplier value capacities.")
        # You might want to write the problem to an .lp file for debugging: prob.writeLP("Procurement_Optimization_Infeasible.lp")
    else:
        print(f"Optimization finished with status: {pulp.LpStatus[prob.status]}. Solution may not be optimal or found.")

if __name__ == "__main__":
    main()
