import pandas as pd
import numpy as np
import argparse
import os
import sys
import json # Added for loading mappings
from datetime import datetime, timedelta

# --- Configuration ---
# Use the same demand period as in optimize_procurement.py
DEMAND_PERIOD_DAYS = 365

# Cost components from evaluate_vendor_material.py
COST_COMPONENTS_BASE = [
    'cost_BasePrice', 'cost_Tariff', 'cost_Holding_LeadTime',
    'cost_Holding_LTVariability', 'cost_Holding_Lateness',
    'cost_Risk_PriceVolatility', 'cost_Impact_PriceTrend',
    'cost_Logistics'
]

# --- Helper Functions ---
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

EXCHANGE_RATES = {
    'EUR': 1.14, 'MXN': 0.051, 'USD': 1.00
}
def convert_to_usd(row, mapped_value_col, mapped_currency_col):
    """Applies exchange rate to convert value to USD using mapped columns."""
    value = row[mapped_value_col]
    currency = row[mapped_currency_col]
    rate = EXCHANGE_RATES.get(currency, 1.0)
    if pd.isna(value) or pd.isna(currency):
        return 0.0 # Return 0 for NaN to allow sum, consider implications
    return value * rate

# --- Main Comparison Logic ---
def main():
    parser = argparse.ArgumentParser(description="Compare historical procurement policy with optimized allocation based on economic impact.")
    parser.add_argument('--ranking-results-path', required=True, help='Full path to the ranking results CSV file (evaluate_vendor_material.py output).')
    parser.add_argument('--optimization-results-path', required=True, help='Full path to the optimization results CSV file (optimize_procurement.py output).')
    parser.add_argument('--tables-dir', required=True, help='Directory containing the mapped table CSV files.')
    parser.add_argument('--comparison-output-path', required=True, help='Full path for the comparison results CSV file.')
    parser.add_argument('--table-map', required=True, help='Path to the JSON file mapping table concepts to filenames.')
    parser.add_argument('--column-map', required=True, help='Path to the JSON file mapping column concepts to actual column names.')
    parser.add_argument('--mode', required=True, choices=['matkl', 'matnr', 'maktx'], help="Grouping mode used for ranking and optimization.")
    parser.add_argument('--costs-config-path', required=True, help='Path to the JSON file specifying which costs to include for comparison.')

    args = parser.parse_args()
    mode = args.mode
    tables_dir = args.tables_dir
    comparison_output_path = args.comparison_output_path
    rank_file_path = args.ranking_results_path
    opt_file_path = args.optimization_results_path

    # Load cost configuration for comparison context
    costs_config_compare = {}
    try:
        with open(args.costs_config_path, 'r') as f:
            config_data = json.load(f)
            # Handle both old and new format
            if 'cost_components' in config_data:
                costs_config_compare = config_data['cost_components']
            else:
                costs_config_compare = config_data
        print(f"Loaded cost configuration for comparison from: {args.costs_config_path}")
    except FileNotFoundError:
        print(f"Warning: Costs configuration file for comparison not found at {args.costs_config_path}. All cost components will be enabled by default for comparison.", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {args.costs_config_path} for comparison. All cost components will be enabled by default.", file=sys.stderr)

    # Load mappings
    try:
        with open(args.table_map, 'r') as f:
            table_map = json.load(f)
        with open(args.column_map, 'r') as f:
            column_map = json.load(f)
        default_table_map = {
            'PO Items': 'SAP_VLY_IL_PO_ITEMS.csv', 'PO Header': 'SAP_VLY_IL_PO_HEADER.csv',
            'Material': 'SAP_VLY_IL_MATERIAL.csv', 'Material Group': 'SAP_VLY_IL_MATERIAL_GROUP.csv',
            'Supplier': 'SAP_VLY_IL_SUPPLIER.csv'
        }
        for key, default_val in default_table_map.items():
            table_map.setdefault(key, default_val)
    except FileNotFoundError as e:
        print(f"Error: Mapping file not found: {e}", file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from mapping file: {e}", file=sys.stderr); sys.exit(1)

    os.makedirs(os.path.dirname(comparison_output_path), exist_ok=True)

    # --- Get mapped column names ---
    col_po_num = column_map.get('PO_Number') or 'EBELN'
    # col_po_item = column_map.get('PO_Item_Number') or 'EBELP' # Not directly used in aggregation here
    col_mat_num = column_map.get('Material_ID') or 'MATNR'
    col_mat_group = column_map.get('Material_Group_Code') or 'MATKL'
    col_item_qty = column_map.get('Item_Ordered_Quantity') or 'MENGE' # Used in items and history
    col_order_unit = column_map.get('Order_Unit') or 'MEINS' # For MEINS check
    col_net_value = column_map.get('Item_Net_Order_Value') or 'NETWR'
    col_supplier_id = column_map.get('Supplier_ID') or 'LIFNR'
    col_po_date = column_map.get('PO_Creation_Date') or 'BEDAT'
    col_currency = column_map.get('PO_Currency_Code') or 'WAERS'
    col_supplier_name = column_map.get('Supplier_Name_1') or 'NAME1'
    col_mat_group_desc = column_map.get('Material_Group_Description') or 'MATKL_DESC'
    col_mat_desc = column_map.get('Material_Description') or 'MAKTX'

    # Columns from ranking/optimization files
    col_eff_cost_per_unit_usd = 'EffectiveCostPerUnit_USD'
    col_avg_unit_price_raw = 'AvgUnitPriceUSD_raw' # For reference
    col_opt_alloc_qty = 'Allocated_Quantity'
    col_opt_total_eff_cost_combo = 'Optimized_Total_Effective_Cost_for_Combo'

    # Determine grouping key and description column based on mode
    if mode == 'matkl':
        grouping_key_mapped = col_mat_group
        desc_key_mapped = col_mat_group # Key to join description table
        desc_col_mapped = col_mat_group_desc
        desc_concept_file_key = 'Material Group'
    elif mode == 'matnr':
        grouping_key_mapped = col_mat_num
        desc_key_mapped = col_mat_num
        desc_col_mapped = col_mat_desc
        desc_concept_file_key = 'Material'
    elif mode == 'maktx':
        grouping_key_mapped = col_mat_desc # Group by description
        desc_key_mapped = col_mat_num     # Still need MATNR to link description file
        desc_col_mapped = col_mat_desc    # The description column is the grouping key itself
        desc_concept_file_key = 'Material' # File containing MAKTX
    else:
        sys.exit(1)

    # Define the unique list of columns to group/merge by, based on mode.
    # This ensures it's defined before any conditional logic might skip it later.
    group_by_hist_cols = list(dict.fromkeys([col_supplier_id, grouping_key_mapped, col_mat_num]))
    print(f"Running comparison in mode: '{mode}', grouping by mapped key: '{grouping_key_mapped}', using merge keys: {group_by_hist_cols}")


    # --- 1. Load Input Data ---
    print(f"Using Ranking File: {rank_file_path}")
    print(f"Using Optimization File: {opt_file_path}")

    # Load Optimization Results
    opt_cols_to_load = [col_supplier_id, grouping_key_mapped, col_mat_num,
                        col_opt_alloc_qty, col_opt_total_eff_cost_combo]
    try:
        header_df_opt = pd.read_csv(opt_file_path, nrows=0, low_memory=False, encoding='utf-8-sig')
        missing_opt_cols = [col for col in opt_cols_to_load if col not in header_df_opt.columns]
        if missing_opt_cols:
             print(f"Error: Required columns missing in optimization file '{opt_file_path}': {missing_opt_cols}", file=sys.stderr); sys.exit(1)
        df_opt_results = load_data(opt_file_path, usecols=opt_cols_to_load)
        # Convert keys to string for reliable merging
        for key_col in [col_supplier_id, grouping_key_mapped, col_mat_num]:
            if key_col in df_opt_results.columns:
                df_opt_results[key_col] = df_opt_results[key_col].astype(str)

    except Exception as e:
        print(f"Error loading or checking optimization file {opt_file_path}: {e}", file=sys.stderr); sys.exit(1)

    # Load Full Ranking Results (for cost structures)
    rank_cols_to_load = [col_supplier_id, grouping_key_mapped, col_mat_num,
                         col_eff_cost_per_unit_usd, col_avg_unit_price_raw] + COST_COMPONENTS_BASE
    try:
        header_df_rank = pd.read_csv(rank_file_path, nrows=0, low_memory=False, encoding='utf-8-sig')
        missing_rank_cols = [col for col in rank_cols_to_load if col not in header_df_rank.columns]
        if missing_rank_cols:
             print(f"Error: Required columns missing in ranking file '{rank_file_path}': {missing_rank_cols}", file=sys.stderr); sys.exit(1)
        df_rank_full = load_data(rank_file_path, usecols=rank_cols_to_load)
        # Convert keys to string
        for key_col in [col_supplier_id, grouping_key_mapped, col_mat_num]:
            if key_col in df_rank_full.columns:
                df_rank_full[key_col] = df_rank_full[key_col].astype(str)

    except Exception as e:
        print(f"Error loading or checking ranking file {rank_file_path}: {e}", file=sys.stderr); sys.exit(1)

    # Load Historical Data
    po_items_filename = table_map.get('PO Items')
    po_header_filename = table_map.get('PO Header')
    desc_filename = table_map.get(desc_concept_file_key)
    supplier_filename = table_map.get('Supplier')

    hist_items_cols_mapped = [col_po_num, col_mat_group, col_mat_num, col_item_qty, col_net_value, col_order_unit]
    hist_header_cols_mapped = [col_po_num, col_supplier_id, col_po_date, col_currency]
    # desc_key_mapped and desc_col_mapped are used for specific description file
    desc_file_cols_mapped = list(set(filter(None, [desc_key_mapped, desc_col_mapped])))
    supplier_cols_mapped = [col_supplier_id, col_supplier_name]

    try:
        df_items_hist = load_data(os.path.join(tables_dir, po_items_filename), usecols=hist_items_cols_mapped)
        df_header_hist = load_data(os.path.join(tables_dir, po_header_filename), usecols=hist_header_cols_mapped)
        df_desc_data = load_data(os.path.join(tables_dir, desc_filename), usecols=desc_file_cols_mapped).drop_duplicates(subset=[desc_key_mapped])
        df_supplier_desc = load_data(os.path.join(tables_dir, supplier_filename), usecols=supplier_cols_mapped).drop_duplicates(subset=[col_supplier_id])
    except Exception as e:
        print(f"Error loading historical or description data: {e}", file=sys.stderr); sys.exit(1)

    # --- 2. Prepare Historical Data ---
    print("Preparing historical data for comparison...")
    # Convert keys to string
    key_cols_to_str = [col_po_num, col_supplier_id, col_mat_group, col_mat_num, desc_key_mapped, col_order_unit]
    for df in [df_items_hist, df_header_hist, df_desc_data, df_supplier_desc]:
        for col in key_cols_to_str:
            if col in df.columns:
                if df[col].dtype == 'float64':
                    try: df[col] = df[col].astype(pd.Int64Dtype()).astype(str)
                    except: df[col] = df[col].astype(str)
                else: df[col] = df[col].astype(str)

    df_items_hist[col_net_value] = pd.to_numeric(df_items_hist[col_net_value], errors='coerce')
    df_items_hist[col_item_qty] = pd.to_numeric(df_items_hist[col_item_qty], errors='coerce')
    df_header_hist[col_po_date] = pd.to_datetime(df_header_hist[col_po_date], errors='coerce')

    # Merge historical items and header
    df_hist_merged_base = pd.merge(
        df_items_hist.dropna(subset=[desc_key_mapped, col_item_qty, col_order_unit, col_mat_num, col_mat_group]),
        df_header_hist.dropna(subset=[col_supplier_id, col_po_date, col_currency]),
        on=col_po_num, how='inner'
    )

    # Merge with item/group descriptions
    df_hist_merged = pd.merge(df_hist_merged_base, df_desc_data, on=desc_key_mapped, how='left')
    # Fill description based on mode
    df_hist_merged[desc_col_mapped] = df_hist_merged[desc_col_mapped].fillna(f'Unknown {mode.capitalize()}')
    if grouping_key_mapped not in df_hist_merged.columns: # e.g. maktx mode where grouping_key_mapped = desc_col_mapped
        if desc_col_mapped in df_hist_merged.columns: # It should be
             df_hist_merged[grouping_key_mapped] = df_hist_merged[desc_col_mapped]
        else: # Should not happen if desc_col_mapped is set correctly
             print(f"Error: Critical description column '{desc_col_mapped}' for grouping key '{grouping_key_mapped}' is missing.", file=sys.stderr); sys.exit(1)

    df_hist_merged.dropna(subset=[grouping_key_mapped, col_mat_num], inplace=True) # Ensure grouping_key and MATNR are present
    df_hist_merged['NETWR_USD'] = df_hist_merged.apply(convert_to_usd, axis=1, mapped_value_col=col_net_value, mapped_currency_col=col_currency)

    # Filter for Demand Period
    demand_end_date = df_hist_merged[col_po_date].max()
    if pd.isna(demand_end_date):
         print("Error: Cannot determine date range for historical comparison.", file=sys.stderr); sys.exit(1)
    demand_start_date = demand_end_date - timedelta(days=DEMAND_PERIOD_DAYS)
    df_hist_demand_period = df_hist_merged[
        (df_hist_merged[col_po_date] >= demand_start_date) & (df_hist_merged[col_po_date] <= demand_end_date)
    ].copy()

    if df_hist_demand_period.empty:
        print("Warning: No historical data found within the demand period.", file=sys.stderr)
        # Create empty df_hist_agg to allow script to proceed and show 0 historicals
        df_hist_agg = pd.DataFrame(columns=[col_supplier_id, grouping_key_mapped, col_mat_num,
                                            'Historical_Allocated_Quantity', 'Historical_Actual_Spend_USD'])
    else:
        # MEINS Consistency Check for Historical Quantity (consistent with other scripts)
        print("Performing MEINS consistency check for historical quantities...")
        # Use the same grouping key as the other scripts for consistency
        # This ensures we filter the same groups that were filtered in vendor evaluation
        meins_check_key = grouping_key_mapped
        if meins_check_key in df_hist_demand_period.columns and col_order_unit in df_hist_demand_period.columns:
            df_check = df_hist_demand_period.dropna(subset=[meins_check_key, col_order_unit])
            if not df_check.empty:
                meins_consistency = df_check.groupby(meins_check_key)[col_order_unit].nunique()
                inconsistent_groups = meins_consistency[meins_consistency > 1].index
                if not inconsistent_groups.empty:
                    print(f"Warning: Excluding {mode} groups with inconsistent MEINS from historical quantity sum: {list(inconsistent_groups)}")
                    df_hist_demand_period = df_hist_demand_period[~df_hist_demand_period[meins_check_key].isin(inconsistent_groups)].copy()
                else:
                    print("MEINS consistency check passed for historical quantities.")
            else: 
                print("Warning: No data for MEINS consistency check after dropping NaNs.")
        else: 
            print(f"Warning: Cannot perform MEINS consistency check. Missing '{meins_check_key}' or '{col_order_unit}'.")

        # COMMENTED OUT: LeadTimeDays == 0 filtering to restore original behavior
        # Filter out transactions that would have LeadTimeDays == 0 for consistency with evaluation script
        # print("Applying LeadTimeDays consistency filter to historical comparison data...")
        # 
        # # Load goods receipt data to calculate lead times for historical filtering
        # hist_gr_filename = table_map.get('PO History') or table_map.get('SAP_VLY_IL_PO_HISTORY.csv') or 'SAP_VLY_IL_PO_HISTORY.csv'
        # col_hist_event_type = column_map.get('PO_History_Transaction_Type') or 'VGABE'
        # col_hist_post_date = column_map.get('History_Posting_Date') or 'BUDAT'
        # col_po_item = column_map.get('PO_Item_Number') or 'EBELP'
        # 
        # hist_gr_cols_comp = [col_po_num, col_po_item, col_hist_event_type, col_hist_post_date]
        # 
        # try:
        #     df_hist_gr_comp = load_data(os.path.join(tables_dir, hist_gr_filename), usecols=hist_gr_cols_comp)
        #     # Convert keys to string and dates
        #     for col in [col_po_num, col_po_item]:
        #         if col in df_hist_gr_comp.columns:
        #             if df_hist_gr_comp[col].dtype == 'float64':
        #                 try: df_hist_gr_comp[col] = df_hist_gr_comp[col].astype(pd.Int64Dtype()).astype(str)
        #                 except: df_hist_gr_comp[col] = df_hist_gr_comp[col].astype(str)
        #             else: df_hist_gr_comp[col] = df_hist_gr_comp[col].astype(str)
        #     
        #     df_hist_gr_comp[col_hist_post_date] = pd.to_datetime(df_hist_gr_comp[col_hist_post_date], errors='coerce')
        #     
        #     # Filter for goods receipts and aggregate to get first GR date
        #     df_gr_filtered_comp = df_hist_gr_comp[df_hist_gr_comp[col_hist_event_type].astype(str) == '1'].copy()
        #     if not df_gr_filtered_comp.empty:
        #         df_gr_agg_comp = df_gr_filtered_comp.groupby([col_po_num, col_po_item])[col_hist_post_date].min().reset_index()
        #         df_gr_agg_comp.rename(columns={col_hist_post_date: 'FirstGRDate'}, inplace=True)
        #         
        #         # Merge with historical demand data to calculate lead times
        #         df_hist_demand_period = pd.merge(
        #             df_hist_demand_period, 
        #             df_gr_agg_comp, 
        #             left_on=[col_po_num], 
        #             right_on=[col_po_num], 
        #             how='left'
        #         )
        #         
        #         # Calculate lead time and filter out 0-day lead times
        #         df_hist_demand_period['LeadTimeDays'] = np.nan
        #         valid_lt_idx_comp = (df_hist_demand_period['FirstGRDate'].notna() & 
        #                        df_hist_demand_period[col_po_date].notna())
        #         
        #         df_hist_demand_period.loc[valid_lt_idx_comp, 'LeadTimeDays'] = (
        #             df_hist_demand_period.loc[valid_lt_idx_comp, 'FirstGRDate'] - 
        #             df_hist_demand_period.loc[valid_lt_idx_comp, col_po_date]
        #         ).dt.days
        #         
        #         # Filter out same-day deliveries for consistency
        #         before_leadtime_comp_filter = len(df_hist_demand_period)
        #         df_hist_demand_period = df_hist_demand_period[
        #             (df_hist_demand_period['LeadTimeDays'].isna()) | 
        #             (df_hist_demand_period['LeadTimeDays'] != 0)
        #         ]
        #         after_leadtime_comp_filter = len(df_hist_demand_period)
        #         print(f"Historical comparison data after LeadTimeDays == 0 filter: {after_leadtime_comp_filter:,} rows (removed {before_leadtime_comp_filter - after_leadtime_comp_filter:,})")
        #     else:
        #         print("Warning: No goods receipt data found for comparison lead time calculation. Proceeding without lead time filtering.")
        # except Exception as e:
        #     print(f"Warning: Could not load goods receipt data for comparison lead time filtering: {e}. Proceeding without lead time filtering.")

        # Aggregate Historical Data
        # The definition of group_by_hist_cols has been moved earlier in the script.
        # We now use the globally defined group_by_hist_cols here.

        # The following block (original lines 255-259) which first defined df_hist_agg
        # caused the error at .reset_index() and seems to be superseded by the
        # subsequent merge operation that redefines df_hist_agg more robustly.
        # We will use the corrected group_by_hist_cols for the subsequent, more robust aggregation.
        # Original problematic block:
        # df_hist_agg = df_hist_demand_period.groupby(group_by_hist_cols).agg(
        #     Historical_Allocated_Quantity=(col_item_qty, 'sum'),
        #     Historical_Actual_Spend_USD=(col_net_value, 'sum')
        # ).reset_index()

        # Re-calculate NETWR_USD sum after grouping to be safe, or sum the NETWR_USD column
        # This uses the corrected group_by_hist_cols
        df_hist_agg_spend_usd = df_hist_demand_period.groupby(group_by_hist_cols)['NETWR_USD'].sum().reset_index()
        df_hist_agg_spend_usd.rename(columns={'NETWR_USD': 'Historical_Actual_Spend_USD'}, inplace=True)

        # This is the primary definition of df_hist_agg, using the corrected group_by_hist_cols
        df_hist_agg = pd.merge(
            df_hist_demand_period.groupby(group_by_hist_cols)[col_item_qty].sum().reset_index(name='Historical_Allocated_Quantity'),
            df_hist_agg_spend_usd,
            on=group_by_hist_cols,
            how='outer'
        ).fillna({'Historical_Allocated_Quantity': 0, 'Historical_Actual_Spend_USD': 0})


    # --- 3. Apply Consistent Filtering to Ranking Data (same as optimization script) ---
    print("Applying consistent filtering to ranking data...")
    
    # Apply MEINS consistency filtering to ranking data (consistent with optimize_procurement.py)
    print("Performing MEINS consistency check for ranking data...")
    inconsistent_groups_ranking = []
    if col_order_unit in df_hist_demand_period.columns and grouping_key_mapped in df_hist_demand_period.columns:
        # Use the same MEINS inconsistent groups identified from historical data
        df_check_hist = df_hist_demand_period.dropna(subset=[grouping_key_mapped, col_order_unit])
        if not df_check_hist.empty:
            meins_consistency_hist = df_check_hist.groupby(grouping_key_mapped)[col_order_unit].nunique()
            inconsistent_groups_ranking = meins_consistency_hist[meins_consistency_hist > 1].index.tolist()
            if inconsistent_groups_ranking:
                print(f"Applying MEINS consistency filter to ranking data: excluding {len(inconsistent_groups_ranking)} groups")
                ranking_before_meins = len(df_rank_full)
                df_rank_full = df_rank_full[~df_rank_full[grouping_key_mapped].isin(inconsistent_groups_ranking)].copy()
                print(f"Ranking data after MEINS filter: {len(df_rank_full):,} rows (removed {ranking_before_meins - len(df_rank_full):,} rows)")
            else:
                print("No MEINS filtering applied to ranking data - all groups are consistent")
        else:
            print("Warning: No data for ranking MEINS consistency check after dropping NaNs.")
    else:
        print(f"Warning: Cannot perform MEINS consistency check on ranking data. Missing '{col_order_unit}' or '{grouping_key_mapped}'.")
    
    # Apply multiple suppliers filtering to ranking data (consistent with optimize_procurement.py)
    print("Filtering ranking data to only include materials with multiple suppliers...")
    suppliers_per_material_rank = df_rank_full.groupby(grouping_key_mapped)[col_supplier_id].nunique()
    multi_supplier_materials_rank = suppliers_per_material_rank[suppliers_per_material_rank > 1].index
    
    ranking_before_multi = len(df_rank_full)
    df_rank_full = df_rank_full[df_rank_full[grouping_key_mapped].isin(multi_supplier_materials_rank)].copy()
    print(f"Materials with multiple suppliers in ranking: {len(multi_supplier_materials_rank):,}")
    print(f"Ranking data after multi-supplier filter: {len(df_rank_full):,} rows (removed {ranking_before_multi - len(df_rank_full):,} rows)")
    
    # Apply the same filtering to historical data to ensure consistency
    print("Applying same filters to historical data for consistency...")
    if inconsistent_groups_ranking:
        hist_before_meins = len(df_hist_agg)
        df_hist_agg = df_hist_agg[~df_hist_agg[grouping_key_mapped].isin(inconsistent_groups_ranking)].copy()
        print(f"Historical data after MEINS filter: {len(df_hist_agg):,} rows (removed {hist_before_meins - len(df_hist_agg):,} rows)")
    
    hist_before_multi = len(df_hist_agg)
    df_hist_agg = df_hist_agg[df_hist_agg[grouping_key_mapped].isin(multi_supplier_materials_rank)].copy()
    print(f"Historical data after multi-supplier filter: {len(df_hist_agg):,} rows (removed {hist_before_multi - len(df_hist_agg):,} rows)")
    print(f"Comparison now constrained to materials with multiple suppliers and MEINS consistency")

    # --- 4. Combine Historical, Ranking, and Optimized Data ---
    print("Combining filtered historical, ranking, and optimized data...")
    # Start with df_rank_full as it contains all potential (supplier, item) cost structures (now filtered)
    # Key columns (LIFNR, MATNR etc.) are already converted to string type during their respective DataFrame loading/creation.
    # The problematic loop that added 'nan' columns and could misinterpret duplicate keys has been removed.

    # Ensure all key columns in the DataFrames to be merged are of string type, just in case.
    # This is a lighter touch than the previous loop.
    for df_to_check in [df_hist_agg, df_rank_full, df_opt_results]:
        for key_c in group_by_hist_cols: # group_by_hist_cols contains unique keys like ['LIFNR', 'MATNR']
            if key_c in df_to_check.columns:
                if df_to_check[key_c].dtype != 'object' and df_to_check[key_c].dtype != 'string':
                     df_to_check[key_c] = df_to_check[key_c].astype(str)
            # else: # If a key column is truly missing from a df, it's a deeper issue.
                  # print(f"Critical Warning: Key column '{key_c}' is missing from one of the DataFrames prior to merge.")


    # Merge historical aggregates with ranking data to get historical effective costs
    df_comparison = pd.merge(
        df_hist_agg,
        df_rank_full,
        on=group_by_hist_cols, # Use unique keys
        how='left', # Keep all historical, get costs if available in rank
        suffixes=('_hist', '_rank_for_hist')
    )
    # Rename rank columns that were brought in for historical context
    df_comparison.rename(columns={
        col_eff_cost_per_unit_usd : 'Historical_EffectiveCostPerUnit_USD',
        col_avg_unit_price_raw: 'Historical_AvgUnitPriceUSD_raw'
    }, inplace=True)
    for comp in COST_COMPONENTS_BASE:
        df_comparison.rename(columns={comp: f'Historical_{comp}_Unit'}, inplace=True)


    # Merge optimized results
    df_comparison = pd.merge(
        df_comparison,
        df_opt_results,
        on=group_by_hist_cols, # Use unique keys
        how='outer', # Keep all historical/ranked and all optimized
        suffixes=('_already_merged', '_opt') # Suffixes for optimized specific columns if names clash
    )
    # Rename optimized columns clearly
    df_comparison.rename(columns={
        col_opt_alloc_qty: 'Optimized_Allocated_Quantity',
        col_opt_total_eff_cost_combo: 'Optimized_Total_Effective_Cost_for_Combo'
    }, inplace=True)


    # Fill NaNs for quantities and costs
    df_comparison['Historical_Allocated_Quantity'] = df_comparison['Historical_Allocated_Quantity'].fillna(0)
    df_comparison['Historical_Actual_Spend_USD'] = df_comparison['Historical_Actual_Spend_USD'].fillna(0)
    df_comparison['Optimized_Allocated_Quantity'] = df_comparison['Optimized_Allocated_Quantity'].fillna(0)
    df_comparison['Optimized_Total_Effective_Cost_for_Combo'] = df_comparison['Optimized_Total_Effective_Cost_for_Combo'].fillna(0)
    df_comparison['Historical_EffectiveCostPerUnit_USD'] = df_comparison['Historical_EffectiveCostPerUnit_USD'].fillna(0) # Or handle if lookup failed

    # For optimized choices, their unit effective cost and components come from df_rank_full
    # If a row was created by 'outer' join from df_opt_results, its cost structure needs to be merged from df_rank_full
    # This happens if an optimized combo wasn't in historical+rank merge
    # We can simplify by ensuring df_rank_full is merged onto the comparison for optimized choices.
    # Since df_rank_full was the source for Historical_EffectiveCostPerUnit_USD, we can use its values for optimized too,
    # but we need to select the correct columns.

    # Let's re-merge df_rank_full for optimized choices explicitly to avoid confusion with historical suffixed cols
    df_rank_for_opt_costs = df_rank_full.rename(columns={
        col_eff_cost_per_unit_usd : 'Optimized_EffectiveCostPerUnit_USD',
        col_avg_unit_price_raw: 'Optimized_AvgUnitPriceUSD_raw'
    })
    for comp in COST_COMPONENTS_BASE:
        df_rank_for_opt_costs.rename(columns={comp: f'Optimized_{comp}_Unit'}, inplace=True)

    # Define the columns to select from df_rank_for_opt_costs, ensuring uniqueness
    opt_cost_cols_to_select = group_by_hist_cols + \
                              ['Optimized_EffectiveCostPerUnit_USD', 'Optimized_AvgUnitPriceUSD_raw'] + \
                              [f'Optimized_{comp}_Unit' for comp in COST_COMPONENTS_BASE]
    unique_opt_cost_cols_to_select = list(dict.fromkeys(opt_cost_cols_to_select))

    # Ensure the selected columns exist in df_rank_for_opt_costs before selecting
    valid_opt_cost_cols = [col for col in unique_opt_cost_cols_to_select if col in df_rank_for_opt_costs.columns]

    df_comparison = pd.merge(
        df_comparison,
        df_rank_for_opt_costs[valid_opt_cost_cols], # Select unique, valid columns
        on=group_by_hist_cols, # Use the unique list of keys defined earlier
        how='left' # Add optimized cost structures
    )
    df_comparison['Optimized_EffectiveCostPerUnit_USD'] = df_comparison['Optimized_EffectiveCostPerUnit_USD'].fillna(0)


    # --- 5. Calculate Full Historical and Optimized Effective Costs and Deltas ---
    print("Calculating total effective costs and deltas...")

    print("Recalculating Effective Costs based on current comparison cost configuration...")
    # COST_COMPONENTS_BASE is already defined in compare_policies.py

    # Recalculate Historical Effective Cost Per Unit for Current Context
    df_comparison['Historical_EffectiveCostPerUnit_USD_CurrentContext'] = 0.0
    active_hist_components_compare = []
    for comp_base_name in COST_COMPONENTS_BASE:
        hist_unit_col = f'Historical_{comp_base_name}_Unit'
        if hist_unit_col not in df_comparison.columns:
            print(f"  Warning: Historical unit cost column '{hist_unit_col}' not found. Assuming 0 for this component.")
            df_comparison[hist_unit_col] = 0.0 # Ensure column exists if logic tries to access it
        df_comparison[hist_unit_col] = df_comparison[hist_unit_col].fillna(0.0)
        
        is_enabled = str(costs_config_compare.get(comp_base_name, "True")).lower() == "true"
        if is_enabled:
            df_comparison['Historical_EffectiveCostPerUnit_USD_CurrentContext'] += df_comparison[hist_unit_col]
            active_hist_components_compare.append(comp_base_name)
    
    df_comparison['Historical_Total_Effective_Cost_for_Combo'] = \
        df_comparison['Historical_Allocated_Quantity'] * df_comparison['Historical_EffectiveCostPerUnit_USD_CurrentContext']
    if active_hist_components_compare:
        print(f"  Historical Effective Cost recalculated using: {active_hist_components_compare}")
    else:
        print("  Warning: All historical cost components disabled/missing for current context. Historical Effective Cost will be 0.")


    # Recalculate Optimized Effective Cost Per Unit for Current Context
    df_comparison['Optimized_EffectiveCostPerUnit_USD_CurrentContext'] = 0.0
    active_opt_components_compare = []
    for comp_base_name in COST_COMPONENTS_BASE:
        opt_unit_col = f'Optimized_{comp_base_name}_Unit'
        if opt_unit_col not in df_comparison.columns:
            print(f"  Warning: Optimized unit cost column '{opt_unit_col}' not found. Assuming 0 for this component.")
            df_comparison[opt_unit_col] = 0.0 # Ensure column exists
        df_comparison[opt_unit_col] = df_comparison[opt_unit_col].fillna(0.0)

        is_enabled = str(costs_config_compare.get(comp_base_name, "True")).lower() == "true"
        if is_enabled:
            df_comparison['Optimized_EffectiveCostPerUnit_USD_CurrentContext'] += df_comparison[opt_unit_col]
            active_opt_components_compare.append(comp_base_name)

    # The original 'Optimized_Total_Effective_Cost_for_Combo' comes from the optimizer's output,
    # which used the EffectiveCost from the *evaluation* step.
    # For the *comparison* context, we should recalculate it based on current toggles.
    df_comparison['Optimized_Total_Effective_Cost_for_Combo'] = \
        df_comparison['Optimized_Allocated_Quantity'] * df_comparison['Optimized_EffectiveCostPerUnit_USD_CurrentContext']
    if active_opt_components_compare:
        print(f"  Optimized Effective Cost recalculated using: {active_opt_components_compare}")
    else:
        print("  Warning: All optimized cost components disabled/missing for current context. Optimized Effective Cost will be 0.")
    
    # Remove the old 'Calculated_Optimized_Total_Effective_Cost_for_Combo' if it exists from previous logic
    if 'Calculated_Optimized_Total_Effective_Cost_for_Combo' in df_comparison.columns:
        df_comparison.drop(columns=['Calculated_Optimized_Total_Effective_Cost_for_Combo'], inplace=True)
    # The np.where logic for Optimized_Total_Effective_Cost_for_Combo is no longer needed as we are explicitly recalculating.


    # Calculate total for each cost component (Historical and Optimized)
    for comp_base_name in COST_COMPONENTS_BASE:
        hist_unit_col = f'Historical_{comp_base_name}_Unit'
        opt_unit_col = f'Optimized_{comp_base_name}_Unit'
        
        # Fill NaNs in unit cost component columns before multiplication
        df_comparison[hist_unit_col] = df_comparison[hist_unit_col].fillna(0)
        df_comparison[opt_unit_col] = df_comparison[opt_unit_col].fillna(0)

        df_comparison[f'Historical_Total_{comp_base_name}'] = df_comparison['Historical_Allocated_Quantity'] * df_comparison[hist_unit_col]
        df_comparison[f'Optimized_Total_{comp_base_name}'] = df_comparison['Optimized_Allocated_Quantity'] * df_comparison[opt_unit_col]

    # Deltas
    df_comparison['Delta_Allocated_Quantity'] = df_comparison['Optimized_Allocated_Quantity'] - df_comparison['Historical_Allocated_Quantity']
    df_comparison['Delta_Total_Effective_Cost_for_Combo'] = df_comparison['Optimized_Total_Effective_Cost_for_Combo'] - df_comparison['Historical_Total_Effective_Cost_for_Combo']

    for comp_base_name in COST_COMPONENTS_BASE:
        df_comparison[f'Delta_Total_{comp_base_name}'] = df_comparison[f'Optimized_Total_{comp_base_name}'] - df_comparison[f'Historical_Total_{comp_base_name}']

    # --- 6. Aggregate Totals for Summary ---
    total_historical_allocated_quantity = df_comparison['Historical_Allocated_Quantity'].sum()
    total_optimized_allocated_quantity = df_comparison['Optimized_Allocated_Quantity'].sum() # Should match target demands sum

    total_historical_actual_spend_usd = df_comparison['Historical_Actual_Spend_USD'].sum()
    # Optimized actual spend can be estimated if needed: Sum(Optimized_Allocated_Quantity * Optimized_AvgUnitPriceUSD_raw)
    df_comparison['Optimized_Actual_Spend_USD_Est_for_Combo'] = df_comparison['Optimized_Allocated_Quantity'] * df_comparison['Optimized_AvgUnitPriceUSD_raw'].fillna(0)
    total_optimized_actual_spend_usd_est = df_comparison['Optimized_Actual_Spend_USD_Est_for_Combo'].sum()


    total_historical_effective_cost = df_comparison['Historical_Total_Effective_Cost_for_Combo'].sum()
    total_optimized_effective_cost = df_comparison['Optimized_Total_Effective_Cost_for_Combo'].sum()

    net_economic_saving_usd = total_historical_effective_cost - total_optimized_effective_cost
    percentage_saving = (net_economic_saving_usd / total_historical_effective_cost) * 100 if total_historical_effective_cost != 0 else 0

    # --- 7. Generate Summary Output ---
    print("\n--- Economic Impact Comparison Summary ---")
    print(f"Comparison Mode: {mode} (Grouping by {grouping_key_mapped}) for period of {DEMAND_PERIOD_DAYS} days")

    print(f"\nTotal Allocated Quantity:")
    print(f"  - Historical: {total_historical_allocated_quantity:,.2f} units")
    print(f"  - Optimized:  {total_optimized_allocated_quantity:,.2f} units")

    print(f"\nTotal Actual Spend (USD):")
    print(f"  - Historical:                          {total_historical_actual_spend_usd:,.2f}")
    print(f"  - Optimized (Estimated Base Price Spend): {total_optimized_actual_spend_usd_est:,.2f}")


    print(f"\nTotal Effective Cost (USD):")
    print(f"  - Historical: {total_historical_effective_cost:,.2f}") # This now reflects current context
    print(f"  - Optimized:  {total_optimized_effective_cost:,.2f}") # This now reflects current context
    print(f"  - Net Economic Saving: {net_economic_saving_usd:,.2f} ({percentage_saving:.2f}%)")

    print("\nBreakdown of Total Effective Cost Changes (Optimized - Historical) based on current cost configuration:") # Updated title
    print(f"{'Cost Component':<30} | {'Historical Total':>18} | {'Optimized Total':>18} | {'Change':>18}")
    print("-" * 90)
    
    summary_total_historical_effective_cost_breakdown = 0.0 # For verification against overall total
    summary_total_optimized_effective_cost_breakdown = 0.0

    for comp_base_name in COST_COMPONENTS_BASE:
        is_enabled = str(costs_config_compare.get(comp_base_name, "True")).lower() == "true"
        
        hist_total_comp_col = f'Historical_Total_{comp_base_name}'
        opt_total_comp_col = f'Optimized_Total_{comp_base_name}'

        # Ensure these columns exist from earlier calculation (they should)
        # df_comparison[f'Historical_Total_{comp_base_name}'] = df_comparison['Historical_Allocated_Quantity'] * df_comparison[hist_unit_col]
        # df_comparison[f'Optimized_Total_{comp_base_name}'] = df_comparison['Optimized_Allocated_Quantity'] * df_comparison[opt_unit_col]
        # These calculations should still happen for ALL components, but we only display/sum enabled ones here.

        if is_enabled:
            hist_total_comp = df_comparison[hist_total_comp_col].sum() if hist_total_comp_col in df_comparison else 0.0
            opt_total_comp = df_comparison[opt_total_comp_col].sum() if opt_total_comp_col in df_comparison else 0.0
            delta_total_comp = opt_total_comp - hist_total_comp
            print(f"{comp_base_name:<30} | {hist_total_comp:>18,.2f} | {opt_total_comp:>18,.2f} | {delta_total_comp:>+18,.2f}")
            summary_total_historical_effective_cost_breakdown += hist_total_comp
            summary_total_optimized_effective_cost_breakdown += opt_total_comp
        else:
            print(f"{comp_base_name:<30} | {'(disabled)':>18} | {'(disabled)':>18} | {'(disabled)':>18}")
    print("-" * 90)
    print(f"{'SUM OF ACTIVE COMPONENTS':<30} | {summary_total_historical_effective_cost_breakdown:>18,.2f} | {summary_total_optimized_effective_cost_breakdown:>18,.2f} | {(summary_total_optimized_effective_cost_breakdown - summary_total_historical_effective_cost_breakdown):>+18,.2f}")


    # --- 8. Add Descriptions and Save Detailed Output ---
    print("\nSaving detailed allocation comparison...")
    # Merge supplier descriptions
    df_comparison = pd.merge(df_comparison, df_supplier_desc, on=col_supplier_id, how='left')
    df_comparison[col_supplier_name] = df_comparison[col_supplier_name].fillna('Unknown Vendor')

    # Merge item/group descriptions
    # df_desc_data has desc_key_mapped and desc_col_mapped
    # Need to ensure grouping_key_mapped (from df_comparison) can join with desc_key_mapped
    # If grouping_key_mapped IS desc_key_mapped (e.g. mode=matkl), simple merge.
    # If mode=maktx, grouping_key_mapped IS desc_col_mapped, desc_key_mapped IS mat_num. Merge on mat_num.
    # If mode=matnr, grouping_key_mapped IS mat_num, desc_key_mapped IS mat_num. Merge on mat_num.

    # Using col_mat_num as the reliable key to get desc_col_mapped (MAKTX) when it's different from grouping_key_mapped
    if desc_col_mapped != grouping_key_mapped: # e.g. mode=matnr, desc_col_mapped=MAKTX, grouping_key_mapped=MATNR
                                              # or mode=matkl, desc_col_mapped=MATKL_DESC, grouping_key_mapped=MATKL
        # df_desc_data is keyed by desc_key_mapped.
        # If desc_key_mapped is MATNR, and grouping_key_mapped is also MATNR (mode=matnr)
        # or desc_key_mapped is MATKL, and grouping_key_mapped is also MATKL (mode=matkl)
        # If desc_key_mapped is MATNR, and grouping_key_mapped is MAKTX (mode=maktx)
        #   Then we need to merge df_desc_data (keyed by MATNR) onto df_comparison (which has MATNR)
        #   to get the MAKTX (desc_col_mapped)
        
        # Standardize: always ensure df_comparison has desc_col_mapped if it's different from grouping_key_mapped.
        # df_desc_data is already unique on desc_key_mapped.
        # If desc_col_mapped is present in df_comparison from earlier merges, ensure it's filled.
        # Otherwise, merge it in.
        
        # If MAKTX is the grouping key, it's already there.
        # If MATNR is the grouping key, we need to bring in MAKTX (desc_col_mapped) using MATNR (desc_key_mapped)
        # If MATKL is the grouping key, we need to bring in MATKL_DESC (desc_col_mapped) using MATKL (desc_key_mapped)

        if desc_col_mapped not in df_comparison.columns:
            # This merge assumes df_desc_data is keyed by `desc_key_mapped` and contains `desc_col_mapped`.
            # `df_comparison` must contain `desc_key_mapped` (e.g., MATNR or MATKL) to join.
            if desc_key_mapped in df_comparison.columns:
                temp_desc_merge = df_desc_data[[desc_key_mapped, desc_col_mapped]].copy()
                df_comparison = pd.merge(df_comparison, temp_desc_merge, on=desc_key_mapped, how='left', suffixes=('', '_desc_final'))
                if desc_col_mapped + '_desc_final' in df_comparison.columns: # if original desc_col_mapped existed
                    df_comparison[desc_col_mapped] = df_comparison[desc_col_mapped + '_desc_final']
                    df_comparison.drop(columns=[desc_col_mapped + '_desc_final'], inplace=True)
            else:
                print(f"Warning: Cannot merge descriptions. Key '{desc_key_mapped}' not in df_comparison.")
        
    if desc_col_mapped in df_comparison.columns:
         df_comparison[desc_col_mapped] = df_comparison[desc_col_mapped].fillna(f'Unknown {mode.capitalize()}')
    elif grouping_key_mapped in df_comparison.columns : # if desc_col_mapped IS grouping_key_mapped (e.g. MAKTX mode)
         df_comparison[grouping_key_mapped] = df_comparison[grouping_key_mapped].fillna(f'Unknown {mode.capitalize()}')


    # Select and order columns for output
    output_cols_keys = [col_supplier_id, col_supplier_name, grouping_key_mapped]
    if desc_col_mapped != grouping_key_mapped and desc_col_mapped in df_comparison.columns:
        output_cols_keys.append(desc_col_mapped)
    output_cols_keys.append(col_mat_num)

    output_cols_quantities = ['Historical_Allocated_Quantity', 'Optimized_Allocated_Quantity', 'Delta_Allocated_Quantity']
    output_cols_spend = ['Historical_Actual_Spend_USD', 'Optimized_Actual_Spend_USD_Est_for_Combo']
    
    output_cols_eff_cost_total = ['Historical_Total_Effective_Cost_for_Combo',
                                  'Optimized_Total_Effective_Cost_for_Combo',
                                  'Delta_Total_Effective_Cost_for_Combo']
    output_cols_eff_cost_unit = ['Historical_EffectiveCostPerUnit_USD', 'Optimized_EffectiveCostPerUnit_USD']
    output_cols_avg_price_unit = ['Historical_AvgUnitPriceUSD_raw', 'Optimized_AvgUnitPriceUSD_raw']


    output_cols_hist_comp_total = [f'Historical_Total_{comp}' for comp in COST_COMPONENTS_BASE]
    output_cols_opt_comp_total = [f'Optimized_Total_{comp}' for comp in COST_COMPONENTS_BASE]
    output_cols_delta_comp_total = [f'Delta_Total_{comp}' for comp in COST_COMPONENTS_BASE]

    output_cols_hist_comp_unit = [f'Historical_{comp}_Unit' for comp in COST_COMPONENTS_BASE]
    output_cols_opt_comp_unit = [f'Optimized_{comp}_Unit' for comp in COST_COMPONENTS_BASE]


    final_output_cols = output_cols_keys + \
                        output_cols_quantities + \
                        output_cols_spend + \
                        output_cols_eff_cost_total + \
                        output_cols_eff_cost_unit + \
                        output_cols_avg_price_unit + \
                        output_cols_hist_comp_total + \
                        output_cols_opt_comp_total + \
                        output_cols_delta_comp_total + \
                        output_cols_hist_comp_unit + \
                        output_cols_opt_comp_unit

    # Filter for rows with some activity
    df_comp_output = df_comparison[
        (df_comparison['Historical_Allocated_Quantity'].abs() > 1e-6) |
        (df_comparison['Optimized_Allocated_Quantity'].abs() > 1e-6)
    ].copy()

    # Ensure all selected columns exist before saving and remove duplicates
    final_output_cols_present = []
    for col in final_output_cols:
        if col in df_comp_output.columns and col not in final_output_cols_present:
            final_output_cols_present.append(col)
        elif col not in df_comp_output.columns:
            print(f"Warning: Output column '{col}' not found in df_comp_output.")


    df_comp_output = df_comp_output[final_output_cols_present]
    df_comp_output.sort_values(by=[grouping_key_mapped, col_mat_num, 'Delta_Total_Effective_Cost_for_Combo'],
                               ascending=[True, True, True], inplace=True)

    try:
        df_comp_output.to_csv(comparison_output_path, index=False, float_format='%.2f')
        print(f"Detailed comparison saved successfully to: {comparison_output_path}")
    except KeyError as e:
         print(f"\nError: Missing column in output selection: {e}", file=sys.stderr)
         print("Available columns:", df_comp_output.columns.tolist())
    except Exception as e:
        print(f"Error saving comparison results to {comparison_output_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
