import pandas as pd
import numpy as np
from scipy import stats # For linear regression (price trend)
import argparse
import os
import sys
import json # Added for loading mappings
from datetime import datetime, timedelta

# --- Configuration ---
# Constants
PRICE_TREND_CUTOFF_DAYS = 180 # Days to consider for price trend calculation

# Economic Impact Parameters (EIPs)
EIP_ANNUAL_HOLDING_COST_RATE = 0.18 # Example: 18%
EIP_DailyHoldingCostRate_Param = EIP_ANNUAL_HOLDING_COST_RATE / 365.0
EIP_SafetyStockMultiplierForLTVar_Param = 1.65 # Corresponds to ~95% CSL
# EIP_AvgDaysLateIfLate_Param will be calculated from data
EIP_RiskPremiumFactorForPriceVolatility_Param = 0.25 # Example: 25%
EIP_PlanningHorizonDaysForPriceTrend_Param = 90 # Example: 90 days

# Currency Conversion Rates (Keep for now, could be externalized later)
EXCHANGE_RATES = {
    'EUR': 1.14, # 1 EUR = 1.14 USD
    'MXN': 0.051, # 1 MXN = 0.051 USD
    'USD': 1.00   # Base currency
    # Add other currencies present in your WAERS column if needed
}

# Define Weights (example, should sum to 1.0 ideally, but can be adjusted)
# These can be loaded from a file or passed as args later
METRIC_WEIGHTS = {
    'AvgUnitPriceUSD_Norm': 0.20,
    'PriceVolatility_Norm': 0.15,
    'PriceTrend_Norm': 0.10,
    'TariffImpact_Norm': 0.15,
    'AvgLeadTimeDays_Norm': 0.10,
    'LeadTimeVariability_Norm': 0.10,
    'OnTimeRate_Norm': 0.10,
    'InFullRate_Norm': 0.10,
}

# Define which metrics are "lower is better" for normalization inversion
LOWER_IS_BETTER_METRICS = [
    'AvgUnitPriceUSD_Norm',
    'PriceVolatility_Norm',
    'PriceTrend_Norm', # More negative slope is better, so treat like lower is better
    'TariffImpact_Norm',
    'AvgLeadTimeDays_Norm',
   'LeadTimeVariability_Norm'
]

# --- Helper Functions ---

def load_data(file_path, usecols=None, dtype=None):
    """Loads a CSV file with basic error handling."""
    # Keep this function as is, path construction happens before calling it.
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

def convert_to_usd(row, mapped_price_col, mapped_currency_col):
    """Applies exchange rate to convert price to USD using mapped column names."""
    price = row[mapped_price_col]
    currency = row[mapped_currency_col]
    rate = EXCHANGE_RATES.get(currency, 1.0) # Default to 1.0 if currency not in map
    if pd.isna(price) or pd.isna(currency): # Check currency NaN too
        return np.nan
    return price * rate

def calculate_price_trend(group, mapped_price_col, mapped_date_col):
    """Calculates the slope of unit price over time using linear regression for a group."""
    series = group[mapped_price_col]
    dates = group[mapped_date_col]
    # Ensure we have enough data points for regression and they are numeric
    valid_indices = pd.notna(series) & pd.notna(dates)
    series_clean = series[valid_indices]
    dates_clean = dates[valid_indices]

    if len(series_clean) < 3: # Need at least 3 points for a meaningful trend
        return 0.0 # Or np.nan, decide how to handle insufficient data

    # Convert dates to numeric representation (e.g., days since first date)
    time_delta = (dates_clean - dates_clean.min()).dt.days

    # Ensure time_delta has variance
    if time_delta.nunique() < 2:
         return 0.0 # Cannot calculate trend if all dates are the same

    try:
        # Perform linear regression: price = slope * time + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_delta, series_clean)
        # We only care about the slope
        return slope if pd.notna(slope) else 0.0
    except Exception as e:
        print(f"Warning: Linregress failed for group - {e}. Returning 0 trend.", file=sys.stderr)
        return 0.0

def normalize_min_max(series, lower_is_better=False):
    """Normalizes a series using Min-Max scaling (0-1 range)."""
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        # Handle cases with NaNs or all values being the same
        return 0.5 # Assign a neutral score or choose another default (e.g., 0)

    if lower_is_better:
        # Invert the scale: higher values get lower scores (closer to 0)
        normalized = (max_val - series) / (max_val - min_val)
    else:
        # Standard scale: higher values get higher scores (closer to 1)
        normalized = (series - min_val) / (max_val - min_val)

    return normalized

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Rank Vendor/Material Group, Vendor/Material Number, or Vendor/Material Description combinations based on performance metrics.")
    # Updated arguments
    parser.add_argument('--tariff-results-path', required=True, help='Full path to the tariff calculation script output CSV.')
    parser.add_argument('--tables-dir', required=True, help='Directory containing the mapped table CSV files.')
    parser.add_argument('--ranking-output-dir', required=True, help='Directory to save the ranking results CSV.')
    parser.add_argument('--table-map', required=True, help='Path to the JSON file mapping table concepts to filenames.')
    parser.add_argument('--column-map', required=True, help='Path to the JSON file mapping column concepts to actual column names.')
    parser.add_argument('--mode', required=True, choices=['matkl', 'matnr', 'maktx'], help="Grouping mode: 'matkl' (Material Group), 'matnr' (Material Number), or 'maktx' (Material Description).")
    parser.add_argument('--metric-weights', required=False, help='JSON string mapping metric names to weights.')
    parser.add_argument('--costs-config-path', required=True, help='Path to the JSON file specifying which costs to include.')

    args = parser.parse_args()

    # Override default metric weights if provided
    if args.metric_weights:
        try:
            # METRIC_WEIGHTS = json.loads(args.metric_weights)  # Loads JSON string into a dictionary
            with open(args.metric_weights, 'r') as f:
                # Assuming the file contains a JSON object
                METRIC_WEIGHTS = json.load(f)
        except json.JSONDecodeError:
            print("Error: Could not parse JSON for metric_weights", file=sys.stderr)
            sys.exit(1)

    # Load cost configuration
    costs_config = {}
    try:
        with open(args.costs_config_path, 'r') as f:
            config_data = json.load(f)
            # Handle both old and new format
            if 'cost_components' in config_data:
                costs_config = config_data['cost_components']
            else:
                costs_config = config_data
        print(f"Loaded cost configuration from: {args.costs_config_path}")
    except FileNotFoundError:
        print(f"Warning: Costs configuration file not found at {args.costs_config_path}. All cost components will be enabled by default.", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {args.costs_config_path}. All cost components will be enabled by default.", file=sys.stderr)
    # Defaulting to True for all known components if file is missing/corrupt is handled in the calculation loop

    mode = args.mode
    tables_dir = args.tables_dir
    ranking_output_dir = args.ranking_output_dir
    tariff_results_path = args.tariff_results_path # Full path now
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
    os.makedirs(ranking_output_dir, exist_ok=True)

    # --- Get mapped column names ---
    col_po_num = column_map.get('PO_Number') or 'EBELN'
    col_po_item = column_map.get('PO_Item_Number') or 'EBELP'
    col_mat_num = column_map.get('Material_ID') or 'MATNR'
    col_mat_group = column_map.get('Material_Group_Code') or 'MATKL'
    col_item_qty = column_map.get('Item_Ordered_Quantity') or 'MENGE' # Used in items and history
    col_net_price = column_map.get('Net_Price_Per_Unit') or 'NETPR'
    col_price_unit = column_map.get('Price_Unit_Quantity') or 'PEINH'
    col_net_value = column_map.get('Item_Net_Order_Value') or 'NETWR'
    col_supplier_id = column_map.get('Supplier_ID') or 'LIFNR'
    col_po_date = column_map.get('PO_Creation_Date') or 'BEDAT'
    col_currency = column_map.get('PO_Currency_Code') or 'WAERS'
    col_hist_event_type = column_map.get('PO_History_Transaction_Type') or 'VGABE'
    col_hist_post_date = column_map.get('History_Posting_Date') or 'BUDAT'
    col_hist_qty = column_map.get('History_Quantity_Base_UoM') or 'MENGE' # Note: Same conceptual name as item qty
    col_plant = column_map.get('Plant_Code') or 'WERKS'
    col_sched_line_num = column_map.get('Schedule_Line_Number') or 'ETENR'
    col_sched_deliv_date = column_map.get('Scheduled_Delivery_Date') or 'EINDT'
    col_supplier_name = column_map.get('Supplier_Name_1') or 'NAME1'
    col_mat_group_desc = column_map.get('Material_Group_Description') or 'MATKL_DESC'
    col_mat_desc = column_map.get('Material_Description') or 'MAKTX'
    col_tariff_percent = column_map.get('Cumulative_Tariff_Percent') or 'Cumulative_Tariff_Percent' # From tariff results
    col_order_unit = column_map.get('Order_Unit') or 'MEINS' # For MEINS check

    # Determine grouping key and description column based on mode and mappings
    if mode == 'matkl':
        grouping_key_mapped = col_mat_group
        desc_key_mapped = col_mat_group # Key to join description table
        desc_col_mapped = col_mat_group_desc
        desc_concept = 'Material Group'
    elif mode == 'matnr':
        grouping_key_mapped = col_mat_num
        desc_key_mapped = col_mat_num
        desc_col_mapped = col_mat_desc
        desc_concept = 'Material'
    elif mode == 'maktx':
        grouping_key_mapped = col_mat_desc # Group by description
        desc_key_mapped = col_mat_num     # Still need MATNR to link description file
        desc_col_mapped = col_mat_desc    # The description column is the grouping key itself
        desc_concept = 'Material' # File containing MAKTX
    else:
        # Should not happen due to argparse choices
        sys.exit(1)

    print(f"Running evaluation in mode: '{mode}', grouping by mapped key: '{grouping_key_mapped}'")

    # --- 1. Load Data using Mappings ---
    # Define required conceptual tables and their essential columns (using mapped names)
    # Ensure col_order_unit is loaded for PO Items
    po_items_cols_to_load = [col_po_num, col_po_item, col_mat_num, col_mat_group, col_item_qty, col_net_price, col_price_unit, col_net_value, col_order_unit]
    # Remove duplicates just in case col_order_unit was already in one of the other mapped variables
    po_items_cols_to_load = sorted(list(set(po_items_cols_to_load)))


    required_data = {
        'PO Items': {'file': table_map.get('SAP_VLY_IL_PO_ITEMS.csv', 'SAP_VLY_IL_PO_ITEMS.csv'), 'cols': po_items_cols_to_load},
        'PO Header': {'file': table_map.get('SAP_VLY_IL_PO_HEADER.csv', 'SAP_VLY_IL_PO_HEADER.csv'), 'cols': [col_po_num, col_supplier_id, col_po_date, col_currency]},
        'PO History': {'file': table_map.get('SAP_VLY_IL_PO_HISTORY.csv', 'SAP_VLY_IL_PO_HISTORY.csv'), 'cols': [col_po_num, col_po_item, col_hist_event_type, col_hist_post_date, col_hist_qty, col_plant]},
        'PO Schedule Lines': {'file': table_map.get('SAP_VLY_IL_PO_SCHEDULE_LINES.csv', 'SAP_VLY_IL_PO_SCHEDULE_LINES.csv'), 'cols': [col_po_num, col_po_item, col_sched_line_num, col_sched_deliv_date]},
        'Supplier': {'file': table_map.get('SAP_VLY_IL_SUPPLIER.csv', 'SAP_VLY_IL_SUPPLIER.csv'), 'cols': [col_supplier_id, col_supplier_name]},
        desc_concept: {'file': table_map.get('SAP_VLY_IL_MATERIAL_GROUP.csv' if desc_concept == 'Material Group' else 'SAP_VLY_IL_MATERIAL.csv', 'SAP_VLY_IL_MATERIAL_GROUP.csv' if desc_concept == 'Material Group' else 'SAP_VLY_IL_MATERIAL.csv'), 'cols': [desc_key_mapped, desc_col_mapped]},
        'Tariff Results': {'file': os.path.basename(tariff_results_path), 'cols': [col_po_num, col_po_item, col_tariff_percent]} # Use basename for tariff file
    }

    dfs_loaded = {}
    for concept, info in required_data.items():
        # Skip missing mapping; default filename used if mapping value is empty or missing
        # Construct full path based on source (tables_dir or tariff_results_path)
        if concept == 'Tariff Results':
            file_path = tariff_results_path # Use the full path provided
        else:
            file_path = os.path.join(tables_dir, info['file'])

        # Remove any empty column names from mapping before checking/loading
        info['cols'] = [col for col in info['cols'] if col]
        # Ensure required columns exist before loading
        try:
             # Quick header check without loading full file
             header_df = pd.read_csv(file_path, nrows=0, low_memory=False, encoding='utf-8-sig')
             missing_cols = [col for col in info['cols'] if col not in header_df.columns]
             if missing_cols:
                  print(f"Error: Required mapped columns missing in file '{info['file']}' for concept '{concept}': {missing_cols}", file=sys.stderr)
                  sys.exit(1)
        except FileNotFoundError:
             print(f"Error: File not found for concept '{concept}': {file_path}", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
             print(f"Error checking header for file '{info['file']}': {e}", file=sys.stderr)
             sys.exit(1)

        # Load the data
        dfs_loaded[concept] = load_data(file_path, usecols=info['cols'])
        # Special handling for description table drop_duplicates
        if concept == desc_concept:
             dfs_loaded[concept] = dfs_loaded[concept].drop_duplicates(subset=[desc_key_mapped])
        elif concept == 'Supplier':
             dfs_loaded[concept] = dfs_loaded[concept].drop_duplicates(subset=[col_supplier_id])


    # Assign loaded dataframes
    df_items = dfs_loaded['PO Items']
    df_header = dfs_loaded['PO Header']
    df_history = dfs_loaded['PO History']
    df_schedule = dfs_loaded['PO Schedule Lines']
    df_supplier = dfs_loaded['Supplier']
    df_desc = dfs_loaded[desc_concept]
    df_tariff = dfs_loaded['Tariff Results']

    # --- 2. Clean and Prepare Data ---
    print("Cleaning and preparing data...")

    # Convert keys to string for reliable merging (using mapped names)
    # Ensure col_order_unit is also converted if it's a key-like field, though typically it's a descriptor
    key_cols_to_convert_to_str = [col_po_num, col_po_item, col_supplier_id, col_mat_group, col_mat_num, col_sched_line_num, col_order_unit]
    dfs_list = [df_items, df_header, df_history, df_schedule, df_supplier, df_desc, df_tariff]
    for df in dfs_list:
        for col in key_cols_to_convert_to_str: # Use the new list
            if col in df.columns:
                # Handle potential float64 if IDs were numeric but read as float due to NaNs elsewhere
                if df[col].dtype == 'float64':
                    # Attempt conversion to Int64 (nullable int) first, then to str
                    # This handles cases like '123.0' becoming '123'
                    try:
                        df[col] = df[col].astype(pd.Int64Dtype()).astype(str)
                    except: # Fallback if direct Int64 conversion fails
                        df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(str)

    # Convert dates (using mapped names)
    df_header[col_po_date] = pd.to_datetime(df_header[col_po_date], errors='coerce')
    df_history[col_hist_post_date] = pd.to_datetime(df_history[col_hist_post_date], errors='coerce')
    df_schedule[col_sched_deliv_date] = pd.to_datetime(df_schedule[col_sched_deliv_date], errors='coerce')

    # Convert numeric (using mapped names)
    num_cols_items_mapped = [col_item_qty, col_net_price, col_price_unit, col_net_value]
    for col in num_cols_items_mapped:
        if col in df_items.columns: # Check if column exists (might not if not mapped/used)
            df_items[col] = pd.to_numeric(df_items[col], errors='coerce')

    if col_hist_qty in df_history.columns:
        df_history[col_hist_qty] = pd.to_numeric(df_history[col_hist_qty], errors='coerce')
    if col_tariff_percent in df_tariff.columns:
        df_tariff[col_tariff_percent] = pd.to_numeric(df_tariff[col_tariff_percent], errors='coerce')

    # Filter bad data needed for calculations (using mapped names)
    required_cols_items_mapped = [col_po_num, col_po_item, col_mat_num, col_net_price, col_price_unit, col_item_qty]
    if mode == 'matkl':
        required_cols_items_mapped.append(col_mat_group)

    if not all(col in df_items.columns for col in required_cols_items_mapped):
         missing_cols = [col for col in required_cols_items_mapped if col not in df_items.columns]
         print(f"Error: Required mapped columns missing in PO Items data: {missing_cols}", file=sys.stderr)
         sys.exit(1)

    df_items.dropna(subset=required_cols_items_mapped, inplace=True)
    # Use mapped columns for filtering
    if col_price_unit in df_items.columns: df_items = df_items[df_items[col_price_unit] > 0]
    if col_item_qty in df_items.columns: df_items = df_items[df_items[col_item_qty] > 0]
    df_header.dropna(subset=[col_po_num, col_supplier_id, col_po_date, col_currency], inplace=True)
    df_history.dropna(subset=[col_po_num, col_po_item, col_hist_post_date, col_hist_qty, col_hist_event_type], inplace=True)
    df_schedule.dropna(subset=[col_po_num, col_po_item, col_sched_deliv_date], inplace=True)
    df_tariff.dropna(subset=[col_po_num, col_po_item, col_tariff_percent], inplace=True)

    # Filter history for Goods Receipts (using mapped column)
    df_gr = df_history[df_history[col_hist_event_type].astype(str) == '1'].copy()
    if df_gr.empty:
         print("Warning: No Goods Receipt records found (Event Type='1'). Lead Time and Delivery metrics will be impacted.", file=sys.stderr)

    # --- 3. Merge Data (using mapped keys) ---
    print("Merging data sources...")
    # Merge items with descriptions first to get grouping key for 'maktx' mode before MEINS check
    print(f"  Merging PO Items with {desc_concept} descriptions...")
    df_items_with_desc = pd.merge(df_items, df_desc, on=desc_key_mapped, how='left')
    # Fill description if missing - important for 'maktx' mode where desc is the key
    df_items_with_desc[desc_col_mapped] = df_items_with_desc[desc_col_mapped].fillna(f'Unknown {mode.capitalize()}')

    # MEINS Consistency Check
    if mode in ['matnr', 'maktx']:
        print(f"Performing MEINS consistency check for mode '{mode}' using column '{col_order_unit}' grouped by '{grouping_key_mapped}'...")
        # Ensure the necessary columns exist in the merged df_items_with_desc
        if col_order_unit in df_items_with_desc.columns and grouping_key_mapped in df_items_with_desc.columns:
            # Drop rows where grouping key or order unit is NaN before checking consistency
            df_check = df_items_with_desc.dropna(subset=[grouping_key_mapped, col_order_unit])
            if not df_check.empty:
                meins_consistency = df_check.groupby(grouping_key_mapped)[col_order_unit].nunique()
                inconsistent_groups = meins_consistency[meins_consistency > 1].index
                if not inconsistent_groups.empty:
                    print(f"Warning: The following {mode} groups have multiple MEINS values and will be excluded from the analysis: {list(inconsistent_groups)}")
                    # Filter the df_items_with_desc DataFrame
                    df_items_with_desc = df_items_with_desc[~df_items_with_desc[grouping_key_mapped].isin(inconsistent_groups)].copy()
                    print(f"Filtered items data to {len(df_items_with_desc)} rows after MEINS check.")
                else:
                    print("MEINS consistency check passed. No groups excluded.")
            else:
                 print("Warning: No valid data for MEINS consistency check after dropping NaNs.")
        else:
            print(f"Warning: Cannot perform MEINS consistency check. Column '{col_order_unit}' or '{grouping_key_mapped}' not found after merging items with descriptions.")
            # Depending on requirements, might exit or proceed cautiously
            sys.exit("Exiting due to missing columns for MEINS check.")

    # Proceed with merging the (potentially filtered) items_with_desc with other data
    print("  Merging with PO Header...")
    df_merged = pd.merge(df_items_with_desc, df_header, on=col_po_num, how='inner')

    print("  Merging with Aggregated Goods Receipts...")
    if not df_gr.empty:
        df_gr_agg = df_gr.groupby([col_po_num, col_po_item]).agg(
            FirstGRDate=(col_hist_post_date, 'min'),
            LastGRDate=(col_hist_post_date, 'max'),
            TotalDeliveredQty=(col_hist_qty, 'sum'),
            GRCount=(col_hist_post_date, 'count')
        ).reset_index()
        df_merged = pd.merge(df_merged, df_gr_agg, on=[col_po_num, col_po_item], how='left')
    else:
        # Ensure these columns exist even if no GR data
        df_merged['FirstGRDate'] = pd.NaT
        df_merged['LastGRDate'] = pd.NaT
        df_merged['TotalDeliveredQty'] = np.nan
        df_merged['GRCount'] = 0

    print("  Merging with Aggregated Schedule Lines...")
    # Find the earliest scheduled delivery date directly
    df_schedule_agg = df_schedule.groupby([col_po_num, col_po_item])[col_sched_deliv_date].min().reset_index()
    df_schedule_agg = df_schedule_agg.rename(columns={col_sched_deliv_date: 'EarliestEINDT'})
    df_merged = pd.merge(df_merged, df_schedule_agg, on=[col_po_num, col_po_item], how='left')

    print("  Merging with Supplier Info...")
    df_merged = pd.merge(df_merged, df_supplier, on=col_supplier_id, how='left')

    print("  Merging with Tariff Results...")
    df_merged = pd.merge(df_merged, df_tariff, on=[col_po_num, col_po_item], how='left')

    # Fill missing descriptions (using mapped names) - Supplier name only now
    df_merged[col_supplier_name] = df_merged[col_supplier_name].fillna('Unknown Vendor')
    # Item/Group description already filled earlier

    # --- 4. Calculate Metrics per Line Item (using mapped names) ---
    print("Calculating metrics per line item...")

    # Calculate Unit Price
    df_merged['UnitPriceRaw'] = df_merged[col_net_price] / df_merged[col_price_unit]

    # Convert Unit Price to USD
    df_merged['UnitPriceUSD'] = df_merged.apply(convert_to_usd, axis=1, mapped_price_col='UnitPriceRaw', mapped_currency_col=col_currency)

    # Calculate Lead Time (days)
    df_merged['LeadTimeDays'] = np.nan
    valid_lt_idx = df_merged['FirstGRDate'].notna() & df_merged[col_po_date].notna()
    df_merged.loc[valid_lt_idx, 'LeadTimeDays'] = (df_merged.loc[valid_lt_idx, 'FirstGRDate'] - df_merged.loc[valid_lt_idx, col_po_date]).dt.days
    df_merged.loc[df_merged['LeadTimeDays'] < 0, 'LeadTimeDays'] = 0

    # Calculate On-Time Status
    df_merged['IsOnTime'] = pd.Series(dtype='boolean')
    valid_ot_idx = df_merged['FirstGRDate'].notna() & df_merged['EarliestEINDT'].notna()
    df_merged.loc[valid_ot_idx, 'IsOnTime'] = df_merged.loc[valid_ot_idx, 'FirstGRDate'] <= df_merged.loc[valid_ot_idx, 'EarliestEINDT']

    # Calculate In-Full Status
    df_merged['IsInFull'] = pd.Series(dtype='boolean')
    valid_if_idx = df_merged['TotalDeliveredQty'].notna() & df_merged[col_item_qty].notna() & (df_merged[col_item_qty] > 0)
    df_merged.loc[valid_if_idx, 'IsInFull'] = df_merged.loc[valid_if_idx, 'TotalDeliveredQty'] >= df_merged.loc[valid_if_idx, col_item_qty]

    # Drop rows where key metrics couldn't be calculated
    if grouping_key_mapped not in df_merged.columns:
         print(f"Error: Mapped grouping key '{grouping_key_mapped}' not found in merged data.", file=sys.stderr)
         sys.exit(1)
    df_merged.dropna(subset=[col_supplier_id, grouping_key_mapped, 'UnitPriceUSD'], inplace=True)

    # --- 4a. Calculate EIP_AvgDaysLateIfLate_Param from historical data ---
    print("Calculating EIP_AvgDaysLateIfLate_Param by material...")
    if 'FirstGRDate' in df_merged.columns and 'EarliestEINDT' in df_merged.columns:
        df_merged['DaysLate'] = (df_merged['FirstGRDate'] - df_merged['EarliestEINDT']).dt.days
        
        # Calculate by material based on grouping mode
        avg_days_late_by_material = df_merged[df_merged['DaysLate'] > 0].groupby(grouping_key_mapped)['DaysLate'].mean()
        
        # Create a mapping dictionary for lookup during cost calculations
        EIP_AvgDaysLateIfLate_Dict = avg_days_late_by_material.to_dict()
        
        # Also calculate global average as fallback
        EIP_AvgDaysLateIfLate_Param = df_merged[df_merged['DaysLate'] > 0]['DaysLate'].mean() if not df_merged[df_merged['DaysLate'] > 0].empty else 0.0
        
        print(f"  Calculated EIP_AvgDaysLateIfLate_Param by material ({len(EIP_AvgDaysLateIfLate_Dict)} materials)")
        print(f"  Global fallback value: {EIP_AvgDaysLateIfLate_Param:.2f} days")
    else:
        EIP_AvgDaysLateIfLate_Dict = {}
        EIP_AvgDaysLateIfLate_Param = 0.0 # Default if columns are missing
        print("  Warning: 'FirstGRDate' or 'EarliestEINDT' not in df_merged. EIP_AvgDaysLateIfLate_Param set to 0.0.")


    # --- 5. Aggregate Metrics per Vendor/Grouping Key (using mapped names) ---
    print(f"Aggregating metrics per Vendor/{grouping_key_mapped}...")

    # Define base aggregation functions
    agg_funcs_mapped = {
        'UnitPriceUSD': ['median', 'std', lambda x: x.std() / x.mean() if pd.notna(x.mean()) and x.mean() != 0 and pd.notna(x.std()) else 0], # median, std, CoV
        'LeadTimeDays': ['median', 'std'],
        'IsOnTime': 'mean',
        'IsInFull': 'mean',
        col_tariff_percent: 'mean',
        col_po_num: 'nunique' # Count unique POs for POLineItemCount
    }

    # Define base grouping columns
    groupby_cols_mapped = [col_supplier_id, col_supplier_name, grouping_key_mapped]

    # Add MATNR to grouping if it's not the primary grouping key, and add aggregation for it
    if col_mat_num != grouping_key_mapped:
        if col_mat_num in df_merged.columns:
            groupby_cols_mapped.append(col_mat_num)
            agg_funcs_mapped[col_mat_num] = 'first' # Assume MATNR is consistent within the group
        else:
            print(f"Warning: Column '{col_mat_num}' not found in df_merged for aggregation.")

    # Add description column to grouping if it's different from the primary grouping key AND MATNR
    if desc_col_mapped != grouping_key_mapped and desc_col_mapped != col_mat_num:
        if desc_col_mapped in df_merged.columns:
            groupby_cols_mapped.append(desc_col_mapped)
            # No specific aggregation needed if it's just for grouping/output,
            # but if we needed it aggregated, add: agg_funcs_mapped[desc_col_mapped] = 'first'
        else:
             print(f"Warning: Description column '{desc_col_mapped}' not found in df_merged for grouping.")

    # Remove duplicates from groupby_cols_mapped while preserving order
    groupby_cols_mapped = sorted(list(set(groupby_cols_mapped)), key=groupby_cols_mapped.index)

    print(f"  Grouping by: {groupby_cols_mapped}")
    print(f"  Aggregating: {list(agg_funcs_mapped.keys())}")

    # Perform aggregation using named aggregations for clear, flat output columns
    # The groupby_cols_mapped (e.g., LIFNR, NAME1, MAKTX, potentially MATNR) will be columns by default due to as_index=False.
    # We only need to define aggregations for other metrics.

    # Note: If col_mat_num or desc_col_mapped were added to groupby_cols_mapped,
    # they are already present as columns in the output of groupby.
    
    agg_definitions = {
        'AvgUnitPriceUSD_raw': pd.NamedAgg(column='UnitPriceUSD', aggfunc='median'),
        'StdDevUnitPriceUSD_raw': pd.NamedAgg(column='UnitPriceUSD', aggfunc='std'),
        'PriceVolatility_raw': pd.NamedAgg(
            column='UnitPriceUSD',
            aggfunc=lambda x: x.std() / x.mean() if pd.notna(x.mean()) and x.mean() != 0 and pd.notna(x.std()) else 0
        ),
        'AvgLeadTimeDays_raw': pd.NamedAgg(column='LeadTimeDays', aggfunc='median'),
        'LeadTimeVariability_raw_days': pd.NamedAgg(column='LeadTimeDays', aggfunc='std'),
        'OnTimeRate_raw': pd.NamedAgg(column='IsOnTime', aggfunc='mean'),
        'InFullRate_raw': pd.NamedAgg(column='IsInFull', aggfunc='mean'),
        # Ensure col_tariff_percent is the correct column name from df_merged
        'TariffImpact_raw_percent': pd.NamedAgg(column=col_tariff_percent, aggfunc='mean'),
        # Ensure col_po_num is the correct column name from df_merged
        'POLineItemCount': pd.NamedAgg(column=col_po_num, aggfunc='nunique')
    }


    df_rank = df_merged.groupby(groupby_cols_mapped, as_index=False).agg(**agg_definitions)

    # Verify essential grouping columns are present
    # These should be directly from groupby_cols_mapped
    if col_supplier_id not in df_rank.columns:
         print(f"Error: '{col_supplier_id}' column is missing after aggregation.")
         print(f"Columns present: {list(df_rank.columns)}")
         sys.exit("Exiting due to missing supplier ID column.")
    if grouping_key_mapped not in df_rank.columns:
         print(f"Error: Grouping key '{grouping_key_mapped}' column is missing after aggregation.")
         print(f"Columns present: {list(df_rank.columns)}")
         sys.exit("Exiting due to missing grouping key column.")
    # If col_mat_num was added to groupby_cols_mapped, it should also be in df_rank.columns
    # If it was aggregated via agg_definitions, its key (e.g. col_mat_num itself) should be in df_rank.columns


    # Calculate Price Trend (using mapped columns) - result is PriceTrend_raw_slope
    print("Calculating price trend (last 6 months)...")
    # Ensure col_po_date is datetime for comparison
    df_merged[col_po_date] = pd.to_datetime(df_merged[col_po_date], errors='coerce')
    
    # Handle cases where df_merged[col_po_date] might be all NaT after coercion
    if df_merged[col_po_date].notna().any():
        cutoff_date = df_merged[col_po_date].max() - timedelta(days=PRICE_TREND_CUTOFF_DAYS)
        df_recent = df_merged[df_merged[col_po_date] >= cutoff_date].copy()
    else:
        print("Warning: PO Creation Date (BEDAT) is all NaT or missing. Price trend cannot be calculated.")
        df_recent = pd.DataFrame() # Empty DataFrame

    if not df_recent.empty:
        if grouping_key_mapped not in df_recent.columns:
            print(f"Error: Mapped grouping key '{grouping_key_mapped}' not found in recent data for price trend.", file=sys.stderr)
            # Attempt merge if mode is maktx and MATNR exists
            if mode == 'maktx' and col_mat_num in df_recent.columns:
                 print("Attempting to merge MAKTX into recent data for trend...")
                 desc_file_path = os.path.join(tables_dir, table_map.get(desc_concept))
                 df_desc_trend = load_data(desc_file_path, usecols=[col_mat_num, col_mat_desc]).drop_duplicates(subset=[col_mat_num])
                 df_desc_trend[col_mat_num] = df_desc_trend[col_mat_num].astype(str)
                 df_recent = pd.merge(df_recent, df_desc_trend, on=col_mat_num, how='left')
                 df_recent[desc_col_mapped] = df_recent[desc_col_mapped].fillna('Unknown Material')
                 if grouping_key_mapped not in df_recent.columns: # Check again
                      print("Error: Failed to merge MAKTX for trend calculation.", file=sys.stderr)
                      # Add a placeholder column to prevent crash, will be filled with 0
                      df_rank['PriceTrend_raw_slope'] = 0.0
                 else: # Mapped key now exists
                    price_trend_series = df_recent.groupby([col_supplier_id, grouping_key_mapped]).apply(
                        lambda g: calculate_price_trend(g, 'UnitPriceUSD', col_po_date),
                        include_groups=False
                    )
                    price_trend_df = price_trend_series.reset_index()
                    price_trend_df.rename(columns={0: 'PriceTrend_raw_slope'}, inplace=True)
                    df_rank = pd.merge(df_rank, price_trend_df, on=[col_supplier_id, grouping_key_mapped], how='left')

            else: # Grouping key not found and cannot be merged
                 df_rank['PriceTrend_raw_slope'] = 0.0 # Add placeholder
        else: # Grouping key was already in df_recent
            price_trend_series = df_recent.groupby([col_supplier_id, grouping_key_mapped]).apply(
                lambda g: calculate_price_trend(g, 'UnitPriceUSD', col_po_date),
                include_groups=False
            )
            price_trend_df = price_trend_series.reset_index()
            price_trend_df.rename(columns={0: 'PriceTrend_raw_slope'}, inplace=True)
            df_rank = pd.merge(df_rank, price_trend_df, on=[col_supplier_id, grouping_key_mapped], how='left')
    else: # df_recent is empty
        df_rank['PriceTrend_raw_slope'] = 0.0

    df_rank['PriceTrend_raw_slope'] = df_rank['PriceTrend_raw_slope'].fillna(0.0)

    # --- Handle Outliers using IQR (Before Filling other NaNs) ---
    # Operates on the _raw metrics
    print("Handling outliers using IQR capping (ignoring NaNs for bounds)...")
    raw_metrics_for_cleaning = [
        'AvgUnitPriceUSD_raw', 'StdDevUnitPriceUSD_raw', 'PriceVolatility_raw', 'PriceTrend_raw_slope',
        'TariffImpact_raw_percent', 'AvgLeadTimeDays_raw', 'LeadTimeVariability_raw_days',
        'OnTimeRate_raw', 'InFullRate_raw'
    ]

    for metric in raw_metrics_for_cleaning:
        if metric in df_rank.columns:
            Q1 = df_rank[metric].quantile(0.25)
            Q3 = df_rank[metric].quantile(0.75)
            if pd.isna(Q1) or pd.isna(Q3):
                print(f"  Skipping outlier capping for '{metric}' (Cannot calculate Q1/Q3).")
                continue
            IQR = Q3 - Q1
            if IQR == 0:
                print(f"  Skipping outlier capping for '{metric}' (IQR is zero).")
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            num_outliers_lower = (df_rank[metric] < lower_bound).sum()
            num_outliers_upper = (df_rank[metric] > upper_bound).sum()
            if num_outliers_lower > 0 or num_outliers_upper > 0:
                print(f"  Capping outliers for '{metric}': Lower (<{lower_bound:.4f}): {num_outliers_lower}, Upper (>{upper_bound:.4f}): {num_outliers_upper}")
                df_rank[metric] = df_rank[metric].clip(lower=lower_bound, upper=upper_bound)
        else:
            print(f"  Warning: Metric '{metric}' not found in df_rank for outlier capping.", file=sys.stderr)

    # --- Fill Remaining NaNs (AFTER Capping Outliers) ---
    # Operates on _raw metrics
    print("Filling remaining NaNs after outlier capping for raw metrics...")
    fill_strategies_raw = {
        'OnTimeRate_raw': 0.0,
        'InFullRate_raw': 0.0,
        'AvgUnitPriceUSD_raw': df_rank['AvgUnitPriceUSD_raw'].median() if 'AvgUnitPriceUSD_raw' in df_rank and df_rank['AvgUnitPriceUSD_raw'].notna().any() else 0,
        'StdDevUnitPriceUSD_raw': df_rank['StdDevUnitPriceUSD_raw'].median() if 'StdDevUnitPriceUSD_raw' in df_rank and df_rank['StdDevUnitPriceUSD_raw'].notna().any() else 0,
        'PriceVolatility_raw': df_rank['PriceVolatility_raw'].median() if 'PriceVolatility_raw' in df_rank and df_rank['PriceVolatility_raw'].notna().any() else 0,
        'TariffImpact_raw_percent': df_rank['TariffImpact_raw_percent'].median() if 'TariffImpact_raw_percent' in df_rank and df_rank['TariffImpact_raw_percent'].notna().any() else 0,
        'AvgLeadTimeDays_raw': df_rank['AvgLeadTimeDays_raw'].median() if 'AvgLeadTimeDays_raw' in df_rank and df_rank['AvgLeadTimeDays_raw'].notna().any() else 999,
        'LeadTimeVariability_raw_days': df_rank['LeadTimeVariability_raw_days'].median() if 'LeadTimeVariability_raw_days' in df_rank and df_rank['LeadTimeVariability_raw_days'].notna().any() else 999,
        # PriceTrend_raw_slope already filled with 0.0 if issues arose
    }
    for col, fill_value in fill_strategies_raw.items():
        if col in df_rank.columns:
            if df_rank[col].isna().any():
                if isinstance(fill_value, (int, float)):
                    print(f"  Filling NaNs in '{col}' with value: {fill_value:.4f}")
                else:
                    print(f"  Filling NaNs in '{col}' with value: {fill_value}")
                df_rank[col] = df_rank[col].fillna(fill_value)
        else:
            print(f"Warning: Column '{col}' not found for NaN filling (raw metrics).", file=sys.stderr)


    # --- Cost Component Calculations (using _raw metrics) ---
    print("Calculating Effective Cost Components...")
    df_rank['cost_BasePrice'] = df_rank['AvgUnitPriceUSD_raw']
    df_rank['cost_Tariff'] = df_rank['AvgUnitPriceUSD_raw'] * (df_rank['TariffImpact_raw_percent'] / 100.0)
    df_rank['cost_Holding_LeadTime'] = df_rank['AvgLeadTimeDays_raw'] * EIP_DailyHoldingCostRate_Param * df_rank['AvgUnitPriceUSD_raw']
    df_rank['cost_Holding_LTVariability'] = (df_rank['LeadTimeVariability_raw_days'] * EIP_SafetyStockMultiplierForLTVar_Param) * \
                                           (EIP_DailyHoldingCostRate_Param * df_rank['AvgUnitPriceUSD_raw'])
    df_rank['cost_Holding_Lateness'] = df_rank.apply(
        lambda row: (1.0 - row['OnTimeRate_raw']) * 
                    EIP_AvgDaysLateIfLate_Dict.get(row[grouping_key_mapped], EIP_AvgDaysLateIfLate_Param) * 
                    EIP_DailyHoldingCostRate_Param * row['AvgUnitPriceUSD_raw'],
        axis=1
    )
    
    # Removed cost_Inefficiency_InFull calculation as it is not a valid metric

    df_rank['cost_Risk_PriceVolatility'] = df_rank['StdDevUnitPriceUSD_raw'] * EIP_RiskPremiumFactorForPriceVolatility_Param
    df_rank['cost_Impact_PriceTrend'] = df_rank['PriceTrend_raw_slope'] * (EIP_PlanningHorizonDaysForPriceTrend_Param / 2.0)

    cost_columns = [
        'cost_BasePrice', 'cost_Tariff', 'cost_Holding_LeadTime',
        'cost_Holding_LTVariability', 'cost_Holding_Lateness',
        'cost_Risk_PriceVolatility', 'cost_Impact_PriceTrend'
    ]
    # Ensure all cost columns exist before summing, fill with 0 if one was missed (e.g. due to all NaN inputs)
    for col in cost_columns:
        if col not in df_rank.columns:
            print(f"Warning: Cost component column '{col}' not found. Filling with 0 for EffectiveCost calculation.")
            df_rank[col] = 0.0
        else: # Fill any NaNs within an existing cost column with 0 before summing
            df_rank[col] = df_rank[col].fillna(0.0)

    # df_rank['EffectiveCostPerUnit_USD'] = df_rank[cost_columns].sum(axis=1) # Old calculation
    df_rank['EffectiveCostPerUnit_USD'] = 0.0
    active_cost_columns_eval = []
    # cost_columns is already defined in your script:
    # cost_columns = [
    #     'cost_BasePrice', 'cost_Tariff', 'cost_Holding_LeadTime',
    #     'cost_Holding_LTVariability', 'cost_Holding_Lateness',
    #     'cost_Risk_PriceVolatility', 'cost_Impact_PriceTrend'
    # ]

    print("Applying cost configuration to EffectiveCostPerUnit_USD calculation:")
    for col_name in cost_columns: # Use the existing cost_columns list
        if col_name not in df_rank.columns:
            print(f"  Warning: Cost component column '{col_name}' not found in df_rank. Skipping for EffectiveCost calculation.")
            continue

        # Ensure NaNs in the component column are handled (e.g., filled with 0) before potential addition
        # This was already done for all cost_columns above, but good to be mindful
        # df_rank[col_name] = df_rank[col_name].fillna(0.0)

        # Default to "True" (enabled) if key is missing in costs_config or value is not explicitly "False"
        is_enabled = str(costs_config.get(col_name, "True")).lower() == "true"

        if is_enabled:
            df_rank['EffectiveCostPerUnit_USD'] += df_rank[col_name]
            active_cost_columns_eval.append(col_name)
            print(f"  Including cost component: {col_name}")
        else:
            print(f"  Excluding cost component: {col_name} (disabled in configuration)")
    
    if not active_cost_columns_eval:
        print("  Warning: All cost components are disabled or missing. EffectiveCostPerUnit_USD will be 0 for all rows.")
    else:
        print(f"  Final EffectiveCostPerUnit_USD calculated using active components: {active_cost_columns_eval}")


    # --- 6. Normalize Metrics (using _raw metrics for normalization input) ---
    print("Normalizing metrics for scoring...")
    # These are the original metric concepts, but we use their _raw versions as input
    # The output _Norm columns will keep their original conceptual names for METRIC_WEIGHTS
    metrics_to_normalize_map = { # Map conceptual metric name to its _raw source column
        'AvgUnitPriceUSD': 'AvgUnitPriceUSD_raw',
        'PriceVolatility': 'PriceVolatility_raw',
        'PriceTrend': 'PriceTrend_raw_slope', # Note: PriceTrend_raw_slope is the source
        'TariffImpact': 'TariffImpact_raw_percent',
        'AvgLeadTimeDays': 'AvgLeadTimeDays_raw',
        'LeadTimeVariability': 'LeadTimeVariability_raw_days',
        'OnTimeRate': 'OnTimeRate_raw',
        'InFullRate': 'InFullRate_raw'
    }

    for metric_concept, raw_metric_col in metrics_to_normalize_map.items():
        norm_col_name = f"{metric_concept}_Norm" # e.g., AvgUnitPriceUSD_Norm
        if raw_metric_col in df_rank.columns:
            is_lower_better = norm_col_name in LOWER_IS_BETTER_METRICS
            df_rank[norm_col_name] = normalize_min_max(df_rank[raw_metric_col], lower_is_better=is_lower_better)
            df_rank[norm_col_name] = df_rank[norm_col_name].fillna(0.5)
        else:
            print(f"Warning: Raw metric source '{raw_metric_col}' for '{norm_col_name}' not found for normalization. Filling with 0.5.", file=sys.stderr)
            df_rank[norm_col_name] = 0.5


    # --- 7. Calculate Final Score (using _Norm columns) ---
    print("Calculating final weighted score...")
    df_rank['FinalScore'] = 0.0
    for norm_col, weight in METRIC_WEIGHTS.items(): # METRIC_WEIGHTS uses conceptual_Norm names
        if norm_col in df_rank.columns:
            df_rank['FinalScore'] += df_rank[norm_col] * weight
        else:
            print(f"Warning: Normalized metric '{norm_col}' not found for weighting. Score may be affected.", file=sys.stderr)

    # --- 8. Rank (Revised: by EffectiveCost then FinalScore) ---
    print("Ranking based on EffectiveCostPerUnit_USD and FinalScore...")
    df_rank.sort_values(by=['EffectiveCostPerUnit_USD', 'FinalScore'], ascending=[True, False], inplace=True)
    df_rank['Rank'] = range(1, len(df_rank) + 1)


    # --- 9. Save Results ---
    print("Saving ranking results...")
    # Construct output filename using the base name from the tariff results path
    tariff_output_basename = os.path.splitext(os.path.basename(tariff_results_path))[0]
    output_filename_base = f"vendor_{mode}_ranking_{tariff_output_basename}.csv"
    output_filename = os.path.join(ranking_output_dir, output_filename_base) # Use argument path

    # Select and order columns for output (using mapped names for keys)
    output_cols_base = ['Rank', col_supplier_id, col_supplier_name]
    # Always include MATNR
    if col_mat_num in df_rank.columns:
         output_cols_base.append(col_mat_num)
    else:
         print(f"Warning: '{col_mat_num}' column not found in final df_rank for output.")

    # Add the grouping key (MAKTX or MATKL or MATNR)
    output_cols_base.append(grouping_key_mapped)
    # Add the specific description column if it's different from the grouping key and MATNR
    if desc_col_mapped != grouping_key_mapped and desc_col_mapped != col_mat_num and desc_col_mapped in df_rank.columns:
         output_cols_base.append(desc_col_mapped)

    # Add the primary new metric and existing score/count
    output_cols_base.extend(['EffectiveCostPerUnit_USD', 'FinalScore', 'POLineItemCount'])

    # Raw metrics (list defined earlier as raw_metrics_for_cleaning)
    # Ensure all columns in raw_metrics_for_cleaning actually exist in df_rank
    output_cols_raw_metrics = [col for col in raw_metrics_for_cleaning if col in df_rank.columns]

    # Cost components (list defined earlier as cost_columns)
    # Ensure all columns in cost_columns actually exist in df_rank
    output_cols_cost_components = [col for col in cost_columns if col in df_rank.columns]

    # Normalized metrics (conceptual names with _Norm suffix, e.g., AvgUnitPriceUSD_Norm)
    # These are the keys from METRIC_WEIGHTS
    output_cols_norm = [norm_col for norm_col in METRIC_WEIGHTS.keys() if norm_col in df_rank.columns]


    output_cols_final = output_cols_base + \
                        output_cols_raw_metrics + \
                        output_cols_cost_components + \
                        output_cols_norm

    # Ensure all selected columns exist before saving and remove duplicates
    output_cols_present = []
    for col in output_cols_final:
        if col in df_rank.columns and col not in output_cols_present:
            output_cols_present.append(col)

    try:
        df_rank[output_cols_present].to_csv(output_filename, index=False, float_format='%.4f')
        print(f"Ranking results saved successfully to: {output_filename}")
    except Exception as e:
        print(f"Error saving results to {output_filename}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Top 10 Vendor/{grouping_key_mapped.capitalize()} Combinations ---")
    print(df_rank[output_cols_present].head(10).to_string())


if __name__ == "__main__":
    main()