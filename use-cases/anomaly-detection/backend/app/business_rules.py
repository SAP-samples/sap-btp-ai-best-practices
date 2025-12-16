import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------------
# Threshold configuration per customer feedback
# For POC:
# - Use 10% growth factor for all products
# - EXCEPT products on the MFC list and exception SKUs (Imlygic, Corlanor),
#   for which NO growth factor is applied.
# -----------------------------------------------------------------------------
POC_GROWTH_FACTOR = 0.10  # 10% growth factor
EXCEPTION_SKU_NAMES = {"IMLYGIC", "CORLANOR"}

# Optional legacy tolerance knobs (no longer used for thresholds but kept for clarity)
TOLERANCE_PERCENT = 0.30
STD_TOLERANCE_MULTIPLIER = 1.75

# Weekly period configuration (ISO week derived from dates)
WEEKLY_PERIOD = "W-SUN"

# MFC reference file for applying growth-factor exceptions in preprocessing
MFC_FILE = "/Users/I760054/Documents/programs/Amgen-Anomaly_detection/ui/datasets/POC_AOM_MFCdata.csv"


def load_and_preprocess_data(data_directory='data'):
    """
    Load CSV files from the data directory and perform initial preprocessing.
    
    Args:
        data_directory (str): Path to the directory containing CSV files
        
    Returns:
        pd.DataFrame: Preprocessed and merged dataframe
    """
    # Get list of CSV files and sort alphabetically for chronological ordering
    filenames = os.listdir(data_directory)
    csv_files = [f for f in filenames if f.lower().endswith('.csv')]
    csv_files.sort()  # Sort alphabetically to ensure chronological order
    print(f"Found {len(csv_files)} CSV files (sorted chronologically): {csv_files}")
    
    # Load and preprocess each file
    dataframes = []
    for file in csv_files[:6]:  # Only process first 6 files
        df = pd.read_csv(f'{data_directory}/{file}', low_memory=False)
        
        # Convert date columns to datetime
        date_columns = [
            "Requested delivery date",
            "Sales Document Created Date",
            "Actual GI Date",
            "Invoice Creation Date",
            "Shelf Life Expiration Date",
            "Original Requested Delivery Date",
            "Header Pricing Date",
            "Item Pricing Date",
            "Batch Manufacture Date"
        ]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')
        
        # Convert time column
        if "Entry time" in df.columns:
            df["Entry time"] = pd.to_datetime(df["Entry time"], format='%I:%M:%S %p', errors='coerce').dt.time
        
        # Convert numeric columns
        number_columns = [
            "Sales Order item qty",
            "Actual quantity delivered",
            "Quantity invoiced",
            "Quanty open to ship (ELC)",
            "Order item value",
            "Invoiced value",
            "Value Open",
            "Unit Price",
            "Actual quantity delivered (ELC only)",
            "Subtotal 1",
            "Subtotal 2",
            "Subtotal 3",
            "Subtotal 4",
            "Subtotal 5",
            "Subtotal 6",
            "Confirmed Quantity"
        ]
        for col in number_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        dataframes.append(df)
        print(f"Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Merge all dataframes (handle single file case)
    if len(dataframes) == 0:
        raise ValueError("No CSV files found in the specified directory")
    elif len(dataframes) == 1:
        merged_df = dataframes[0]
        print(f"\nUsing single DataFrame with shape: {merged_df.shape}")
    else:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nMerged {len(dataframes)} DataFrames into shape: {merged_df.shape}")

    # Consolidate duplicate material lines within the same sales document
    merged_df = consolidate_order_lines(merged_df)

    return merged_df


def consolidate_order_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate duplicate material lines within the same Sales Document.

    For each `Sales Document Number`, lines that refer to the same product
    (same `Material Number` and `Material Description`) are combined:
    - Sum `Sales Order item qty` across duplicates
    - Keep the smallest `Sales Document Item` as the representative item
    - Keep the first non-null value for other columns

    This ensures order-level checks use the total quantity per material.
    """
    if 'Sales Document Number' not in df.columns or 'Material Number' not in df.columns:
        return df

    # Prepare helper to select first non-null
    def first_non_null(series: pd.Series):
        return series.dropna().iloc[0] if series.dropna().shape[0] > 0 else np.nan

    group_cols = ['Sales Document Number', 'Material Number', 'Material Description']
    agg_dict = {col: first_non_null for col in df.columns if col not in group_cols}
    # Explicit aggregations
    if 'Sales Order item qty' in df.columns:
        agg_dict['Sales Order item qty'] = 'sum'
    if 'Sales Document Item' in df.columns:
        agg_dict['Sales Document Item'] = 'min'

    consolidated = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    print(f"Consolidated duplicate material lines: {len(df)} -> {len(consolidated)} rows")
    return consolidated


def add_90day_customer_material_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 90-day average check for each Customer/Material pair.

    For each row, compute the mean and std of `Sales Order item qty` for the same
    (Sold To number, Material Number) over the prior 90 days (strictly before the
    current `Sales Document Created Date`). Then compute the upper threshold as:
        mean + max(0.30*mean, 1.75*std)
    and flag whether the current line quantity is within this threshold.
    """
    if 'Sales Document Created Date' not in df.columns:
        return df

    work = df.copy()
    work['Sales Document Created Date'] = pd.to_datetime(work['Sales Document Created Date'], errors='coerce')

    # Sort for efficient look-back joins
    work = work.sort_values(['Sold To number', 'Material Number', 'Sales Document Created Date'])

    # Precompute rolling 90-day stats using expanding window filtered by 90-day lookback
    # Approach: for each group, compute for each date the stats on records in the last 90 days (excluding current row)
    def compute_rolling_90d(group: pd.DataFrame) -> pd.DataFrame:
        dates = group['Sales Document Created Date']
        qty = group['Sales Order item qty']
        mean_list = []
        std_list = []
        for i in range(len(group)):
            cutoff_start = dates.iloc[i] - pd.Timedelta(days=90)
            hist_mask = (dates < dates.iloc[i]) & (dates >= cutoff_start)
            hist_vals = qty[hist_mask]
            mean_list.append(hist_vals.mean() if len(hist_vals) > 0 else np.nan)
            std_list.append(hist_vals.std() if len(hist_vals) > 1 else 0.0)
        out = group.copy()
        out['cm_90d_mean_qty'] = mean_list
        out['cm_90d_std_qty'] = std_list
        return out

    work = work.groupby(['Sold To number', 'Material Number'], group_keys=False).apply(compute_rolling_90d)

    # Load MFC table to flag MFC items for growth-factor exception
    try:
        mfc = pd.read_csv(MFC_FILE)
        if 'Product Description' in mfc.columns:
            mfc_names = set(mfc['Product Description'].astype(str).str.strip().str.upper().unique())
        else:
            mfc_names = set()
    except Exception:
        mfc_names = set()

    # Identify exception SKUs (MFC + named exceptions)
    mat_names = work.get('Trade Name', pd.Series(index=work.index, dtype=object)).astype(str).str.strip().str.upper()
    is_exception_sku = mat_names.isin(mfc_names.union(EXCEPTION_SKU_NAMES))

    # Compute thresholds per POC rule: mean + growth_factor*mean (or no growth for exceptions)
    base_mean = work['cm_90d_mean_qty'].fillna(0)
    growth = np.where(is_exception_sku.values, 0.0, POC_GROWTH_FACTOR)
    work['cm_90d_upper_threshold'] = base_mean * (1.0 + growth)
    work['qty_minus_cm_90d_upper'] = work['Sales Order item qty'] - work['cm_90d_upper_threshold']
    work['qty_over_cm_90d_upper_ratio'] = np.where(work['cm_90d_upper_threshold'] > 0,
                                                   work['Sales Order item qty'] / work['cm_90d_upper_threshold'],
                                                   np.nan)
    work['is_outside_cm_90d_threshold'] = work['Sales Order item qty'] > work['cm_90d_upper_threshold']

    # Merge selected columns back to original df order
    cols = ['cm_90d_mean_qty', 'cm_90d_std_qty', 'cm_90d_upper_threshold',
            'qty_minus_cm_90d_upper', 'qty_over_cm_90d_upper_ratio', 'is_outside_cm_90d_threshold']
    df = work.reindex(columns=list(df.columns) + [c for c in cols if c not in df.columns])
    for c in cols:
        df[c] = work[c]

    print("Added 90-day customer/material check features.")
    return df


def add_weekly_average_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 13-week weekly average checks for:
    - Customer/Material (cm)
    - Material-only (mat)

    Uses ISO week derivation. Rolling window excludes current week (shifted).
    Threshold = mean + max(0.30*mean, 1.75*std). For sparse history (<4 weeks),
    fall back to p95 of prior weekly totals.
    """
    if 'Sales Document Created Date' not in df.columns:
        return df

    work = df.copy()
    work['Sales Document Created Date'] = pd.to_datetime(work['Sales Document Created Date'], errors='coerce')
    iso = work['Sales Document Created Date'].dt.isocalendar()
    work['iso_year'] = iso.year.astype(int)
    work['iso_week'] = iso.week.astype(int)
    work['iso_year_week'] = work['iso_year'] * 100 + work['iso_week']

    # Build weekly aggregates for cm and mat
    weekly_cm = work.groupby(['Sold To number', 'Material Number', 'iso_year', 'iso_week', 'iso_year_week'])['Sales Order item qty'].sum().reset_index(name='weekly_qty')
    weekly_mat = work.groupby(['Material Number', 'iso_year', 'iso_week', 'iso_year_week'])['Sales Order item qty'].sum().reset_index(name='weekly_qty')

    def add_rolling_weekly_stats(weekly_df: pd.DataFrame, group_cols: list, prefix: str) -> pd.DataFrame:
        weekly_df = weekly_df.sort_values(group_cols + ['iso_year_week'])
        # Rolling 13-week mean excluding current week (POC uses 10% growth on mean)
        weekly_df[f'{prefix}_13w_mean'] = weekly_df.groupby(group_cols)['weekly_qty'].transform(lambda s: s.shift(1).rolling(window=13, min_periods=1).mean())  # Using min_periods=1 to handle sparse history
        
        # Determine growth factor exceptions (MFC or named exceptions) using latest material name per group if available
        # Check if 'Material Description' column exists in the working DataFrame
        if 'Trade Name' in work.columns:
            # Determine grouping keys: use both customer and material if available, otherwise just material
            if 'Sold To number' in group_cols:
                key_cols = ['Sold To number', 'Material Number']
            else:
                key_cols = ['Material Number']
            # For each group, get the latest (by iso_year_week) non-null Material Description, convert to uppercase and strip whitespace
            latest_names = (
                work
                .dropna(subset=['Trade Name'])  # Remove rows where Material Description is missing
                .sort_values(['iso_year_week'])           # Sort by week to get the latest
                .groupby(key_cols)['Trade Name']
                .last()                                  # Take the last (latest) Material Description per group
                .astype(str).str.strip().str.upper()     # Standardize the name for matching
            )
            # Merge the latest material names into the weekly_df for exception checking
            weekly_df = weekly_df.merge(
                latest_names.rename('Trade Name_latest'),
                left_on=key_cols,
                right_index=True,
                how='left'
            )
            # Use the merged column for material name checks, fill missing with empty string
            mat_names = weekly_df['Trade Name_latest'].fillna('')
        else:
            # If Trade Name is not available, use empty strings for all rows
            mat_names = pd.Series([''] * len(weekly_df))

        try:
            mfc = pd.read_csv(MFC_FILE)
            mfc_names = set(mfc['Product Description'].astype(str).str.strip().str.upper().unique()) if 'Product Description' in mfc.columns else set()
        except Exception:
            mfc_names = set()

        is_exception = mat_names.astype(str).str.upper().isin(set(EXCEPTION_SKU_NAMES).union(mfc_names))
        growth = np.where(is_exception.values, 0.0, POC_GROWTH_FACTOR)

        # Upper threshold per POC rule
        weekly_df[f'{prefix}_13w_upper_threshold'] = weekly_df[f'{prefix}_13w_mean'].fillna(0) * (1.0 + growth)
        return weekly_df

    weekly_cm = add_rolling_weekly_stats(weekly_cm, ['Sold To number', 'Material Number'], 'cm_weekly')
    weekly_mat = add_rolling_weekly_stats(weekly_mat, ['Material Number'], 'mat_weekly')

    # Merge weekly thresholds back to line level by current iso_year_week context
    work = work.merge(weekly_cm[['Sold To number','Material Number','iso_year_week','cm_weekly_13w_mean','cm_weekly_13w_upper_threshold']],
                      on=['Sold To number','Material Number','iso_year_week'], how='left')
    work = work.merge(weekly_mat[['Material Number','iso_year_week','mat_weekly_13w_mean','mat_weekly_13w_upper_threshold']],
                      on=['Material Number','iso_year_week'], how='left')

    # Line-level comparisons (compare line qty against weekly thresholds)
    work['qty_minus_cm_weekly_upper'] = work['Sales Order item qty'] - work['cm_weekly_13w_upper_threshold']
    work['qty_minus_mat_weekly_upper'] = work['Sales Order item qty'] - work['mat_weekly_13w_upper_threshold']
    work['is_outside_cm_weekly_threshold'] = work['Sales Order item qty'] > work['cm_weekly_13w_upper_threshold']
    work['is_outside_mat_weekly_threshold'] = work['Sales Order item qty'] > work['mat_weekly_13w_upper_threshold']

    # Merge selected back
    cols = [
        'cm_weekly_13w_mean','cm_weekly_13w_upper_threshold',
        'mat_weekly_13w_mean','mat_weekly_13w_upper_threshold',
        'qty_minus_cm_weekly_upper','qty_minus_mat_weekly_upper',
        'is_outside_cm_weekly_threshold','is_outside_mat_weekly_threshold'
    ]
    df = work.reindex(columns=list(df.columns) + [c for c in cols if c not in df.columns])
    for c in cols:
        df[c] = work[c]

    print("Added 13-week weekly average check features for customer/material and material-only.")
    return df


def add_focused_anomaly_explanations(df):
    """
    Add human-readable explanations for detected anomalies (90-day and weekly checks only).
    
    Args:
        df (pd.DataFrame): Input dataframe with anomaly features
        
    Returns:
        pd.DataFrame: Dataframe with focused anomaly explanations added
    """
    def explain_anomaly(row):
        reasons = []
        
        # Helper to format numbers
        def format_num(val, precision=2):
            if pd.isna(val):
                return "N/A"
            if isinstance(val, float):
                return f"{val:.{precision}f}"
            return str(val)
        
        # 90-day CM threshold explanations
        if row.get('is_outside_cm_90d_threshold', False):
            qty = format_num(row.get('Sales Order item qty'))
            upper = format_num(row.get('cm_90d_upper_threshold'))
            diff = format_num(row.get('qty_minus_cm_90d_upper'))
            reasons.append(f'90-day CM threshold exceeded: qty {qty} > upper {upper} (diff {diff})')

        # Weekly threshold explanations
        if row.get('is_outside_cm_weekly_threshold', False):
            qty = format_num(row.get('Sales Order item qty'))
            upper = format_num(row.get('cm_weekly_13w_upper_threshold'))
            diff = format_num(row.get('qty_minus_cm_weekly_upper'))
            reasons.append(f'13-week CM weekly threshold exceeded: qty {qty} > upper {upper} (diff {diff})')

        if row.get('is_outside_mat_weekly_threshold', False):
            qty = format_num(row.get('Sales Order item qty'))
            upper = format_num(row.get('mat_weekly_13w_upper_threshold'))
            diff = format_num(row.get('qty_minus_mat_weekly_upper'))
            reasons.append(f'13-week material weekly threshold exceeded: qty {qty} > upper {upper} (diff {diff})')
        
        return '; '.join(reasons) if reasons else ''
    
    df['focused_anomaly_explanation'] = df.apply(explain_anomaly, axis=1)
    print("Added focused anomaly explanations (90-day and weekly checks only).")
    return df


def export_focused_features(df, output_path):
    """
    Export selected and ordered columns to CSV (focused on 90-day and weekly checks).
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path for output CSV file
        
    Returns:
        None
    """
    columns_to_export = [
        # Identifiers
        'Sales Document Number',
        'Sales Document Item',
        'Customer PO number',
        'Sold To number',
        'Ship-To Party',
        'Plant',
        'Sales Document Created Date',
        'Entry time',
        'Actual GI Date',
        
        # Material Information
        'Material Number',
        'Material Description',
        'Trade Name',
        
        # Quantity Information
        'Sales Order item qty',
        'Sales unit',
        
        # 90-Day CM Check
        'cm_90d_mean_qty',
        'cm_90d_std_qty',
        'cm_90d_upper_threshold',
        'qty_minus_cm_90d_upper',
        'qty_over_cm_90d_upper_ratio',
        'is_outside_cm_90d_threshold',

        # Weekly Avg Checks (13-week)
        'cm_weekly_13w_mean',
        'cm_weekly_13w_upper_threshold',
        'qty_minus_cm_weekly_upper',
        'is_outside_cm_weekly_threshold',
        'mat_weekly_13w_mean',
        'mat_weekly_13w_upper_threshold',
        'qty_minus_mat_weekly_upper',
        'is_outside_mat_weekly_threshold',
        
        # Final Explanation
        'focused_anomaly_explanation'
    ]
    
    # Filter to only include existing columns
    columns_to_export = [col for col in columns_to_export if col in df.columns]
    
    # Export
    df[columns_to_export].to_csv(output_path, index=False)
    print(f"Focused features saved to {output_path}. Exported {len(columns_to_export)} columns.")


def main():
    """
    Main function to execute the focused data preprocessing pipeline (90-day and weekly checks only).
    """
    print("Starting focused data preprocessing pipeline (90-day and weekly checks only)...")
    print("=" * 70)
    
    # Load and merge data
    path = "/Users/I760054/Documents/programs/Amgen-Anomaly_detection/ui/datasets/new_data_feedback/"
    df = load_and_preprocess_data(path)
    
    # Add only the focused features (90-day and weekly checks)
    df = add_90day_customer_material_check(df)
    df = add_weekly_average_checks(df)
    df = add_focused_anomaly_explanations(df)
    
    # Calculate average orders per day
    orders_per_day = df.groupby(df['Sales Document Created Date'].dt.date)['Sales Document Number'].nunique()
    avg_orders_per_day = orders_per_day.mean()
    print(f"\nAverage number of orders per day: {avg_orders_per_day:.2f}")
    
    # Save complete dataset
    df.to_csv('data/focused_90day_weekly_with_features.csv', index=False)
    print("\nFocused features generated and saved to data/focused_90day_weekly_with_features.csv")
    
    # Save selected features
    export_focused_features(df, path + 'focused_90day_weekly_features_selected_ordered.csv')
    
    print("\nFocused data preprocessing completed successfully!")
    return df


if __name__ == "__main__":
    main()
