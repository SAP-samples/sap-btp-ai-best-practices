import numpy as np
import pandas as pd
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Robust statistics and feature engineering parameters
# ---------------------------------------------------------------------------
# The following constants define minimum-history gates and robust estimators
# used throughout quantity/price/fulfillment checks. The goal is to avoid
# unstable flags when there is too little data for a customer–material pair.
#
# Quantity deviation
# - For n >= MIN_QTY_HISTORY_STRICT we use mean/std and 5th/95th percentiles
# - For MIN_QTY_HISTORY_ROBUST <= n < MIN_QTY_HISTORY_STRICT we use
#   median/MAD (robust z) and IQR whiskers for out-of-range detection
# - For n < MIN_QTY_HISTORY_ROBUST we set neutral defaults (z=0, flags=False)
MIN_QTY_HISTORY_ROBUST = 10
MIN_QTY_HISTORY_STRICT = 30

# Price anomaly robustness for (Material, UoM) and per-customer-material
MIN_PRICE_SAMPLES_ROBUST = 20

# UoM, ship-to and fulfillment stability gates
MIN_UOM_SAMPLES = 10
MIN_SHIPTO_ORDERS = 50
MIN_FULFILL_SAMPLES_ROBUST = 20

# Rolling quantity trend (orders, not months)
TREND_WINDOW_ORDERS = 5
TREND_MIN_PERIODS = 3

# Comparison epsilon to avoid floating point boundary artifacts in range checks
COMPARISON_EPS = 1e-6

# Minimum threshold for divisors (std, MAD) to avoid division by near-zero values
# that can occur due to floating-point artifacts when all values are identical
MIN_DIVISOR_THRESHOLD = 1e-6


def _get_month_order(filename):
    """Extract month from filename and return sort order for chronological processing."""
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    fname_lower = filename.lower()
    for month, order in month_order.items():
        if month in fname_lower:
            return order
    return 99  # Unknown months sort last


def load_and_preprocess_data(data_directory='data'):
    """
    Load CSV files from the data directory and perform initial preprocessing.
    
    Args:
        data_directory (str): Path to the directory containing CSV files
        
    Returns:
        pd.DataFrame: Preprocessed and merged dataframe
    """
    # Get list of CSV files ending with '.csv' and sort them chronologically by month
    filenames = os.listdir(data_directory)
    csv_files = sorted([f for f in filenames if f.lower().endswith('.csv')], key=_get_month_order)
    print(f"Found {len(csv_files)} CSV files (sorted): {csv_files}")
    
    # Load and preprocess each file
    dataframes = []
    for file in csv_files:
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
    
    # Filter out cancelled orders and negative order values
    merged_df = filter_cancelled_and_negative_orders(merged_df)
    print(f"Filtered cancelled/negative-value rows: {len(merged_df)} rows")
    
    # Consolidate duplicate material lines within an order/ship-to (and UoM)
    merged_df = consolidate_duplicate_material_lines(merged_df)
    print(f"Consolidated duplicate material lines: {len(merged_df)} rows")
    return merged_df


def filter_cancelled_and_negative_orders(df):
    """
    Remove rows that are not relevant for downstream anomaly detection.
    Specifically removes:
    - Rows where BillingStatus Desc indicates the order was cancelled
    - Rows where Order item value is negative
    """
    before_rows = len(df)
    # Cancelled flag (case-insensitive, trimmed)
    if 'BillingStatus Desc' in df.columns:
        billing_norm = df['BillingStatus Desc'].astype(str).str.strip().str.casefold()
        mask_cancelled = billing_norm == 'cancelled'
    else:
        mask_cancelled = pd.Series(False, index=df.index)
    
    # Negative order item value flag
    if 'Order item value' in df.columns:
        mask_negative_value = df['Order item value'] < 0
    else:
        mask_negative_value = pd.Series(False, index=df.index)
    
    removed_cancelled = int(mask_cancelled.sum())
    removed_negative = int(mask_negative_value.sum())
    df = df[~(mask_cancelled | mask_negative_value)].copy()
    after_rows = len(df)
    print(f"Filtered cancelled/negative-value rows: removed {removed_cancelled} cancelled, {removed_negative} negative. Remaining: {after_rows} rows.")
    return df


def consolidate_duplicate_material_lines(df):
    """
    Consolidate multiple lines for the same material within the same sales order and Ship-To.
    The line chosen keeps the Sales Document Item from the first occurrence in time; numeric
    quantities/monetary values are summed. Unit Price is recomputed as total value / total qty
    when available, falling back to the first line's unit price otherwise.
    """
    # Define grouping keys (avoid mixing different units of measure)
    group_keys = ['Sales Document Number', 'Material Number', 'Ship-To Party']
    group_keys_with_uom = group_keys + ['Sales unit'] if 'Sales unit' in df.columns else group_keys
    
    # If there is nothing to consolidate, return early
    if not all(key in df.columns for key in group_keys):
        return df
    
    before_rows = len(df)
    dup_counts = df.groupby(group_keys_with_uom, dropna=False).size()
    groups_merged = int((dup_counts > 1).sum())
    
    # Stable sort ensures "first" aggregator is deterministic and corresponds to earliest occurrence
    sort_cols = [col for col in ['Sales Document Number', 'Sales Document Created Date', 'Entry time', 'Sales Document Item'] if col in df.columns]
    df_sorted = df.sort_values(sort_cols, kind='mergesort') if sort_cols else df
    
    # Columns to sum when consolidating
    sum_candidates = [
        'Sales Order item qty',
        'Order item value',
        'Invoiced value',
        'Actual quantity delivered',
        'Actual quantity delivered (ELC only)',
        'Quantity invoiced',
        'Value Open',
        'Subtotal 1',
        'Subtotal 2',
        'Subtotal 3',
        'Subtotal 4',
        'Subtotal 5',
        'Subtotal 6',
        'Confirmed Quantity'
    ]
    present_sum_cols = [c for c in sum_candidates if c in df_sorted.columns]
    
    # Build aggregation dictionary dynamically
    agg_dict = {}
    for col in df_sorted.columns:
        if col in group_keys_with_uom:
            continue  # group keys will be retained automatically
        if col in present_sum_cols:
            agg_dict[col] = 'sum'
        else:
            # keep metadata of the earliest occurrence within the group
            agg_dict[col] = 'first'
    
    consolidated = df_sorted.groupby(group_keys_with_uom, dropna=False).agg(agg_dict).reset_index()
    
    # Recompute Unit Price as weighted average price when possible
    if ('Unit Price' in consolidated.columns) and ('Order item value' in consolidated.columns) and ('Sales Order item qty' in consolidated.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            recomputed_price = np.where(consolidated['Sales Order item qty'] > 0,
                                        consolidated['Order item value'] / consolidated['Sales Order item qty'],
                                        consolidated['Unit Price'])
        consolidated['Unit Price'] = recomputed_price
    
    after_rows = len(consolidated)
    print(f"Consolidated duplicate material lines by order/ship-to/UoM: merged groups {groups_merged}; rows {before_rows} -> {after_rows}.")
    return consolidated


def add_first_time_order_features(df):
    """
    Add features related to first-time customer-material orders.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with first-time order features added
    """
    # Group by Sold To number and Material Number to find first order dates
    first_order_dates = df.groupby(['Sold To number', 'Material Number'])['Sales Document Created Date'].min().reset_index()
    first_order_dates.rename(columns={'Sales Document Created Date': 'first_order_date'}, inplace=True)
    
    # Merge back to original dataframe
    df = pd.merge(df, first_order_dates, on=['Sold To number', 'Material Number'], how='left')
    
    # Flag if current order is the first
    df['is_first_time_cust_material_order'] = (df['Sales Document Created Date'] == df['first_order_date'])
    
    # Drop temporary column
    df.drop(columns=['first_order_date'], inplace=True)
    
    print("Added 'is_first_time_cust_material_order' feature.")
    return df


def add_rare_material_features(df):
    """
    Add features for rare materials (appearing less than 3 times).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with rare material features added
    """
    # Count material occurrences
    material_counts = df.groupby(['Material Number', 'Material Description']).size().reset_index(name='count')
    
    # Find rare materials
    rarest_materials = material_counts[material_counts['count'] < 3]
    
    num_unique_materials = material_counts['Material Number'].nunique()
    num_rare_materials = len(rarest_materials)
    
    print(f"Number of different Material Numbers: {num_unique_materials}")
    print(f"Number of Material Numbers appearing less than 3 times: {num_rare_materials}")
    
    # Flag rare materials
    df['is_rare_material'] = df['Material Number'].isin(rarest_materials['Material Number'])
    
    return df


def add_quantity_deviation_features(df):
    """
    Add features related to quantity pattern deviations.

    Design notes (robustness and minimum history):
    - We compute statistics per (Sold To number, Material Number).
    - When history is small (n < MIN_QTY_HISTORY_ROBUST), we avoid fragile signals by
      setting neutral defaults: z=0 and range flag=False.
    - For medium history (MIN_QTY_HISTORY_ROBUST ≤ n < MIN_QTY_HISTORY_STRICT), we use
      robust statistics: median and MAD for a robust z-score, and IQR whiskers for the
      range flag. This reduces false positives driven by a few extreme historical values.
    - For sufficiently large history (n ≥ MIN_QTY_HISTORY_STRICT), we revert to
      mean/std and percentile range (p05–p95), which provide finer granularity.
    This gating generally lowers false positives in sparse customer–material pairs,
    which improves ML precision without materially hurting recall because borderline
    cases are still reviewed by AI in the −0.05..0 band.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with quantity deviation features added
    """
    group_cols = ['Sold To number', 'Material Number']
    qty = 'Sales Order item qty'

    # Compute multi-regime stats per customer–material using explicit named agg
    stats = df.groupby(group_cols, dropna=False)[qty].agg(
        hist_count='size',
        hist_mean='mean',
        hist_std='std',
        p05=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
        hist_median='median',
        hist_mad=lambda x: np.median(np.abs(x - x.median())),
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()
    # IQR whiskers
    stats['iqr_low'] = stats['q1'] - 1.5 * (stats['q3'] - stats['q1'])
    stats['iqr_high'] = stats['q3'] + 1.5 * (stats['q3'] - stats['q1'])
    df = pd.merge(df, stats, on=group_cols, how='left')

    # Default neutral values
    df['qty_deviation_from_mean'] = 0.0
    df['qty_z_score'] = 0.0
    df['qty_range_low'] = np.nan
    df['qty_range_high'] = np.nan
    df['qty_range_method'] = 'none'
    df['is_qty_outside_typical_range'] = False

    # Large history: mean/std and percentile band
    large = df['hist_count'] >= MIN_QTY_HISTORY_STRICT
    df.loc[large, 'qty_deviation_from_mean'] = df.loc[large, qty] - df.loc[large, 'hist_mean']
    std_pos = large & (df['hist_std'] > MIN_DIVISOR_THRESHOLD)
    df.loc[std_pos, 'qty_z_score'] = df.loc[std_pos, 'qty_deviation_from_mean'] / df.loc[std_pos, 'hist_std']
    df.loc[large, 'qty_range_low'] = df.loc[large, 'p05']
    df.loc[large, 'qty_range_high'] = df.loc[large, 'p95']
    df.loc[large, 'qty_range_method'] = 'percentile'
    df.loc[large, 'is_qty_outside_typical_range'] = (
        (df.loc[large, qty] < (df.loc[large, 'qty_range_low'] - COMPARISON_EPS)) |
        (df.loc[large, qty] > (df.loc[large, 'qty_range_high'] + COMPARISON_EPS))
    )

    # Medium history: robust z via median/MAD; DO NOT raise an out-of-range flag
    # (keep the flag conservative to reduce false positives on sparse history)
    medium = (df['hist_count'] >= MIN_QTY_HISTORY_ROBUST) & (df['hist_count'] < MIN_QTY_HISTORY_STRICT)
    mad_pos = medium & (df['hist_mad'] > MIN_DIVISOR_THRESHOLD)
    df.loc[medium, 'qty_deviation_from_mean'] = df.loc[medium, qty] - df.loc[medium, 'hist_median']
    df.loc[mad_pos, 'qty_z_score'] = 0.6745 * (df.loc[mad_pos, qty] - df.loc[mad_pos, 'hist_median']) / df.loc[mad_pos, 'hist_mad']
    # For medium history we record the IQR band for reference only but do not flag
    df.loc[medium, 'qty_range_low'] = df.loc[medium, 'iqr_low']
    df.loc[medium, 'qty_range_high'] = df.loc[medium, 'iqr_high']
    df.loc[medium, 'qty_range_method'] = 'iqr'
    df.loc[medium, 'is_qty_outside_typical_range'] = False

    # Small history: keep neutral defaults (already set)
    
    print("Added quantity deviation features.")
    return df


def add_unusual_uom_features(df):
    """
    Add features for unusual unit of measure.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with UoM features added
    """
    # Find most common UoM for each customer-material
    common_uom_full = df.groupby(['Sold To number', 'Material Number'])['Sales unit'].agg(
        count='size',
        mode=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).reset_index().rename(columns={'mode': 'historical_common_uom', 'count': 'uom_hist_count'})
    
    # Merge back
    df = pd.merge(df, common_uom_full, on=['Sold To number', 'Material Number'], how='left')
    
    # Flag unusual UoM
    # Only flag unusual UoM when there is enough history
    df['is_unusual_uom'] = (df['Sales unit'] != df['historical_common_uom']) & (df['uom_hist_count'] >= MIN_UOM_SAMPLES)
    
    print("Added unusual UoM features.")
    return df


def add_duplicate_order_features(df):
    """
    Add features for suspected duplicate orders (within 24 hours).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicate order features added
    """
    # Ensure datetime type
    df['Sales Document Created Date'] = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')
    
    # Sort for groupby/shift logic
    df = df.sort_values(['Sold To number', 'Material Number', 'Sales Order item qty', 'Sales Document Created Date'])
    
    # Group and shift to get previous order info
    group_cols = ['Sold To number', 'Material Number', 'Sales Order item qty']
    df['prev_order_date'] = df.groupby(group_cols)['Sales Document Created Date'].shift(1)
    df['prev_sales_doc_number'] = df.groupby(group_cols)['Sales Document Number'].shift(1)
    df['prev_sales_doc_item'] = df.groupby(group_cols)['Sales Document Item'].shift(1)
    
    # Calculate time difference
    df['time_diff_hours'] = (df['Sales Document Created Date'] - df['prev_order_date']).dt.total_seconds() / 3600
    
    # Flag duplicates
    df['is_suspected_duplicate_order'] = (df['time_diff_hours'] > 0) & (df['time_diff_hours'] <= 24)
    
    # Create string version of previous date for display
    df['prev_order_date_str'] = df['prev_order_date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    
    num_duplicates = df['is_suspected_duplicate_order'].sum()
    print(f"Number of duplicates (same customer, material, qty, within 24h): {int(num_duplicates)}")
    
    return df


def add_quantity_trend_features(df):
    """
    Add rolling slope of quantity over the last N orders per (Sold-To, Material).

    This feature captures local trend without requiring long history. We fit a
    simple least-squares line over the last TREND_WINDOW_ORDERS observations of
    `Sales Order item qty` in order creation time for each customer–material and
    record its slope. For groups with insufficient data (< TREND_MIN_PERIODS),
    slope is set to 0.0 (neutral).
    """
    work = df.copy()
    work['Sales Document Created Date'] = pd.to_datetime(work['Sales Document Created Date'], errors='coerce')
    work = work.sort_values(['Sold To number', 'Material Number', 'Sales Document Created Date'])

    slopes = np.zeros(len(work), dtype=float)

    def _rolling_slope(g: pd.DataFrame) -> pd.Series:
        vals = g['Sales Order item qty'].astype(float).values
        # Use order index as x; scale to [0,1] for numeric stability
        x = np.arange(len(vals), dtype=float)
        out = np.zeros_like(vals, dtype=float)
        for i in range(len(vals)):
            start = max(0, i - TREND_WINDOW_ORDERS + 1)
            y = vals[start:i+1]
            n = y.size
            if n < TREND_MIN_PERIODS:
                out[i] = 0.0
                continue
            xw = x[start:i+1]
            # Normalize x to reduce scale effects
            xw = (xw - xw.min()) / (xw.max() - xw.min()) if xw.max() > xw.min() else xw * 0.0
            # Least squares slope
            denom = (xw**2).sum() - (xw.sum()**2)/n
            if denom == 0:
                out[i] = 0.0
            else:
                slope = ((xw*y).sum() - xw.sum()*y.sum()/n) / denom
                out[i] = slope
        return pd.Series(out, index=g.index)

    slopes_series = work.groupby(['Sold To number', 'Material Number'], group_keys=False).apply(_rolling_slope)
    work['qty_trend_slope_lastN'] = slopes_series.values

    # Merge back in original order
    df = work.sort_index()
    print("Added rolling quantity trend slope feature.")
    return df


def add_monthly_volume_features(df):
    """
    Add features for monthly accumulated volume with rolling z-score context and per-order share.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with monthly volume features added
    """
    # Parameters (conservative defaults for 6-month horizon)
    MONTHLY_ROLLING_WINDOW = 6
    MONTHLY_ROLLING_MIN_PERIODS = 3
    ORDER_Z_HIGH_THRESHOLD = 2.0

    # Create year-month column
    df['year_month'] = df['Sales Document Created Date'].dt.to_period('M')

    # Calculate monthly totals
    monthly_totals = df.groupby(['Sold To number', 'Material Number', 'year_month'])['Sales Order item qty'].sum().reset_index()
    monthly_totals.rename(columns={'Sales Order item qty': 'current_month_total_qty'}, inplace=True)

    # Compute rolling baseline per (customer, material), excluding current month from baseline
    def _apply_rolling(g):
        g = g.sort_values('year_month').copy()
        vals = g['current_month_total_qty'].astype(float)
        roll_mean = vals.shift(1).rolling(
            window=MONTHLY_ROLLING_WINDOW,
            min_periods=MONTHLY_ROLLING_MIN_PERIODS
        ).mean()
        roll_std = vals.shift(1).rolling(
            window=MONTHLY_ROLLING_WINDOW,
            min_periods=MONTHLY_ROLLING_MIN_PERIODS
        ).std(ddof=0)
        z = np.where(roll_std > MIN_DIVISOR_THRESHOLD,
                     (vals - roll_mean) / roll_std,
                     0.0)
        g['month_rolling_mean'] = roll_mean
        g['month_rolling_std'] = roll_std
        g['month_rolling_z'] = z
        return g

    monthly_totals = monthly_totals.groupby(['Sold To number', 'Material Number'], group_keys=False).apply(_apply_rolling)

    # Merge rolling z and month totals back to orders
    cols_to_merge = ['Sold To number', 'Material Number', 'year_month',
                     'current_month_total_qty', 'month_rolling_mean', 'month_rolling_std', 'month_rolling_z']
    df = pd.merge(df, monthly_totals[cols_to_merge],
                  on=['Sold To number', 'Material Number', 'year_month'],
                  how='left')

    # Per-order share of month quantity
    df['order_share_of_month'] = np.where(
        df['current_month_total_qty'] > MIN_DIVISOR_THRESHOLD,
        df['Sales Order item qty'] / df['current_month_total_qty'],
        0.0
    )

    # Order-level high z flag (uses previously computed qty_z_score)
    if 'qty_z_score' in df.columns:
        df['is_order_qty_high_z'] = df['qty_z_score'].abs() >= ORDER_Z_HIGH_THRESHOLD
    else:
        df['is_order_qty_high_z'] = False

    # Cleanup
    df.drop(columns=['year_month'], inplace=True)

    print("Added monthly rolling z-score and per-order share features.")
    return df


def add_unusual_delivery_features(df):
    """
    Add features for unusual delivery destinations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with delivery features added
    """
    # Calculate ship-to percentages
    ship_to_counts = df.groupby(['Sold To number', 'Ship-To Party'], dropna=False).size().reset_index(name='ship_to_count_for_sold_to_ship_to')
    total_orders_sold_to = df.groupby('Sold To number', dropna=False).size().reset_index(name='total_orders_for_sold_to')
    
    ship_to_counts = pd.merge(ship_to_counts, total_orders_sold_to, on='Sold To number', how='left')
    ship_to_counts['ship_to_percentage_for_sold_to'] = (
        ship_to_counts['ship_to_count_for_sold_to_ship_to'] / ship_to_counts['total_orders_for_sold_to']
    )
    
    # Merge back
    df = pd.merge(df, ship_to_counts[['Sold To number', 'Ship-To Party', 'ship_to_percentage_for_sold_to', 'total_orders_for_sold_to']], 
                  on=['Sold To number', 'Ship-To Party'], how='left')
    
    # Flag unusual ship-to (threshold: 1%)
    df['is_unusual_ship_to_for_sold_to'] = (
        (df['ship_to_percentage_for_sold_to'] < 0.01) &
        df['ship_to_percentage_for_sold_to'].notna() &
        (df['total_orders_for_sold_to'].fillna(0) >= MIN_SHIPTO_ORDERS)
    )
    
    print("Added unusual delivery destination features.")
    return df


def add_pricing_features(df):
    """
    Add features for pricing anomalies.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with pricing features added
    """
    # Compute robust price stats per (Material, UoM)
    price_stats = df.groupby(['Material Number', 'Sales unit'], dropna=False)['Unit Price'].agg(
        price_count='size',
        price_q1=lambda x: x.quantile(0.25),
        price_q3=lambda x: x.quantile(0.75),
        price_p05=lambda x: x.quantile(0.05),
        price_p95=lambda x: x.quantile(0.95),
        price_median='median'
    ).reset_index()
    price_stats['price_iqr_low'] = price_stats['price_q1'] - 1.5 * (price_stats['price_q3'] - price_stats['price_q1'])
    price_stats['price_iqr_high'] = price_stats['price_q3'] + 1.5 * (price_stats['price_q3'] - price_stats['price_q1'])
    df = pd.merge(df, price_stats, on=['Material Number', 'Sales unit'], how='left')

    # Robust flagging depending on history size
    enough = df['price_count'] >= MIN_PRICE_SAMPLES_ROBUST
    # default False
    df['is_unusual_unit_price'] = False
    # small/medium n: IQR whiskers. If IQR degenerate (low>=high), don't flag.
    iqr_mask = ~enough
    iqr_valid = iqr_mask & (df['price_iqr_low'].notna()) & (df['price_iqr_high'].notna()) & (df['price_iqr_high'] > df['price_iqr_low'] + COMPARISON_EPS)
    df.loc[iqr_valid, 'is_unusual_unit_price'] = (
        (df.loc[iqr_valid, 'Unit Price'] < (df.loc[iqr_valid, 'price_iqr_low'] - COMPARISON_EPS)) |
        (df.loc[iqr_valid, 'Unit Price'] > (df.loc[iqr_valid, 'price_iqr_high'] + COMPARISON_EPS))
    )
    # large n: percentile band. If p05==p95 (all values equal), don't flag.
    pct_mask = enough & df['price_p05'].notna() & df['price_p95'].notna() & (df['price_p95'] > df['price_p05'] + COMPARISON_EPS)
    df.loc[pct_mask, 'is_unusual_unit_price'] = (
        (df.loc[pct_mask, 'Unit Price'] < (df.loc[pct_mask, 'price_p05'] - COMPARISON_EPS)) |
        (df.loc[pct_mask, 'Unit Price'] > (df.loc[pct_mask, 'price_p95'] + COMPARISON_EPS))
    )
    
    # Check for value mismatches
    df['expected_order_item_value'] = df['Unit Price'] * df['Sales Order item qty']
    value_mismatch_tolerance = 0.01
    df['is_value_mismatch_price_qty'] = (
        abs(df['Order item value'] - df['expected_order_item_value']) > value_mismatch_tolerance
    ) & df['Order item value'].notna() & df['expected_order_item_value'].notna()

    # Customer-specific price z-score: compare order unit price vs customer's median price
    # for this material. This provides sensitivity to customer-level pricing norms.
    price_grp = df.groupby(['Sold To number', 'Material Number'])['Unit Price']
    cust_price_stats = price_grp.agg(
        cust_price_count='size',
        cust_price_median='median',
        cust_price_mad=lambda x: np.median(np.abs(x - np.median(x)))
    ).reset_index()
    df = pd.merge(df, cust_price_stats, on=['Sold To number', 'Material Number'], how='left')
    df['price_z_vs_customer'] = 0.0
    enough_cust = df['cust_price_count'] >= MIN_PRICE_SAMPLES_ROBUST
    # robust z with MAD when MAD > 0, else fallback to 0 (neutral)
    mad_pos = enough_cust & (df['cust_price_mad'] > MIN_DIVISOR_THRESHOLD)
    df.loc[mad_pos, 'price_z_vs_customer'] = 0.6745 * (
        (df.loc[mad_pos, 'Unit Price'] - df.loc[mad_pos, 'cust_price_median']) / df.loc[mad_pos, 'cust_price_mad']
    )
    
    print("Added pricing features.")
    return df


def add_fulfillment_time_features(df):
    """
    Add features for unusual fulfillment times.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with fulfillment time features added
    """
    # Ensure datetime columns
    df['Actual GI Date'] = pd.to_datetime(df['Actual GI Date'], errors='coerce')
    df['Sales Document Created Date'] = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')
    
    # Calculate fulfillment duration
    df['fulfillment_duration_days'] = (df['Actual GI Date'] - df['Sales Document Created Date']).dt.days
    
    # Compute robust stats per (Material, Ship-To) using explicit named aggregation
    fulfillment_stats = df.groupby(['Material Number', 'Ship-To Party'], dropna=False)['fulfillment_duration_days'].agg(
        fulfill_count='size',
        fulfillment_q1=lambda x: x.quantile(0.25),
        fulfillment_q3=lambda x: x.quantile(0.75),
        fulfillment_p05=lambda x: x.quantile(0.05),
        fulfillment_p95=lambda x: x.quantile(0.95),
        fulfillment_median='median',
        fulfillment_mad=lambda x: np.median(np.abs(x - x.median()))
    ).reset_index()
    fulfillment_stats['fulfillment_iqr_low'] = fulfillment_stats['fulfillment_q1'] - 1.5 * (fulfillment_stats['fulfillment_q3'] - fulfillment_stats['fulfillment_q1'])
    fulfillment_stats['fulfillment_iqr_high'] = fulfillment_stats['fulfillment_q3'] + 1.5 * (fulfillment_stats['fulfillment_q3'] - fulfillment_stats['fulfillment_q1'])
    df = pd.merge(df, fulfillment_stats, on=['Material Number', 'Ship-To Party'], how='left')

    # Robust flagging depending on history size
    enough = df['fulfill_count'] >= MIN_FULFILL_SAMPLES_ROBUST
    df['fulfillment_range_low'] = np.nan
    df['fulfillment_range_high'] = np.nan
    df['fulfillment_range_method'] = 'none'
    df['is_unusual_fulfillment_time'] = False
    # small/medium n: IQR whiskers
    iqr_mask = ~enough
    df.loc[iqr_mask, 'fulfillment_range_low'] = df.loc[iqr_mask, 'fulfillment_iqr_low']
    df.loc[iqr_mask, 'fulfillment_range_high'] = df.loc[iqr_mask, 'fulfillment_iqr_high']
    df.loc[iqr_mask, 'fulfillment_range_method'] = 'iqr'
    df.loc[iqr_mask, 'is_unusual_fulfillment_time'] = (
        (df.loc[iqr_mask, 'fulfillment_duration_days'] < (df.loc[iqr_mask, 'fulfillment_range_low'] - COMPARISON_EPS)) |
        (df.loc[iqr_mask, 'fulfillment_duration_days'] > (df.loc[iqr_mask, 'fulfillment_range_high'] + COMPARISON_EPS))
    )
    # large n: percentile band
    pct_mask = enough & df['fulfillment_p05'].notna() & df['fulfillment_p95'].notna()
    df.loc[pct_mask, 'fulfillment_range_low'] = df.loc[pct_mask, 'fulfillment_p05']
    df.loc[pct_mask, 'fulfillment_range_high'] = df.loc[pct_mask, 'fulfillment_p95']
    df.loc[pct_mask, 'fulfillment_range_method'] = 'percentile'
    df.loc[pct_mask, 'is_unusual_fulfillment_time'] = (
        (df.loc[pct_mask, 'fulfillment_duration_days'] < (df.loc[pct_mask, 'fulfillment_range_low'] - COMPARISON_EPS)) |
        (df.loc[pct_mask, 'fulfillment_duration_days'] > (df.loc[pct_mask, 'fulfillment_range_high'] + COMPARISON_EPS))
    )
    
    print("Added fulfillment time features.")
    return df


def add_anomaly_explanations(df):
    """
    Add human-readable explanations for detected anomalies.
    
    Args:
        df (pd.DataFrame): Input dataframe with anomaly features
        
    Returns:
        pd.DataFrame: Dataframe with anomaly explanations added
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
        
        if row.get('is_first_time_cust_material_order', False):
            reasons.append('First-time customer-material order')
        
        if row.get('is_rare_material', False):
            mat_num = row.get('Material Number', 'N/A')
            mat_desc = row.get('Material Description', 'N/A')
            reasons.append(f'Rare material (appears less than 3 times): {mat_num} ({mat_desc})')
        
        if row.get('is_qty_outside_typical_range', False):
            qty = format_num(row.get('Sales Order item qty'))
            p05_qty = format_num(row.get('p05'))
            p95_qty = format_num(row.get('p95'))
            reasons.append(f'Qty (actual: {qty}) outside typical range [{p05_qty}-{p95_qty}]')
        
        if row.get('is_unusual_uom', False):
            current_uom = row.get('Sales unit', 'N/A')
            hist_uom = row.get('historical_common_uom', 'N/A')
            reasons.append(f"Unusual UoM: '{current_uom}' (expected '{hist_uom}')")
        
        if row.get('is_suspected_duplicate_order', False):
            mat_num = row.get('Material Number', 'N/A')
            qty = format_num(row.get('Sales Order item qty'))
            prev_date_str = row.get('prev_order_date_str', 'N/A')
            time_diff_hours = format_num(row.get('time_diff_hours', 'N/A'), 1)
            prev_doc_num = format_num(row.get('prev_sales_doc_number', 'N/A'), 0)
            prev_doc_item = format_num(row.get('prev_sales_doc_item', 'N/A'), 0)
            reasons.append(f"Suspected duplicate: Material '{mat_num}' qty {qty} (duplicates order {prev_doc_num} item {prev_doc_item} from {prev_date_str}, {time_diff_hours}h prior)")
        
        # Month context (rolling z-score)
        if pd.notna(row.get('month_rolling_z', np.nan)):
            reasons.append(f"Month rolling z-score: {format_num(row.get('month_rolling_z'), 2)}")
        
        if row.get('is_unusual_ship_to_for_sold_to', False):
            ship_to = row.get('Ship-To Party', 'N/A')
            sold_to = row.get('Sold To number', 'N/A')
            perc = format_num(row.get('ship_to_percentage_for_sold_to', 0) * 100, 1)
            reasons.append(f"Unusual ship-to: '{ship_to}' for sold-to '{sold_to}' (usage: {perc}%)")
        
        if row.get('is_unusual_unit_price', False):
            price = format_num(row.get('Unit Price'))
            p05_price = format_num(row.get('price_p05'))
            p95_price = format_num(row.get('price_p95'))
            reasons.append(f'Unusual unit price: {price} (expected range [{p05_price}-{p95_price}])')
        
        if row.get('is_value_mismatch_price_qty', False):
            actual_val = format_num(row.get('Order item value'))
            expected_val = format_num(row.get('expected_order_item_value'))
            reasons.append(f'Order item value mismatch (actual: {actual_val}, expected: {expected_val})')
        
        if row.get('is_unusual_fulfillment_time', False):
            days = format_num(row.get('fulfillment_duration_days'), 0)
            p05_days = format_num(row.get('fulfillment_p05'), 0)
            p95_days = format_num(row.get('fulfillment_p95'), 0)
            reasons.append(f'Unusual fulfillment time: {days} days (expected range [{p05_days}-{p95_days} days])')
        
        return '; '.join(reasons) if reasons else ''
    
    df['anomaly_explanation'] = df.apply(explain_anomaly, axis=1)
    print("Added anomaly explanations.")
    return df


def export_selected_features(df, output_path):
    """
    Export selected and ordered columns to CSV.
    
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
        'Sales Document Created Date',
        'Entry time',
        'Actual GI Date',
        
        # First-time order
        'is_first_time_cust_material_order',
        
        # Rare Material
        'Material Number',
        'Material Description',
        'is_rare_material',
        
        # Quantity Anomaly
        'Sales Order item qty',
        'hist_mean',
        'hist_std',
        'p05',
        'p95',
        'qty_deviation_from_mean',
        'qty_z_score',
        'is_qty_outside_typical_range',
        
        # Unit of Measure Anomaly
        'Sales unit',
        'historical_common_uom',
        'is_unusual_uom',
        
        # Duplicate Order Anomaly
        'is_suspected_duplicate_order',
        
        # Monthly Volume Anomaly
        'qty_trend_slope_lastN',
        'current_month_total_qty',
        'month_rolling_mean',
        'month_rolling_std',
        'month_rolling_z',
        'order_share_of_month',
        'is_order_qty_high_z',
        
        # Unusual Delivery Destination Anomaly
        'ship_to_percentage_for_sold_to',
        'is_unusual_ship_to_for_sold_to',
        
        # Pricing Anomaly - Unit Price
        'Unit Price',
        'price_p05',
        'price_p95',
        'is_unusual_unit_price',
        'price_z_vs_customer',
        
        # Pricing Anomaly - Value Mismatch
        'Order item value',
        'expected_order_item_value',
        'is_value_mismatch_price_qty',
        
        # Fulfillment Time Anomaly
        'fulfillment_duration_days',
        'fulfillment_p05',
        'fulfillment_p95',
        'is_unusual_fulfillment_time',
        
        # Final Explanation
        'anomaly_explanation'
    ]
    
    # Filter to only include existing columns
    columns_to_export = [col for col in columns_to_export if col in df.columns]
    
    # Export
    df[columns_to_export].to_csv(output_path, index=False)
    print(f"Selected features saved to {output_path}. Exported {len(columns_to_export)} columns.")


def main():
    """
    Main function to execute the complete data preprocessing pipeline.
    """
    print("Starting data preprocessing pipeline...")
    print("=" * 50)
    
    # Load and merge data
    path = "/Users/I760054/Documents/programs/Amgen-Anomaly_detection/ui/datasets/us"
    df = load_and_preprocess_data(path)
    
    # Add all features
    df = add_first_time_order_features(df)
    df = add_rare_material_features(df)
    df = add_quantity_deviation_features(df)
    df = add_unusual_uom_features(df)
    df = add_duplicate_order_features(df)
    df = add_quantity_trend_features(df)
    df = add_monthly_volume_features(df)
    df = add_unusual_delivery_features(df)
    df = add_pricing_features(df)
    df = add_fulfillment_time_features(df)
    df = add_anomaly_explanations(df)
    
    # Calculate average orders per day
    orders_per_day = df.groupby(df['Sales Document Created Date'].dt.date)['Sales Document Number'].nunique()
    avg_orders_per_day = orders_per_day.mean()
    print(f"\nAverage number of orders per day: {avg_orders_per_day:.2f}")
    
    # Save complete dataset
    df.to_csv('data/merged_with_features.csv', index=False)
    print("\nFeatures generated and saved to data/merged_with_features.csv")
    
    # Save selected features
    # Ensure correct path join for selected features export
    export_selected_features(df, os.path.join(path, 'merged_with_features_selected_ordered.csv'))
    
    print("\nData preprocessing completed successfully!")
    return df


if __name__ == "__main__":
    main()