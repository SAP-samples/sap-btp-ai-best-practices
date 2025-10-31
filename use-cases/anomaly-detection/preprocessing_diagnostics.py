#!/usr/bin/env python3
"""
Preprocessing diagnostics for anomaly feature flags.

This script replicates the checks performed during analysis to help validate
whether the initial anomaly flags from data_preprocessing.py are sensible.

It loads an exported CSV (typically merged_with_features_selected_ordered.csv)
and prints:
  - Count of non-empty anomaly_explanation vs. union of flag columns
  - Per-flag True counts
  - Rare material stats (<3 occurrences)
  - Ship-to distribution stats (5% threshold context)
  - Value mismatch diagnostics (absolute diff stats)
  - Quantity p05–p95 coverage and outside rate
  - Monthly volume flags: amplification from month-grain flags to row-grain
  - Fulfillment time flags vs. small group sizes

Usage:
  python ui/preprocessing_diagnostics.py --file /absolute/path/to/your.csv

Notes:
  - The script is read-only; it prints diagnostics to stdout.
  - It handles missing columns gracefully and skips checks when prerequisites
    are not present in the CSV.
"""

import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd


def _print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV into a pandas DataFrame with minimal assumptions.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from: {csv_path}")
    return df


def check_explanations_and_flags(df: pd.DataFrame) -> None:
    """
    Verify that non-empty anomaly_explanation aligns with the union of all
    boolean anomaly flags and print per-flag True counts.
    """
    _print_header("ANOMALY EXPLANATIONS AND FLAG UNION")

    # All flags referenced in data_preprocessing.explain_anomaly
    candidate_flags: List[str] = [
        'is_first_time_cust_material_order',
        'is_rare_material',
        'is_qty_outside_typical_range',
        'is_unusual_uom',
        'is_suspected_duplicate_order',
        'is_monthly_qty_outside_typical_range',
        'is_unusual_ship_to_for_sold_to',
        'is_unusual_unit_price',
        'is_value_mismatch_price_qty',
        'is_unusual_fulfillment_time',
    ]

    present_flags = [c for c in candidate_flags if c in df.columns]
    missing_flags = [c for c in candidate_flags if c not in df.columns]
    print(f"Flags present: {len(present_flags)}; Missing: {missing_flags}")

    if 'anomaly_explanation' in df.columns:
        non_empty_exp = (
            df['anomaly_explanation']
            .fillna('')
            .astype(str)
            .str.strip()
            .ne('')
            .sum()
        )
        print(f"Non-empty anomaly_explanation: {int(non_empty_exp):,}")
    else:
        non_empty_exp = None
        print("Column 'anomaly_explanation' not found. Skipping direct comparison.")

    # Per-flag counts and union
    flag_counts = {}
    for col in present_flags:
        s = df[col]
        if s.dtype == bool:
            flag_counts[col] = int(s.sum())
        else:
            flag_counts[col] = int((s == True).sum())  # noqa: E712

    if present_flags:
        any_flag = int(df[present_flags].fillna(False).any(axis=1).sum())
        print("\nFlag counts (True rows):")
        for k in present_flags:
            print(f"- {k}: {flag_counts[k]:,}")
        print(f"\nUnion(any flag True): {any_flag:,}")
        if non_empty_exp is not None:
            print(f"Matches non-empty anomaly_explanation? {any_flag == non_empty_exp}")
    else:
        print("No known flag columns found; cannot compute union.")


def check_rare_materials(df: pd.DataFrame) -> None:
    """
    Compute basic rare material diagnostics (<3 occurrences per Material Number).
    """
    _print_header("RARE MATERIAL STATISTICS")
    if 'Material Number' not in df.columns:
        print("Missing column 'Material Number'. Skipping.")
        return

    counts = df.groupby('Material Number').size()
    rare_mats = counts[counts < 3]
    print(f"Unique materials: {int(counts.shape[0]):,}")
    print(f"Rare materials (unique, <3 rows): {int(rare_mats.shape[0]):,}")
    if 'is_rare_material' in df.columns:
        rare_rows = int(df['is_rare_material'].fillna(False).sum())
        print(f"Rows flagged is_rare_material: {rare_rows:,}")
    else:
        print("Column 'is_rare_material' not found. Skipping row-level count.")


def check_ship_to_distribution(df: pd.DataFrame) -> None:
    """
    Analyze the ship-to distribution per Sold To to contextualize the 1% threshold.
    """
    _print_header("SHIP-TO DISTRIBUTION (1% THRESHOLD CONTEXT)")
    required = {'Sold To number', 'Ship-To Party'}
    if not required.issubset(df.columns):
        print(f"Missing columns for ship-to analysis: {sorted(required - set(df.columns))}")
        return

    ship_counts = df.groupby(['Sold To number', 'Ship-To Party']).size().reset_index(name='ct')
    sold_totals = ship_counts.groupby('Sold To number')['ct'].sum().rename('total')
    ship_counts = ship_counts.merge(sold_totals, on='Sold To number')
    ship_counts['pct'] = ship_counts['ct'] / ship_counts['total']

    avg_ship_to_per_sold = ship_counts.groupby('Sold To number').size().mean()
    # Updated threshold: 1%
    frac_under_1 = float((ship_counts['pct'] < 0.01).mean())
    print(f"Average ship-to destinations per Sold To: {avg_ship_to_per_sold:.3f}")
    print(f"Fraction of (Sold To, Ship-To) combos under 1% usage: {frac_under_1:.2%}")

    if 'is_unusual_ship_to_for_sold_to' in df.columns:
        flagged = int(df['is_unusual_ship_to_for_sold_to'].fillna(False).sum())
        print(f"Rows flagged is_unusual_ship_to_for_sold_to (1% threshold): {flagged:,}")


def check_value_mismatch(df: pd.DataFrame) -> None:
    """
    Summarize absolute mismatch between 'Order item value' and expected value.
    """
    _print_header("VALUE MISMATCH DIAGNOSTICS")
    required = {'Order item value', 'expected_order_item_value'}
    if not required.issubset(df.columns):
        print(f"Missing columns for value mismatch: {sorted(required - set(df.columns))}")
        return

    diffs = (df['Order item value'] - df['expected_order_item_value']).dropna()
    if len(diffs) == 0:
        print("No comparable rows to compute differences.")
        return
    abs_diffs = diffs.abs()
    median = float(abs_diffs.median())
    p95 = float(abs_diffs.quantile(0.95))
    print(f"Absolute difference median: {median}")
    print(f"Absolute difference p95: {p95}")
    if 'is_value_mismatch_price_qty' in df.columns:
        flagged = int(df['is_value_mismatch_price_qty'].fillna(False).sum())
        print(f"Rows flagged is_value_mismatch_price_qty: {flagged:,}")


def check_quantity_range(df: pd.DataFrame) -> None:
    """
    Evaluate the p05–p95 quantity rule coverage and the outside rate where defined.
    """
    _print_header("QUANTITY RANGE (p05–p95) COVERAGE")
    required = {'p05', 'p95', 'Sales Order item qty'}
    if not required.issubset(df.columns):
        print(f"Missing columns for quantity range: {sorted(required - set(df.columns))}")
        return

    mask = df['p05'].notna() & df['p95'].notna()
    outside = ((df['Sales Order item qty'] < df['p05']) | (df['Sales Order item qty'] > df['p95'])) & mask
    defined_rows = int(mask.sum())
    outside_rate = float(outside.sum() / mask.sum()) if mask.sum() > 0 else float('nan')
    print(f"Rows with defined p05/p95: {defined_rows:,}")
    print(f"Outside typical range rate (given defined): {outside_rate:.2%}")
    if 'is_qty_outside_typical_range' in df.columns:
        flagged = int(df['is_qty_outside_typical_range'].fillna(False).sum())
        print(f"Rows flagged is_qty_outside_typical_range: {flagged:,}")


def check_monthly_volume_flags(df: pd.DataFrame, date_format: str = None) -> None:
    """
    Quantify amplification from month-level flags to row-level counts and examine
    small-history effects for (Sold To, Material).
    """
    _print_header("MONTHLY VOLUME FLAGS (AMPLIFICATION AND HISTORY EFFECTS)")

    if 'Sales Document Created Date' not in df.columns:
        print("Missing 'Sales Document Created Date'. Skipping monthly analysis.")
        return

    try:
        if date_format:
            dt = pd.to_datetime(df['Sales Document Created Date'], format=date_format, errors='coerce')
        else:
            dt = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')
    except Exception:
        print("Failed to parse 'Sales Document Created Date' as datetime.")
        return

    work = df.copy()
    # Convert to monthly period and also keep calendar month label for clarity
    work['year_month'] = dt.dt.to_period('M')
    work['calendar_month'] = dt.dt.to_period('M').astype(str)

    flag_col = 'is_monthly_qty_outside_typical_range'
    if flag_col not in work.columns:
        print(f"Column '{flag_col}' not found. Skipping.")
        return

    monthly_rows_true = int(work[flag_col].fillna(False).sum())
    flagged_months_df = work[work[flag_col].fillna(False)].dropna(subset=['year_month'])
    key_cols = ['Sold To number', 'Material Number', 'year_month']
    if not set(key_cols[:-1]).issubset(work.columns):
        print("Missing 'Sold To number' or 'Material Number'. Skipping detailed monthly analysis.")
        print(f"Monthly rows flagged: {monthly_rows_true:,}")
        return

    flagged_keys = flagged_months_df[key_cols].drop_duplicates()
    num_flagged_months = len(flagged_keys)

    all_months = work.dropna(subset=['year_month'])[key_cols].drop_duplicates()
    total_unique_months = len(all_months)

    avg_orders_per_flagged_month = monthly_rows_true / num_flagged_months if num_flagged_months else 0.0
    frac_flagged_months = num_flagged_months / total_unique_months if total_unique_months else float('nan')

    print(f"Monthly rows flagged: {monthly_rows_true:,}")
    print(f"Distinct flagged months (Sold To, Material, year-month): {num_flagged_months:,}")
    print(f"Avg orders per flagged month: {avg_orders_per_flagged_month:.2f}")
    print(f"Total unique (Sold To, Material, year-month): {total_unique_months:,}")
    print(f"Fraction of months flagged: {frac_flagged_months:.2%}")

    # Clarify calendar month coverage across dataset
    unique_calendar_months = work['calendar_month'].dropna().unique()
    print(f"Unique calendar months present in dataset: {len(unique_calendar_months)} -> {sorted(unique_calendar_months)[:6]}{' ...' if len(unique_calendar_months)>6 else ''}")

    # Distribution of months per pair
    months_per_pair = all_months.groupby(['Sold To number', 'Material Number']).size()
    desc = months_per_pair.describe()
    print("\nMonths per (Sold To, Material) pair - summary:")
    print(desc.to_string())

    # Small-sample prevalence
    small_pairs = int((months_per_pair < 6).sum())
    total_pairs = int(months_per_pair.shape[0])
    frac_small = small_pairs / total_pairs if total_pairs else float('nan')
    print(f"\nPairs with < 6 months history: {small_pairs}/{total_pairs} ({frac_small:.2%})")

    # Flag rate by history bucket
    buckets = pd.cut(
        months_per_pair,
        bins=[0, 3, 6, 12, 24, 999],
        right=False,
        labels=['<3', '3-5', '6-11', '12-23', '24+']
    )
    pair_bucket = buckets.reset_index()
    pair_bucket.columns = ['Sold To number', 'Material Number', 'bucket']

    month_keys = all_months.copy()
    month_keys['flag'] = False
    flag_key_set = set([tuple(x) for x in flagged_keys[key_cols].values])
    month_keys['flag'] = month_keys[key_cols].apply(lambda r: tuple(r) in flag_key_set, axis=1)
    month_keys = month_keys.merge(pair_bucket, on=['Sold To number', 'Material Number'], how='left')

    bucket_flag_rates = month_keys.groupby('bucket')['flag'].mean().sort_index()
    print("\nFlagged month rate by history size bucket:")
    for b, rate in bucket_flag_rates.items():
        # Handle potential NaN for empty buckets
        rate_str = f"{rate:.2%}" if pd.notna(rate) else "nan%"
        print(f"  {b}: {rate_str}")

    # Orders per month comparison for flagged vs non-flagged months
    orders_per_month = work.groupby(key_cols).size().rename('orders_in_month')
    orders_df = orders_per_month.reset_index()
    orders_df['flag'] = orders_df[key_cols].apply(lambda r: tuple(r) in flag_key_set, axis=1)
    med_flag = float(orders_df.loc[orders_df['flag'], 'orders_in_month'].median()) if (orders_df['flag'].any()) else float('nan')
    med_nonflag = float(orders_df.loc[~orders_df['flag'], 'orders_in_month'].median()) if (~orders_df['flag']).any() else float('nan')
    print(f"\nMedian orders per flagged month: {med_flag}")
    print(f"Median orders per non-flagged month: {med_nonflag}")


def check_monthly_volume_rolling(df: pd.DataFrame, window: int = 3, z_thresh: float = 2.0, date_format: str = None) -> None:
    """
    Rolling-month diagnostics for monthly totals per (Sold To, Material).

    This computes monthly totals of 'Sales Order item qty' per (Sold To, Material),
    then flags months where the current month's total deviates from the rolling
    mean by more than z_thresh * rolling std, using a trailing window of size
    'window' (excluding the current month from the mean/std).

    Purpose: stabilize anomalies when there are few months of history.
    """
    _print_header("ROLLING MONTHLY DIAGNOSTICS (TRAILING WINDOW Z-SCORE)")

    required = {'Sales Document Created Date', 'Sold To number', 'Material Number', 'Sales Order item qty'}
    if not required.issubset(df.columns):
        print(f"Missing columns for rolling monthly analysis: {sorted(required - set(df.columns))}")
        return

    if date_format:
        dt = pd.to_datetime(df['Sales Document Created Date'], format=date_format, errors='coerce')
    else:
        dt = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')
    work = df.copy()
    work['year_month'] = dt.dt.to_period('M')

    # Monthly totals per (Sold To, Material, month)
    monthly = (
        work.groupby(['Sold To number', 'Material Number', 'year_month'])['Sales Order item qty']
        .sum()
        .rename('current_month_total_qty')
        .reset_index()
    )

    # Sort for rolling calculations
    monthly = monthly.sort_values(['Sold To number', 'Material Number', 'year_month'])

    # For rolling mean/std, operate within each (Sold To, Material)
    def _apply_rolling(g: pd.DataFrame) -> pd.DataFrame:
        # Convert period to timestamp end-of-month for proper ordering and rolling
        ts = g['year_month'].dt.to_timestamp('M')
        vals = g['current_month_total_qty'].astype(float)
        # Use shift(1) so the current month isn't included in its baseline
        roll_mean = vals.rolling(window=window, min_periods=max(2, window-1)).mean().shift(1)
        roll_std = vals.rolling(window=window, min_periods=max(2, window-1)).std(ddof=0).shift(1)
        z = (vals - roll_mean) / roll_std
        g = g.copy()
        g['roll_mean'] = roll_mean
        g['roll_std'] = roll_std
        g['roll_z'] = z
        # Anomaly when we have a baseline (mean/std not NaN) and |z|>threshold
        g['is_rolling_monthly_anomaly'] = (g['roll_std'].notna()) & (g['roll_std'] > 0) & (g['roll_z'].abs() > z_thresh)
        return g

    monthly_rolled = monthly.groupby(['Sold To number', 'Material Number'], group_keys=False).apply(_apply_rolling)

    # Summary
    total_months = len(monthly_rolled)
    flagged_rows = int(monthly_rolled['is_rolling_monthly_anomaly'].sum())
    frac_flagged = flagged_rows / total_months if total_months else float('nan')
    print(f"Window: {window} months, Z-threshold: {z_thresh}")
    print(f"Monthly rows evaluated: {total_months:,}")
    print(f"Flagged months (rolling rule): {flagged_rows:,} ({frac_flagged:.2%})")

    # Compare overlap with existing fixed p05–p95 month flags, if present
    if 'is_monthly_qty_outside_typical_range' in work.columns:
        # Merge flags at month grain
        month_key = ['Sold To number', 'Material Number', 'year_month']
        fixed_flags = work[['Sold To number', 'Material Number', 'year_month', 'is_monthly_qty_outside_typical_range']].drop_duplicates()
        joined = monthly_rolled.merge(fixed_flags, on=month_key, how='left')
        fixed_true = joined['is_monthly_qty_outside_typical_range'].fillna(False)
        rolling_true = joined['is_rolling_monthly_anomaly'].fillna(False)

        both = int((fixed_true & rolling_true).sum())
        fixed_only = int((fixed_true & ~rolling_true).sum())
        rolling_only = int((~fixed_true & rolling_true).sum())
        print("\nOverlap with fixed monthly p05–p95 flags:")
        print(f"  Both fixed and rolling True: {both:,}")
        print(f"  Fixed-only True: {fixed_only:,}")
        print(f"  Rolling-only True: {rolling_only:,}")

    # Top-level intuition: how many (Sold To, Material) pairs have enough history to compute a baseline
    hist_counts = monthly_rolled.groupby(['Sold To number', 'Material Number'])['roll_mean'].apply(lambda s: s.notna().sum())
    enough_hist_pairs = int((hist_counts >= 1).sum())
    total_pairs = int(hist_counts.shape[0])
    print(f"\nPairs with sufficient rolling baseline (>=1 baseline month): {enough_hist_pairs}/{total_pairs} ({(enough_hist_pairs/total_pairs if total_pairs else float('nan')):.2%})")


def check_monthly_volume_min_history(
    df: pd.DataFrame,
    min_months: int = 6,
    min_orders_in_month: int = 0,
    date_format: str = None,
) -> None:
    """
    Apply a minimum-history gate to existing fixed monthly flags:
    - Keep flagged months only for (Sold To, Material) pairs with >= min_months of history
    - Optionally require at least min_orders_in_month orders in that flagged month

    This shows the impact of gating without recomputing quantiles.
    """
    _print_header("MONTHLY FLAGS WITH MIN HISTORY GATING")

    required = {'Sales Document Created Date', 'Sold To number', 'Material Number', 'is_monthly_qty_outside_typical_range'}
    if not required.issubset(df.columns):
        print(f"Missing columns for min-history analysis: {sorted(required - set(df.columns))}")
        return

    if date_format:
        dt = pd.to_datetime(df['Sales Document Created Date'], format=date_format, errors='coerce')
    else:
        dt = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')

    work = df.copy()
    work['year_month'] = dt.dt.to_period('M')

    # All pair-month keys
    month_keys = ['Sold To number', 'Material Number', 'year_month']
    all_months = work.dropna(subset=['year_month'])[month_keys].drop_duplicates()

    # Compute history length per pair
    months_per_pair = all_months.groupby(['Sold To number', 'Material Number']).size().rename('months_history')

    # Existing fixed monthly flags at month-grain
    fixed_flags = (
        work[['Sold To number', 'Material Number', 'year_month', 'is_monthly_qty_outside_typical_range']]
        .drop_duplicates()
    )

    # Merge history size
    fixed_flags = fixed_flags.merge(
        months_per_pair.reset_index(),
        on=['Sold To number', 'Material Number'],
        how='left'
    )

    # Optional: orders per flagged month
    orders_in_month = work.groupby(month_keys).size().rename('orders_in_month').reset_index()
    fixed_flags = fixed_flags.merge(orders_in_month, on=month_keys, how='left')

    # Apply gating
    fixed_true = fixed_flags['is_monthly_qty_outside_typical_range'].fillna(False)
    hist_ok = fixed_flags['months_history'].fillna(0) >= min_months
    orders_ok = fixed_flags['orders_in_month'].fillna(0) >= min_orders_in_month
    gated_true = fixed_true & hist_ok & orders_ok

    # Counts at month grain
    total_months = len(fixed_flags)
    original_true = int(fixed_true.sum())
    gated_true_count = int(gated_true.sum())
    print(f"Original fixed-flagged months: {original_true:,}")
    print(f"After gating (min_months={min_months}, min_orders_in_month={min_orders_in_month}): {gated_true_count:,}")
    print(f"Reduction: {original_true - gated_true_count:,} ({((original_true - gated_true_count)/original_true if original_true else 0):.2%})")

    # Row-level amplification after gating
    # Build set of gated flagged month keys
    flagged_keys = set([
        (r['Sold To number'], r['Material Number'], r['year_month'])
        for _, r in fixed_flags.loc[gated_true].iterrows()
    ])
    # Count rows in original df belonging to these month keys
    row_mask = work[month_keys].apply(lambda r: (r['Sold To number'], r['Material Number'], r['year_month']) in flagged_keys, axis=1)
    print(f"Rows in gated flagged months: {int(row_mask.sum()):,}")


def check_monthly_volume_material_baseline(
    df: pd.DataFrame,
    date_format: str = None,
) -> None:
    """
    Material-level baseline diagnostic:
    - Compute monthly totals per material across all customers
    - Compute material-level p05/p95 across months
    - Flag pair-months where current total is outside the material baseline
    - Report overlap with fixed pair-level flags, focusing on pairs with insufficient history
    """
    _print_header("MATERIAL-LEVEL MONTHLY BASELINE (p05–p95)")

    required = {'Sales Document Created Date', 'Sold To number', 'Material Number', 'Sales Order item qty'}
    if not required.issubset(df.columns):
        print(f"Missing columns for material baseline: {sorted(required - set(df.columns))}")
        return

    if date_format:
        dt = pd.to_datetime(df['Sales Document Created Date'], format=date_format, errors='coerce')
    else:
        dt = pd.to_datetime(df['Sales Document Created Date'], errors='coerce')

    work = df.copy()
    work['year_month'] = dt.dt.to_period('M')

    # Pair-month totals
    pair_month = (
        work.groupby(['Sold To number', 'Material Number', 'year_month'])['Sales Order item qty']
        .sum().rename('current_month_total_qty').reset_index()
    )

    # Material-month totals
    material_month = (
        work.groupby(['Material Number', 'year_month'])['Sales Order item qty']
        .sum().rename('material_month_total_qty').reset_index()
    )

    # Material-level p05/p95 across months
    material_stats = material_month.groupby('Material Number')['material_month_total_qty'].agg(
        mat_p05=lambda x: x.quantile(0.05),
        mat_p95=lambda x: x.quantile(0.95)
    ).reset_index()

    # Join stats to pair-month
    joined = pair_month.merge(material_stats, on='Material Number', how='left')
    joined['is_material_outlier'] = (
        (joined['current_month_total_qty'] < joined['mat_p05']) |
        (joined['current_month_total_qty'] > joined['mat_p95'])
    ) & joined['mat_p05'].notna() & joined['mat_p95'].notna()

    total_months = len(joined)
    mat_flagged = int(joined['is_material_outlier'].sum())
    print(f"Material-baseline flagged months: {mat_flagged:,} of {total_months:,} ({(mat_flagged/total_months if total_months else 0):.2%})")

    # Overlap with fixed pair-level monthly flags for pairs with insufficient history (<6 months)
    month_keys = ['Sold To number', 'Material Number', 'year_month']
    all_months = work.dropna(subset=['year_month'])[month_keys].drop_duplicates()
    months_per_pair = all_months.groupby(['Sold To number', 'Material Number']).size().rename('months_history').reset_index()

    fixed_flags = work[month_keys + ['is_monthly_qty_outside_typical_range']].drop_duplicates()
    fixed_flags = fixed_flags.merge(months_per_pair, on=['Sold To number', 'Material Number'], how='left')
    small_hist = fixed_flags['months_history'].fillna(0) < 6

    comp = joined.merge(fixed_flags, on=month_keys, how='left')
    fixed_true = comp['is_monthly_qty_outside_typical_range'].fillna(False)
    mat_true = comp['is_material_outlier'].fillna(False)
    small_mask = comp['months_history'].fillna(0) < 6

    both_small = int((fixed_true & mat_true & small_mask).sum())
    fixed_only_small = int((fixed_true & ~mat_true & small_mask).sum())
    mat_only_small = int((~fixed_true & mat_true & small_mask).sum())
    print("Overlap on pairs with <6 months:")
    print(f"  Both fixed and material True: {both_small:,}")
    print(f"  Fixed-only True: {fixed_only_small:,}")
    print(f"  Material-only True: {mat_only_small:,}")

def check_fulfillment_time_flags(df: pd.DataFrame) -> None:
    """
    Examine how 'is_unusual_fulfillment_time' varies with group sizes for
    (Material Number, Ship-To Party), highlighting small-sample inflation.
    """
    _print_header("FULFILLMENT TIME FLAGS VS. GROUP SIZE")
    required = {'Material Number', 'Ship-To Party'}
    flag_col = 'is_unusual_fulfillment_time'
    if not required.issubset(df.columns):
        print(f"Missing columns for fulfillment analysis: {sorted(required - set(df.columns))}")
        return
    if flag_col not in df.columns:
        print(f"Missing column '{flag_col}'. Skipping.")
        return

    grp = df.groupby(['Material Number', 'Ship-To Party'])
    sizes = grp.size().rename('n').reset_index()
    flag_counts = df.groupby(['Material Number', 'Ship-To Party'])[flag_col].sum().rename('flag_true').reset_index()
    stats = sizes.merge(flag_counts, on=['Material Number', 'Ship-To Party'], how='left')
    stats['flag_true'] = stats['flag_true'].fillna(0)
    stats['flag_rate'] = stats['flag_true'] / stats['n']

    buckets = pd.cut(
        stats['n'],
        bins=[0, 5, 10, 20, 50, 999999],
        right=True,
        labels=['1-5', '6-10', '11-20', '21-50', '51+']
    )
    stats['bucket'] = buckets

    bucket_summary = stats.groupby('bucket').agg(
        n_groups=('n', 'count'),
        mean_size=('n', 'mean'),
        mean_flag_rate=('flag_rate', 'mean')
    )
    print(bucket_summary.to_string())

    small_groups_rows = int(stats.loc[stats['n'] <= 5, 'n'].sum())
    total_rows = int(len(df))
    share = small_groups_rows / total_rows if total_rows else float('nan')
    print(f"\nRows from groups with <=5 samples: {small_groups_rows}/{total_rows} ({share:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostics for anomaly preprocessing flags")
    parser.add_argument("--file", required=True, help="Absolute path to exported CSV with features")
    # Rolling monthly diagnostics options
    parser.add_argument("--rolling-window", type=int, default=3, help="Trailing rolling window (months) for monthly totals")
    parser.add_argument("--rolling-z", type=float, default=2.0, help="Z-score threshold for rolling anomaly flag")
    parser.add_argument("--date-format", type=str, default=None, help="Explicit datetime format for 'Sales Document Created Date' (e.g. %m/%d/%y)")
    parser.add_argument("--min-months", type=int, default=6, help="Minimum history months for gating fixed monthly flags")
    parser.add_argument("--min-orders-in-month", type=int, default=0, help="Minimum orders in a flagged month for gating")
    args = parser.parse_args()

    df = load_dataframe(args.file)

    check_explanations_and_flags(df)
    check_rare_materials(df)
    check_ship_to_distribution(df)
    check_value_mismatch(df)
    check_quantity_range(df)
    check_monthly_volume_flags(df, date_format=args.date_format)
    check_fulfillment_time_flags(df)

    # Optional: Rolling monthly anomaly diagnostics
    try:
        check_monthly_volume_rolling(df, window=args.rolling_window, z_thresh=args.rolling_z, date_format=args.date_format)
    except Exception as e:
        _print_header("ROLLING MONTHLY DIAGNOSTICS")
        print(f"Skipping rolling diagnostics due to error: {e}")

    # Optional: Minimum-history gating for fixed monthly flags
    try:
        check_monthly_volume_min_history(
            df,
            min_months=args.min_months,
            min_orders_in_month=args.min_orders_in_month,
            date_format=args.date_format,
        )
    except Exception as e:
        _print_header("MONTHLY FLAGS WITH MIN HISTORY GATING")
        print(f"Skipping min-history gating due to error: {e}")

    # Optional: Material-level baseline comparison
    try:
        check_monthly_volume_material_baseline(df, date_format=args.date_format)
    except Exception as e:
        _print_header("MATERIAL-LEVEL MONTHLY BASELINE (p05–p95)")
        print(f"Skipping material baseline due to error: {e}")


if __name__ == "__main__":
    main()


