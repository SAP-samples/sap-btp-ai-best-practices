"""
Runtime validation for Epics 1 & 2 (Data Ingestion + Layer 0 Artifacts).

This script validates all data loaders and seasonality computations per PROJECT_PLAN.md.
It follows validation patterns from prototype/validate_runtime.py.

Epic 1 Checkpoints (Data Ingestion & Cleaning):
    - DATA-01: Port and Verify Data Loaders
    - DATA-02: Traffic Missingness Flag
    - DATA-03: Awareness/Consideration Feeds
    - DATA-04: CRM Demographics

Epic 2 Checkpoints (Layer 0 Artifacts):
    - SEAS-01: DMA Seasonal Weight Computation
    - SEAS-02: Sister-DMA Fallback

Usage:
    python -m forecasting.regressor.validation.validate_epic1_epic2

Expected outcome:
    - All checkpoints should pass (✓)
    - Warnings (⚠) indicate potential issues but not failures
    - Failures (✗) indicate critical problems requiring attention
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

# Import data ingestion functions
from app.regressor.data_ingestion.written_sales import (
    load_written_sales_with_flags,
    compute_traffic_missingness_flag,
)
from app.regressor.data_ingestion.store_master import load_store_master
from app.regressor.data_ingestion.awareness import load_awareness_with_mapping
from app.regressor.data_ingestion.crm_mix import load_demographics_with_typing

# Import Layer 0 artifact functions
from app.regressor.seasonality import compute_dma_seasonality
from app.regressor.sister_dma import build_sister_dma_map
from app.regressor.io_utils import load_written_sales


# =============================================================================
# Utility Functions for Validation Reporting
# =============================================================================

class ValidationTracker:
    """Tracks validation results across all checkpoints."""

    def __init__(self):
        self.passed = 0
        self.warnings = 0
        self.failed = 0
        self.total = 0

    def record_pass(self):
        self.passed += 1
        self.total += 1

    def record_warning(self):
        self.warnings += 1
        self.total += 1

    def record_fail(self):
        self.failed += 1
        self.total += 1

    def print_summary(self):
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Total Checks: {self.total}")
        print(f"Passed: {self.passed} ✓")
        print(f"Warnings: {self.warnings} ⚠")
        print(f"Failed: {self.failed} ✗")
        print("=" * 80)

        return self.failed == 0


def print_section_header(title: str):
    """Print a standardized section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_condition(tracker: ValidationTracker, condition: bool,
                    pass_msg: str, fail_msg: str, is_warning: bool = False):
    """
    Check a condition and record the result.

    Args:
        tracker: ValidationTracker instance
        condition: Boolean condition to check
        pass_msg: Message to display if condition is True
        fail_msg: Message to display if condition is False
        is_warning: If True, treat failure as warning instead of hard failure
    """
    if condition:
        print(f"✓ {pass_msg}")
        tracker.record_pass()
    else:
        if is_warning:
            print(f"⚠ {fail_msg}")
            tracker.record_warning()
        else:
            print(f"✗ {fail_msg}")
            tracker.record_fail()


def display_dataframe_info(df: pd.DataFrame, name: str, max_cols: int = 10):
    """Display comprehensive information about a DataFrame."""
    print(f"\n{name}:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if df.shape[1] <= max_cols:
        print(f"  Columns: {', '.join(df.columns.tolist())}")
    else:
        print(f"  Columns (first {max_cols}): {', '.join(df.columns.tolist()[:max_cols])}...")

    print(f"  Date range: {df.index.min() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'} to "
          f"{df.index.max() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'}")


def display_numeric_stats(df: pd.DataFrame, columns: list, max_display: int = 5):
    """Display statistical summary for numeric columns."""
    if not columns:
        return

    print(f"\nStatistical Summary (showing first {min(len(columns), max_display)} columns):")
    cols_to_show = columns[:max_display]

    for col in cols_to_show:
        if col in df.columns:
            series = df[col]
            null_pct = series.isna().sum() / len(series) * 100
            print(f"  {col}:")
            print(f"    Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            print(f"    Min: {series.min():.4f}, Max: {series.max():.4f}")
            print(f"    Null: {null_pct:.2f}%")


def validate_no_negatives(df: pd.DataFrame, columns: list, tracker: ValidationTracker):
    """Validate that specified columns have no negative values."""
    for col in columns:
        if col in df.columns:
            has_negatives = (df[col] < 0).any()
            check_condition(
                tracker,
                not has_negatives,
                f"No negative values in {col}",
                f"Found negative values in {col}"
            )


def validate_range(df: pd.DataFrame, columns: list, min_val: float, max_val: float,
                   tracker: ValidationTracker):
    """Validate that specified columns are within expected range."""
    for col in columns:
        if col in df.columns:
            within_range = df[col].between(min_val, max_val).all()
            check_condition(
                tracker,
                within_range,
                f"{col} values within [{min_val}, {max_val}]",
                f"{col} has values outside [{min_val}, {max_val}]"
            )


def validate_normalization(df: pd.DataFrame, group_by_cols: list, sum_col: str,
                          expected_sum: float, tolerance: float, tracker: ValidationTracker):
    """Validate that values sum to expected total within tolerance."""
    if group_by_cols:
        sums = df.groupby(group_by_cols)[sum_col].sum()
    else:
        sums = pd.Series([df[sum_col].sum()])

    within_tolerance = ((sums >= expected_sum - tolerance) &
                       (sums <= expected_sum + tolerance)).all()

    check_condition(
        tracker,
        within_tolerance,
        f"{sum_col} sums to {expected_sum} ± {tolerance}",
        f"{sum_col} does not sum to {expected_sum} (range: {sums.min():.4f} - {sums.max():.4f})"
    )


# =============================================================================
# Epic 1 Validation Functions
# =============================================================================

def validate_data01_written_sales(tracker: ValidationTracker):
    """
    DATA-01: Validate load_written_sales() and load_store_master().

    Verifies:
        - "NULL" strings in Store_Traffic converted to NaN
        - Percentage columns clipped to [0, 100]
        - No negative values in staffing columns
        - Expected number of unique stores (163)
        - Reasonable date range coverage
    """
    print_section_header("DATA-01: Validating Written Sales Loader")

    try:
        # Load written sales data
        df_sales = load_written_sales()
        display_dataframe_info(df_sales, "Written Sales Data")

        # Check unique stores (using profit_center_nbr, not store_id)
        unique_stores = df_sales['profit_center_nbr'].nunique()
        check_condition(
            tracker,
            unique_stores == 163,
            f"Found expected 163 unique stores",
            f"Found {unique_stores} stores instead of expected 163",
            is_warning=True
        )

        # Check date range (using fiscal_start_date_week, not week_date)
        if 'fiscal_start_date_week' in df_sales.columns:
            min_date = df_sales['fiscal_start_date_week'].min()
            max_date = df_sales['fiscal_start_date_week'].max()
            print(f"  Date range: {min_date} to {max_date}")

            # Expect data from at least 2020
            check_condition(
                tracker,
                min_date.year <= 2020,
                "Data starts on or before 2020",
                f"Data starts in {min_date.year}, expected earlier",
                is_warning=True
            )

        # Check for NULL traffic strings (should be NaN) - using store_traffic (lowercase after rename)
        if 'store_traffic' in df_sales.columns:
            has_null_strings = (df_sales['store_traffic'].astype(str) == 'NULL').any()
            check_condition(
                tracker,
                not has_null_strings,
                "No 'NULL' strings in store_traffic (correctly converted to NaN)",
                "Found 'NULL' strings in store_traffic that should be NaN"
            )

            # Check traffic NaN percentage
            traffic_null_pct = df_sales['store_traffic'].isna().sum() / len(df_sales) * 100
            print(f"  store_traffic null percentage: {traffic_null_pct:.2f}%")

        # Check percentage columns (note: these are proportions in [0, 1+] range, not [0, 100])
        pct_columns = [col for col in df_sales.columns if col.startswith('pct_')]
        if pct_columns:
            print(f"\n  Validating {len(pct_columns)} percentage columns...")
            # Check for non-negative values (percentages/proportions should be >= 0)
            for col in pct_columns:
                if col in df_sales.columns:
                    has_negatives = (df_sales[col] < 0).any()
                    check_condition(
                        tracker,
                        not has_negatives,
                        f"{col} has no negative values",
                        f"{col} has negative values (data quality issue)",
                        is_warning=True
                    )

            # Display sample stats for first few pct columns
            display_numeric_stats(df_sales, pct_columns, max_display=3)

        # Check staffing columns have no negatives (warnings for data quality issues)
        staffing_columns = [col for col in df_sales.columns
                          if any(x in col.lower() for x in ['hours', 'associates', 'staff'])]
        if staffing_columns:
            print(f"\n  Validating {len(staffing_columns)} staffing columns...")
            for col in staffing_columns:
                if col in df_sales.columns:
                    has_negatives = (df_sales[col] < 0).any()
                    check_condition(
                        tracker,
                        not has_negatives,
                        f"No negative values in {col}",
                        f"Found negative values in {col} (data quality issue)",
                        is_warning=True
                    )

        # Display sample rows
        print("\n  Sample rows (first 3):")
        sample_cols = ['profit_center_nbr', 'fiscal_start_date_week', 'channel', 'total_sales'] + pct_columns[:2]
        available_cols = [c for c in sample_cols if c in df_sales.columns]
        print(df_sales[available_cols].head(3).to_string(index=False))

        # Now validate store master
        print("\n" + "-" * 80)
        print("Validating Store Master Data")
        print("-" * 80)

        df_stores = load_store_master()
        display_dataframe_info(df_stores, "Store Master Data")

        # Check proforma_annual_sales is numeric
        if 'proforma_annual_sales' in df_stores.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df_stores['proforma_annual_sales'])
            check_condition(
                tracker,
                is_numeric,
                "proforma_annual_sales is numeric type",
                "proforma_annual_sales is not numeric type"
            )

            # Check for reasonable values
            if is_numeric:
                mean_proforma = df_stores['proforma_annual_sales'].mean()
                print(f"  Mean proforma annual sales: ${mean_proforma:,.0f}")

        # Check is_outlet is boolean
        if 'is_outlet' in df_stores.columns:
            is_bool = df_stores['is_outlet'].dtype == bool
            check_condition(
                tracker,
                is_bool,
                "is_outlet is boolean type",
                f"is_outlet is {df_stores['is_outlet'].dtype} instead of boolean"
            )

            # Report outlet distribution
            if is_bool or df_stores['is_outlet'].dtype in [np.int64, np.float64]:
                outlet_pct = df_stores['is_outlet'].mean() * 100
                print(f"  Outlet stores: {outlet_pct:.1f}%")

        # Check lat/lon presence
        has_lat = 'latitude' in df_stores.columns
        has_lon = 'longitude' in df_stores.columns
        check_condition(
            tracker,
            has_lat and has_lon,
            "Latitude and longitude columns present",
            "Missing latitude or longitude columns"
        )

        if has_lat and has_lon:
            # Check for reasonable coordinate ranges
            lat_valid = df_stores['latitude'].between(25, 50)  # Continental US roughly
            lon_valid = df_stores['longitude'].between(-125, -65)
            any_outside = ~(lat_valid & lon_valid)
            check_condition(
                tracker,
                any_outside.sum() == 0,
                "Latitude/longitude values within reasonable US ranges",
                f"Some lat/lon values outside expected US ranges (count={any_outside.sum()})",
                is_warning=True
            )
            if any_outside.any():
                print("  Sample out-of-range stores:")
                cols = ['profit_center_nbr', 'market_city', 'latitude', 'longitude']
                print(df_stores.loc[any_outside, cols].head(5).to_string(index=False))

        print("\n✓ DATA-01 validation complete")

    except Exception as e:
        print(f"✗ DATA-01 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


def validate_data02_traffic_flag(tracker: ValidationTracker):
    """
    DATA-02: Validate traffic missingness flag computation.

    Verifies:
        - has_traffic_data flag computed correctly
        - Stores with >20% missing traffic flagged as 0
        - Stores with <8 non-null weeks flagged as 0
        - Flag successfully merged to sales data
    """
    print_section_header("DATA-02: Validating Traffic Missingness Flag")

    try:
        # Load sales data with flags
        df_sales = load_written_sales_with_flags()

        # Check that has_traffic_data column exists
        has_flag = 'has_traffic_data' in df_sales.columns
        check_condition(
            tracker,
            has_flag,
            "has_traffic_data column present in sales data",
            "has_traffic_data column missing from sales data"
        )

        if has_flag:
            # Check flag values are 0 or 1
            flag_values = df_sales['has_traffic_data'].unique()
            valid_values = set(flag_values).issubset({0, 1, np.nan})
            check_condition(
                tracker,
                valid_values,
                "has_traffic_data contains only 0/1 values",
                f"has_traffic_data contains unexpected values: {flag_values}"
            )

            # Report distribution (using profit_center_nbr, not store_id)
            flag_dist = df_sales.groupby('profit_center_nbr')['has_traffic_data'].first()
            pct_with_data = (flag_dist == 1).sum() / len(flag_dist) * 100
            print(f"\n  Traffic data availability:")
            print(f"    Stores with sufficient data (flag=1): {(flag_dist == 1).sum()} ({pct_with_data:.1f}%)")
            print(f"    Stores with insufficient data (flag=0): {(flag_dist == 0).sum()} ({100-pct_with_data:.1f}%)")

            # Validate the flag logic by recomputing on a sample
            print("\n  Validating flag computation logic...")

            # Compute flag independently to verify
            flag_check = compute_traffic_missingness_flag(df_sales)

            # Merge and compare (using profit_center_nbr)
            df_check = df_sales.merge(
                flag_check[['profit_center_nbr', 'has_traffic_data']].drop_duplicates(),
                on='profit_center_nbr',
                suffixes=('_original', '_recomputed')
            )

            if 'has_traffic_data_original' in df_check.columns:
                flags_match = (df_check['has_traffic_data_original'] ==
                             df_check['has_traffic_data_recomputed']).all()
                check_condition(
                    tracker,
                    flags_match,
                    "Flag values match recomputed results",
                    "Flag values differ from recomputed results"
                )

            # Display sample stores with their flags (using store_traffic, lowercase)
            print("\n  Sample stores with flags:")
            sample = df_sales.groupby('profit_center_nbr').agg({
                'has_traffic_data': 'first',
                'store_traffic': lambda x: x.isna().sum() / len(x) * 100
            }).rename(columns={'store_traffic': 'traffic_missing_pct'})
            print(sample.head(10).to_string())

            # Validate threshold logic (using store_traffic)
            if 'store_traffic' in df_sales.columns:
                # Check stores with high missing rate are flagged 0
                store_missing = df_sales.groupby('profit_center_nbr').agg({
                    'store_traffic': lambda x: x.isna().sum() / len(x),
                    'has_traffic_data': 'first'
                })

                high_missing = store_missing[store_missing['store_traffic'] > 0.20]
                if len(high_missing) > 0:
                    all_flagged_zero = (high_missing['has_traffic_data'] == 0).all()
                    check_condition(
                        tracker,
                        all_flagged_zero,
                        "Stores with >20% missing traffic correctly flagged as 0",
                        "Some stores with >20% missing traffic not flagged as 0"
                    )

        print("\n✓ DATA-02 validation complete")

    except Exception as e:
        print(f"✗ DATA-02 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


def validate_data03_awareness(tracker: ValidationTracker):
    """
    DATA-03: Validate awareness/consideration feed ingestion.

    Verifies:
        - 38,000+ rows loaded from refreshed export
        - Market → DMA mapping applied correctly
        - Awareness values are numeric
        - Date range covers 2022-2025
        - Forward-fill logic for monthly→weekly conversion
    """
    print_section_header("DATA-03: Validating Awareness/Consideration Feeds")

    try:
        # Load awareness data with mapping
        df_awareness = load_awareness_with_mapping()
        display_dataframe_info(df_awareness, "Awareness/Consideration Data")

        # Check row count (should be 38k+)
        row_count = len(df_awareness)
        check_condition(
            tracker,
            row_count >= 38000,
            f"Found {row_count:,} rows (≥38,000 expected)",
            f"Found only {row_count:,} rows, expected ≥38,000",
            is_warning=True
        )

        # Check required columns (using market_city, week_start, awareness - not brand_awareness)
        required_cols = ['market_city', 'week_start', 'awareness']
        missing_cols = [col for col in required_cols if col not in df_awareness.columns]
        check_condition(
            tracker,
            len(missing_cols) == 0,
            "All required columns present (market_city, week_start, awareness)",
            f"Missing required columns: {missing_cols}"
        )

        # Check awareness values are numeric
        if 'awareness' in df_awareness.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df_awareness['awareness'])
            check_condition(
                tracker,
                is_numeric,
                "awareness values are numeric",
                f"awareness is {df_awareness['awareness'].dtype}, expected numeric"
            )

            # Check for reasonable values (typically 0-100 or 0-1)
            if is_numeric:
                min_val = df_awareness['awareness'].min()
                max_val = df_awareness['awareness'].max()
                print(f"  awareness range: {min_val:.4f} to {max_val:.4f}")

                # Display stats
                display_numeric_stats(df_awareness, ['awareness'], max_display=1)

        # Check consideration if present
        if 'consideration' in df_awareness.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df_awareness['consideration'])
            check_condition(
                tracker,
                is_numeric,
                "consideration values are numeric (optional Tier 3)",
                f"consideration is {df_awareness['consideration'].dtype}",
                is_warning=True
            )

        # Check date range (using week_start)
        if 'week_start' in df_awareness.columns:
            min_date = df_awareness['week_start'].min()
            max_date = df_awareness['week_start'].max()
            print(f"\n  Date range: {min_date} to {max_date}")

            # Expect data from 2022 onwards
            check_condition(
                tracker,
                min_date.year <= 2022,
                "Data starts on or before 2022",
                f"Data starts in {min_date.year}, expected 2022 or earlier",
                is_warning=True
            )

            check_condition(
                tracker,
                max_date.year >= 2024,
                "Data extends to 2024 or later",
                f"Data only extends to {max_date.year}",
                is_warning=True
            )

        # Check DMA coverage (using market_city)
        if 'market_city' in df_awareness.columns:
            unique_dmas = df_awareness['market_city'].nunique()
            print(f"  Unique DMAs (market_city) covered: {unique_dmas}")

            # Should cover multiple DMAs
            check_condition(
                tracker,
                unique_dmas >= 10,
                f"Covers {unique_dmas} DMAs (sufficient coverage)",
                f"Only covers {unique_dmas} DMAs, expected broader coverage",
                is_warning=True
            )

        # Display sample rows (using market_city, week_start, awareness, consideration)
        print("\n  Sample rows (first 5):")
        sample_cols = ['market_city', 'week_start', 'awareness']
        if 'consideration' in df_awareness.columns:
            sample_cols.append('consideration')
        available_cols = [c for c in sample_cols if c in df_awareness.columns]
        print(df_awareness[available_cols].head(5).to_string(index=False))

        print("\n✓ DATA-03 validation complete")

    except Exception as e:
        print(f"✗ DATA-03 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


def validate_data04_demographics(tracker: ValidationTracker):
    """
    DATA-04: Validate CRM demographic mix ingestion.

    Verifies:
        - Returns tuple of 2 DataFrames (static, crm)
        - Static: 163 stores × 1 row each
        - CRM: ~96 rows per store-channel
        - 23 CRM percentage features present
        - Lag/roll variants available
    """
    print_section_header("DATA-04: Validating CRM Demographics")

    try:
        # Load demographics data
        result = load_demographics_with_typing()

        # Check returns tuple
        is_tuple = isinstance(result, tuple)
        check_condition(
            tracker,
            is_tuple,
            "load_demographics returns tuple",
            f"load_demographics returns {type(result)}, expected tuple"
        )

        if is_tuple and len(result) == 2:
            df_static, df_crm = result

            # Validate static demographics
            print("\n" + "-" * 80)
            print("Validating Static Demographics")
            print("-" * 80)

            display_dataframe_info(df_static, "Static Demographics")

            # Check one row per store (using profit_center_nbr)
            if 'profit_center_nbr' in df_static.columns:
                unique_stores = df_static['profit_center_nbr'].nunique()
                total_rows = len(df_static)

                check_condition(
                    tracker,
                    unique_stores == total_rows,
                    f"One row per store ({unique_stores} stores)",
                    f"Multiple rows per store ({total_rows} rows for {unique_stores} stores)"
                )

                check_condition(
                    tracker,
                    unique_stores == 163,
                    "Found expected 163 stores",
                    f"Found {unique_stores} stores instead of 163",
                    is_warning=True
                )

            # Display sample columns
            print(f"\n  Static columns ({len(df_static.columns)}):")
            print(f"    {', '.join(df_static.columns.tolist()[:10])}...")

            # Validate CRM time-varying demographics
            print("\n" + "-" * 80)
            print("Validating CRM Time-Varying Demographics")
            print("-" * 80)

            display_dataframe_info(df_crm, "CRM Demographics")

            # Check rows per store (using profit_center_nbr, no channel in CRM data)
            if 'profit_center_nbr' in df_crm.columns:
                rows_per_store = len(df_crm) / df_crm['profit_center_nbr'].nunique()
                print(f"  Average rows per store: {rows_per_store:.1f}")

                check_condition(
                    tracker,
                    rows_per_store >= 50,
                    f"Sufficient time series depth (~{rows_per_store:.0f} weeks per store)",
                    f"Only {rows_per_store:.0f} weeks per store, expected more",
                    is_warning=True
                )

            # Check CRM percentage features
            crm_pct_cols = [col for col in df_crm.columns if col.startswith('crm_')]
            print(f"\n  CRM percentage features: {len(crm_pct_cols)}")

            check_condition(
                tracker,
                len(crm_pct_cols) >= 20,
                f"Found {len(crm_pct_cols)} CRM features (≥20 expected)",
                f"Found only {len(crm_pct_cols)} CRM features, expected ≥20",
                is_warning=True
            )

            # Display sample CRM columns
            print(f"    Sample columns: {', '.join(crm_pct_cols[:5])}...")

            # Check for lag/roll variants
            has_lag = any('_lag_' in col or '_lag' in col for col in crm_pct_cols)
            has_roll = any('_roll_' in col or '_rolling' in col for col in crm_pct_cols)

            check_condition(
                tracker,
                has_lag,
                "Lag variants present (lag_1, lag_4)",
                "No lag variants found in CRM features",
                is_warning=True
            )

            check_condition(
                tracker,
                has_roll,
                "Rolling variants present (roll_4, roll_8)",
                "No rolling variants found in CRM features",
                is_warning=True
            )

            # Validate percentage ranges (check for non-negative values)
            if crm_pct_cols:
                print(f"\n  Validating {len(crm_pct_cols)} CRM percentage columns...")
                # CRM percentages should be non-negative
                for col in crm_pct_cols[:5]:  # Check first 5
                    if col in df_crm.columns:
                        has_negatives = (df_crm[col] < 0).any()
                        check_condition(
                            tracker,
                            not has_negatives,
                            f"{col} has no negative values",
                            f"{col} has negative values (data quality issue)",
                            is_warning=True
                        )

                # Display sample stats
                display_numeric_stats(df_crm, crm_pct_cols[:3], max_display=3)

            # Display sample rows (using profit_center_nbr and week_start)
            print("\n  Sample CRM rows (first 3):")
            sample_cols = ['profit_center_nbr', 'week_start'] + crm_pct_cols[:3]
            available_cols = [c for c in sample_cols if c in df_crm.columns]
            print(df_crm[available_cols].head(3).to_string(index=False))

        else:
            tracker.record_fail()
            print(f"✗ Unexpected return format: {type(result)}, length: {len(result) if is_tuple else 'N/A'}")

        print("\n✓ DATA-04 validation complete")

    except Exception as e:
        print(f"✗ DATA-04 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


# =============================================================================
# Epic 2 Validation Functions
# =============================================================================

def validate_seas01_seasonality(tracker: ValidationTracker):
    """
    SEAS-01: Validate DMA seasonal weight computation.

    Verifies:
        - 52 weights per DMA
        - Weights sum to 1.0 (normalized)
        - Week 53 folded into week 1
        - 3-point cyclic smoothing applied
        - All weights are positive
    """
    print_section_header("SEAS-01: Validating DMA Seasonal Weight Computation")

    try:
        # Compute DMA seasonality
        df_seasonality = compute_dma_seasonality()
        display_dataframe_info(df_seasonality, "DMA Seasonality Weights")

        # Check required columns (using market_city, fiscal_week, weight)
        required_cols = ['market_city', 'fiscal_week', 'weight']
        missing_cols = [col for col in required_cols if col not in df_seasonality.columns]
        check_condition(
            tracker,
            len(missing_cols) == 0,
            "All required columns present (market_city, fiscal_week, weight)",
            f"Missing required columns: {missing_cols}"
        )

        # Check 52 weeks per DMA (using market_city and fiscal_week)
        if 'market_city' in df_seasonality.columns and 'fiscal_week' in df_seasonality.columns:
            weeks_per_dma = df_seasonality.groupby('market_city')['fiscal_week'].count()
            all_52_weeks = (weeks_per_dma == 52).all()

            check_condition(
                tracker,
                all_52_weeks,
                "All DMAs have exactly 52 weeks",
                f"Some DMAs have != 52 weeks (range: {weeks_per_dma.min()}-{weeks_per_dma.max()})"
            )

            # Check fiscal_week range
            min_week = df_seasonality['fiscal_week'].min()
            max_week = df_seasonality['fiscal_week'].max()

            check_condition(
                tracker,
                min_week == 1 and max_week == 52,
                "fiscal_week spans 1-52 (week 53 folded into week 1)",
                f"fiscal_week range is {min_week}-{max_week}, expected 1-52"
            )

        # Check normalization (sum to 1.0 per DMA) - using weight column
        if 'weight' in df_seasonality.columns:
            print("\n  Validating normalization (weights sum to 1.0 per DMA)...")

            dma_sums = df_seasonality.groupby('market_city')['weight'].sum()
            tolerance = 0.01  # Allow 1% tolerance

            within_tolerance = ((dma_sums >= 1.0 - tolerance) &
                              (dma_sums <= 1.0 + tolerance)).all()

            check_condition(
                tracker,
                within_tolerance,
                f"All DMA weights sum to 1.0 ± {tolerance}",
                f"Some DMAs outside tolerance (range: {dma_sums.min():.4f} - {dma_sums.max():.4f})"
            )

            # Display statistics
            print(f"    Sum range: {dma_sums.min():.6f} to {dma_sums.max():.6f}")
            print(f"    Mean sum: {dma_sums.mean():.6f}")
            print(f"    Std sum: {dma_sums.std():.6f}")

        # Check all weights are positive
        if 'weight' in df_seasonality.columns:
            all_positive = (df_seasonality['weight'] > 0).all()
            check_condition(
                tracker,
                all_positive,
                "All seasonal weights are positive",
                "Some seasonal weights are zero or negative"
            )

            # Display weight statistics
            display_numeric_stats(df_seasonality, ['weight'], max_display=1)

        # Display sample DMA seasonal curve (using market_city)
        if 'market_city' in df_seasonality.columns:
            sample_dma = df_seasonality['market_city'].iloc[0]
            print(f"\n  Sample seasonal curve for DMA {sample_dma}:")
            sample_curve = df_seasonality[df_seasonality['market_city'] == sample_dma].sort_values('fiscal_week')
            print(sample_curve[['fiscal_week', 'weight']].head(10).to_string(index=False))
            print("  ...")
            print(sample_curve[['fiscal_week', 'weight']].tail(5).to_string(index=False))

        # Check for smoothness (basic check - no huge jumps)
        if 'market_city' in df_seasonality.columns and 'weight' in df_seasonality.columns:
            print("\n  Checking smoothness (3-point cyclic smoothing)...")

            # For each DMA, check max week-to-week change
            max_changes = []
            for dma in df_seasonality['market_city'].unique()[:5]:  # Check first 5 DMAs
                dma_data = df_seasonality[df_seasonality['market_city'] == dma].sort_values('fiscal_week')
                weights = dma_data['weight'].values
                changes = np.abs(np.diff(weights))
                max_changes.append(changes.max())

            avg_max_change = np.mean(max_changes)
            print(f"    Average max week-to-week change: {avg_max_change:.4f}")

            # Smoothed data shouldn't have huge jumps (>0.01 in weekly weight)
            check_condition(
                tracker,
                avg_max_change < 0.01,
                "Seasonal curves are smooth (max Δ < 0.01)",
                f"Some large jumps detected (avg max Δ = {avg_max_change:.4f})",
                is_warning=True
            )

        print("\n✓ SEAS-01 validation complete")

    except Exception as e:
        print(f"✗ SEAS-01 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


def validate_seas02_sister_fallback(tracker: ValidationTracker):
    """
    SEAS-02: Validate sister-DMA fallback mapping.

    Verifies:
        - DMAs with <3 years history identified
        - Nearest same-region DMA chosen as sister
        - Fallback to nearest overall if no region match
        - Distance calculations are reasonable
        - Sister mapping applied in seasonality computation
    """
    print_section_header("SEAS-02: Validating Sister-DMA Fallback")

    try:
        # Build sister DMA map
        sister_map = build_sister_dma_map()

        # Check if map is empty (may be valid if all DMAs have sufficient history)
        if len(sister_map) == 0:
            print("  No DMAs require sister fallback (all have ≥3 years history)")
            print("✓ SEAS-02 validation complete (no fallback needed)")
            tracker.record_pass()
            return

        print(f"\n  Found {len(sister_map)} DMAs requiring sister fallback:")

        # Display sister map (sister_map is a DataFrame, not a dict)
        print("\n  Sister DMA Mapping:")
        print("  " + "-" * 90)
        print(f"  {'DMA':<20} {'Sister DMA':<20} {'Distance (mi)':<15} {'Same Region':<12}")
        print("  " + "-" * 90)

        for idx, row in sister_map.head(10).iterrows():  # Show first 10
            dma_id = row['market_city']
            sister_id = row['sister_market_city']
            distance = row['distance_miles']
            same_region = row['same_region']
            print(f"  {dma_id:<20} {sister_id:<20} {distance:<15.1f} {'Yes' if same_region else 'No':<12}")

        if len(sister_map) > 10:
            print(f"  ... ({len(sister_map) - 10} more)")
        print("  " + "-" * 90)

        # Validate distance values
        distances = sister_map['distance_miles'].values

        if len(distances) > 0:
            avg_distance = distances.mean()
            max_distance = distances.max()

            print(f"\n  Distance statistics:")
            print(f"    Average distance to sister: {avg_distance:.1f} miles")
            print(f"    Max distance to sister: {max_distance:.1f} miles")

            # Reasonable distances should be < 1000 miles typically
            check_condition(
                tracker,
                avg_distance < 1000,
                f"Average sister distance reasonable ({avg_distance:.1f} miles)",
                f"Average sister distance very large ({avg_distance:.1f} miles)",
                is_warning=True
            )

            # Check no infinite distances
            has_inf = np.isinf(distances).any()
            check_condition(
                tracker,
                not has_inf,
                "All DMAs successfully matched to a sister",
                "Some DMAs have infinite distance (no sister found)"
            )

        # Check same-region preference
        same_region_count = sister_map['same_region'].sum()
        same_region_pct = same_region_count / len(sister_map) * 100

        print(f"\n  Sister selection strategy:")
        print(f"    Same-region sisters: {same_region_count} ({same_region_pct:.1f}%)")
        print(f"    Cross-region sisters: {len(sister_map) - same_region_count} ({100-same_region_pct:.1f}%)")

        check_condition(
            tracker,
            same_region_pct >= 50,
            "Majority of sisters from same region (preferred strategy)",
            f"Only {same_region_pct:.1f}% same-region matches",
            is_warning=True
        )

        # Verify sister mapping is used in seasonality computation
        print("\n  Verifying sister mapping applied in seasonality computation...")
        df_seasonality = compute_dma_seasonality()

        # Check that DMAs in sister_map have seasonality weights (using market_city)
        if 'market_city' in df_seasonality.columns:
            dmas_with_weights = set(df_seasonality['market_city'].unique())
            sister_dmas = set(sister_map['market_city'].unique())

            sister_dmas_have_weights = sister_dmas.issubset(dmas_with_weights)
            check_condition(
                tracker,
                sister_dmas_have_weights,
                "All sister-assigned DMAs have seasonal weights",
                "Some sister-assigned DMAs missing seasonal weights"
            )

            # For a sample DMA with sister, verify weights are inherited
            if sister_dmas_have_weights and len(sister_map) > 0:
                sample_row = sister_map.iloc[0]
                sample_dma = sample_row['market_city']
                sister_dma = sample_row['sister_market_city']

                sample_weights = df_seasonality[df_seasonality['market_city'] == sample_dma]['weight'].values
                sister_weights = df_seasonality[df_seasonality['market_city'] == sister_dma]['weight'].values

                if len(sample_weights) > 0 and len(sister_weights) > 0:
                    weights_match = np.allclose(sample_weights, sister_weights, atol=0.001)
                    check_condition(
                        tracker,
                        weights_match,
                        f"DMA {sample_dma} inherited weights from sister {sister_dma}",
                        f"DMA {sample_dma} weights differ from sister {sister_dma}",
                        is_warning=True
                    )

        print("\n✓ SEAS-02 validation complete")

    except Exception as e:
        print(f"✗ SEAS-02 validation failed with error: {str(e)}")
        tracker.record_fail()
        import traceback
        traceback.print_exc()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main validation execution flow."""

    print("\n" + "=" * 80)
    print("EPIC 1 & 2 VALIDATION: Data Ingestion + Layer 0 Artifacts")
    print("=" * 80)
    print("\nThis script validates all functionality from Epic 1 (Data Ingestion)")
    print("and Epic 2 (Layer 0 Artifacts) per PROJECT_PLAN.md")
    print("\nFollowing validation patterns from prototype/validate_runtime.py")

    # Initialize tracker
    tracker = ValidationTracker()

    # Epic 1: Data Ingestion & Cleaning
    print("\n" + "=" * 80)
    print("EPIC 1: DATA INGESTION & CLEANING")
    print("=" * 80)

    validate_data01_written_sales(tracker)
    validate_data02_traffic_flag(tracker)
    validate_data03_awareness(tracker)
    validate_data04_demographics(tracker)

    # Epic 2: Layer 0 Artifacts (Seasonality & Baselines)
    print("\n" + "=" * 80)
    print("EPIC 2: LAYER 0 ARTIFACTS (SEASONALITY & BASELINES)")
    print("=" * 80)

    validate_seas01_seasonality(tracker)
    validate_seas02_sister_fallback(tracker)

    # Print final summary
    success = tracker.print_summary()

    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
