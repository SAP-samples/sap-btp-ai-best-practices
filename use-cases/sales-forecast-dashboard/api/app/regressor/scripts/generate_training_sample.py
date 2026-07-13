"""
Generate Training Table Sample - Interactive Script

This script generates a sample canonical training table with user-specified parameters
for manual validation and inspection of the EPIC 4 feature engineering pipeline.

Usage:
    # Interactive mode (prompts for parameters)
    python -m forecasting.regressor.scripts.generate_training_sample

    # Command-line mode
    python -m forecasting.regressor.scripts.generate_training_sample \
        --model-variant B \
        --horizons 1 4 \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --output sample_model_b.csv

Examples:
    # Generate Model B with horizons 1-4 for recent data
    python -m forecasting.regressor.scripts.generate_training_sample -m B --horizons 1 4

    # Generate Model A with horizons 1-13 for full year 2024
    python -m forecasting.regressor.scripts.generate_training_sample \
        -m A --horizons 1 13 --start-date 2024-01-01 --end-date 2024-12-31

    # Quick test with just horizon=1
    python -m forecasting.regressor.scripts.generate_training_sample -m B --horizons 1 1
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from app.regressor.etl.canonical_table import build_canonical_training_table


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_table_summary(df: pd.DataFrame, model_variant: str):
    """Print detailed summary of the generated table."""
    print_section(f"Model {model_variant} Training Table Summary")

    # Basic dimensions
    print(f"\nDataFrame Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Date range
    if 'origin_week_date' in df.columns and 'target_week_date' in df.columns:
        print(f"\nOrigin Date Range:")
        print(f"  First: {df['origin_week_date'].min()}")
        print(f"  Last:  {df['origin_week_date'].max()}")
        print(f"\nTarget Date Range:")
        print(f"  First: {df['target_week_date'].min()}")
        print(f"  Last:  {df['target_week_date'].max()}")

    # Horizon distribution
    if 'horizon' in df.columns:
        print(f"\nHorizon Distribution:")
        horizon_counts = df['horizon'].value_counts().sort_index()
        for h, count in horizon_counts.items():
            print(f"  h={h:2d}: {count:6,} rows")

    # Channel distribution
    if 'channel' in df.columns:
        print(f"\nChannel Distribution:")
        for channel, count in df['channel'].value_counts().items():
            pct = 100 * count / len(df)
            print(f"  {channel:5s}: {count:6,} rows ({pct:5.1f}%)")

    # Store count
    if 'profit_center_nbr' in df.columns:
        n_stores = df['profit_center_nbr'].nunique()
        print(f"\nUnique Stores: {n_stores}")

    # Column categories
    print(f"\nColumn Categories:")

    key_cols = [c for c in df.columns if c in [
        'profit_center_nbr', 'dma', 'channel', 'origin_week_date',
        'target_week_date', 'horizon'
    ]]
    print(f"  Keys:     {len(key_cols):2d} - {key_cols}")

    label_cols = [c for c in df.columns if c.startswith('label_')]
    print(f"  Labels:   {len(label_cols):2d} - {label_cols}")

    feature_cols = [c for c in df.columns
                    if c not in key_cols + label_cols
                    and c not in ['has_traffic_data', 'total_sales', 'order_count',
                                  'store_traffic', 'aur', 'allocated_web_traffic']]
    print(f"  Features: {len(feature_cols):2d}")

    # Feature breakdown by category
    print(f"\nFeature Breakdown:")

    feature_patterns = {
        'Time-Varying': ['dma_seasonal_weight', 'woy', 'sin_woy', 'cos_woy',
                        'is_holiday', 'fiscal_year', 'fiscal_period'],
        'Sales/AOV Lags': ['log_sales_lag_', 'log_sales_roll_', 'AOV_roll_'],
        'Conversion': ['ConversionRate_'],
        'Web Traffic': ['allocated_web_traffic_'],
        'Omnichannel': ['pct_omni_channel_'],
        'Product Mix': ['pct_value_product_', 'pct_premium_product_', 'pct_white_glove_'],
        'Static DNA': ['proforma_annual_sales', 'is_outlet', 'weeks_since_open', 'sq_ft'],
        'Cannibalization': ['cannibalization_pressure', 'min_dist_new_store', 'num_new_stores_'],
        'Awareness': ['brand_awareness', 'brand_consideration'],
        'CRM': ['crm_']
    }

    for category, patterns in feature_patterns.items():
        matches = [f for f in feature_cols if any(p in f for p in patterns)]
        if matches:
            nan_rate = df[matches].isna().mean().mean() * 100
            print(f"  {category:20s}: {len(matches):2d} features (avg {nan_rate:5.1f}% NaN)")

    # Overall NaN rate
    print(f"\nOverall NaN Rate:")
    total_nan_rate = df[feature_cols].isna().mean().mean() * 100
    print(f"  All features: {total_nan_rate:.1f}%")

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\nMemory Usage: {memory_mb:.1f} MB")


def print_sample_rows(df: pd.DataFrame, n: int = 5):
    """Print sample rows from the table."""
    print_section(f"Sample Rows (first {n})")

    # Select key columns for display
    display_cols = [
        'profit_center_nbr', 'channel', 'origin_week_date', 'horizon',
        'target_week_date', 'label_log_sales'
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # Add a few feature columns
    feature_cols = [c for c in df.columns
                    if c not in display_cols
                    and not c.startswith('label_')
                    and c not in ['has_traffic_data', 'total_sales', 'order_count',
                                  'store_traffic', 'aur']]
    display_cols.extend(feature_cols[:5])

    print("\n" + df[display_cols].head(n).to_string(index=False))


def generate_training_table(
    model_variant: str,
    horizons: range,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_crm: bool = False,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate canonical training table with specified parameters.

    Parameters
    ----------
    model_variant : str
        'A' (actionable/explainability) or 'B' (production/full)
    horizons : range
        Range of horizons to generate (e.g., range(1, 5) for h=1-4)
    start_date : Optional[str]
        Start date for origin_week_date filter (YYYY-MM-DD)
    end_date : Optional[str]
        End date for origin_week_date filter (YYYY-MM-DD)
    include_crm : bool, default=False
        Include CRM demographic features
    output_path : Optional[str]
        Path to save CSV output (optional)

    Returns
    -------
    pd.DataFrame
        Generated training table
    """
    print_section(f"Generating Model {model_variant} Training Table")

    print(f"\nParameters:")
    print(f"  Model Variant:  {model_variant}")
    print(f"  Horizons:       {list(horizons)}")
    print(f"  Start Date:     {start_date or 'All available'}")
    print(f"  End Date:       {end_date or 'All available'}")
    print(f"  Include CRM:    {include_crm}")

    print(f"\nBuilding canonical training table...")
    start_time = datetime.now()

    # Build table
    df = build_canonical_training_table(
        horizons=horizons,
        include_features=True,
        model_variant=model_variant,
        include_crm=include_crm
    )

    # Apply date filters if specified
    if start_date or end_date:
        print(f"\nApplying date filters...")
        original_rows = len(df)

        if start_date:
            df = df[df['origin_week_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['origin_week_date'] <= pd.to_datetime(end_date)]

        filtered_rows = len(df)
        print(f"  Filtered from {original_rows:,} to {filtered_rows:,} rows")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {elapsed:.1f} seconds")

    # Print summary
    print_table_summary(df, model_variant)
    print_sample_rows(df)

    # Save if requested
    if output_path:
        print_section("Saving Output")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving to: {output_file}")
        df.to_csv(output_file, index=False)

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"File size: {file_size_mb:.1f} MB")

    return df


def interactive_mode() -> Tuple[str, range, Optional[str], Optional[str], bool, Optional[str]]:
    """Interactive prompts for parameters."""
    print_section("Generate Training Table - Interactive Mode")

    # Model variant
    print("\nModel Variant:")
    print("  A - Actionable/Explainability (business levers only, no autoregressive features)")
    print("  B - Production/Full (all features including lags/rolls)")
    while True:
        model_variant = input("\nSelect model variant [A/B]: ").strip().upper()
        if model_variant in ['A', 'B']:
            break
        print("Invalid choice. Please enter 'A' or 'B'.")

    # Horizons
    print("\nHorizon Range:")
    print("  Examples: 1-4 (near-term), 1-13 (quarter), 1-52 (full year)")
    while True:
        try:
            h_start = int(input("  Start horizon: ").strip())
            h_end = int(input("  End horizon: ").strip())
            if 1 <= h_start <= h_end <= 52:
                horizons = range(h_start, h_end + 1)
                break
            print("Invalid range. Start and end must be between 1 and 52.")
        except ValueError:
            print("Invalid input. Please enter numbers.")

    # Date filters
    print("\nDate Filters (optional, press Enter to skip):")
    start_date = input("  Start date (YYYY-MM-DD): ").strip() or None
    end_date = input("  End date (YYYY-MM-DD): ").strip() or None

    # CRM features
    print("\nCRM Demographics:")
    include_crm_str = input("  Include CRM features? [y/N]: ").strip().lower()
    include_crm = include_crm_str in ['y', 'yes']

    # Output path
    print("\nOutput (optional):")
    output_path = input("  Save to CSV (path or press Enter to skip): ").strip() or None

    return model_variant, horizons, start_date, end_date, include_crm, output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate canonical training table samples for validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '-m', '--model-variant',
        choices=['A', 'B'],
        help="Model variant: A (actionable) or B (production)"
    )

    parser.add_argument(
        '--horizons',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help="Horizon range (e.g., 1 4 for h=1-4)"
    )

    parser.add_argument(
        '--start-date',
        help="Start date filter (YYYY-MM-DD)"
    )

    parser.add_argument(
        '--end-date',
        help="End date filter (YYYY-MM-DD)"
    )

    parser.add_argument(
        '--include-crm',
        action='store_true',
        help="Include CRM demographic features"
    )

    parser.add_argument(
        '-o', '--output',
        help="Output CSV file path"
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help="Run in interactive mode (prompts for parameters)"
    )

    # Parse args
    args = parser.parse_args()

    # Determine mode
    if args.interactive or not (args.model_variant and args.horizons):
        # Interactive mode
        model_variant, horizons, start_date, end_date, include_crm, output_path = interactive_mode()
    else:
        # Command-line mode
        model_variant = args.model_variant
        horizons = range(args.horizons[0], args.horizons[1] + 1)
        start_date = args.start_date
        end_date = args.end_date
        include_crm = args.include_crm
        output_path = args.output

    # Generate table
    try:
        df = generate_training_table(
            model_variant=model_variant,
            horizons=horizons,
            start_date=start_date,
            end_date=end_date,
            include_crm=include_crm,
            output_path=output_path
        )

        print_section("Generation Complete")
        print(f"\n✓ Successfully generated {len(df):,} rows for Model {model_variant}")
        print(f"✓ Columns: {len(df.columns)}")

        if output_path:
            print(f"✓ Saved to: {output_path}")

        print("\nNext Steps:")
        print("  1. Inspect the sample rows above")
        print("  2. Check feature coverage and NaN rates")
        print("  3. Verify date ranges and horizon distribution")
        if output_path:
            print(f"  4. Open {output_path} in Excel/Pandas for detailed inspection")

        return df

    except Exception as e:
        print_section("ERROR")
        print(f"\n✗ Failed to generate training table: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
