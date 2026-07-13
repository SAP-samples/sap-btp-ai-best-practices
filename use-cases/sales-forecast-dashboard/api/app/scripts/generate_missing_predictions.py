"""
Generate Missing Predictions for 2025 (Nov-Dec)

This script generates predictions for horizons that are missing from the
FIRST JANUARY 2025 ORIGIN in PREDICTIONS_BM table.

The January 2025 origin may only have predictions through October (horizons 1-42).
This script generates the missing horizons 43-52 to complete the 52-week forecast.

The script:
1. Finds the FIRST 2025 origin_week_date per store in PREDICTIONS_BM
2. Identifies missing horizons for that origin (expected: 52 weeks, actual: may be 42)
3. Generates baseline features for missing weeks using seasonal naive approach
   (same business levers as same week in 2024)
4. Runs the inference pipeline to generate predictions
5. Inserts results into PREDICTIONS_BM in HANA with the January 2025 origin

Usage:
    python -m app.scripts.generate_missing_predictions
    python -m app.scripts.generate_missing_predictions --dry-run
    python -m app.scripts.generate_missing_predictions --channel B&M

Regeneration mode (to fix data quality issues like missing dma_seasonal_weight):
    python -m app.scripts.generate_missing_predictions --regenerate-from 2025-11-01
    python -m app.scripts.generate_missing_predictions --regenerate-from 2025-11-01 --regenerate-to 2025-12-29
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.hana_loader import (
    get_hana_connection,
    load_predictions_filtered,
    load_model_b_filtered,
    load_calendar,
    insert_predictions_bm,
    delete_predictions_by_date_range,
    close_connection,
    parse_fiscal_week,
)
from app.regressor.baseline_generator import BaselineGenerator, BaselineConfig
from app.regressor.pipelines.inference import InferencePipeline
from app.regressor.configs import InferenceConfig
from app.regressor.features.model_views import MODEL_A_FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "agent" / "input" / "checkpoints"
END_OF_YEAR_DATE = "2025-12-29"  # Last Monday of 2025

# Columns that exist in PREDICTIONS_BM table (used to filter before insert)
# This prevents inserting derived/intermediate columns that don't exist in the table
PREDICTIONS_BM_COLUMNS = [
    'profit_center_nbr', 'dma', 'channel', 'origin_week_date', 'target_week_date',
    'horizon',
    'pred_sales_mean',  # Needed by export.py for Sales
    'pred_sales_p50', 'pred_sales_p90',
    'pred_aov_mean', 'pred_aov_p50', 'pred_aov_p90',
    'pred_traffic_p10', 'pred_traffic_p50', 'pred_traffic_p90',
    'pred_log_orders',  # Needed by export.py for Orders calculation
    'pred_logit_conversion',  # Needed by export.py for Conversion Rate (B&M only)
    'is_outlet', 'is_new_store', 'is_comp_store', 'store_design_sf', 'merchandising_sf',
    'top_features_pred_log_sales',
    'dma_seasonal_weight',  # Needed by export.py for Seasonal Weight (AUV analysis)
]


def get_all_2025_target_weeks() -> List[date]:
    """
    Get all target week dates (Mondays) for calendar year 2025.

    Returns:
        List of Monday dates from 2025-01-06 to 2025-12-29
    """
    # Generate all Mondays in 2025
    start = date(2025, 1, 6)  # First Monday of 2025
    end = date(2025, 12, 29)  # Last Monday of 2025

    weeks = []
    current = start
    while current <= end:
        weeks.append(current)
        current += timedelta(days=7)

    logger.info(f"Total target weeks in 2025: {len(weeks)} (from {start} to {end})")
    return weeks


def find_existing_predictions(channel: str = "B&M") -> pd.DataFrame:
    """
    Load existing predictions from PREDICTIONS_BM for 2025.

    Args:
        channel: Channel to query ("B&M" or "WEB")

    Returns:
        DataFrame with profit_center_nbr and target_week_date columns
    """
    logger.info(f"Loading existing {channel} predictions for 2025...")

    df = load_predictions_filtered(
        channel=channel,
        min_target_date="2025-01-01",
        columns=["profit_center_nbr", "target_week_date", "channel"],
    )

    # Convert dates
    df["target_week_date"] = pd.to_datetime(df["target_week_date"]).dt.date

    logger.info(f"Found {len(df)} existing prediction rows for {channel}")
    logger.info(f"Stores with predictions: {df['profit_center_nbr'].nunique()}")

    if len(df) > 0:
        min_date = df["target_week_date"].min()
        max_date = df["target_week_date"].max()
        logger.info(f"Date range: {min_date} to {max_date}")

    return df


def find_missing_weeks_per_store(
    existing_predictions: pd.DataFrame,
    all_2025_weeks: List[date],
) -> Dict[int, List[date]]:
    """
    For each store with existing predictions, find which 2025 weeks are missing
    AFTER their last existing prediction.

    We only look forward from the last prediction date, not backward.
    A store that opened mid-year legitimately has no data for earlier weeks.

    Args:
        existing_predictions: DataFrame with profit_center_nbr and target_week_date
        all_2025_weeks: List of all Monday dates in 2025

    Returns:
        Dict mapping store_id -> list of missing target_week_dates (forward only)
    """
    end_of_year = max(all_2025_weeks)
    missing_by_store: Dict[int, List[date]] = {}

    # Group by store
    for store_id, group in existing_predictions.groupby("profit_center_nbr"):
        existing_weeks = set(group["target_week_date"].tolist())
        last_prediction_date = max(existing_weeks)

        # Only find missing weeks AFTER the last prediction date
        future_weeks = [w for w in all_2025_weeks if w > last_prediction_date]
        missing_weeks = sorted(set(future_weeks) - existing_weeks)

        if missing_weeks:
            missing_by_store[int(store_id)] = missing_weeks

    # Summary statistics
    total_stores = existing_predictions["profit_center_nbr"].nunique()
    stores_with_missing = len(missing_by_store)
    total_missing_weeks = sum(len(weeks) for weeks in missing_by_store.values())

    logger.info(f"Stores with missing future weeks: {stores_with_missing}/{total_stores}")
    logger.info(f"Total missing store-weeks to generate: {total_missing_weeks}")

    if missing_by_store:
        # Show range of missing weeks
        all_missing_dates = [d for weeks in missing_by_store.values() for d in weeks]
        logger.info(f"Missing week range: {min(all_missing_dates)} to {max(all_missing_dates)}")

    return missing_by_store


def get_latest_origin_date_per_store(channel: str = "B&M") -> Dict[int, date]:
    """
    Get the latest origin_week_date per store from MODEL_B.

    This is used as the origin date for generating baselines for missing weeks.

    Args:
        channel: Channel to query

    Returns:
        Dict mapping store_id -> latest origin_week_date
    """
    logger.info(f"Finding latest origin dates per store for {channel}...")

    # Load MODEL_B filtered by channel (loads all columns)
    df = load_model_b_filtered(channel=channel)

    # Convert and find max per store
    df["origin_week_date"] = pd.to_datetime(df["origin_week_date"]).dt.date
    latest_dates = df.groupby("profit_center_nbr")["origin_week_date"].max().to_dict()

    logger.info(f"Found latest origin dates for {len(latest_dates)} stores")

    return latest_dates


def get_first_2025_origin_per_store(channel: str = "B&M") -> Dict[int, date]:
    """
    Get the first origin_week_date >= 2025-01-01 per store from PREDICTIONS_BM.

    This is used as the origin date for generating missing horizons.
    The January 2025 origin provides the consistent starting point for 52-week forecasts.

    Args:
        channel: Channel to query

    Returns:
        Dict mapping store_id -> first 2025 origin_week_date
    """
    logger.info(f"Finding first 2025 origin dates per store for {channel}...")

    # Load predictions to find existing origins
    df = load_predictions_filtered(
        channel=channel,
        min_target_date="2025-01-01",
        columns=["profit_center_nbr", "origin_week_date"],
    )

    df["origin_week_date"] = pd.to_datetime(df["origin_week_date"]).dt.date

    # Filter to origins >= 2025-01-01
    jan_2025 = date(2025, 1, 1)
    df_2025 = df[df["origin_week_date"] >= jan_2025]

    if df_2025.empty:
        logger.warning("No 2025 origins found in PREDICTIONS_BM, falling back to latest origins")
        return df.groupby("profit_center_nbr")["origin_week_date"].max().to_dict()

    # Get FIRST (minimum) 2025 origin per store
    first_dates = df_2025.groupby("profit_center_nbr")["origin_week_date"].min().to_dict()

    logger.info(f"Found first 2025 origin dates for {len(first_dates)} stores")

    # Log sample of origin dates
    sample_dates = list(first_dates.values())[:5]
    logger.info(f"Sample origin dates: {sample_dates}")

    return first_dates


def find_missing_horizons_for_origin(
    channel: str,
    origin_dates: Dict[int, date],
    all_2025_weeks: List[date],
) -> Dict[int, List[date]]:
    """
    Find target weeks that are missing for a specific origin per store.

    This checks if the given origin has predictions for all weeks of 2025.
    If not, identifies which target weeks (horizons) are missing.

    Args:
        channel: Channel to query
        origin_dates: Dict mapping store_id -> origin_week_date to check
        all_2025_weeks: List of all Monday dates in 2025

    Returns:
        Dict mapping store_id -> list of missing target_week_dates
    """
    logger.info(f"Finding missing horizons for specified origins...")

    # Load existing predictions with origin filter
    df = load_predictions_filtered(
        channel=channel,
        min_target_date="2025-01-01",
        columns=["profit_center_nbr", "origin_week_date", "target_week_date"],
    )

    df["origin_week_date"] = pd.to_datetime(df["origin_week_date"]).dt.date
    df["target_week_date"] = pd.to_datetime(df["target_week_date"]).dt.date

    missing_by_store: Dict[int, List[date]] = {}

    for store_id, origin_date in origin_dates.items():
        # Filter to this store and origin
        store_df = df[
            (df["profit_center_nbr"] == store_id) &
            (df["origin_week_date"] == origin_date)
        ]

        existing_weeks = set(store_df["target_week_date"].tolist())

        # Find all 2025 weeks AFTER the origin that should have predictions
        expected_weeks = [w for w in all_2025_weeks if w > origin_date]
        missing_weeks = sorted(set(expected_weeks) - existing_weeks)

        if missing_weeks:
            missing_by_store[int(store_id)] = missing_weeks

    total_missing = sum(len(w) for w in missing_by_store.values())
    logger.info(f"Stores with missing horizons: {len(missing_by_store)}")
    logger.info(f"Total missing store-weeks: {total_missing}")

    if missing_by_store:
        all_missing = [d for w in missing_by_store.values() for d in w]
        logger.info(f"Missing week range: {min(all_missing)} to {max(all_missing)}")

        # Log average missing weeks per store
        avg_missing = total_missing / len(missing_by_store)
        logger.info(f"Average missing weeks per store: {avg_missing:.1f}")

    return missing_by_store


def create_baseline_skeleton(
    missing_by_store: Dict[int, List[date]],
    latest_origins: Dict[int, date],
    channel: str,
) -> pd.DataFrame:
    """
    Create skeleton DataFrame for baseline generation.

    Args:
        missing_by_store: Dict mapping store_id -> missing target weeks
        latest_origins: Dict mapping store_id -> latest origin_week_date
        channel: Channel ("B&M" or "WEB")

    Returns:
        DataFrame with rows for each (store, channel, target_week) combination
    """
    rows = []

    for store_id, missing_weeks in missing_by_store.items():
        origin_date = latest_origins.get(store_id)
        if origin_date is None:
            logger.warning(f"No origin date found for store {store_id}, skipping")
            continue

        for target_week in missing_weeks:
            # Calculate horizon (weeks from origin to target)
            horizon = (target_week - origin_date).days // 7

            if horizon <= 0:
                # Target is before or at origin, skip
                continue

            rows.append({
                "profit_center_nbr": store_id,
                "channel": channel,
                "origin_week_date": origin_date,
                "target_week_date": target_week,
                "horizon": horizon,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Created baseline skeleton with {len(df)} rows")
    logger.info(f"Horizon range: {df['horizon'].min()} to {df['horizon'].max()} weeks")

    return df


def generate_baselines_for_missing_weeks(
    skeleton_df: pd.DataFrame,
    channel: str,
    checkpoint_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Model A and Model B baselines for missing weeks.

    Uses the BaselineGenerator with seasonal naive approach for business levers.

    Args:
        skeleton_df: DataFrame with store/channel/origin/target/horizon columns
        channel: Channel ("B&M" or "WEB")
        checkpoint_dir: Path to model checkpoints

    Returns:
        Tuple of (model_a_df, model_b_df)
    """
    logger.info(f"Generating baselines for {len(skeleton_df)} missing rows...")

    # Get unique stores
    store_ids = skeleton_df["profit_center_nbr"].unique().tolist()
    logger.info(f"Processing {len(store_ids)} stores")

    # Configure baseline generator
    # Note: We need to create a custom approach since BaselineGenerator expects
    # to generate for all horizons from a single origin, but we have variable origins

    # For now, we'll process store by store with their specific origin dates
    model_b_rows = []

    # Load historical MODEL_B data for seasonal naive lookups
    logger.info("Loading historical MODEL_B data for seasonal naive lookups...")
    historical_model_b = load_model_b_filtered(channel=channel)
    historical_model_b["target_week_date"] = pd.to_datetime(historical_model_b["target_week_date"])

    # Load calendar for fiscal/holiday features
    calendar_df = load_calendar()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # Group skeleton by store
    for store_id, store_group in skeleton_df.groupby("profit_center_nbr"):
        store_historical = historical_model_b[
            historical_model_b["profit_center_nbr"] == store_id
        ]

        if store_historical.empty:
            logger.warning(f"No historical data for store {store_id}, skipping")
            continue

        # Get the latest row for this store (to use as template for static features)
        latest_row = store_historical.sort_values("target_week_date").iloc[-1]

        for _, row in store_group.iterrows():
            target_week = pd.Timestamp(row["target_week_date"])
            lookback_week = target_week - pd.Timedelta(weeks=52)

            # Find the seasonal naive row (same week last year)
            seasonal_row = store_historical[
                store_historical["target_week_date"] == lookback_week
            ]

            if seasonal_row.empty:
                # Fall back to DMA average for this week-of-year
                woy = target_week.isocalendar()[1]
                dma = latest_row["dma"]
                dma_woy_avg = historical_model_b[
                    (historical_model_b["dma"] == dma)
                    & (historical_model_b["target_week_date"].dt.isocalendar().week == woy)
                ]
                if not dma_woy_avg.empty:
                    seasonal_row = dma_woy_avg.iloc[[0]]  # Use first match as template
                else:
                    # Final fallback: use latest row
                    seasonal_row = pd.DataFrame([latest_row])

            # Build the baseline row
            baseline_row = seasonal_row.iloc[0].to_dict()

            # Update key fields
            baseline_row["profit_center_nbr"] = store_id
            baseline_row["channel"] = channel
            baseline_row["origin_week_date"] = row["origin_week_date"]
            baseline_row["target_week_date"] = row["target_week_date"]
            baseline_row["horizon"] = row["horizon"]

            # Update calendar-dependent features from target week
            calendar_row = calendar_df[calendar_df["date"] == target_week]
            if not calendar_row.empty:
                cal = calendar_row.iloc[0]
                # Parse fiscal_week from HANA format ("Week 45" -> 45)
                fiscal_week = parse_fiscal_week(cal.get("fiscal_week"))
                baseline_row["woy"] = fiscal_week if fiscal_week is not None else target_week.isocalendar()[1]
                # Add sin/cos encoding
                woy_val = baseline_row["woy"]
                baseline_row["sin_woy"] = np.sin(2 * np.pi * woy_val / 52)
                baseline_row["cos_woy"] = np.cos(2 * np.pi * woy_val / 52)

            model_b_rows.append(baseline_row)

    model_b_df = pd.DataFrame(model_b_rows)
    logger.info(f"Generated {len(model_b_df)} baseline rows")

    # Model A is a subset of Model B with only business lever features
    # Key columns needed to join Model A with predictions
    key_columns = ['profit_center_nbr', 'channel', 'origin_week_date', 'target_week_date', 'horizon']

    # First, check if model_b_df has duplicate columns and deduplicate if needed
    if model_b_df.columns.duplicated().any():
        logger.warning(f"Model B has duplicate columns, removing duplicates")
        model_b_df = model_b_df.loc[:, ~model_b_df.columns.duplicated()]

    # Build Model A by subsetting Model B columns to only include MODEL_A_FEATURES
    # Use a set to avoid duplicate columns, then convert back to list preserving key column order
    feature_cols = [f for f in MODEL_A_FEATURES if f in model_b_df.columns and f not in key_columns]
    model_a_columns = key_columns + feature_cols
    available_cols = [c for c in model_a_columns if c in model_b_df.columns]

    if len(available_cols) > len(key_columns):
        model_a_df = model_b_df[available_cols].copy()
        logger.info(f"Created Model A with {len(model_a_df)} rows and {len(available_cols)} columns for explainability")
    else:
        model_a_df = pd.DataFrame()
        logger.warning("Could not create Model A data - insufficient columns for explainability")

    return model_a_df, model_b_df


def run_inference(
    model_b_df: pd.DataFrame,
    model_a_df: pd.DataFrame,
    channel: str,
    checkpoint_dir: Path,
) -> pd.DataFrame:
    """
    Run inference pipeline on baseline data.

    Args:
        model_b_df: Model B features DataFrame
        model_a_df: Model A features DataFrame for SHAP explainability
        channel: Channel ("B&M" or "WEB")
        checkpoint_dir: Path to model checkpoints

    Returns:
        DataFrame with predictions added
    """
    logger.info(f"Running inference for {len(model_b_df)} rows...")

    # Enable explainability if Model A data is available
    has_model_a = model_a_df is not None and not model_a_df.empty

    config = InferenceConfig(
        checkpoint_dir=checkpoint_dir,
        channels=[channel],
        run_explainability=has_model_a,  # Enable SHAP if Model A data available
    )

    pipeline = InferencePipeline(config)

    result = pipeline.run(
        model_b_data=model_b_df,
        model_a_data=model_a_df if has_model_a else None,
        channels=[channel],
    )

    # Get the predictions for our channel
    if channel == "B&M":
        predictions_df = result.bm_predictions
    else:
        predictions_df = result.web_predictions

    if predictions_df is None:
        raise RuntimeError(f"No predictions generated for channel {channel}")

    logger.info(f"Generated predictions for {len(predictions_df)} rows")

    return predictions_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate missing predictions for 2025 (Nov-Dec)"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="B&M",
        choices=["B&M", "WEB", "both"],
        help="Channel to process (default: B&M)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Path to model checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--min-target-date",
        type=str,
        default=None,
        help="Only generate predictions for target weeks >= this date (YYYY-MM-DD). "
             "Use to focus on specific gaps, e.g., --min-target-date 2025-11-03 for Nov-Dec only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--regenerate-from",
        type=str,
        default=None,
        help="Delete and regenerate predictions from this date (YYYY-MM-DD). "
             "Use to fix data quality issues in existing predictions.",
    )
    parser.add_argument(
        "--regenerate-to",
        type=str,
        default=None,
        help="Delete and regenerate predictions up to this date (YYYY-MM-DD). "
             "Defaults to end of year if not specified. Requires --regenerate-from.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate regenerate arguments
    if args.regenerate_to and not args.regenerate_from:
        parser.error("--regenerate-to requires --regenerate-from to be specified")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Determine channels to process
    channels = ["B&M", "WEB"] if args.channel == "both" else [args.channel]

    try:
        # Handle regeneration mode - delete existing predictions first
        if args.regenerate_from:
            from_date = args.regenerate_from
            to_date = args.regenerate_to or END_OF_YEAR_DATE
            logger.info(f"Regeneration mode: will delete predictions from {from_date} to {to_date}")

            if not args.dry_run:
                for channel in channels:
                    table_name = "PREDICTIONS_BM" if channel == "B&M" else "PREDICTIONS_WEB"
                    logger.info(f"Deleting existing {channel} predictions from {from_date} to {to_date}...")
                    deleted = delete_predictions_by_date_range(
                        table_name=table_name,
                        min_target_date=from_date,
                        max_target_date=to_date,
                    )
                    logger.info(f"Deleted {deleted} rows from {table_name}")
            else:
                logger.info("[DRY RUN] Would delete existing predictions (skipped)")

        # Get all 2025 target weeks
        all_2025_weeks = get_all_2025_target_weeks()

        for channel in channels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing channel: {channel}")
            logger.info(f"{'='*60}")

            # Step 1: Get FIRST 2025 origin dates (not latest)
            # This gives us the January 2025 origin for each store, which is the
            # consistent starting point for 52-week forecasts
            first_2025_origins = get_first_2025_origin_per_store(channel)

            if not first_2025_origins:
                logger.warning(f"No 2025 origins found for {channel}, skipping")
                continue

            # Step 2: Find missing horizons for the January 2025 origins
            # This checks if each store's January origin has 52 weeks of predictions
            missing_by_store = find_missing_horizons_for_origin(
                channel, first_2025_origins, all_2025_weeks
            )

            if not missing_by_store:
                logger.info(f"No missing horizons found for {channel}")
                continue

            # Step 2b: Filter to only weeks >= min_target_date if specified
            if args.min_target_date:
                min_date = datetime.strptime(args.min_target_date, "%Y-%m-%d").date()
                logger.info(f"Filtering to only target weeks >= {min_date}")

                filtered_missing = {}
                for store_id, weeks in missing_by_store.items():
                    filtered_weeks = [w for w in weeks if w >= min_date]
                    if filtered_weeks:
                        filtered_missing[store_id] = filtered_weeks

                before_count = sum(len(w) for w in missing_by_store.values())
                after_count = sum(len(w) for w in filtered_missing.values())
                logger.info(f"Filtered from {before_count} to {after_count} missing store-weeks")

                if filtered_missing:
                    all_filtered = [d for w in filtered_missing.values() for d in w]
                    logger.info(f"Filtered missing week range: {min(all_filtered)} to {max(all_filtered)}")

                missing_by_store = filtered_missing

                if not missing_by_store:
                    logger.info(f"No missing weeks after filtering for {channel}")
                    continue

            # Step 3: Create skeleton with January 2025 origin
            skeleton_df = create_baseline_skeleton(missing_by_store, first_2025_origins, channel)

            if skeleton_df.empty:
                logger.warning(f"No valid skeleton rows for {channel}")
                continue

            if args.dry_run:
                logger.info(f"[DRY RUN] Would generate baselines for {len(skeleton_df)} rows")
                logger.info(f"[DRY RUN] Sample stores: {skeleton_df['profit_center_nbr'].unique()[:5].tolist()}")
                continue

            # Step 5: Generate baselines
            model_a_df, model_b_df = generate_baselines_for_missing_weeks(
                skeleton_df, channel, checkpoint_dir
            )

            if model_b_df.empty:
                logger.warning(f"No baseline rows generated for {channel}")
                continue

            # Step 6: Run inference (with SHAP explainability if Model A data available)
            predictions_df = run_inference(model_b_df, model_a_df, channel, checkpoint_dir)

            # Step 7: Filter to only columns that exist in PREDICTIONS_BM
            # This removes intermediate columns like woy, sin_woy, cos_woy
            valid_columns = [col for col in predictions_df.columns if col.lower() in PREDICTIONS_BM_COLUMNS]
            predictions_df_filtered = predictions_df[valid_columns].copy()
            logger.info(f"Filtered to {len(valid_columns)} valid columns for HANA insert")

            # Step 8: Insert to HANA
            logger.info(f"Inserting {len(predictions_df_filtered)} predictions to HANA...")
            table_name = "PREDICTIONS_BM" if channel == "B&M" else "PREDICTIONS_WEB"
            rows_inserted = insert_predictions_bm(predictions_df_filtered, table_name=table_name)
            logger.info(f"Successfully inserted {rows_inserted} rows to {table_name}")

        logger.info("\nDone!")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        close_connection()


if __name__ == "__main__":
    main()
