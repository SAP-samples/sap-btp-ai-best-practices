"""
Export tools for the forecasting agent.

Tools for exporting precomputed baseline forecasts to CSV.

Per Agent_plan.md: Export baseline predictions from checkpoint file
for external analysis or integration.

Memory Optimization:
This module uses chunked/batched processing to handle large exports
without exceeding memory limits. Instead of loading all data at once,
it processes stores in batches and writes incrementally to CSV.
"""

from __future__ import annotations

import gc
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from app.agent.hana_loader import (
    load_predictions_filtered,
    get_unique_store_ids,
    clear_cache as clear_hana_cache,
    load_calendar,
    load_yougov_dma_map,
)
from app.agent.session import get_session

# Batch size for chunked processing (number of stores per batch)
# Lower values use less memory but require more database queries
# Reduced from 50 to 10 to prevent OOM on memory-constrained CF instances (1.5GB)
BATCH_SIZE = 10


def _clear_global_caches() -> None:
    """
    Clear all global LRU caches to free memory before export.

    This is safe because:
    - Export uses filtered queries (load_predictions_filtered) which are not cached
    - Session-level caches are separate and unaffected
    - Other tools primarily use filtered queries too
    - Any cleared data can be re-fetched from HANA if needed later

    Called at the start of export to ensure maximum memory is available.
    """
    clear_hana_cache()
    gc.collect()


def _add_fiscal_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fiscal calendar fields by joining on target_week_date.

    Joins with CALENDAR table to add fiscal_year, fiscal_quarter,
    fiscal_month, and fiscal_week columns.

    Args:
        df: DataFrame with target_week_date column (datetime)

    Returns:
        DataFrame with fiscal fields added
    """
    calendar = load_calendar()
    # Select only needed columns and ensure date column is datetime
    calendar_subset = calendar[['date', 'fiscal_year', 'fiscal_quarter', 'fiscal_month', 'fiscal_week']].copy()
    calendar_subset['date'] = pd.to_datetime(calendar_subset['date'])

    # Convert fiscal_week from "Week 03" format to numeric (3)
    # Handle both string format "Week XX" and already-numeric values
    if calendar_subset['fiscal_week'].dtype == object:
        calendar_subset['fiscal_week'] = (
            calendar_subset['fiscal_week']
            .astype(str)
            .str.replace(r'^Week\s*', '', regex=True)
            .astype(int)
        )

    # Join on target_week_date
    df = df.merge(
        calendar_subset,
        left_on='target_week_date',
        right_on='date',
        how='left'
    )

    # Drop the redundant date column from calendar
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    return df


def _add_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add region field by joining on DMA.

    Joins with YOUGOV_DMA_MAP table to add region column.
    Uses case-insensitive matching to handle variations in DMA naming.

    Args:
        df: DataFrame with dma column

    Returns:
        DataFrame with region field added
    """
    dma_map = load_yougov_dma_map()
    # Select only needed columns and deduplicate
    dma_subset = dma_map[['market_city', 'region']].drop_duplicates()

    # Normalize market_city to uppercase for case-insensitive join
    # This matches the pattern used in awareness_features.py and budget.py
    dma_subset = dma_subset.copy()
    dma_subset['market_city_upper'] = dma_subset['market_city'].str.upper()

    # Create uppercase join key in predictions data
    df['dma_upper'] = df['dma'].str.upper()

    # Join on uppercase keys for case-insensitive matching
    df = df.merge(
        dma_subset[['market_city_upper', 'region']],
        left_on='dma_upper',
        right_on='market_city_upper',
        how='left'
    )

    # Drop the temporary join columns
    df = df.drop(columns=['dma_upper', 'market_city_upper'], errors='ignore')

    return df


# Columns to export (internal names used for DataFrame operations)
EXPORT_COLUMNS = [
    "profit_center_nbr",
    "origin_week_date",
    "horizon",
    "target_week_date",
    "channel",
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_month",
    "fiscal_week",
    "dma",
    "region",
    "dma_seasonal_weight",
    "pred_sales_mean",
    "pred_aov_mean",
    "pred_traffic_p50",
    "pred_orders",
    "conversion_rate",
]

# Column rename mapping for user-friendly CSV headers
EXPORT_COLUMN_RENAME = {
    "pred_sales_mean": "Sales",
    "pred_aov_mean": "AOV",
    "pred_traffic_p50": "Traffic",
    "pred_orders": "Orders",
    "conversion_rate": "Conversion Rate",
    "fiscal_year": "Fiscal Year",
    "fiscal_quarter": "Fiscal Quarter",
    "fiscal_month": "Fiscal Month",
    "fiscal_week": "Fiscal Week",
    "dma": "DMA",
    "region": "Region",
    "dma_seasonal_weight": "Seasonal Weight",
}

# Base columns to read from HANA (common to both channels)
_READ_COLUMNS_BASE = [
    "profit_center_nbr",
    "origin_week_date",
    "horizon",
    "target_week_date",
    "channel",
    "pred_sales_mean",
    "pred_aov_mean",
    "pred_log_orders",
    "dma",
    "dma_seasonal_weight",
]

# B&M-specific columns (traffic and conversion metrics)
_READ_COLUMNS_BM_ONLY = [
    "pred_traffic_p50",
    "pred_logit_conversion",
]

def _get_read_columns(channel: str) -> List[str]:
    """Get the appropriate columns to read based on channel."""
    if channel.upper() == "B&M":
        return _READ_COLUMNS_BASE + _READ_COLUMNS_BM_ONLY
    else:
        # WEB channel doesn't have traffic/conversion columns
        return _READ_COLUMNS_BASE


def _process_batch(
    df: pd.DataFrame,
    origin_week_date_parsed: Optional[date],
    dmas: Optional[List[str]],
) -> tuple[pd.DataFrame, str]:
    """
    Process a batch of predictions data.

    Applies transformations, filtering by origin_week_date and DMAs,
    and deduplication.

    Args:
        df: DataFrame batch to process
        origin_week_date_parsed: Optional specific origin date filter
        dmas: Optional list of DMAs to filter by

    Returns:
        Tuple of (processed DataFrame, origin_week_mode string)
    """
    if len(df) == 0:
        return df, "per_store_earliest"

    # Convert date columns
    df["target_week_date"] = pd.to_datetime(df["target_week_date"])
    df["origin_week_date"] = pd.to_datetime(df["origin_week_date"])

    # Transform log-scale predictions back to original scale
    df["pred_orders"] = np.exp(df["pred_log_orders"])

    # Calculate conversion rate from log-odds (inverse logit / sigmoid)
    # Only B&M has conversion data; WEB channel doesn't have this column
    if "pred_logit_conversion" in df.columns:
        df["conversion_rate"] = 1 / (1 + np.exp(-df["pred_logit_conversion"]))

    # Filter by origin_week_date
    if origin_week_date_parsed is not None:
        df = df[df["origin_week_date"] == pd.Timestamp(origin_week_date_parsed)]
        origin_week_mode = "user_specified"
    else:
        # Default: use earliest origin_week_date per store (constrained to 2025)
        df_2025 = df[df["origin_week_date"] >= pd.Timestamp("2025-01-01")]

        if len(df_2025) == 0:
            return pd.DataFrame(columns=EXPORT_COLUMNS), "per_store_earliest"

        # Get earliest origin_week_date per store in this batch
        earliest_origins = (
            df_2025.groupby("profit_center_nbr")["origin_week_date"]
            .min()
            .reset_index()
        )
        earliest_origins.columns = ["profit_center_nbr", "earliest_origin"]

        # Join back to filter rows to only the earliest origin per store
        df = df.merge(earliest_origins, on="profit_center_nbr")
        df = df[df["origin_week_date"] == df["earliest_origin"]]
        df = df.drop(columns=["earliest_origin"])
        origin_week_mode = "per_store_earliest"

    # Deduplicate by key columns
    dedup_keys = ["profit_center_nbr", "horizon", "target_week_date", "channel"]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")

    # Filter by DMAs if provided
    if dmas is not None and len(dmas) > 0:
        df = df[df["dma"].isin(dmas)]

    # Add fiscal and region fields via joins
    if len(df) > 0:
        df = _add_fiscal_fields(df)
        df = _add_region(df)

        # Select only export columns and rename to user-friendly names
        # Use intersection to handle any missing columns gracefully
        available_export_cols = [c for c in EXPORT_COLUMNS if c in df.columns]
        df = df[available_export_cols].copy()
        df = df.rename(columns=EXPORT_COLUMN_RENAME)

    return df, origin_week_mode


@tool
def export_baseline_forecasts(
    max_horizon: int = 52,
    start_date: Optional[str] = None,
    store_ids: Optional[List[int]] = None,
    dmas: Optional[List[str]] = None,
    origin_week_date: Optional[str] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export precomputed baseline forecasts for stores to CSV.

    Reads from precomputed predictions and exports filtered data based on
    target date, horizon, and optional store/DMA filters. Only works for
    baseline scenario. Uses memory-efficient chunked processing to handle
    large exports without running out of memory.

    By default, uses the earliest origin_week_date in 2025 per store to ensure
    consistent forecasts. This handles new stores that may only have recent data.
    The origin_week_date parameter can override this to use a specific date.

    Output is deduplicated by (profit_center_nbr, horizon, target_week_date, channel)
    and sorted by profit_center_nbr and horizon.

    Exported columns:
        - profit_center_nbr: Store identifier
        - origin_week_date: Week when the forecast was generated
        - horizon: Weeks ahead being predicted (1-52)
        - target_week_date: Week being forecasted
        - channel: Sales channel (B&M or WEB)
        - Fiscal Year: Fiscal year of the target week
        - Fiscal Quarter: Fiscal quarter (1-4) of the target week
        - Fiscal Month: Fiscal month of the target week
        - Fiscal Week: Fiscal week (1-52) of the target week
        - DMA: Designated Market Area (market city)
        - Region: Geographic region
        - Seasonal Weight: Weekly seasonality weight (% of annual volume)
        - Sales: Predicted mean sales
        - AOV: Predicted mean average order value
        - Traffic: Predicted median traffic (B&M only)
        - Orders: Predicted number of orders
        - Conversion Rate: Predicted conversion rate, 0-1 scale (B&M only)

    Args:
        max_horizon: Maximum horizon weeks to include (1-52). Default: 52.
        start_date: Start date for target_week_date filter (YYYY-MM-DD format).
            Defaults to current date if not provided.
        store_ids: Optional list of profit_center_nbr values to filter by.
            If not provided, includes all stores.
        dmas: Optional list of DMA names to filter by.
            If not provided, includes all DMAs.
        origin_week_date: Specific origin week to use (YYYY-MM-DD format).
            If not provided, uses the earliest 2025 origin_week_date per store.
        channel: Channel to export. Options:
            - "B&M": Export only Brick & Mortar predictions
            - "WEB": Export only Web/E-commerce predictions
            - "ALL" or None (default): Export both channels

    Returns:
        Dictionary containing:
        - status: "exported" on success
        - file: Path to exported CSV file
        - num_stores: Number of unique stores in export
        - num_rows: Total rows exported
        - channels_exported: List of channels included (e.g., ["B&M", "WEB"])
        - date_range: Actual date range in exported data
        - horizon_range: Actual horizon range in exported data
        - origin_week_mode: "per_store_earliest" or "user_specified"
        - output_dir: Path to output directory
        - batches_processed: Number of batches processed

    Example:
        >>> export_baseline_forecasts(max_horizon=4)
        {"status": "exported", "file": "...", "num_stores": 400, ...}

        >>> export_baseline_forecasts(max_horizon=4, start_date="2025-01-06", store_ids=[1, 2, 3])
        {"status": "exported", "file": "...", "num_stores": 3, ...}

        >>> export_baseline_forecasts(max_horizon=13, origin_week_date="2025-01-06")
        {"status": "exported", "origin_week_mode": "user_specified", ...}

        >>> export_baseline_forecasts(max_horizon=52, channel="WEB")
        {"status": "exported", "channels_exported": ["WEB"], ...}
    """
    # Clear global caches to free memory before starting export
    # This prevents OOM when prior operations have loaded large cached DataFrames
    _clear_global_caches()

    # Validate and determine channels to export
    if channel is None or channel.upper() == "ALL":
        channels_to_export = ["B&M", "WEB"]
    elif channel.upper() in ["B&M", "WEB"]:
        channels_to_export = [channel.upper()]
    else:
        return {
            "error": f"Invalid channel: '{channel}'. Must be 'B&M', 'WEB', 'ALL', or None.",
            "hint": "Use 'ALL' or omit parameter to export both channels.",
        }

    # Validate max_horizon
    if not 1 <= max_horizon <= 52:
        return {
            "error": f"max_horizon must be between 1 and 52, got {max_horizon}",
            "hint": "Valid horizon range is 1-52 weeks.",
        }

    # Parse and validate start_date (default to current date)
    if start_date is None:
        start_date_parsed = date.today()
    else:
        try:
            start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            return {
                "error": f"Invalid date format: '{start_date}'. Expected YYYY-MM-DD.",
                "hint": "Example: '2025-01-06'",
            }

    # Parse and validate origin_week_date if provided
    origin_week_date_parsed = None
    if origin_week_date is not None:
        try:
            origin_week_date_parsed = datetime.strptime(origin_week_date, "%Y-%m-%d").date()
        except ValueError:
            return {
                "error": f"Invalid origin_week_date format: '{origin_week_date}'. Expected YYYY-MM-DD.",
                "hint": "Example: '2025-01-06'",
            }

    # Create output directory and file path early
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"baseline_forecasts_{timestamp}.csv"

    # Initialize statistics tracking
    total_rows = 0
    unique_stores = set()
    channels_exported = set()
    min_date = None
    max_date = None
    min_horizon_val = None
    max_horizon_val = None
    origin_week_mode = "per_store_earliest"
    batches_processed = 0
    header_written = False

    # Process each channel
    try:
        for current_channel in channels_to_export:
            # Step 1: Get list of unique store IDs for this channel (lightweight query)
            try:
                if store_ids:
                    # User specified stores - use those directly
                    channel_store_ids = store_ids
                else:
                    # Query unique store IDs with filters applied
                    channel_store_ids = get_unique_store_ids(
                        channel=current_channel,
                        min_target_date=str(start_date_parsed),
                        max_horizon=max_horizon,
                        dmas=dmas,
                    )
            except Exception as e:
                return {
                    "error": f"Failed to query store IDs from HANA for channel {current_channel}: {str(e)}",
                    "hint": "Check HANA connection and ensure predictions table exists.",
                }

            if len(channel_store_ids) == 0:
                # No stores for this channel, continue to next channel
                continue

            # Step 2: Process stores in batches for this channel
            for i in range(0, len(channel_store_ids), BATCH_SIZE):
                batch_store_ids = channel_store_ids[i : i + BATCH_SIZE]

                # Load predictions for this batch only
                # Use channel-specific columns (WEB doesn't have traffic/conversion)
                df_batch = load_predictions_filtered(
                    channel=current_channel,
                    profit_center_nbrs=batch_store_ids,
                    min_target_date=str(start_date_parsed),
                    max_horizon=max_horizon,
                    columns=_get_read_columns(current_channel),
                )

                if len(df_batch) == 0:
                    continue

                # Process the batch (transformations, filtering, deduplication)
                df_processed, batch_origin_mode = _process_batch(
                    df_batch, origin_week_date_parsed, dmas
                )

                # Update origin_week_mode from first batch with data
                if batches_processed == 0:
                    origin_week_mode = batch_origin_mode

                if len(df_processed) == 0:
                    # Release memory
                    del df_batch, df_processed
                    gc.collect()
                    continue

                # Sort batch by profit_center_nbr and horizon
                df_processed = df_processed.sort_values(
                    ["profit_center_nbr", "horizon"]
                ).reset_index(drop=True)

                # Append to CSV file
                df_processed.to_csv(
                    output_path,
                    mode="a",
                    header=not header_written,
                    index=False,
                )
                header_written = True

                # Update statistics
                total_rows += len(df_processed)
                unique_stores.update(df_processed["profit_center_nbr"].unique())
                channels_exported.add(current_channel)

                batch_min_date = df_processed["target_week_date"].min()
                batch_max_date = df_processed["target_week_date"].max()
                batch_min_horizon = int(df_processed["horizon"].min())
                batch_max_horizon = int(df_processed["horizon"].max())

                if min_date is None or batch_min_date < min_date:
                    min_date = batch_min_date
                if max_date is None or batch_max_date > max_date:
                    max_date = batch_max_date
                if min_horizon_val is None or batch_min_horizon < min_horizon_val:
                    min_horizon_val = batch_min_horizon
                if max_horizon_val is None or batch_max_horizon > max_horizon_val:
                    max_horizon_val = batch_max_horizon

                batches_processed += 1

                # Release memory before next batch
                del df_batch, df_processed
                gc.collect()

    except Exception as e:
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        return {
            "error": f"Failed during batch processing: {str(e)}",
            "hint": "Check HANA connection and available memory.",
            "batches_completed": batches_processed,
        }

    # Check if any data was exported
    if total_rows == 0:
        # Clean up empty file if it exists
        if output_path.exists():
            output_path.unlink()
        return {
            "error": "No data matches the specified filters after processing.",
            "hint": "Try adjusting start_date, max_horizon, store_ids, or dmas parameters.",
            "filters_applied": {
                "start_date": str(start_date_parsed),
                "max_horizon": max_horizon,
                "store_ids": store_ids,
                "dmas": dmas,
                "origin_week_date": origin_week_date,
                "channel": channel,
            },
        }

    # Track exported file for chat attachments
    try:
        session = get_session()
        session.add_export_file(str(output_path.absolute()))
    except RuntimeError:
        pass  # No session context (e.g., running in isolation)

    return {
        "status": "exported",
        "file": str(output_path.absolute()),
        "num_stores": len(unique_stores),
        "num_rows": total_rows,
        "channels_exported": sorted(list(channels_exported)),
        "date_range": {
            "start": str(min_date.date()) if hasattr(min_date, "date") else str(min_date),
            "end": str(max_date.date()) if hasattr(max_date, "date") else str(max_date),
        },
        "horizon_range": {
            "min": min_horizon_val,
            "max": max_horizon_val,
        },
        "origin_week_mode": origin_week_mode,
        "output_dir": str(output_dir.absolute()),
        "batches_processed": batches_processed,
        "hint": "File can be opened in Excel or Python for analysis.",
    }


__all__ = ["export_baseline_forecasts"]
