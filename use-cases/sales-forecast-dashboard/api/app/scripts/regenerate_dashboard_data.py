"""
Regenerate Dashboard Data from SAP HANA

This script pulls data from SAP HANA and regenerates the JSON files
(stores.json, dma_summary.json, timeseries/) used by the Sales Forecast Dashboard API.

Usage:
    python -m app.scripts.regenerate_dashboard_data --verbose
    python -m app.scripts.regenerate_dashboard_data --skip-timeseries
    python -m app.scripts.regenerate_dashboard_data --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.hana_loader import (
    get_hana_connection,
    query_to_dataframe,
    close_connection,
)
from app.services.explanation_generator import (
    generate_static_explanation,
    safe_yoy_percentage,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Output directories
DATA_DIR = Path(__file__).parent.parent / "data"
TIMESERIES_DIR = DATA_DIR / "timeseries"

# Business features config for filtering SHAP values
BUSINESS_FEATURES_CONFIG = Path(__file__).parent.parent / "config" / "business_features.json"

# YoY threshold config for traffic light color system
YOY_THRESHOLDS_CONFIG = Path(__file__).parent.parent / "config" / "yoy_thresholds.json"

# Feature display config for explanation generation
FEATURE_DISPLAY_CONFIG = Path(__file__).parent.parent / "config" / "feature_display.json"


def load_business_features() -> set:
    """
    Load whitelist of business features from shared config.

    Returns:
        Set of feature names that are business levers (from Scenario Maker)
    """
    if BUSINESS_FEATURES_CONFIG.exists():
        with open(BUSINESS_FEATURES_CONFIG) as f:
            config = json.load(f)
            features = set(config.get("business_features", {}).keys())
            logger.info(f"Loaded {len(features)} business features from config")
            return features
    logger.warning("Business features config not found, using all features")
    return set()


def load_yoy_thresholds() -> dict:
    """
    Load YoY threshold config for traffic light colors.

    Returns:
        Dict with red_threshold and green_threshold values
    """
    if YOY_THRESHOLDS_CONFIG.exists():
        with open(YOY_THRESHOLDS_CONFIG) as f:
            return json.load(f)
    return {"red_threshold": -5, "green_threshold": 5}


def load_feature_display() -> dict:
    """
    Load feature display config for explanation generation.

    Returns:
        Dict mapping feature names to display config (name, unit, templates)
    """
    if FEATURE_DISPLAY_CONFIG.exists():
        with open(FEATURE_DISPLAY_CONFIG) as f:
            config = json.load(f)
            logger.info(f"Loaded display config for {len(config)} features")
            return config
    logger.warning("Feature display config not found, using defaults")
    return {}


def create_baseline_sales_lookup(predictions_2024: pd.DataFrame) -> Dict:
    """
    Create lookup for 2024 baseline sales by store and week_of_year.

    Key: (profit_center_nbr, week_of_year)
    Value: pred_sales_p50

    This lookup enables showing 2024 sales on the time series chart
    and calculating dollar impacts from SHAP deltas.

    Args:
        predictions_2024: DataFrame with 2024 predictions

    Returns:
        Dictionary for O(1) baseline sales lookup
    """
    logger.info("Building 2024 baseline sales lookup...")

    if predictions_2024.empty:
        logger.warning("  No 2024 baseline data available")
        return {}

    baseline_sales = {}

    for _, row in predictions_2024.iterrows():
        store = row['profit_center_nbr']
        week_of_year = row['target_week_date'].isocalendar()[1]
        sales = row.get('pred_sales_p50')

        if sales is not None and not pd.isna(sales):
            key = (store, week_of_year)
            # If duplicate, keep the value (later rows overwrite)
            baseline_sales[key] = float(sales)

    logger.info(f"  Built lookup with {len(baseline_sales):,} store-week entries")
    return baseline_sales


# =============================================================================
# YoY Sales Change Computation
# =============================================================================

def compute_yoy_sales_change(
    predictions_2025: pd.DataFrame,
    predictions_2024: pd.DataFrame
) -> Dict[int, Optional[float]]:
    """
    DEPRECATED: Use compute_yoy_auv_change instead for AUV-based comparison.

    Compute YoY sales change percentage for each store (weekly average based).

    Formula: ((avg_2025 - avg_2024) / avg_2024) * 100

    Args:
        predictions_2025: DataFrame with 2025 predictions
        predictions_2024: DataFrame with 2024 predictions

    Returns:
        Dict mapping profit_center_nbr -> yoy_change_percent (None if no baseline)
    """
    logger.info("Computing YoY sales change for each store (weekly average)...")

    # Average sales per store for 2025
    avg_2025 = predictions_2025.groupby('profit_center_nbr')['pred_sales_p50'].mean()

    # Average sales per store for 2024
    avg_2024 = predictions_2024.groupby('profit_center_nbr')['pred_sales_p50'].mean()

    # Compute YoY change
    yoy_change: Dict[int, Optional[float]] = {}
    stores_with_baseline = 0
    stores_without_baseline = 0

    for store_id in avg_2025.index:
        if store_id in avg_2024.index and avg_2024[store_id] > 0:
            change = ((avg_2025[store_id] - avg_2024[store_id]) / avg_2024[store_id]) * 100
            yoy_change[int(store_id)] = round(change, 2)
            stores_with_baseline += 1
        else:
            yoy_change[int(store_id)] = None
            stores_without_baseline += 1

    logger.info(f"  Computed YoY changes for {stores_with_baseline} stores with baseline")
    if stores_without_baseline > 0:
        logger.info(f"  {stores_without_baseline} stores have no 2024 baseline data")

    return yoy_change


# =============================================================================
# AUV (Annualized Unit Volume) Computation
# =============================================================================

def filter_to_single_origin(predictions: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Filter predictions to the first origin_week_date in the target year per store.

    Each store may have multiple inference runs stored in PREDICTIONS_BM.
    To avoid double-counting when computing AUV, select only ONE
    origin_week_date per store - specifically the first forecast made in
    the target year (e.g., first week of January 2025).

    For stores that opened during the year, use their first available origin.

    Args:
        predictions: DataFrame with predictions (must have origin_week_date column)
        target_year: The year to use for filtering origins (e.g., 2025)

    Returns:
        DataFrame filtered to single origin_week_date per store
    """
    required_cols = ['origin_week_date', 'profit_center_nbr']
    missing = [c for c in required_cols if c not in predictions.columns]
    if missing:
        logger.warning(f"Missing columns {missing}, skipping origin filter")
        return predictions

    if predictions.empty:
        logger.warning("  Empty predictions DataFrame, skipping origin filter")
        return predictions

    original_count = len(predictions)
    year_start = pd.Timestamp(f"{target_year}-01-01")

    # Filter to origins in the target year
    origins_in_year = predictions[predictions["origin_week_date"] >= year_start]

    if origins_in_year.empty:
        # Fallback: if no origins in target year, use earliest available
        logger.warning(f"  No origins >= {target_year}-01-01, using earliest available")
        selected_origin = predictions.groupby("profit_center_nbr")["origin_week_date"].min()
    else:
        # Get the earliest origin in the target year per store
        # This gives us the "first forecast of 2025" for each store
        selected_origin = origins_in_year.groupby("profit_center_nbr")["origin_week_date"].min()

    selected_df = selected_origin.reset_index().rename(columns={"origin_week_date": "selected_origin"})

    # Merge and filter to keep only rows with selected origin per store
    df_merged = predictions.merge(selected_df, on="profit_center_nbr")
    df_filtered = df_merged[df_merged["origin_week_date"] == df_merged["selected_origin"]]
    df_filtered = df_filtered.drop(columns=["selected_origin"])

    # Log the result
    unique_stores = df_filtered['profit_center_nbr'].nunique()
    unique_origins = df_filtered['origin_week_date'].nunique()
    avg_weeks = len(df_filtered) / max(unique_stores, 1)
    logger.info(f"  Filtered to single origin per store: {original_count:,} -> {len(df_filtered):,} rows")
    logger.info(f"  Unique stores: {unique_stores}, Unique origin dates: {unique_origins}, Avg weeks/store: {avg_weeks:.1f}")

    return df_filtered


def deduplicate_by_store_week(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate predictions to one row per (store, target_week_date).

    This prevents inflated AUV calculations when there are multiple rows
    for the same store-week combination (e.g., from multiple inference runs
    or duplicate data insertions).

    Args:
        predictions: DataFrame with predictions

    Returns:
        DataFrame with at most one row per (store, target_week_date)
    """
    required_cols = ['profit_center_nbr', 'target_week_date']
    missing = [c for c in required_cols if c not in predictions.columns]
    if missing:
        logger.warning(f"Missing columns {missing} for deduplication, skipping")
        return predictions

    if predictions.empty:
        return predictions

    before_count = len(predictions)
    predictions_deduped = predictions.drop_duplicates(
        subset=['profit_center_nbr', 'target_week_date'],
        keep='first'
    )
    after_count = len(predictions_deduped)

    if before_count != after_count:
        removed = before_count - after_count
        duplication_factor = before_count / max(after_count, 1)
        logger.warning(
            f"  Removed {removed:,} duplicate store-week rows "
            f"({before_count:,} -> {after_count:,}, {duplication_factor:.1f}x duplication)"
        )
    else:
        logger.info(f"  No duplicate store-week rows found ({after_count:,} rows)")

    return predictions_deduped


def compute_auv(predictions: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Compute AUV (Annualized Unit Volume) for each store.

    AUV = Sum of pred_sales_p50 across all forecast weeks (up to 52 weeks).
    This represents the total annual sales volume for each store.

    Args:
        predictions: DataFrame with predictions (must have profit_center_nbr,
                    pred_sales_p50, pred_sales_p90, target_week_date columns)

    Returns:
        Dict mapping profit_center_nbr -> {
            'auv_p50': float (sum of weekly p50),
            'auv_p90': float (sum of weekly p90),
            'weeks_count': int (number of weeks summed)
        }
    """
    logger.info("Computing AUV (52-week sum) for each store...")

    auv_data = predictions.groupby('profit_center_nbr').agg({
        'pred_sales_p50': 'sum',
        'pred_sales_p90': 'sum',
        'target_week_date': 'nunique'
    }).reset_index()
    auv_data.columns = ['profit_center_nbr', 'auv_p50', 'auv_p90', 'weeks_count']

    result = {}
    for _, row in auv_data.iterrows():
        store_id = int(row['profit_center_nbr'])
        result[store_id] = {
            'auv_p50': float(row['auv_p50']),
            'auv_p90': float(row['auv_p90']),
            'weeks_count': int(row['weeks_count'])
        }

    logger.info(f"  Computed AUV for {len(result)} stores")
    if result:
        avg_weeks = sum(d['weeks_count'] for d in result.values()) / len(result)
        logger.info(f"  Average weeks per store: {avg_weeks:.1f}")

    return result


def compute_yoy_auv_change(
    predictions_2025: pd.DataFrame,
    predictions_2024: pd.DataFrame
) -> Dict[int, Optional[float]]:
    """
    Compute YoY AUV change percentage for each store.

    Compares total annual sales (AUV) between 2025 and 2024.
    Formula: ((auv_2025 - auv_2024) / auv_2024) * 100

    Args:
        predictions_2025: DataFrame with 2025 predictions
        predictions_2024: DataFrame with 2024 predictions

    Returns:
        Dict mapping profit_center_nbr -> yoy_auv_change_percent (None if no baseline)
    """
    logger.info("Computing YoY AUV change for each store (52-week sum comparison)...")

    # Sum sales per store for 2025 (AUV)
    auv_2025 = predictions_2025.groupby('profit_center_nbr')['pred_sales_p50'].sum()

    # Sum sales per store for 2024 (baseline AUV)
    auv_2024 = predictions_2024.groupby('profit_center_nbr')['pred_sales_p50'].sum()

    # Compute YoY change
    yoy_change: Dict[int, Optional[float]] = {}
    stores_with_baseline = 0
    stores_without_baseline = 0

    for store_id in auv_2025.index:
        if store_id in auv_2024.index and auv_2024[store_id] > 0:
            change = ((auv_2025[store_id] - auv_2024[store_id]) / auv_2024[store_id]) * 100
            yoy_change[int(store_id)] = round(change, 2)
            stores_with_baseline += 1
        else:
            yoy_change[int(store_id)] = None
            stores_without_baseline += 1

    logger.info(f"  Computed YoY AUV changes for {stores_with_baseline} stores with baseline")
    if stores_without_baseline > 0:
        logger.info(f"  {stores_without_baseline} stores have no 2024 baseline data")

    return yoy_change


# =============================================================================
# HANA Query Functions
# =============================================================================

def load_predictions_from_hana(
    year: int,
    channel: str = "B&M",
    include_all_columns: bool = True
) -> pd.DataFrame:
    """
    Load predictions for a specific year from PREDICTIONS_BM or PREDICTIONS_WEB table.

    CRITICAL: Only selects needed columns to minimize data transfer.
    The full table has 80 columns and can be 7GB+.

    Args:
        year: Year to load (2024 or 2025)
        channel: Channel to load ("B&M" or "WEB")
        include_all_columns: If True, include all prediction columns.
                            If False, only load columns needed for SHAP baseline.

    Returns:
        DataFrame with predictions for the specified year and channel
    """
    schema = os.getenv('HANA_SCHEMA', 'AICOE')
    table_name = "PREDICTIONS_BM" if channel == "B&M" else "PREDICTIONS_WEB"

    # WEB channel does not have traffic/conversion predictions
    is_bm = channel == "B&M"

    if include_all_columns:
        # Full set of columns for 2025 (or current year) data
        # Traffic columns only exist in B&M
        traffic_columns = """
            PRED_TRAFFIC_P10,
            PRED_TRAFFIC_P50,
            PRED_TRAFFIC_P90,""" if is_bm else ""

        columns = f"""
            PROFIT_CENTER_NBR,
            ORIGIN_WEEK_DATE,
            DMA,
            CHANNEL,
            TARGET_WEEK_DATE,
            HORIZON,
            PRED_SALES_P50,
            PRED_SALES_P90,
            PRED_AOV_MEAN,
            PRED_AOV_P50,
            PRED_AOV_P90,{traffic_columns}
            IS_OUTLET,
            IS_NEW_STORE,
            IS_COMP_STORE,
            STORE_DESIGN_SF,
            MERCHANDISING_SF,
            TOP_FEATURES_PRED_LOG_SALES
        """
    else:
        # Minimal columns for baseline SHAP computation
        traffic_col = "PRED_TRAFFIC_P50," if is_bm else ""
        columns = f"""
            PROFIT_CENTER_NBR,
            ORIGIN_WEEK_DATE,
            CHANNEL,
            TARGET_WEEK_DATE,
            HORIZON,
            PRED_SALES_P50,
            PRED_AOV_P50,
            {traffic_col}
            TOP_FEATURES_PRED_LOG_SALES
        """

    query = f"""
        SELECT {columns}
        FROM "{schema}"."{table_name}"
        WHERE TARGET_WEEK_DATE >= '{year}-01-01'
          AND TARGET_WEEK_DATE < '{year + 1}-01-01'
    """

    logger.info(f"Loading {year} {channel} predictions from HANA (this may take a few minutes)...")
    df = query_to_dataframe(query)

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Convert date columns
    if 'target_week_date' in df.columns:
        df['target_week_date'] = pd.to_datetime(df['target_week_date'])
    if 'origin_week_date' in df.columns:
        df['origin_week_date'] = pd.to_datetime(df['origin_week_date'])

    logger.info(f"  Loaded {len(df):,} rows for {year} {channel}")
    return df


def load_profit_centers_from_hana() -> pd.DataFrame:
    """
    Load store master data from PROFIT_CENTER table.

    Returns:
        DataFrame with store locations and metadata
    """
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    query = f"""
        SELECT
            PROFIT_CENTER_NBR,
            PROFIT_CENTER_NAME,
            LATITUDE,
            LONGITUDE,
            CITY,
            STATE,
            MARKET___CITY,
            OUTLET,
            MERCHANDISING_SF,
            STORE_DESIGN_SF,
            BCG_STORE_CATEGORY
        FROM "{schema}"."PROFIT_CENTER"
        WHERE LATITUDE IS NOT NULL
          AND LONGITUDE IS NOT NULL
    """

    logger.info("Loading profit centers from HANA...")
    df = query_to_dataframe(query)

    # Column mapping for normalization
    column_mapping = {
        'PROFIT_CENTER_NBR': 'profit_center_nbr',
        'PROFIT_CENTER_NAME': 'store_name',
        'LATITUDE': 'lat',
        'LONGITUDE': 'lng',
        'CITY': 'city',
        'STATE': 'state',
        'MARKET___CITY': 'dma',
        'OUTLET': 'is_outlet',
        'MERCHANDISING_SF': 'merchandising_sf',
        'STORE_DESIGN_SF': 'store_design_sf',
        'BCG_STORE_CATEGORY': 'bcg_category',
    }

    # Rename columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert lat/lng to float
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')

    # Filter out invalid coordinates
    df = df[df['lat'].notna() & df['lng'].notna()]

    logger.info(f"  Loaded {len(df):,} stores with valid coordinates")
    return df


# =============================================================================
# SHAP Processing Functions
# =============================================================================

def parse_shap_string(shap_str: str) -> List[Dict]:
    """
    Parse SHAP string format: "feature=value:impact; feature2=value2:impact2"

    Args:
        shap_str: Raw SHAP string from HANA

    Returns:
        List of dicts: [{'feature': 'name', 'value': '...', 'impact': float}, ...]
    """
    if pd.isna(shap_str) or shap_str == '' or not isinstance(shap_str, str):
        return []

    features = []
    for part in shap_str.split('; '):
        part = part.strip()
        if ':' in part and '=' in part:
            try:
                feature_val, impact = part.rsplit(':', 1)
                feature_name, value = feature_val.split('=', 1)
                features.append({
                    'feature': feature_name.strip(),
                    'value': value.strip(),
                    'impact': float(impact)
                })
            except (ValueError, IndexError):
                continue
    return features


def create_baseline_lookup(predictions_2024: pd.DataFrame) -> Dict:
    """
    Create lookup dictionary for 2024 baseline SHAP values and feature values.

    Key: (profit_center_nbr, channel, week_of_year)
    Value: {
        'shap_dict': {feature: impact},
        'value_dict': {feature: value},
        'shap_features': [...]
    }

    Note: horizon is intentionally excluded from the key to enable YoY comparisons
    even when 2024 and 2025 predictions have different forecast horizons.

    Args:
        predictions_2024: DataFrame with 2024 predictions

    Returns:
        Dictionary for O(1) baseline lookup with both impacts and values
    """
    logger.info("Building 2024 baseline lookup for SHAP delta computation...")

    if predictions_2024.empty:
        logger.warning("  No 2024 baseline data available")
        return {}

    baseline = {}
    row_count = 0

    for _, row in predictions_2024.iterrows():
        row_count += 1

        store = row['profit_center_nbr']
        channel = row.get('channel', 'B&M')
        # Extract ISO week of year for seasonal matching
        week_of_year = row['target_week_date'].isocalendar()[1]

        key = (store, channel, week_of_year)

        # Parse SHAP string for this row
        shap_str = row.get('top_features_pred_log_sales', '')
        shap_features = parse_shap_string(shap_str)

        # Impact dict (existing): {feature_name: impact}
        shap_dict = {f['feature']: f['impact'] for f in shap_features}

        # Value dict (NEW): {feature_name: value} for "X to Y" descriptions
        value_dict = {f['feature']: f['value'] for f in shap_features}

        # Store the baseline (if duplicate keys, later rows overwrite)
        baseline[key] = {
            'shap_dict': shap_dict,
            'value_dict': value_dict,
            'shap_features': shap_features
        }

        if row_count % 100000 == 0:
            logger.info(f"  Processed {row_count:,} baseline rows...")

    logger.info(f"  Built lookup with {len(baseline):,} unique baseline entries")
    return baseline


def compute_shap_delta(
    shap_2025: List[Dict],
    baseline_lookup: Dict,
    store_id: int,
    channel: str,
    target_date: datetime
) -> List[Dict]:
    """
    Compute SHAP delta: SHAP_2025 - SHAP_2024.

    Matches by store, channel, and ISO week_of_year.
    Preserves both current (2025) and baseline (2024) feature values
    for "changed from X to Y" explanations.

    Note: horizon is intentionally excluded from matching to enable YoY comparisons
    even when 2024 and 2025 predictions have different forecast horizons.

    Args:
        shap_2025: Parsed 2025 SHAP features
        baseline_lookup: Dict from create_baseline_lookup()
        store_id, channel: Matching keys
        target_date: 2025 target_week_date

    Returns:
        List of dicts with delta impacts, baseline values, and has_baseline flag
    """
    # Get week of year from 2025 date to find matching 2024 week
    week_of_year = target_date.isocalendar()[1]

    key = (store_id, channel, week_of_year)
    baseline_data = baseline_lookup.get(key)

    if not baseline_data:
        # No 2024 baseline found - return original values marked as no baseline
        return [{
            'feature': f['feature'],
            'value': f['value'],
            'baseline_value': None,
            'impact': f['impact'],  # Keep raw impact if no baseline
            'has_baseline': False
        } for f in shap_2025]

    baseline_shap = baseline_data['shap_dict']
    baseline_values = baseline_data['value_dict']

    # Create 2025 lookup
    shap_2025_dict = {f['feature']: f for f in shap_2025}

    # Compute deltas for all features present in either year
    all_features = set(baseline_shap.keys()) | set(shap_2025_dict.keys())

    delta_features = []
    for feature in all_features:
        impact_2025 = shap_2025_dict.get(feature, {}).get('impact', 0)
        impact_2024 = baseline_shap.get(feature, 0)

        value_2025 = shap_2025_dict.get(feature, {}).get('value', None)
        value_2024 = baseline_values.get(feature, None)

        delta = impact_2025 - impact_2024

        delta_features.append({
            'feature': feature,
            'value': value_2025,
            'baseline_value': value_2024,
            'impact': round(delta, 6),  # Delta value
            'has_baseline': True
        })

    # Sort by absolute delta (descending)
    delta_features.sort(key=lambda x: abs(x['impact']), reverse=True)

    return delta_features


# =============================================================================
# Output Generation Functions
# =============================================================================

def generate_stores_json(
    predictions: pd.DataFrame,
    profit_centers: pd.DataFrame,
    auv_data_bm: Dict[int, Dict[str, float]] = None,
    auv_data_web: Dict[int, Dict[str, float]] = None,
    yoy_auv_changes_bm: Dict[int, Optional[float]] = None,
    yoy_auv_changes_web: Dict[int, Optional[float]] = None,
    comp_stores_only: bool = True
) -> List[Dict]:
    """
    Generate stores.json data structure with separate B&M and WEB AUV fields.

    Merges AUV predictions with profit center location data.
    Uses DMA from predictions if available, falls back to profit centers.
    Filters to comp stores only by default.

    Stores are omnichannel - each store has both B&M and WEB AUV values.

    Args:
        predictions: 2025 predictions DataFrame (can be either channel)
        profit_centers: Store master DataFrame
        auv_data_bm: Dict mapping store_id -> {auv_p50, auv_p90, weeks_count} for B&M
        auv_data_web: Dict mapping store_id -> {auv_p50, auv_p90, weeks_count} for WEB
        yoy_auv_changes_bm: Dict mapping store_id -> YoY AUV change % for B&M
        yoy_auv_changes_web: Dict mapping store_id -> YoY AUV change % for WEB
        comp_stores_only: If True, only include comp stores (is_comp_store=1)

    Returns:
        List of store dictionaries matching Store model schema
    """
    if auv_data_bm is None:
        auv_data_bm = {}
    if auv_data_web is None:
        auv_data_web = {}
    if yoy_auv_changes_bm is None:
        yoy_auv_changes_bm = {}
    if yoy_auv_changes_web is None:
        yoy_auv_changes_web = {}

    logger.info("Generating stores.json with AUV...")

    # Get store attributes from any row per store
    # (attributes like dma, channel, is_comp_store are consistent across all rows for a store)
    store_attrs = predictions.groupby('profit_center_nbr').first().reset_index()

    # Merge with profit center locations
    stores = profit_centers.merge(
        store_attrs[[
            'profit_center_nbr', 'dma', 'channel',
            'is_outlet', 'is_new_store', 'is_comp_store',
            'store_design_sf', 'merchandising_sf'
        ]],
        on='profit_center_nbr',
        how='inner',
        suffixes=('_pc', '_pred')
    )

    # Filter to comp stores only if requested
    total_stores = len(stores)
    if comp_stores_only:
        stores = stores[stores['is_comp_store'] == True]
        logger.info(f"  Filtered to {len(stores)} comp stores (from {total_stores} total)")
    else:
        logger.info(f"  Including all {total_stores} stores (comp_stores_only=False)")

    # Use DMA from predictions if available, otherwise from profit centers
    if 'dma_pred' in stores.columns and 'dma_pc' in stores.columns:
        stores['dma_final'] = stores['dma_pred'].fillna(stores['dma_pc'])
    elif 'dma_pred' in stores.columns:
        stores['dma_final'] = stores['dma_pred']
    elif 'dma_pc' in stores.columns:
        stores['dma_final'] = stores['dma_pc']
    elif 'dma' in stores.columns:
        stores['dma_final'] = stores['dma']
    else:
        stores['dma_final'] = 'UNKNOWN'

    stores_list = []
    for _, row in stores.iterrows():
        store_id = int(row['profit_center_nbr'])

        # Get B&M AUV data for this store
        bm_auv = auv_data_bm.get(store_id, {'auv_p50': 0, 'auv_p90': 0, 'weeks_count': 0})
        bm_auv_p50 = bm_auv['auv_p50']
        bm_auv_p90 = bm_auv['auv_p90']
        bm_weeks_count = bm_auv['weeks_count']

        # Get WEB AUV data for this store
        web_auv = auv_data_web.get(store_id, {'auv_p50': 0, 'auv_p90': 0, 'weeks_count': 0})
        web_auv_p50 = web_auv['auv_p50']
        web_auv_p90 = web_auv['auv_p90']
        web_weeks_count = web_auv['weeks_count']

        # Combined AUV for backward compatibility
        combined_auv_p50 = bm_auv_p50 + web_auv_p50
        combined_auv_p90 = bm_auv_p90 + web_auv_p90
        combined_weeks = max(bm_weeks_count, web_weeks_count)

        # Compute weekly average for backward compatibility
        weekly_avg_p50 = combined_auv_p50 / max(combined_weeks, 1) if combined_auv_p50 > 0 else 0
        weekly_avg_p90 = combined_auv_p90 / max(combined_weeks, 1) if combined_auv_p90 > 0 else 0

        # Get channel-specific YoY values
        bm_yoy = yoy_auv_changes_bm.get(store_id)
        web_yoy = yoy_auv_changes_web.get(store_id)

        # Compute combined YoY as weighted average of channels (weighted by AUV volume)
        combined_yoy = None
        if bm_yoy is not None or web_yoy is not None:
            if bm_yoy is not None and web_yoy is not None:
                # Both channels have data - weighted average by AUV volume
                total_auv = bm_auv_p50 + web_auv_p50
                if total_auv > 0:
                    combined_yoy = (bm_yoy * bm_auv_p50 + web_yoy * web_auv_p50) / total_auv
                else:
                    # Fallback to simple average if no AUV data
                    combined_yoy = (bm_yoy + web_yoy) / 2
            elif bm_yoy is not None:
                combined_yoy = bm_yoy
            else:
                combined_yoy = web_yoy

        store_data = {
            'id': store_id,
            'name': str(row['store_name']) if pd.notna(row.get('store_name')) else f"Store #{store_id}",
            'dma': str(row['dma_final']),
            'lat': float(row['lat']),
            'lng': float(row['lng']),
            'city': str(row['city']) if pd.notna(row.get('city')) else '',
            'state': str(row['state']) if pd.notna(row.get('state')) else '',
            # B&M channel AUV fields
            'bm_auv_p50': bm_auv_p50,
            'bm_auv_p90': bm_auv_p90,
            'bm_auv_weeks_count': bm_weeks_count,
            'bm_yoy_auv_change': yoy_auv_changes_bm.get(store_id),
            # WEB channel AUV fields
            'web_auv_p50': web_auv_p50,
            'web_auv_p90': web_auv_p90,
            'web_auv_weeks_count': web_weeks_count,
            'web_yoy_auv_change': yoy_auv_changes_web.get(store_id),
            # Combined AUV fields (for backward compatibility and "All" view)
            'auv_p50': combined_auv_p50,
            'auv_p90': combined_auv_p90,
            'auv_weeks_count': combined_weeks,
            'yoy_auv_change': round(combined_yoy, 2) if combined_yoy is not None else None,
            # DEPRECATED: Weekly average for backward compatibility
            'pred_sales_p50': weekly_avg_p50,
            'pred_sales_p90': weekly_avg_p90,
            'yoy_sales_change': round(combined_yoy, 2) if combined_yoy is not None else None,
            # Store attributes - check suffixed columns first (from merge), then plain column
            'is_outlet': bool(row.get('is_outlet_pred', row.get('is_outlet_pc', row.get('is_outlet')))) if pd.notna(row.get('is_outlet_pred', row.get('is_outlet_pc', row.get('is_outlet')))) else False,
            'is_new_store': bool(row.get('is_new_store_pred', row.get('is_new_store_pc', row.get('is_new_store')))) if pd.notna(row.get('is_new_store_pred', row.get('is_new_store_pc', row.get('is_new_store')))) else False,
            'is_comp_store': bool(row.get('is_comp_store_pred', row.get('is_comp_store_pc', row.get('is_comp_store')))) if pd.notna(row.get('is_comp_store_pred', row.get('is_comp_store_pc', row.get('is_comp_store')))) else False,
            'store_design_sf': float(row.get('store_design_sf_pred', row.get('store_design_sf'))) if pd.notna(row.get('store_design_sf_pred', row.get('store_design_sf'))) else None,
            'merchandising_sf': float(row.get('merchandising_sf_pred', row.get('merchandising_sf'))) if pd.notna(row.get('merchandising_sf_pred', row.get('merchandising_sf'))) else None,
            'bcg_category': str(row['bcg_category']) if pd.notna(row.get('bcg_category')) else None,
        }
        stores_list.append(store_data)

    logger.info(f"  Generated data for {len(stores_list)} stores")
    return stores_list


def compute_dma_yoy_status(stores_in_dma: List[Dict], channel: str = None) -> Optional[str]:
    """
    Compute worst-case YoY status for a DMA based on its stores.

    Logic (alert-based - shows worst case):
    - If ANY store has yoy < -5 → "decrease" (red)
    - Else if ANY store has -5 <= yoy <= 5 → "stable" (yellow)
    - Else if ALL stores have yoy > 5 → "increase" (green)
    - If all stores have None → None

    Args:
        stores_in_dma: List of store dicts with yoy_auv_change fields
        channel: Optional channel filter ("B&M", "WEB", or None for combined)

    Returns:
        "decrease", "stable", "increase", or None
    """
    # Get channel-specific YoY values
    if channel == "B&M":
        yoy_values = [s.get('bm_yoy_auv_change') for s in stores_in_dma]
    elif channel == "WEB":
        yoy_values = [s.get('web_yoy_auv_change') for s in stores_in_dma]
    else:
        # Combined: use B&M values as primary (stores are omnichannel)
        yoy_values = [s.get('bm_yoy_auv_change') for s in stores_in_dma]

    # Filter out None values
    valid_values = [v for v in yoy_values if v is not None]

    if not valid_values:
        return None

    # Check for any red (decrease) - worst case first
    if any(v < -5 for v in valid_values):
        return "decrease"

    # Check for any yellow (stable)
    if any(-5 <= v <= 5 for v in valid_values):
        return "stable"

    # All remaining must be green (increase)
    return "increase"


def compute_dma_yoy_auv_percentage(stores_in_dma: List[Dict], channel: str = None) -> Optional[float]:
    """
    Compute weighted average YoY AUV percentage for a DMA.

    Weights by store's AUV as proxy for store importance.
    This gives larger stores more influence on the DMA's overall YoY change.

    Args:
        stores_in_dma: List of store dicts with yoy_auv_change and auv_p50 fields
        channel: Optional channel filter ("B&M", "WEB", or None for combined)

    Returns:
        Weighted average YoY AUV percentage, or None if no valid data
    """
    # Get channel-specific field names
    if channel == "B&M":
        yoy_field = 'bm_yoy_auv_change'
        auv_field = 'bm_auv_p50'
    elif channel == "WEB":
        yoy_field = 'web_yoy_auv_change'
        auv_field = 'web_auv_p50'
    else:
        yoy_field = 'bm_yoy_auv_change'  # Default to B&M for combined
        auv_field = 'auv_p50'

    valid_stores = [
        s for s in stores_in_dma
        if s.get(yoy_field) is not None and s.get(auv_field, 0) > 0
    ]

    if not valid_stores:
        return None

    total_weighted_yoy = sum(
        s[yoy_field] * s[auv_field]
        for s in valid_stores
    )
    total_auv = sum(s[auv_field] for s in valid_stores)

    return round(total_weighted_yoy / total_auv, 2) if total_auv > 0 else None


def compute_dma_yoy_percentage(stores_in_dma: List[Dict]) -> Optional[float]:
    """
    DEPRECATED: Use compute_dma_yoy_auv_percentage instead.

    Compute weighted average YoY percentage for a DMA (weekly average based).

    Args:
        stores_in_dma: List of store dicts with yoy_sales_change and pred_sales_p50 fields

    Returns:
        Weighted average YoY percentage, or None if no valid data
    """
    valid_stores = [
        s for s in stores_in_dma
        if s.get('yoy_sales_change') is not None and s.get('pred_sales_p50', 0) > 0
    ]

    if not valid_stores:
        return None

    total_weighted_yoy = sum(
        s['yoy_sales_change'] * s['pred_sales_p50']
        for s in valid_stores
    )
    total_sales = sum(s['pred_sales_p50'] for s in valid_stores)

    return round(total_weighted_yoy / total_sales, 2) if total_sales > 0 else None


def generate_dma_summary(
    predictions: pd.DataFrame,
    stores_list: List[Dict]
) -> List[Dict]:
    """
    Generate dma_summary.json data structure with AUV aggregates.

    Computes DMA centroids from store coordinates.
    Aggregates AUV (Annualized Unit Volume) at DMA level.
    Computes worst-case YoY status for traffic light coloring.

    Note: stores_list should already be filtered to comp stores only.

    Args:
        predictions: 2025 predictions DataFrame
        stores_list: Output from generate_stores_json() (comp stores only)

    Returns:
        List of DMA summary dictionaries
    """
    logger.info("Generating dma_summary.json with AUV...")

    # Convert stores list to DataFrame for easier processing
    stores_df = pd.DataFrame(stores_list)

    if stores_df.empty:
        logger.warning("  No stores to aggregate")
        return []

    # Compute DMA centroids and AUV aggregates from stores (including channel-specific)
    dma_agg = stores_df.groupby('dma').agg({
        'lat': 'mean',
        'lng': 'mean',
        'id': 'count',
        # Combined AUV
        'auv_p50': 'sum',
        'auv_p90': 'sum',
        # B&M channel AUV
        'bm_auv_p50': 'sum',
        'bm_auv_p90': 'sum',
        # WEB channel AUV
        'web_auv_p50': 'sum',
        'web_auv_p90': 'sum',
        # DEPRECATED: Weekly avg sum for backward compat
        'pred_sales_p50': 'sum',
        'pred_sales_p90': 'sum',
    }).reset_index()

    dma_agg.columns = [
        'dma', 'lat', 'lng', 'store_count',
        'total_auv_p50', 'total_auv_p90',
        'bm_total_auv_p50', 'bm_total_auv_p90',
        'web_total_auv_p50', 'web_total_auv_p90',
        'total_pred_sales_p50', 'total_pred_sales_p90'
    ]

    # Build lookup of stores by DMA for YoY status computation
    stores_by_dma = {}
    for store in stores_list:
        dma_name = store.get('dma')
        if dma_name not in stores_by_dma:
            stores_by_dma[dma_name] = []
        stores_by_dma[dma_name].append(store)

    # Convert to list of dicts
    result = []
    status_counts = {"decrease": 0, "stable": 0, "increase": 0, None: 0}

    for _, row in dma_agg.iterrows():
        dma_name = row['dma']
        stores_in_dma = stores_by_dma.get(dma_name, [])
        store_count = int(row['store_count'])

        # Compute YoY status and percentages for each channel
        yoy_status = compute_dma_yoy_status(stores_in_dma)  # Combined (uses B&M)
        yoy_auv_pct = compute_dma_yoy_auv_percentage(stores_in_dma)  # Combined
        yoy_pct = compute_dma_yoy_percentage(stores_in_dma)  # DEPRECATED
        status_counts[yoy_status] = status_counts.get(yoy_status, 0) + 1

        # Channel-specific YoY status and percentages
        bm_yoy_status = compute_dma_yoy_status(stores_in_dma, channel="B&M")
        bm_yoy_auv_pct = compute_dma_yoy_auv_percentage(stores_in_dma, channel="B&M")
        web_yoy_status = compute_dma_yoy_status(stores_in_dma, channel="WEB")
        web_yoy_auv_pct = compute_dma_yoy_auv_percentage(stores_in_dma, channel="WEB")

        # Compute average AUV per store (combined and by channel)
        total_auv_p50 = float(row['total_auv_p50'])
        avg_auv_p50 = total_auv_p50 / store_count if store_count > 0 else 0

        bm_total_auv_p50 = float(row['bm_total_auv_p50'])
        bm_avg_auv_p50 = bm_total_auv_p50 / store_count if store_count > 0 else 0

        web_total_auv_p50 = float(row['web_total_auv_p50'])
        web_avg_auv_p50 = web_total_auv_p50 / store_count if store_count > 0 else 0

        result.append({
            'dma': dma_name,
            'lat': float(row['lat']),
            'lng': float(row['lng']),
            'store_count': store_count,
            # Combined AUV fields (52-week sum aggregates)
            'total_auv_p50': total_auv_p50,
            'total_auv_p90': float(row['total_auv_p90']),
            'avg_auv_p50': avg_auv_p50,
            'yoy_auv_change_pct': yoy_auv_pct,
            'yoy_status': yoy_status,
            # B&M channel AUV aggregates
            'bm_total_auv_p50': bm_total_auv_p50,
            'bm_total_auv_p90': float(row['bm_total_auv_p90']),
            'bm_avg_auv_p50': bm_avg_auv_p50,
            'bm_yoy_auv_change_pct': bm_yoy_auv_pct,
            'bm_yoy_status': bm_yoy_status,
            # WEB channel AUV aggregates
            'web_total_auv_p50': web_total_auv_p50,
            'web_total_auv_p90': float(row['web_total_auv_p90']),
            'web_avg_auv_p50': web_avg_auv_p50,
            'web_yoy_auv_change_pct': web_yoy_auv_pct,
            'web_yoy_status': web_yoy_status,
            # DEPRECATED: Weekly average totals for backward compatibility
            'total_pred_sales_p50': float(row['total_pred_sales_p50']),
            'total_pred_sales_p90': float(row['total_pred_sales_p90']),
            'yoy_sales_change_pct': yoy_pct,
        })

    logger.info(f"  Generated summary for {len(result)} DMAs")
    logger.info(f"  YoY status breakdown: {status_counts['increase']} green, {status_counts['stable']} yellow, {status_counts['decrease']} red")
    return result


def generate_store_timeseries(
    predictions: pd.DataFrame,
    baseline_lookup: Dict,
    baseline_sales_lookup: Dict,
    feature_config: Dict,
    output_dir: Path,
    channel: str = "B&M"
) -> int:
    """
    Generate individual store_{id}_{channel}.json timeseries files.

    Pre-computes SHAP deltas for each time point.
    Adds 2024 baseline sales, YoY change percentage, and
    natural language explanations for each week.
    Deduplicates by date (keeps first occurrence).

    Args:
        predictions: 2025 predictions DataFrame for the specified channel
        baseline_lookup: SHAP baseline lookup dict (for delta computation)
        baseline_sales_lookup: 2024 sales lookup {(store, week_of_year): sales_p50}
        feature_config: Feature display config for explanation generation
        output_dir: Path to timeseries/ directory
        channel: Channel name ("B&M" or "WEB")

    Returns:
        Number of store files generated
    """
    channel_suffix = "bm" if channel == "B&M" else "web"
    logger.info(f"Generating store timeseries files for {channel} channel...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load business features whitelist for filtering
    business_features = load_business_features()

    store_count = 0
    total_stores = predictions['profit_center_nbr'].nunique()
    baseline_hits = 0
    baseline_misses = 0

    for store_id, group in predictions.groupby('profit_center_nbr'):
        # Sort by date
        ts_data = group.sort_values('target_week_date')

        # Build time series data with SHAP delta values per time point
        timeseries = []
        for _, row in ts_data.iterrows():
            # Parse raw 2025 SHAP features
            shap_str = row.get('top_features_pred_log_sales', '')
            shap_2025 = parse_shap_string(shap_str)

            # Compute SHAP delta (2025 - 2024)
            shap_deltas = compute_shap_delta(
                shap_2025=shap_2025,
                baseline_lookup=baseline_lookup,
                store_id=store_id,
                channel=row.get('channel', 'B&M'),
                target_date=row['target_week_date']
            )

            # Track baseline coverage
            if shap_deltas and shap_deltas[0].get('has_baseline', False):
                baseline_hits += 1
            else:
                baseline_misses += 1

            # Filter to business features only (if whitelist is loaded)
            if business_features:
                shap_deltas = [
                    f for f in shap_deltas
                    if f['feature'] in business_features
                ]
                # Sort by absolute impact and limit to top 10
                shap_deltas = sorted(
                    shap_deltas,
                    key=lambda x: abs(x['impact']),
                    reverse=True
                )[:10]

            # Get baseline sales for this week (2024 same week_of_year)
            week_of_year = row['target_week_date'].isocalendar()[1]
            baseline_sales = baseline_sales_lookup.get((store_id, week_of_year))

            # Current sales
            current_sales = float(row['pred_sales_p50']) if pd.notna(row.get('pred_sales_p50')) else None

            # Compute YoY percentage change
            yoy_pct = safe_yoy_percentage(current_sales, baseline_sales)

            # Generate explanation
            explanation = None
            if current_sales is not None:
                explanation = generate_static_explanation(
                    shap_features=shap_deltas,
                    baseline_sales=baseline_sales,
                    current_sales=current_sales,
                    feature_config=feature_config
                )

            ts_entry = {
                'date': row['target_week_date'].strftime('%Y-%m-%d'),
                'pred_sales_p50': current_sales,
                'pred_sales_p90': float(row['pred_sales_p90']) if pd.notna(row.get('pred_sales_p90')) else None,
                'baseline_sales_p50': baseline_sales,
                'yoy_change_pct': round(yoy_pct, 2) if yoy_pct is not None else None,
                'pred_aov_mean': float(row['pred_aov_mean']) if pd.notna(row.get('pred_aov_mean')) else None,
                'pred_aov_p50': float(row['pred_aov_p50']) if pd.notna(row.get('pred_aov_p50')) else None,
                'pred_aov_p90': float(row['pred_aov_p90']) if pd.notna(row.get('pred_aov_p90')) else None,
                'shap_features': shap_deltas,
                'explanation': explanation
            }
            # Traffic predictions only exist for B&M channel
            if channel == "B&M":
                ts_entry['pred_traffic_p10'] = float(row['pred_traffic_p10']) if pd.notna(row.get('pred_traffic_p10')) else None
                ts_entry['pred_traffic_p50'] = float(row['pred_traffic_p50']) if pd.notna(row.get('pred_traffic_p50')) else None
                ts_entry['pred_traffic_p90'] = float(row['pred_traffic_p90']) if pd.notna(row.get('pred_traffic_p90')) else None
            timeseries.append(ts_entry)

        # Remove duplicates by date (keep first occurrence)
        seen_dates = set()
        unique_timeseries = []
        for ts in timeseries:
            if ts['date'] not in seen_dates:
                seen_dates.add(ts['date'])
                unique_timeseries.append(ts)

        store_ts = {
            'store_id': int(store_id),
            'channel': channel,
            'timeseries': unique_timeseries
        }

        output_path = output_dir / f'store_{int(store_id)}_{channel_suffix}.json'
        with open(output_path, 'w') as f:
            json.dump(store_ts, f, indent=2)

        store_count += 1
        if store_count % 50 == 0:
            logger.info(f"  Processed {store_count}/{total_stores} stores...")

    total_rows = baseline_hits + baseline_misses
    if total_rows > 0:
        coverage = 100 * baseline_hits / total_rows
        logger.info(f"  Baseline coverage: {baseline_hits:,}/{total_rows:,} ({coverage:.1f}%)")
    logger.info(f"  Generated {store_count} store timeseries files")

    return store_count


def generate_dma_timeseries(
    predictions: pd.DataFrame,
    stores_list: List[Dict],
    output_dir: Path
) -> int:
    """
    Generate dma_{name}.json aggregated timeseries files.

    Aggregates pred_sales_p50 and pred_sales_p90 across stores in each DMA.
    Uses URL-safe filenames (replaces / and spaces with _).

    Args:
        predictions: 2025 predictions DataFrame
        stores_list: List with store-to-DMA mapping
        output_dir: Path to timeseries/ directory

    Returns:
        Number of DMA files generated
    """
    logger.info("Generating DMA timeseries files...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping of store_id to DMA from stores_list
    store_to_dma = {store['id']: store['dma'] for store in stores_list}

    # Add DMA column to predictions based on store mapping
    predictions_copy = predictions.copy()
    predictions_copy['dma_mapped'] = predictions_copy['profit_center_nbr'].map(store_to_dma)

    # Filter to only stores we have DMA mappings for
    predictions_with_dma = predictions_copy[predictions_copy['dma_mapped'].notna()]

    # Deduplicate: keep only one row per (store, target_week_date) to prevent
    # summing multiple origins for the same store-date combination.
    # This mirrors the deduplication logic in generate_store_timeseries().
    if len(predictions_with_dma) > 0:
        before_dedup = len(predictions_with_dma)
        predictions_with_dma = predictions_with_dma.drop_duplicates(
            subset=['profit_center_nbr', 'target_week_date'],
            keep='first'
        )
        after_dedup = len(predictions_with_dma)
        if before_dedup != after_dedup:
            logger.warning(f"  Removed {before_dedup - after_dedup} duplicate store-date rows before DMA aggregation")

    dma_count = 0
    total_dmas = predictions_with_dma['dma_mapped'].nunique()

    for dma_name, group in predictions_with_dma.groupby('dma_mapped'):
        # Aggregate by date across all stores in this DMA
        ts_agg = group.groupby('target_week_date').agg({
            'pred_sales_p50': 'sum',
            'pred_sales_p90': 'sum'
        }).reset_index().sort_values('target_week_date')

        timeseries = []
        for _, row in ts_agg.iterrows():
            ts_entry = {
                'date': row['target_week_date'].strftime('%Y-%m-%d'),
                'pred_sales_p50': float(row['pred_sales_p50']) if pd.notna(row['pred_sales_p50']) else 0,
                'pred_sales_p90': float(row['pred_sales_p90']) if pd.notna(row['pred_sales_p90']) else 0
            }
            timeseries.append(ts_entry)

        dma_ts = {
            'dma': str(dma_name),
            'timeseries': timeseries
        }

        # Use URL-safe filename (replace special chars)
        safe_name = str(dma_name).replace('/', '_').replace(' ', '_')
        output_path = output_dir / f'dma_{safe_name}.json'
        with open(output_path, 'w') as f:
            json.dump(dma_ts, f, indent=2)

        dma_count += 1
        if dma_count % 10 == 0:
            logger.info(f"  Processed {dma_count}/{total_dmas} DMAs...")

    logger.info(f"  Generated {dma_count} DMA timeseries files")
    return dma_count


# =============================================================================
# Main Function
# =============================================================================

def regenerate_data(
    output_dir: Optional[Path] = None,
    skip_timeseries: bool = False,
    verbose: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Main entry point to regenerate all JSON data files from HANA.

    Args:
        output_dir: Override default output directory
        skip_timeseries: If True, only regenerate stores.json and dma_summary.json
        verbose: If True, print progress messages
        dry_run: If True, test HANA connection only without writing files

    Returns:
        Summary dict with file counts and duration
    """
    start_time = datetime.now()

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    # Set output directories
    data_dir = output_dir or DATA_DIR
    timeseries_dir = data_dir / "timeseries"

    logger.info("=" * 60)
    logger.info("Sales Forecast Dashboard - Data Regeneration from HANA")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Test HANA connection
    logger.info("Connecting to HANA...")
    try:
        cc = get_hana_connection()
        schema = os.getenv('HANA_SCHEMA', 'AICOE')
        logger.info(f"  Connected to {schema} schema")
    except Exception as e:
        logger.error(f"  Failed to connect to HANA: {e}")
        return {'error': str(e)}

    if dry_run:
        logger.info("Dry run mode - connection successful, exiting without writing files")
        close_connection()
        return {'status': 'dry_run_success'}

    # Create output directories
    data_dir.mkdir(parents=True, exist_ok=True)
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Load B&M Channel Data
    # ==========================================================================
    logger.info("\n--- Loading B&M Channel Data ---")

    # Load 2025 B&M predictions (full columns)
    predictions_2025_bm = load_predictions_from_hana(2025, channel="B&M", include_all_columns=True)

    # Load 2024 B&M predictions for baseline (minimal columns)
    predictions_2024_bm = load_predictions_from_hana(2024, channel="B&M", include_all_columns=False)

    # Filter and deduplicate B&M predictions
    logger.info("Filtering B&M predictions to single origin per store...")
    predictions_2025_bm = filter_to_single_origin(predictions_2025_bm, target_year=2025)
    predictions_2024_bm = filter_to_single_origin(predictions_2024_bm, target_year=2024)

    logger.info("Deduplicating B&M predictions by store and target week...")
    predictions_2025_bm = deduplicate_by_store_week(predictions_2025_bm)
    predictions_2024_bm = deduplicate_by_store_week(predictions_2024_bm)

    # Compute B&M AUV (Annualized Unit Volume) = 52-week sum
    auv_data_bm = compute_auv(predictions_2025_bm)

    # Compute B&M YoY AUV changes
    yoy_auv_changes_bm = compute_yoy_auv_change(predictions_2025_bm, predictions_2024_bm)

    # Build B&M baseline lookups
    baseline_lookup_bm = create_baseline_lookup(predictions_2024_bm)
    baseline_sales_lookup_bm = create_baseline_sales_lookup(predictions_2024_bm)

    # Free 2024 B&M memory
    del predictions_2024_bm
    gc.collect()
    logger.info("Freed 2024 B&M DataFrame memory")

    # ==========================================================================
    # Load WEB Channel Data
    # ==========================================================================
    logger.info("\n--- Loading WEB Channel Data ---")

    # Load 2025 WEB predictions (full columns)
    predictions_2025_web = load_predictions_from_hana(2025, channel="WEB", include_all_columns=True)

    # Load 2024 WEB predictions for baseline (minimal columns)
    predictions_2024_web = load_predictions_from_hana(2024, channel="WEB", include_all_columns=False)

    # Filter and deduplicate WEB predictions
    logger.info("Filtering WEB predictions to single origin per store...")
    predictions_2025_web = filter_to_single_origin(predictions_2025_web, target_year=2025)
    predictions_2024_web = filter_to_single_origin(predictions_2024_web, target_year=2024)

    logger.info("Deduplicating WEB predictions by store and target week...")
    predictions_2025_web = deduplicate_by_store_week(predictions_2025_web)
    predictions_2024_web = deduplicate_by_store_week(predictions_2024_web)

    # Compute WEB AUV (Annualized Unit Volume) = 52-week sum
    auv_data_web = compute_auv(predictions_2025_web)

    # Compute WEB YoY AUV changes
    yoy_auv_changes_web = compute_yoy_auv_change(predictions_2025_web, predictions_2024_web)

    # Build WEB baseline lookups
    baseline_lookup_web = create_baseline_lookup(predictions_2024_web)
    baseline_sales_lookup_web = create_baseline_sales_lookup(predictions_2024_web)

    # Free 2024 WEB memory
    del predictions_2024_web
    gc.collect()
    logger.info("Freed 2024 WEB DataFrame memory")

    # Load feature display config for explanation generation
    feature_config = load_feature_display()

    # Load profit centers
    profit_centers = load_profit_centers_from_hana()

    # ==========================================================================
    # Generate Output Files
    # ==========================================================================
    logger.info("\n--- Generating Output Files ---")

    # Generate stores JSON with both B&M and WEB AUV (filtered to comp stores only)
    # Use B&M predictions as the base since all stores have B&M data
    stores = generate_stores_json(
        predictions=predictions_2025_bm,
        profit_centers=profit_centers,
        auv_data_bm=auv_data_bm,
        auv_data_web=auv_data_web,
        yoy_auv_changes_bm=yoy_auv_changes_bm,
        yoy_auv_changes_web=yoy_auv_changes_web,
        comp_stores_only=True  # Default: only comp stores (open 60+ weeks)
    )
    stores_output_path = data_dir / 'stores.json'
    with open(stores_output_path, 'w') as f:
        json.dump(stores, f, indent=2)
    logger.info(f"Saved: {stores_output_path}")

    # Generate DMA summary JSON (uses stores list which has both channel AUV data)
    dma_summary = generate_dma_summary(predictions_2025_bm, stores)
    dma_output_path = data_dir / 'dma_summary.json'
    with open(dma_output_path, 'w') as f:
        json.dump(dma_summary, f, indent=2)
    logger.info(f"Saved: {dma_output_path}")

    # Generate timeseries files for both channels
    store_ts_count_bm = 0
    store_ts_count_web = 0
    dma_ts_count = 0

    if not skip_timeseries:
        # Generate B&M timeseries files (store_{id}_bm.json)
        store_ts_count_bm = generate_store_timeseries(
            predictions_2025_bm,
            baseline_lookup_bm,
            baseline_sales_lookup_bm,
            feature_config,
            timeseries_dir,
            channel="B&M"
        )

        # Generate WEB timeseries files (store_{id}_web.json)
        store_ts_count_web = generate_store_timeseries(
            predictions_2025_web,
            baseline_lookup_web,
            baseline_sales_lookup_web,
            feature_config,
            timeseries_dir,
            channel="WEB"
        )

        # DMA timeseries uses B&M predictions (combined could be added later)
        dma_ts_count = generate_dma_timeseries(predictions_2025_bm, stores, timeseries_dir)
    else:
        logger.info("Skipping timeseries generation (--skip-timeseries flag)")

    # Total store timeseries count
    store_ts_count = store_ts_count_bm + store_ts_count_web

    # Close connection
    close_connection()

    # Free remaining DataFrames memory
    del predictions_2025_bm
    del predictions_2025_web
    gc.collect()

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info("Data Regeneration Complete!")
    logger.info(f"Duration: {duration}")
    logger.info(f"Output files:")
    logger.info(f"  - {dma_output_path}")
    logger.info(f"  - {stores_output_path}")
    if not skip_timeseries:
        total_ts = store_ts_count + dma_ts_count
        logger.info(f"  - {timeseries_dir}/ ({total_ts} files)")
        logger.info(f"    - B&M store timeseries: {store_ts_count_bm}")
        logger.info(f"    - WEB store timeseries: {store_ts_count_web}")
        logger.info(f"    - DMA timeseries: {dma_ts_count}")
    logger.info("=" * 60)

    return {
        'status': 'success',
        'duration': str(duration),
        'stores_count': len(stores),
        'dma_count': len(dma_summary),
        'store_timeseries_count': store_ts_count,
        'store_timeseries_count_bm': store_ts_count_bm,
        'store_timeseries_count_web': store_ts_count_web,
        'dma_timeseries_count': dma_ts_count,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate dashboard JSON files from SAP HANA"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: api/app/data/)"
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip timeseries file generation (faster, for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test HANA connection only without writing files"
    )

    args = parser.parse_args()

    verbose = not args.quiet if args.quiet else args.verbose

    result = regenerate_data(
        output_dir=args.output_dir,
        skip_timeseries=args.skip_timeseries,
        verbose=verbose,
        dry_run=args.dry_run
    )

    if result.get('error'):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
