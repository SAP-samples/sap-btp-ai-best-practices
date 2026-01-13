"""
Data loading utilities for the forecasting agent.

Wraps forecasting.regressor functions to provide:
- Cached store master data with coordinates
- Time-varying DMA awareness/consideration metrics
- Pre-computed cannibalization lookup from model_b.csv
- Fresh cannibalization computation for new hypothetical locations

This module bridges the agent tools with the real data infrastructure
in forecasting.regressor, replacing the mock data previously used.
"""

from __future__ import annotations

from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from app.agent.hana_loader import (
    load_store_master,
    load_awareness_consideration,
    load_yougov_dma_map,
    load_model_b,
)
from app.regressor.geo import haversine_miles
from app.regressor.features.model_views import get_model_a_features_for_channel
from app.agent.paths import (
    get_checkpoint_dir,
    get_model_b_path,
    CHECKPOINT_DIR as AGENT_CHECKPOINT_DIR,
    MODEL_B_PATH as AGENT_MODEL_B_PATH,
)

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from app.regressor.models import SurrogateExplainer


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

CHECKPOINT_DIR = get_checkpoint_dir()
MODEL_B_PATH = get_model_b_path()


# =============================================================================
# STORE MASTER DATA
# =============================================================================

@lru_cache(maxsize=1)
def get_store_master() -> pd.DataFrame:
    """Load and cache store master data with coordinates.

    Returns DataFrame with columns:
    - profit_center_nbr (int)
    - profit_center_name (str) - human-readable store name
    - store_address (str) - physical address
    - market_city (str) - DMA label
    - latitude, longitude (float)
    - date_opened (datetime)
    - is_outlet (bool)
    - merchandising_sf (float)
    - proforma_annual_sales (float)
    """
    df = load_store_master()
    # Filter to stores with valid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    return df


@lru_cache(maxsize=1)
def get_bm_stores() -> pd.DataFrame:
    """Get B&M stores only (have coordinates and are retail locations).

    Filters out Distribution Centers (DCs) and stores without coordinates.
    """
    stores = get_store_master()
    # Filter to retail/outlet (not DC)
    stores = stores[stores['location_type'].astype(str).str.lower() != 'dc']
    return stores


@lru_cache(maxsize=1)
def get_dmas() -> List[str]:
    """Get list of valid DMAs from store master."""
    stores = get_store_master()
    return sorted(stores['market_city'].dropna().unique().tolist())


def get_store_info(profit_center_nbr: int) -> Optional[Dict[str, Any]]:
    """Get store info by profit center number.

    Returns dict with store metadata or None if not found.
    """
    stores = get_store_master()
    store_row = stores[stores['profit_center_nbr'] == profit_center_nbr]

    if len(store_row) == 0:
        return None

    row = store_row.iloc[0]
    return {
        'profit_center_nbr': int(row['profit_center_nbr']),
        'profit_center_name': row.get('profit_center_name', 'Unknown'),
        'store_address': row.get('store_address', 'Unknown'),
        'dma': row.get('market_city', 'Unknown'),
        'latitude': row.get('latitude'),
        'longitude': row.get('longitude'),
        'date_opened': row.get('date_opened'),
        'is_outlet': bool(row.get('is_outlet', False)),
        'merchandising_sf': row.get('merchandising_sf'),
        'proforma_annual_sales': row.get('proforma_annual_sales'),
    }


# =============================================================================
# DMA AWARENESS/CONSIDERATION DATA
# =============================================================================

@lru_cache(maxsize=1)
def get_awareness_data() -> pd.DataFrame:
    """Load and cache awareness/consideration data."""
    return load_awareness_consideration()


@lru_cache(maxsize=1)
def get_yougov_dma_map() -> pd.DataFrame:
    """Load and cache YouGov DMA mapping."""
    return load_yougov_dma_map()


def get_dma_metrics_for_date(dma: str, target_date: datetime) -> Dict[str, float]:
    """Get brand awareness/consideration for a DMA at a specific date.

    Parameters
    ----------
    dma : str
        DMA name (market_city from store master)
    target_date : datetime
        Target date to find closest awareness data for

    Returns
    -------
    dict
        {'brand_awareness': float, 'brand_consideration': float}
        Values are scaled to 0-1 range.
    """
    awareness_df = get_awareness_data()
    yougov_map = get_yougov_dma_map()

    # Map market_city (DMA) to yougov market
    dma_row = yougov_map[yougov_map['market_city'] == dma]
    if len(dma_row) == 0:
        # Fallback to defaults
        return {'brand_awareness': 0.60, 'brand_consideration': 0.40}

    market = dma_row.iloc[0]['market']

    # Filter to this market
    market_awareness = awareness_df[awareness_df['market'] == market].copy()
    if len(market_awareness) == 0:
        return {'brand_awareness': 0.60, 'brand_consideration': 0.40}

    # Find closest week to target_date
    target_ts = pd.Timestamp(target_date)
    market_awareness['date_diff'] = abs(market_awareness['week_start'] - target_ts)
    closest = market_awareness.loc[market_awareness['date_diff'].idxmin()]

    # Scale to 0-1 (awareness data is in percentage 0-100)
    awareness = closest.get('awareness', 60)
    consideration = closest.get('consideration', 40)

    # Handle potential NaN values
    if pd.isna(awareness):
        awareness = 60
    if pd.isna(consideration):
        consideration = 40

    return {
        'brand_awareness': float(awareness) / 100,
        'brand_consideration': float(consideration) / 100
    }


# =============================================================================
# MODEL B DATA AND PRE-COMPUTED CANNIBALIZATION
# =============================================================================

@lru_cache(maxsize=1)
def get_model_b_data() -> pd.DataFrame:
    """Load full Model B feature data for inference from HANA."""
    return load_model_b()


def get_model_b_row(
    profit_center_nbr: int,
    channel: str,
    origin_week_date: datetime,
    horizon: int = 13,
) -> Optional[pd.Series]:
    """Return a single Model B row for the requested store/channel/date/horizon."""
    df = get_model_b_data()
    origin_str = pd.Timestamp(origin_week_date).strftime("%Y-%m-%d")
    channel_upper = (channel or "").upper()

    mask = (
        (df["profit_center_nbr"] == profit_center_nbr)
        & (df["channel"] == channel_upper)
        & (df["origin_week_date"] == origin_str)
        & (df["horizon"] == horizon)
    )
    subset = df[mask]
    if len(subset) == 0:
        return None
    return subset.iloc[0]


@lru_cache(maxsize=1)
def get_precomputed_cannibalization() -> pd.DataFrame:
    """Load pre-computed cannibalization from model_b.csv.

    Returns subset of columns needed for cannibalization lookup.
    """
    df = get_model_b_data()
    cols = [
        'profit_center_nbr', 'origin_week_date',
        'cannibalization_pressure', 'min_dist_new_store_km',
        'num_new_stores_within_10mi_last_52wk', 'num_new_stores_within_20mi_last_52wk'
    ]
    # Only keep columns that exist
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].drop_duplicates()


def lookup_cannibalization_for_existing_store(
    profit_center_nbr: int,
    origin_date: datetime
) -> Optional[Dict[str, Any]]:
    """Lookup pre-computed cannibalization for an existing store.

    Parameters
    ----------
    profit_center_nbr : int
        Store ID
    origin_date : datetime
        Origin week date to look up

    Returns
    -------
    dict or None
        Cannibalization features dict, or None if not found.
    """
    df = get_precomputed_cannibalization()
    origin_str = pd.Timestamp(origin_date).strftime('%Y-%m-%d')

    match = df[
        (df['profit_center_nbr'] == profit_center_nbr) &
        (df['origin_week_date'] == origin_str)
    ]

    if len(match) > 0:
        row = match.iloc[0]
        return {
            'cannibalization_pressure': float(row.get('cannibalization_pressure', 0)),
            'min_dist_new_store_km': float(row.get('min_dist_new_store_km', np.nan)),
            'num_new_stores_within_10mi_last_52wk': int(row.get('num_new_stores_within_10mi_last_52wk', 0)),
            'num_new_stores_within_20mi_last_52wk': int(row.get('num_new_stores_within_20mi_last_52wk', 0))
        }

    return None


def get_baseline_features_for_stores(
    store_list: List[int],
    origin_date: datetime,
    horizons: Optional[List[int]] = None,
    channel: str = 'B&M',
    fallback_scenario: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Get baseline Model B features for a list of stores at a given origin date.

    Used to prepare feature data for running inference with the real model.

    Parameters
    ----------
    store_list : List[int]
        List of profit_center_nbr values
    origin_date : datetime
        Origin week date
    horizons : List[int], optional
        Filter to specific horizons. If None, returns all horizons.
    channel : str, default 'B&M'
        Channel filter

    Returns
    -------
    pd.DataFrame
        Model B features ready for inference. If no rows are found and
        fallback_scenario is provided, attempts to build synthetic rows
        using donor rows from the same channel/DMA.
    """
    df = get_model_b_data()
    origin_str = pd.Timestamp(origin_date).strftime('%Y-%m-%d')

    # Filter to requested stores, origin date, and channel
    mask = (
        (df['profit_center_nbr'].isin(store_list)) &
        (df['origin_week_date'] == origin_str) &
        (df['channel'] == channel)
    )

    if horizons:
        mask &= df['horizon'].isin(horizons)

    result = df[mask].copy()

    # Attempt synthetic construction for new stores when no baseline rows exist
    if len(result) == 0 and fallback_scenario is not None:
        synthetic = build_synthetic_features_for_new_store(
            scenario=fallback_scenario,
            horizons=horizons or fallback_scenario.get('horizons', [1])
        )
        if synthetic is not None:
            return synthetic

    return result


def build_synthetic_features_for_new_store(
    scenario: Dict[str, Any],
    horizons: List[int],
) -> Optional[pd.DataFrame]:
    """Construct synthetic Model B rows for a new store not in model_b.csv.

    Uses median donor rows from the same channel/DMA (and outlet flag when
    available) to approximate the full feature set, then overlays the
    scenario's Model A fields. This is a pragmatic fallback to enable
    inference for brand-new stores.
    """
    df = get_model_b_data()

    channel = scenario.get('channel', 'B&M')
    dma = scenario.get('dma')
    features = scenario.get('features', {})
    origin_week_date = scenario.get('origin_week_date')

    if origin_week_date is None:
        return None

    origin_ts = pd.Timestamp(origin_week_date)

    candidates = df[df['channel'] == channel].copy()
    if dma and 'dma' in candidates.columns:
        subset = candidates[candidates['dma'] == dma]
        if len(subset) > 0:
            candidates = subset

    # Match outlet flag when possible
    if 'is_outlet' in df.columns and 'is_outlet' in features:
        outlet_flag = features.get('is_outlet')
        subset = candidates[candidates['is_outlet'] == outlet_flag]
        if len(subset) > 0:
            candidates = subset

    if len(candidates) == 0:
        return None

    rows: List[pd.Series] = []
    for h in horizons:
        sub = candidates[candidates['horizon'] == h]
        if len(sub) == 0:
            sub = candidates

        # Start from a representative row then replace numerics with medians
        base = sub.iloc[0].copy()
        numeric_cols = sub.select_dtypes(include=[np.number]).columns
        base.loc[numeric_cols] = sub[numeric_cols].median()

        base['profit_center_nbr'] = scenario.get('profit_center_nbr')
        if dma:
            base['dma'] = dma
        base['channel'] = channel
        base['origin_week_date'] = origin_ts.strftime('%Y-%m-%d')
        base['horizon'] = h

        if 'target_week_date' in base.index:
            base['target_week_date'] = (origin_ts + pd.Timedelta(weeks=h)).strftime('%Y-%m-%d')

        for key, value in features.items():
            if key in base.index:
                base[key] = value

        rows.append(base)

    if not rows:
        return None

    return pd.DataFrame(rows)


# =============================================================================
# CANNIBALIZATION COMPUTATION
# =============================================================================

def compute_cannibalization_for_new_location(
    latitude: float,
    longitude: float,
    opening_date: datetime,
    lambda_km: float = 8.0,
    tau_weeks: float = 13.0,
    max_distance_miles: float = 20.0,
    lookback_weeks: int = 52
) -> Dict[str, Any]:
    """Compute cannibalization pressure for a hypothetical new store location.

    Uses the validated formula from app.regressor.features.cannibalization:
    pressure = SUM_j[ exp(-dist_ij_km / lambda) * (1 + weeks_since_j_open / tau) ]

    This computes how much cannibalization pressure the NEW store would experience
    from existing nearby stores that opened recently.

    Parameters
    ----------
    latitude : float
        New store latitude
    longitude : float
        New store longitude
    opening_date : datetime
        New store opening date
    lambda_km : float, default 8.0
        Distance decay parameter in kilometers
    tau_weeks : float, default 13.0
        Time intensification parameter in weeks
    max_distance_miles : float, default 20.0
        Maximum distance to consider for cannibalization
    lookback_weeks : int, default 52
        Only consider stores opened within this many weeks

    Returns
    -------
    dict
        Cannibalization features:
        - cannibalization_pressure: Continuous pressure metric
        - min_dist_new_store_km: Distance to nearest recently opened store
        - num_new_stores_within_10mi_last_52wk: Count within 10 miles
        - num_new_stores_within_20mi_last_52wk: Count within 20 miles
    """
    stores = get_bm_stores()
    opening_ts = pd.Timestamp(opening_date)

    # Compute distances to all existing stores
    distances_miles = haversine_miles(
        latitude, longitude,
        stores['latitude'].values,
        stores['longitude'].values
    )
    distances_km = distances_miles * 1.60934

    # Get store opening dates
    stores = stores.copy()
    stores['distance_miles'] = distances_miles
    stores['distance_km'] = distances_km
    stores['weeks_since_open'] = (opening_ts - stores['date_opened']).dt.days / 7.0

    # Filter to "new" stores (opened within lookback period, within max distance)
    nearby_new = stores[
        (stores['weeks_since_open'] > 0) &
        (stores['weeks_since_open'] <= lookback_weeks) &
        (stores['distance_miles'] <= max_distance_miles) &
        (stores['distance_miles'] > 0)  # Exclude self if somehow present
    ].copy()

    if len(nearby_new) == 0:
        return {
            'cannibalization_pressure': 0.0,
            'min_dist_new_store_km': np.nan,
            'num_new_stores_within_10mi_last_52wk': 0,
            'num_new_stores_within_20mi_last_52wk': 0
        }

    # Apply validated formula
    nearby_new['distance_decay'] = np.exp(-nearby_new['distance_km'] / lambda_km)
    nearby_new['time_intensification'] = 1 + (nearby_new['weeks_since_open'] / tau_weeks)
    nearby_new['pressure_contribution'] = (
        nearby_new['distance_decay'] * nearby_new['time_intensification']
    )

    total_pressure = nearby_new['pressure_contribution'].sum()
    min_dist_km = nearby_new['distance_km'].min()
    num_within_10mi = int((nearby_new['distance_miles'] <= 10).sum())
    num_within_20mi = int((nearby_new['distance_miles'] <= 20).sum())

    return {
        'cannibalization_pressure': float(total_pressure),
        'min_dist_new_store_km': float(min_dist_km),
        'num_new_stores_within_10mi_last_52wk': num_within_10mi,
        'num_new_stores_within_20mi_last_52wk': num_within_20mi
    }


def compute_cannibalization_with_new_store(
    existing_store_id: int,
    new_store_lat: float,
    new_store_lon: float,
    new_store_opening_date: datetime,
    origin_date: datetime,
    lambda_km: float = 8.0,
    tau_weeks: float = 13.0,
) -> Dict[str, Any]:
    """Compute updated cannibalization for an existing store given a new store opening.

    This adds the new store's contribution to the existing store's cannibalization
    pressure. Used for what-if analysis: "If we open a new store here, how does
    it affect existing stores' cannibalization pressure?"

    Parameters
    ----------
    existing_store_id : int
        Profit center number of the existing store
    new_store_lat : float
        Latitude of the hypothetical new store
    new_store_lon : float
        Longitude of the hypothetical new store
    new_store_opening_date : datetime
        Opening date of the hypothetical new store
    origin_date : datetime
        Origin date for the analysis (when predictions are made)
    lambda_km : float, default 8.0
        Distance decay parameter
    tau_weeks : float, default 13.0
        Time intensification parameter

    Returns
    -------
    dict
        Updated cannibalization features for the existing store
    """
    # Get existing store's current cannibalization from model_b.csv
    existing_cannib = lookup_cannibalization_for_existing_store(
        existing_store_id, origin_date
    )

    if existing_cannib is None:
        existing_cannib = {
            'cannibalization_pressure': 0.0,
            'min_dist_new_store_km': np.nan,
            'num_new_stores_within_10mi_last_52wk': 0,
            'num_new_stores_within_20mi_last_52wk': 0
        }

    # Get existing store's coordinates
    stores = get_bm_stores()
    store_row = stores[stores['profit_center_nbr'] == existing_store_id]
    if len(store_row) == 0:
        return existing_cannib

    existing_lat = float(store_row.iloc[0]['latitude'])
    existing_lon = float(store_row.iloc[0]['longitude'])

    # Compute distance to new store
    distance_miles = haversine_miles(existing_lat, existing_lon, new_store_lat, new_store_lon)
    distance_km = distance_miles * 1.60934

    # If new store is beyond 20 miles, no additional cannibalization
    if distance_miles > 20:
        return existing_cannib

    # Compute weeks since new store opened (relative to origin_date)
    origin_ts = pd.Timestamp(origin_date)
    opening_ts = pd.Timestamp(new_store_opening_date)
    weeks_since_new_open = (origin_ts - opening_ts).days / 7.0

    # Only add cannibalization if new store has opened (positive weeks)
    if weeks_since_new_open <= 0:
        return existing_cannib

    # Compute new store's contribution to cannibalization pressure
    distance_decay = np.exp(-distance_km / lambda_km)
    time_intensification = 1 + (weeks_since_new_open / tau_weeks)
    new_pressure_contribution = distance_decay * time_intensification

    # Update cannibalization features
    updated = existing_cannib.copy()
    updated['cannibalization_pressure'] = float(
        existing_cannib['cannibalization_pressure'] + new_pressure_contribution
    )

    # Update min distance if new store is closer
    current_min = existing_cannib['min_dist_new_store_km']
    if pd.isna(current_min) or distance_km < current_min:
        updated['min_dist_new_store_km'] = float(distance_km)

    # Update counts
    if distance_miles <= 10:
        updated['num_new_stores_within_10mi_last_52wk'] = int(
            existing_cannib['num_new_stores_within_10mi_last_52wk'] + 1
        )
    if distance_miles <= 20:
        updated['num_new_stores_within_20mi_last_52wk'] = int(
            existing_cannib['num_new_stores_within_20mi_last_52wk'] + 1
        )

    return updated


def get_stores_within_radius(
    center_lat: float,
    center_lon: float,
    radius_miles: float = 20.0
) -> List[int]:
    """Get list of store IDs within a radius of a location.

    Parameters
    ----------
    center_lat : float
        Center latitude
    center_lon : float
        Center longitude
    radius_miles : float, default 20.0
        Radius in miles

    Returns
    -------
    List[int]
        List of profit_center_nbr values for stores within radius
    """
    stores = get_bm_stores()

    distances = haversine_miles(
        center_lat, center_lon,
        stores['latitude'].values,
        stores['longitude'].values
    )

    within_radius = stores[distances <= radius_miles]
    return within_radius['profit_center_nbr'].astype(int).tolist()


# =============================================================================
# INFERENCE PIPELINE INTEGRATION
# =============================================================================

_inference_pipeline = None
_surrogates: Dict[str, Optional[Any]] = {}


def get_inference_pipeline():
    """Get or create the singleton InferencePipeline instance.

    Uses lazy loading to avoid import overhead when inference is not needed.
    The pipeline is configured with the default checkpoint directory.

    Returns
    -------
    InferencePipeline or None
        Configured pipeline, or None if models not available.
    """
    global _inference_pipeline

    if _inference_pipeline is not None:
        return _inference_pipeline

    try:
        from app.regressor.pipelines import InferencePipeline
        from app.regressor.configs import InferenceConfig, BiasCorrection

        # Default checkpoint location
        checkpoint_dir = CHECKPOINT_DIR
        output_dir = Path('output_agent_inference')

        # Check if checkpoints exist
        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
            return None

        config = InferenceConfig(
            name="agent_inference",
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            channels=["B&M", "WEB"],
            bias_correction=BiasCorrection(correct_bm=False, correct_web=False),
            run_explainability=False,
        )

        _inference_pipeline = InferencePipeline(config)
        return _inference_pipeline

    except ImportError as e:
        print(f"Warning: Could not import InferencePipeline: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize InferencePipeline: {e}")
        return None


def run_inference(
    model_b_features: pd.DataFrame,
    channels: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Run model inference on the provided features.

    This is the main entry point for getting model predictions from the agent.
    It handles lazy loading of the InferencePipeline and returns predictions.

    Parameters
    ----------
    model_b_features : pd.DataFrame
        Model B features to predict on. Must include columns like:
        - profit_center_nbr
        - channel
        - origin_week_date
        - horizon
        - All required feature columns

    channels : List[str], optional
        Channels to predict. Default is to predict on whatever channels
        are present in the data.

    Returns
    -------
    pd.DataFrame or None
        Predictions with columns like pred_sales_p50, pred_aov_p50, etc.
        Returns None if inference fails or models not available.

    Notes
    -----
    The predictions include:
    - pred_sales_p50, pred_sales_p90: Sales quantiles
    - pred_aov_p50, pred_aov_p90: AOV quantiles
    - pred_traffic_p50, pred_traffic_p90: Traffic quantiles (B&M only)
    """
    pipeline = get_inference_pipeline()

    if pipeline is None:
        return None

    try:
        result = pipeline.run(model_b_features, channels=channels)

        # Combine B&M and WEB predictions
        dfs = []
        if result.bm_predictions is not None:
            dfs.append(result.bm_predictions)
        if result.web_predictions is not None:
            dfs.append(result.web_predictions)

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        print(f"Inference failed: {e}")
        return None


def run_whatif_inference(
    baseline_features: pd.DataFrame,
    modified_features: pd.DataFrame,
    channels: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Run what-if comparison inference.

    Runs inference on both baseline and modified features, then computes
    the delta between them.

    Parameters
    ----------
    baseline_features : pd.DataFrame
        Original Model B features (baseline scenario)
    modified_features : pd.DataFrame
        Modified Model B features (what-if scenario)
    channels : List[str], optional
        Channels to predict on

    Returns
    -------
    dict or None
        Dictionary with:
        - 'baseline_predictions': DataFrame of baseline predictions
        - 'whatif_predictions': DataFrame of what-if predictions
        - 'delta': DataFrame with delta columns (pred_delta_sales, etc.)

    Notes
    -----
    The dataframes are aligned by key columns (profit_center_nbr, channel,
    origin_week_date, horizon) for proper comparison.
    """
    baseline_preds = run_inference(baseline_features, channels)
    whatif_preds = run_inference(modified_features, channels)

    if baseline_preds is None or whatif_preds is None:
        return None

    # Key columns for merging
    keys = ['profit_center_nbr', 'channel', 'origin_week_date', 'horizon']
    keys = [k for k in keys if k in baseline_preds.columns]

    # Merge predictions
    baseline_subset = baseline_preds[keys + ['pred_sales_p50', 'pred_aov_p50']].copy()
    baseline_subset = baseline_subset.rename(columns={
        'pred_sales_p50': 'baseline_sales_p50',
        'pred_aov_p50': 'baseline_aov_p50',
    })

    whatif_subset = whatif_preds[keys + ['pred_sales_p50', 'pred_aov_p50']].copy()
    whatif_subset = whatif_subset.rename(columns={
        'pred_sales_p50': 'whatif_sales_p50',
        'pred_aov_p50': 'whatif_aov_p50',
    })

    merged = baseline_subset.merge(whatif_subset, on=keys, how='inner')

    # Compute deltas
    merged['delta_sales'] = merged['whatif_sales_p50'] - merged['baseline_sales_p50']
    merged['delta_sales_pct'] = (
        merged['delta_sales'] / merged['baseline_sales_p50'] * 100
    ).fillna(0)
    merged['delta_aov'] = merged['whatif_aov_p50'] - merged['baseline_aov_p50']

    return {
        'baseline_predictions': baseline_preds,
        'whatif_predictions': whatif_preds,
        'delta': merged,
    }


# =============================================================================
# SURROGATE EXPLAINER (SHAP) HELPERS
# =============================================================================

def get_surrogate_explainer(channel: str) -> Optional[Any]:
    """Lazy-load and cache surrogate explainer for a channel.

    Parameters
    ----------
    channel : str
        "B&M" or "WEB"

    Returns
    -------
    SurrogateExplainer or None
    """
    global _surrogates
    ch = (channel or "").upper()
    if ch in _surrogates:
        return _surrogates[ch]

    try:
        from app.regressor.models import SurrogateExplainer
        model_path = CHECKPOINT_DIR / ("surrogate_bm.cbm" if ch == "B&M" else "surrogate_web.cbm")
        meta_path = CHECKPOINT_DIR / ("surrogate_bm.meta.json" if ch == "B&M" else "surrogate_web.meta.json")
        if not model_path.exists():
            return None
        explainer = SurrogateExplainer(channel=ch)
        explainer.load_model(model_path, meta_path=str(meta_path) if meta_path.exists() else None)
        _surrogates[ch] = explainer
        return explainer
    except Exception as exc:
        print(f"Warning: SurrogateExplainer unavailable for channel {ch}: {exc}")
        return None


def compute_surrogate_shap(
    channel: str,
    feature_row: Dict[str, Any],
    name: str = "Sales",
    top_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Compute SHAP using surrogate for a single feature row.

    Parameters
    ----------
    channel : str
        Channel name ("B&M" or "WEB")
    feature_row : dict
        Model A feature values
    name : str
        Name used in explainer output
    top_k : int
        Number of top contributors to return

    Returns
    -------
    dict or None
        SHAP output formatted for storage:
        {"top_features": [...], "feature_contributions": {...}, "expected_value": float, "surrogate_r2": float}
    """
    explainer = get_surrogate_explainer(channel)
    if explainer is None:
        return None

    # Build feature row using Model A features for this channel
    feat_list = get_model_a_features_for_channel(channel)
    row = {}
    for k in feat_list:
        val = feature_row.get(k)
        if isinstance(val, bool):
            val = int(val)
        if val is None:
            val = 0
        row[k] = val

    df = pd.DataFrame([row]).reindex(columns=feat_list, fill_value=0)

    # Compute SHAP values directly using the surrogate explainer
    # The explainer's compute_shap_values handles data preparation internally
    print(f"Generating SHAP explanation for {name}...")
    try:
        shap_values = explainer.compute_shap_values(df)
    except Exception as e:
        print(f"Warning: SHAP computation failed: {e}")
        return None

    # shap_values shape: (1, num_features) for single row
    if shap_values is None or len(shap_values) == 0:
        return None

    # Get feature names from the explainer (in the order used during training)
    feature_names = explainer.feature_names
    if not feature_names:
        print("Warning: Explainer has no feature_names")
        return None

    # Build feature-level SHAP data
    shap_row = shap_values[0]  # Single row
    feature_shap_pairs = list(zip(feature_names, shap_row))

    # Sort by absolute SHAP value (descending) to get top contributors
    feature_shap_pairs_sorted = sorted(feature_shap_pairs, key=lambda x: abs(x[1]), reverse=True)

    # Build top_features list (top-k contributors)
    top_features = []
    for feat_name, shap_val in feature_shap_pairs_sorted[:top_k]:
        feat_value = feature_row.get(feat_name)
        top_features.append({
            "feature": feat_name,
            "value": feat_value,
            "shap_value": float(shap_val),
        })

    # Build feature_contributions dict (all features)
    total_abs_shap = sum(abs(sv) for _, sv in feature_shap_pairs)
    feature_contributions = {}
    for feat_name, shap_val in feature_shap_pairs:
        feat_value = feature_row.get(feat_name)
        pct_contrib = (abs(shap_val) / total_abs_shap * 100) if total_abs_shap > 0 else 0
        feature_contributions[feat_name] = {
            "value": feat_value,
            "shap": float(shap_val),
            "pct_contribution": pct_contrib,
        }

    # Get surrogate R2 score
    surrogate_r2 = getattr(explainer, "r2_score_value", None)

    return {
        "top_features": top_features,
        "feature_contributions": feature_contributions,
        "expected_value": None,  # Not easily available from TreeExplainer
        "surrogate_r2": surrogate_r2,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Store master
    'get_store_master',
    'get_bm_stores',
    'get_dmas',
    'get_store_info',
    # DMA metrics
    'get_awareness_data',
    'get_dma_metrics_for_date',
    # Model B data
    'get_model_b_data',
    'get_model_b_row',
    'get_precomputed_cannibalization',
    'lookup_cannibalization_for_existing_store',
    'get_baseline_features_for_stores',
    'build_synthetic_features_for_new_store',
    # Cannibalization computation
    'compute_cannibalization_for_new_location',
    'compute_cannibalization_with_new_store',
    'get_stores_within_radius',
    # Inference
    'get_inference_pipeline',
    'run_inference',
    'run_whatif_inference',
    # Surrogate explainability
    'get_surrogate_explainer',
    'compute_surrogate_shap',
]
