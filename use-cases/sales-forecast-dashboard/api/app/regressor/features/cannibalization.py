"""
Cannibalization Pressure Features (FE-06)

Quantifies competitive pressure from nearby store openings.
Empirically validated formula from event study analysis.

TIER 2-3 Feature:
- 5-10% improvement for affected stores
- Impact: -13.8% to -22.0% for stores <10 miles from new openings

Formula:
    cannibalization_pressure_i = Σ_j exp(-dist_ij/8km) × (1 + weeks_since_j_open/13)

Validated Parameters:
- λ = 8 km (primary impact zone)
- τ = 13 weeks (effects intensify, not decay)
- Only stores j opened within last 52 weeks
- Only stores j within 20 miles (32 km) of store i

Author: EPIC 4 Feature Engineering
Status: Phase 2.6 - Cannibalization Features
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from app.regressor.features.transforms import compute_lag
from app.regressor.geo import haversine_miles
from app.regressor.io_utils import load_store_master


def compute_cannibalization_pressure(
    canonical_df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    lambda_km: float = 8.0,
    tau_weeks: float = 13.0,
    max_distance_miles: float = 20.0,
    lookback_weeks: int = 52
) -> pd.DataFrame:
    """
    Compute cannibalization pressure from nearby store openings.

    TIME ALIGNMENT: TARGET-ALIGNED (competitive pressure at forecast target for What-If scenarios)

    Validated formula (from event study analysis):

        cannibalization_pressure_i = sum_j exp(-dist_ij/lambda) * (1 + weeks_since_j_open/tau)

    Where:
    - lambda = 8 km (primary impact zone, empirically validated)
    - tau = 13 weeks (effects intensify, not decay)
    - Only sum over stores j opened within last 52 weeks (relative to target date)
    - Only include stores j within 20 miles (32 km) of store i

    Empirical Impact Magnitudes (from RESULTS_INTERPRETATION_GUIDE.md):
    - 0-10 miles: -13.8% immediate, intensifying to -22.0% by week 13
    - 10-20 miles: -4.8% immediate, growing to -7.6% by week 13
    - >20 miles: ~0% (no measurable effect)

    Returns Features:
    - cannibalization_pressure: Continuous pressure metric
    - min_dist_new_store_km: Distance to nearest store opened <52 weeks ago
    - num_new_stores_within_10mi_last_52wk: Count of new stores within 10 miles
    - num_new_stores_within_20mi_last_52wk: Count of new stores within 20 miles

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with (store, origin_week_date, target_week_date) rows
    store_master_df : pd.DataFrame
        Store master with columns:
        - profit_center_nbr (or store_id)
        - latitude, longitude (or lat, lon)
        - date_opened (or open_date)
    target_col : str, default='target_week_date'
        Name of target week column (features aligned to forecast target)
    store_col : str, default='profit_center_nbr'
        Name of store column
    lambda_km : float, default=8.0
        Distance decay parameter in kilometers (lambda)
    tau_weeks : float, default=13.0
        Time intensification parameter in weeks (tau)
    max_distance_miles : float, default=20.0
        Maximum distance to consider for cannibalization (miles)
    lookback_weeks : int, default=52
        Only consider stores opened within last N weeks

    Returns
    -------
    pd.DataFrame
        Canonical table with cannibalization features attached

    Examples
    --------
    >>> from app.regressor.io_utils import load_store_master
    >>> store_master = load_store_master()
    >>> df = compute_cannibalization_pressure(canonical_df, store_master)
    >>> # Validation: Check stores <10 miles from recent openings
    >>> high_pressure = df[df['cannibalization_pressure'] > 0.5]
    >>> print(f"Stores under high pressure: {high_pressure['profit_center_nbr'].nunique()}")

    Notes
    -----
    - Pressure = 0 for stores >20 miles from any recent opening
    - Effects intensify over first 13 weeks (not decay)
    - Validated against event study showing -13.8% to -22% impact <10 miles
    """
    df = canonical_df.copy()

    # Prepare store master
    store_df = _prepare_store_master_for_cannibalization(store_master_df, store_col)

    # Compute pairwise distances between all stores
    distance_matrix = _compute_store_distance_matrix(store_df, store_col)

    # Get unique target dates from canonical table - TARGET-ALIGNED
    target_dates = df[target_col].unique()

    # Compute cannibalization features for each target date
    cannibalization_features = []

    for target_date in target_dates:
        target_date_pd = pd.to_datetime(target_date)

        # For each store, compute pressure at this target date
        for store_id in store_df[store_col].unique():
            features = _compute_pressure_for_store_date(
                store_id=store_id,
                reference_date=target_date_pd,
                store_df=store_df,
                distance_matrix=distance_matrix,
                store_col=store_col,
                lambda_km=lambda_km,
                tau_weeks=tau_weeks,
                max_distance_miles=max_distance_miles,
                lookback_weeks=lookback_weeks
            )

            features[store_col] = store_id
            features[target_col] = target_date
            cannibalization_features.append(features)

    # Convert to dataframe
    cannibalization_df = pd.DataFrame(cannibalization_features)

    # Join to canonical table on (store, target_week_date) - TARGET-ALIGNED
    df = df.merge(
        cannibalization_df,
        on=[store_col, target_col],
        how='left'
    )

    # Fill NaN with 0 (stores with no pressure)
    pressure_cols = [
        'cannibalization_pressure',
        'num_new_stores_within_10mi_last_52wk',
        'num_new_stores_within_20mi_last_52wk'
    ]
    for col in pressure_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def _prepare_store_master_for_cannibalization(
    store_master_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Prepare store master with standardized columns for cannibalization computation.

    Parameters
    ----------
    store_master_df : pd.DataFrame
        Raw store master
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Cleaned store master with: store_id, latitude, longitude, date_opened
    """
    df = store_master_df.copy()

    # Standardize store column
    if store_col not in df.columns and 'store_id' in df.columns:
        df = df.rename(columns={'store_id': store_col})

    # Standardize lat/lon columns
    if 'latitude' not in df.columns and 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    if 'longitude' not in df.columns and 'lon' in df.columns:
        df = df.rename(columns={'lon': 'longitude'})

    # Standardize date column
    if 'date_opened' not in df.columns and 'open_date' in df.columns:
        df = df.rename(columns={'open_date': 'date_opened'})

    # Ensure date_opened is datetime
    if 'date_opened' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date_opened']):
            df['date_opened'] = pd.to_datetime(df['date_opened'], errors='coerce')

    # Filter to stores with valid lat/lon and opening date
    required_cols = [store_col, 'latitude', 'longitude', 'date_opened']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Store master missing required columns: {missing_cols}")

    # Drop stores with missing lat/lon or date_opened
    df = df.dropna(subset=['latitude', 'longitude', 'date_opened'])

    return df


def _compute_store_distance_matrix(
    store_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Compute pairwise distance matrix between all stores.

    Parameters
    ----------
    store_df : pd.DataFrame
        Store master with latitude, longitude
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Distance matrix with columns: store_i, store_j, distance_miles, distance_km
    """
    stores = store_df[[store_col, 'latitude', 'longitude']].drop_duplicates()

    # Cartesian product (all pairs)
    stores_i = stores.copy()
    stores_j = stores.copy()
    stores_i['_key'] = 1
    stores_j['_key'] = 1

    pairs = stores_i.merge(stores_j, on='_key', suffixes=('_i', '_j'))
    pairs = pairs.drop(columns=['_key'])

    # Compute distances
    pairs['distance_miles'] = haversine_miles(
        pairs['latitude_i'],
        pairs['longitude_i'],
        pairs['latitude_j'],
        pairs['longitude_j']
    )

    # Convert to km
    pairs['distance_km'] = pairs['distance_miles'] * 1.60934

    # Keep only needed columns
    distance_matrix = pairs[[
        f'{store_col}_i',
        f'{store_col}_j',
        'distance_miles',
        'distance_km'
    ]]

    return distance_matrix


def _compute_pressure_for_store_date(
    store_id: int,
    reference_date: pd.Timestamp,
    store_df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    store_col: str,
    lambda_km: float,
    tau_weeks: float,
    max_distance_miles: float,
    lookback_weeks: int
) -> dict:
    """
    Compute cannibalization pressure for a single store at a single reference date.

    Parameters
    ----------
    store_id : int
        Store to compute pressure for
    reference_date : pd.Timestamp
        Reference date for pressure calculation (target_week_date for target-aligned)
    store_df : pd.DataFrame
        Store master
    distance_matrix : pd.DataFrame
        Pairwise distances
    store_col : str
        Store column name
    lambda_km : float
        Distance decay parameter
    tau_weeks : float
        Time intensification parameter
    max_distance_miles : float
        Maximum distance to consider
    lookback_weeks : int
        Only consider stores opened within last N weeks

    Returns
    -------
    dict
        Features: cannibalization_pressure, min_dist_new_store_km, num_new_stores_within_{10,20}mi
    """
    # Get distances from this store to all other stores
    distances = distance_matrix[
        distance_matrix[f'{store_col}_i'] == store_id
    ].copy()

    if len(distances) == 0:
        return {
            'cannibalization_pressure': 0.0,
            'min_dist_new_store_km': np.nan,
            'num_new_stores_within_10mi_last_52wk': 0,
            'num_new_stores_within_20mi_last_52wk': 0
        }

    # Merge store opening dates
    distances = distances.merge(
        store_df[[store_col, 'date_opened']],
        left_on=f'{store_col}_j',
        right_on=store_col,
        how='left'
    )

    # Compute weeks since each store j opened (at reference_date)
    distances['weeks_since_j_open'] = (
        (reference_date - distances['date_opened']).dt.days / 7.0
    )

    # Filter to "new" stores (opened within lookback_weeks)
    new_stores = distances[
        (distances['weeks_since_j_open'] > 0) &  # Opened before reference_date
        (distances['weeks_since_j_open'] <= lookback_weeks) &  # Within lookback window
        (distances['distance_miles'] > 0) &  # Exclude self (distance=0)
        (distances['distance_miles'] <= max_distance_miles)  # Within max distance
    ].copy()

    if len(new_stores) == 0:
        return {
            'cannibalization_pressure': 0.0,
            'min_dist_new_store_km': np.nan,
            'num_new_stores_within_10mi_last_52wk': 0,
            'num_new_stores_within_20mi_last_52wk': 0
        }

    # Compute cannibalization pressure formula
    # pressure_j = exp(-dist_ij/λ) × (1 + weeks_since_j_open/τ)
    new_stores['distance_decay'] = np.exp(-new_stores['distance_km'] / lambda_km)
    new_stores['time_intensification'] = 1 + (new_stores['weeks_since_j_open'] / tau_weeks)
    new_stores['pressure_contribution'] = (
        new_stores['distance_decay'] * new_stores['time_intensification']
    )

    # Sum pressure from all nearby new stores
    total_pressure = new_stores['pressure_contribution'].sum()

    # Min distance to new store
    min_dist_km = new_stores['distance_km'].min()

    # Count new stores within 10 and 20 miles
    num_within_10mi = (new_stores['distance_miles'] <= 10).sum()
    num_within_20mi = (new_stores['distance_miles'] <= 20).sum()

    return {
        'cannibalization_pressure': total_pressure,
        'min_dist_new_store_km': min_dist_km,
        'num_new_stores_within_10mi_last_52wk': num_within_10mi,
        'num_new_stores_within_20mi_last_52wk': num_within_20mi
    }


def validate_cannibalization_impact(
    df: pd.DataFrame,
    store_col: str = 'profit_center_nbr',
    target_col: str = 'label_log_sales'
) -> pd.DataFrame:
    """
    Validate cannibalization pressure against empirical impact benchmarks.

    Expected Impacts (from event study):
    - 0-10 miles: -13.8% to -22.0% by week 13
    - 10-20 miles: -4.8% to -7.6% by week 13
    - >20 miles: ~0%

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cannibalization features and targets
    store_col : str, default='profit_center_nbr'
        Store column
    target_col : str, default='label_log_sales'
        Target column to measure impact on

    Returns
    -------
    pd.DataFrame
        Validation results by distance bucket

    Examples
    --------
    >>> results = validate_cannibalization_impact(canonical_df)
    >>> print(results)
    """
    if 'cannibalization_pressure' not in df.columns:
        raise ValueError("Dataframe must contain 'cannibalization_pressure' column")

    if 'min_dist_new_store_km' not in df.columns:
        raise ValueError("Dataframe must contain 'min_dist_new_store_km' column")

    # Convert km to miles for bucket analysis
    df['min_dist_new_store_miles'] = df['min_dist_new_store_km'] / 1.60934

    # Define distance buckets
    df['distance_bucket'] = pd.cut(
        df['min_dist_new_store_miles'],
        bins=[0, 10, 20, 999],
        labels=['0-10 miles', '10-20 miles', '>20 miles']
    )

    # Compute average target by bucket
    impact_analysis = df.groupby('distance_bucket').agg({
        target_col: ['mean', 'std', 'count'],
        'cannibalization_pressure': ['mean', 'std']
    }).round(4)

    return impact_analysis
