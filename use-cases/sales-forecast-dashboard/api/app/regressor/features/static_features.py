"""
Static Store DNA Features (FE-05)

Store-level characteristics that are time-invariant or slowly-changing.
Provides cross-sectional lift and baseline adjustments.

TIER 1 Cross-Sectional Predictors:
- proforma_annual_sales: Pooled ρ=+0.84 (STRONGEST predictor)
- is_outlet: Pooled ρ=-0.54 (large negative effect)

TIER 2 Store Maturity (time-aware at t0+h):
- weeks_since_open: FE ρ=+0.30 for conversion (CRITICAL for new stores)
- Computed at target_week_date for accurate forecast horizon maturity

Author: EPIC 4 Feature Engineering
Status: Phase 2.5 - Static Features
"""

import numpy as np
import pandas as pd
from typing import Optional


def attach_static_store_features(
    canonical_df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr'
) -> pd.DataFrame:
    """
    Attach static store DNA features and time-aware maturity features.

    TIER 1 Cross-Sectional Predictors:
    - proforma_annual_sales: Pooled ρ=+0.84 (STRONGEST predictor for cross-sectional variation)
    - is_outlet: Pooled ρ=-0.54 (large negative effect on sales/conversion)

    TIER 2 Store Maturity (time-aware at t0+h):
    - weeks_since_open (at target_week_date):
      * FE ρ=+0.30 for conversion (CRITICAL for new stores)
      * Computed from date_opened to target_week_date
    - weeks_since_open_capped_13: min(weeks_since_open, 13)
    - weeks_since_open_capped_52: min(weeks_since_open, 52)
    - is_new_store: Binary (weeks_since_open < 52)
    - is_comp_store: Binary (weeks_since_open >= 60, standard comp threshold)

    TIER 2 Store Characteristics:
    - sq_ft: Total square footage
    - merchandising_sf: Merchandising space
    - store_design_sf: Design consultation space
    - region: Geographic region (categorical for CatBoost)
    - format: Store format (categorical)
    - store_id, dma_id: Categorical IDs (CatBoost handles natively)

    TIER 3 (Optional):
    - cohort: Opening year/quarter
    - parking: Parking availability
    - open_date: Store opening date

    CRITICAL: weeks_since_open must be computed at target_week_date (t0 + h),
    NOT origin_week_date, to get accurate maturity at forecast horizon.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with (store, origin, horizon, target_week_date) rows
    store_master_df : pd.DataFrame
        Store master data with columns:
        - profit_center_nbr (or store_id)
        - proforma_annual_sales
        - is_outlet
        - date_opened (or open_date)
        - sq_ft
        - region (optional)
        - format (optional)
        - dma (or dma_id)
    target_col : str, default='target_week_date'
        Name of target date column (t0 + h)
    store_col : str, default='profit_center_nbr'
        Name of store column

    Returns
    -------
    pd.DataFrame
        Canonical table with static store DNA features attached

    Examples
    --------
    >>> from app.regressor.io_utils import load_store_master
    >>> store_master = load_store_master()
    >>> df = attach_static_store_features(canonical_df, store_master)
    >>> # Validation
    >>> assert 'proforma_annual_sales' in df.columns
    >>> assert 'weeks_since_open' in df.columns

    Notes
    -----
    - weeks_since_open varies by horizon (different target_week_date)
    - All other features are truly static (constant across horizons)
    - Validates: proforma_annual_sales Pooled ρ ≈ +0.84
    """
    df = canonical_df.copy()

    # Prepare store master
    store_df = _prepare_store_master(store_master_df, store_col)

    # Ensure target_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[target_col]):
        df[target_col] = pd.to_datetime(df[target_col])

    # Join static features (constant across all horizons)
    static_features = _extract_static_features(store_df, store_col)

    df = df.merge(
        static_features,
        on=store_col,
        how='left'
    )

    # Compute time-aware maturity features at target_week_date
    df = _compute_maturity_features(df, store_df, target_col, store_col)

    return df


def _prepare_store_master(
    store_master_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Prepare store master dataframe with standardized column names.

    Parameters
    ----------
    store_master_df : pd.DataFrame
        Raw store master data
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Cleaned store master
    """
    df = store_master_df.copy()

    # Standardize store column
    if store_col not in df.columns and 'store_id' in df.columns:
        df = df.rename(columns={'store_id': store_col})

    # Standardize date column
    if 'date_opened' not in df.columns and 'open_date' in df.columns:
        df = df.rename(columns={'open_date': 'date_opened'})

    # Ensure date_opened is datetime
    if 'date_opened' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date_opened']):
            df['date_opened'] = pd.to_datetime(df['date_opened'], errors='coerce')

    # Standardize DMA column
    if 'dma' not in df.columns and 'dma_id' in df.columns:
        df = df.rename(columns={'dma_id': 'dma'})

    return df


def _extract_static_features(
    store_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Extract truly static features (constant across all time periods).

    Parameters
    ----------
    store_df : pd.DataFrame
        Store master dataframe
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Dataframe with static features
    """
    # Define columns to extract (if they exist)
    feature_cols = [store_col]

    # TIER 1: proforma_annual_sales, is_outlet
    tier1_cols = ['proforma_annual_sales', 'is_outlet']
    for col in tier1_cols:
        if col in store_df.columns:
            feature_cols.append(col)

    # TIER 2: Store characteristics
    tier2_cols = [
        'sq_ft',
        'merchandising_sf',
        'store_design_sf',
        'region',
        'format',
        'dma'
    ]
    for col in tier2_cols:
        if col in store_df.columns:
            feature_cols.append(col)

    # TIER 3: Cohort info
    tier3_cols = ['cohort', 'parking']
    for col in tier3_cols:
        if col in store_df.columns:
            feature_cols.append(col)

    # Keep date_opened for maturity calculation (but don't include as feature)
    if 'date_opened' in store_df.columns and 'date_opened' not in feature_cols:
        feature_cols.append('date_opened')

    # Extract unique store records
    static_df = store_df[feature_cols].drop_duplicates(subset=[store_col])

    return static_df


def _compute_maturity_features(
    df: pd.DataFrame,
    store_df: pd.DataFrame,
    target_col: str,
    store_col: str
) -> pd.DataFrame:
    """
    Compute time-aware maturity features at target_week_date (t0 + h).

    CRITICAL: Must use target_week_date, NOT origin_week_date, to get
    accurate store maturity at the forecast horizon.

    Features:
    - weeks_since_open: Continuous weeks from opening to target_week_date
    - weeks_since_open_capped_13: min(weeks_since_open, 13) for learning ramp
    - weeks_since_open_capped_52: min(weeks_since_open, 52) for first-year effect
    - is_new_store: Binary, weeks_since_open < 52
    - is_comp_store: Binary, weeks_since_open >= 60 (standard comp definition)

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table with target_week_date
    store_df : pd.DataFrame
        Store master with date_opened
    target_col : str
        Target date column
    store_col : str
        Store column

    Returns
    -------
    pd.DataFrame
        Dataframe with maturity features
    """
    # If date_opened not available, return with NaN maturity features
    if 'date_opened' not in store_df.columns:
        df['weeks_since_open'] = np.nan
        df['weeks_since_open_capped_13'] = np.nan
        df['weeks_since_open_capped_52'] = np.nan
        df['is_new_store'] = 0
        df['is_comp_store'] = 0
        return df

    # Merge date_opened from store_df if not already present
    if 'date_opened' not in df.columns:
        df = df.merge(
            store_df[[store_col, 'date_opened']].drop_duplicates(),
            on=store_col,
            how='left'
        )

    # Compute weeks_since_open at target_week_date
    df['weeks_since_open'] = (
        (df[target_col] - df['date_opened']).dt.days / 7.0
    )

    # Handle negative values (target before opening - shouldn't happen but be safe)
    df['weeks_since_open'] = df['weeks_since_open'].clip(lower=0)

    # Capped versions
    df['weeks_since_open_capped_13'] = df['weeks_since_open'].clip(upper=13)
    df['weeks_since_open_capped_52'] = df['weeks_since_open'].clip(upper=52)

    # Binary indicators
    df['is_new_store'] = (df['weeks_since_open'] < 52).astype(int)
    df['is_comp_store'] = (df['weeks_since_open'] >= 60).astype(int)

    # Drop temporary date_opened column (keep only as feature if explicitly requested)
    # Actually, keep it in case it's useful downstream
    # df = df.drop(columns=['date_opened'], errors='ignore')

    return df


def create_cohort_features(
    df: pd.DataFrame,
    store_df: pd.DataFrame,
    store_col: str = 'profit_center_nbr'
) -> pd.DataFrame:
    """
    Create cohort features from store opening dates (TIER 3).

    Cohort features capture systematic differences between stores opened
    in different time periods (e.g., format changes, market conditions).

    Features:
    - cohort_year: Year of opening
    - cohort_quarter: Quarter of opening (e.g., "2022-Q3")
    - cohort_half: Half-year of opening (e.g., "2022-H1")

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add cohort features
    store_df : pd.DataFrame
        Store master with date_opened
    store_col : str, default='profit_center_nbr'
        Store column name

    Returns
    -------
    pd.DataFrame
        Dataframe with cohort features

    Examples
    --------
    >>> df = create_cohort_features(canonical_df, store_master)
    >>> df['cohort_year'].value_counts()
    """
    if 'date_opened' not in store_df.columns:
        df['cohort_year'] = np.nan
        df['cohort_quarter'] = np.nan
        df['cohort_half'] = np.nan
        return df

    # Extract cohort info from store_df
    cohort_df = store_df[[store_col, 'date_opened']].drop_duplicates()

    cohort_df['cohort_year'] = cohort_df['date_opened'].dt.year
    cohort_df['cohort_quarter'] = (
        cohort_df['date_opened'].dt.year.astype(str) + '-Q' +
        cohort_df['date_opened'].dt.quarter.astype(str)
    )
    cohort_df['cohort_half'] = (
        cohort_df['date_opened'].dt.year.astype(str) + '-H' +
        ((cohort_df['date_opened'].dt.month - 1) // 6 + 1).astype(str)
    )

    # Merge to main dataframe
    df = df.merge(
        cohort_df[[store_col, 'cohort_year', 'cohort_quarter', 'cohort_half']],
        on=store_col,
        how='left'
    )

    return df


def validate_static_features(
    df: pd.DataFrame,
    expected_pooled_corr: Optional[dict] = None
) -> dict:
    """
    Validate static features against expected correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features and targets
    expected_pooled_corr : Optional[dict]
        Expected pooled correlations:
        {
            'proforma_annual_sales': 0.84,
            'is_outlet': -0.54,
            'weeks_since_open': 0.30  # for conversion target
        }

    Returns
    -------
    dict
        Validation results

    Examples
    --------
    >>> expected = {
    ...     'proforma_annual_sales': 0.84,
    ...     'is_outlet': -0.54
    ... }
    >>> results = validate_static_features(df, expected)
    >>> print(results['proforma_annual_sales'])
    """
    if expected_pooled_corr is None:
        expected_pooled_corr = {
            'proforma_annual_sales': 0.84,
            'is_outlet': -0.54,
            'weeks_since_open': 0.30  # for conversion
        }

    results = {}

    for feature, expected_corr in expected_pooled_corr.items():
        if feature not in df.columns:
            results[feature] = {
                'status': 'MISSING',
                'expected': expected_corr,
                'actual': None
            }
            continue

        # Find target columns
        target_cols = [col for col in df.columns if col.startswith('label_')]

        if len(target_cols) == 0:
            results[feature] = {
                'status': 'NO_TARGETS',
                'expected': expected_corr,
                'actual': None
            }
            continue

        # Compute correlations with each target
        correlations = {}
        for target_col in target_cols:
            valid_mask = df[feature].notna() & df[target_col].notna()
            if valid_mask.sum() > 10:
                corr = df.loc[valid_mask, feature].corr(df.loc[valid_mask, target_col])
                correlations[target_col] = corr

        results[feature] = {
            'status': 'COMPUTED',
            'expected': expected_corr,
            'actual_correlations': correlations,
            'max_abs_corr': max([abs(c) for c in correlations.values()]) if correlations else None
        }

    return results
