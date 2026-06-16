"""
CRM Demographic Mix Features (FE-08)

Customer composition features from CRM data.
Treat as static/cross-sectional predictors with strong pooled correlations
but weak fixed-effects (FE) predictive power.

TIER 3 Cross-Sectional Predictors (Pooled ρ = 0.15-0.21, FE ρ < 0.05):

Top CRM Features:
- crm_dwelling_single_family_dwelling_unit_pct: ρ=+0.21 for WEB AOV
- crm_owner_renter_owner_pct: ρ=+0.21 for WEB AOV
- crm_income_150k_plus_pct: ρ=+0.20 for WEB sales
- crm_education_college_pct: ρ=+0.18 for WEB sales
- crm_age_55_64_pct: ρ=+0.14 for B&M conversion

Treatment:
- Static/slow-moving features (FE ρ<0.05)
- Provide cross-sectional lift for baseline adjustments
- NOT actionable operational levers

Author: EPIC 4 Feature Engineering
Status: Phase 2.8 - CRM Features
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import warnings

from app.regressor.features.transforms import compute_lag, compute_rolling_mean


def attach_crm_features(
    canonical_df: pd.DataFrame,
    crm_df: pd.DataFrame,
    store_col: str = 'profit_center_nbr',
    origin_col: str = 'origin_week_date',
    include_lags: bool = False,
    include_rolls: bool = False
) -> pd.DataFrame:
    """
    Attach CRM demographic mix features.

    TIER 3 Cross-Sectional Predictors (Pooled ρ = 0.15-0.21, FE ρ < 0.05):

    Top 23 CRM percent features:
    - Dwelling type: single_family, multi_family, mobile_home
    - Ownership: owner vs renter
    - Income bands: <25k, 25-50k, 50-75k, 75-100k, 100-150k, 150k+
    - Education: high_school, some_college, college_graduate, post_graduate
    - Age bands: 18-24, 25-34, 35-44, 45-54, 55-64, 65+

    Include:
    - Latest snapshot (static cross-sectional) - ALWAYS
    - Lags {1,4} (optional, FE signal weak)
    - Rolling means {4,8} (optional, FE signal weak)

    Guidance from driver screening:
    - Strong pooled correlations (ρ = 0.15-0.21) despite weak FE
    - Cross-sectional selection effects (good stores have favorable demographics)
    - NOT short-term levers (FE ρ < 0.05)

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table
    crm_df : pd.DataFrame
        CRM demographic data with columns:
        - profit_center_nbr (or store_id)
        - week_date
        - crm_* features (23 percent features)
    store_col : str, default='profit_center_nbr'
        Store column
    origin_col : str, default='origin_week_date'
        Origin week column (for join)
    include_lags : bool, default=False
        Include lag features (weak FE signal, optional)
    include_rolls : bool, default=False
        Include rolling mean features (weak FE signal, optional)

    Returns
    -------
    pd.DataFrame
        Canonical table with CRM features

    Examples
    --------
    >>> from app.regressor.data_ingestion.crm_mix import load_crm_mix
    >>> crm_data = load_crm_mix()
    >>> df = attach_crm_features(canonical_df, crm_data)
    >>> # Validation: Check pooled correlations
    >>> from features.validation import validate_feature_correlations
    >>> benchmarks = {
    ...     'crm_dwelling_single_family_pct': {'label_log_aov_web': 0.21},
    ...     'crm_owner_renter_owner_pct': {'label_log_aov_web': 0.21}
    ... }
    >>> results = validate_feature_correlations(df, df, benchmarks)

    Notes
    -----
    - Treats CRM percentages as Tier 3 cross-sectional predictors
    - Include in feature schema but segregate from time-varying operational levers
    - Validates pooled correlations remain within ±0.02 of screening values
    - Documents top segments: homeowners, single-family, high income
    """
    df = canonical_df.copy()

    # Prepare CRM data
    crm_clean = _prepare_crm_data(crm_df, store_col)

    # Identify CRM percent features
    crm_feature_cols = [col for col in crm_clean.columns if col.startswith('crm_')]

    if len(crm_feature_cols) == 0:
        warnings.warn("No CRM features (crm_*) found in crm_df")
        return df

    # Always include latest snapshot (static)
    df = _attach_crm_snapshot(df, crm_clean, store_col, origin_col, crm_feature_cols, channel_col='channel')

    # Optional: Include lags
    if include_lags:
        crm_clean = _compute_crm_lags(crm_clean, store_col, crm_feature_cols)

    # Optional: Include rolling means
    if include_rolls:
        crm_clean = _compute_crm_rolls(crm_clean, store_col, crm_feature_cols)

    # Join lag/roll features if computed
    if include_lags or include_rolls:
        lag_roll_cols = [col for col in crm_clean.columns
                        if any(x in col for x in ['_lag_', '_roll_mean_'])]

        if len(lag_roll_cols) > 0:
            join_cols = [store_col, 'week_date'] + lag_roll_cols

            df = df.merge(
                crm_clean[join_cols],
                left_on=[store_col, origin_col],
                right_on=[store_col, 'week_date'],
                how='left'
            )

            if 'week_date' in df.columns:
                df = df.drop(columns=['week_date'])

    return df


def _prepare_crm_data(
    crm_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Prepare CRM data with standardized column names.

    Parameters
    ----------
    crm_df : pd.DataFrame
        Raw CRM data
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Cleaned CRM data
    """
    df = crm_df.copy()

    # Standardize store column
    if store_col not in df.columns and 'store_id' in df.columns:
        df = df.rename(columns={'store_id': store_col})

    # Standardize date column - handle all common names
    if 'week_date' not in df.columns:
        if 'week_start' in df.columns:
            df = df.rename(columns={'week_start': 'week_date'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'week_date'})

    # Ensure week_date is datetime
    if 'week_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['week_date']):
            df['week_date'] = pd.to_datetime(df['week_date'])

    # Sort by store and date
    if 'week_date' in df.columns:
        df = df.sort_values([store_col, 'week_date'])

    return df


def _attach_crm_snapshot(
    df: pd.DataFrame,
    crm_df: pd.DataFrame,
    store_col: str,
    origin_col: str,
    crm_feature_cols: List[str],
    channel_col: str = 'channel'
) -> pd.DataFrame:
    """
    Attach latest CRM snapshot (static cross-sectional features).

    For each (store, channel, origin_week_date), use the most recent CRM values
    available at or before origin_week_date.

    Joins on both store AND channel to prevent Cartesian product explosion.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table
    crm_df : pd.DataFrame
        CRM data
    store_col : str
        Store column
    origin_col : str
        Origin week column
    crm_feature_cols : List[str]
        List of CRM feature columns to include
    channel_col : str, default='channel'
        Channel column in canonical table

    Returns
    -------
    pd.DataFrame
        Dataframe with CRM snapshot features
    """
    # Determine CRM channel column name
    crm_channel_col = 'channel_norm' if 'channel_norm' in crm_df.columns else 'channel'
    has_crm_channel = crm_channel_col in crm_df.columns

    if 'week_date' in crm_df.columns:
        # Get most recent CRM values per store+channel
        group_cols = [store_col]
        if has_crm_channel:
            group_cols.append(crm_channel_col)

        crm_latest = (crm_df
            .sort_values([store_col, 'week_date'])
            .groupby(group_cols)
            .tail(1)
        )
    else:
        # No date column, assume data is already latest snapshot
        crm_latest = crm_df

    # Build base join columns
    join_cols = [store_col] + crm_feature_cols
    join_cols = [col for col in join_cols if col in crm_latest.columns]

    # Include channel column if present in CRM data
    if has_crm_channel:
        # Add channel to join columns if not already present
        join_cols_with_channel = [store_col, crm_channel_col] + [
            c for c in join_cols if c not in [store_col, crm_channel_col]
        ]
        join_cols_with_channel = [col for col in join_cols_with_channel if col in crm_latest.columns]

        # Normalize canonical table channel to match CRM format (uppercase)
        df['_crm_channel_join'] = df[channel_col].str.upper().str.strip()

        # Join on store + channel
        df = df.merge(
            crm_latest[join_cols_with_channel],
            left_on=[store_col, '_crm_channel_join'],
            right_on=[store_col, crm_channel_col],
            how='left'
        )

        # Clean up temporary columns
        df = df.drop(columns=['_crm_channel_join'])
        if crm_channel_col in df.columns and crm_channel_col != channel_col:
            df = df.drop(columns=[crm_channel_col])
    else:
        # Fallback: no channel in CRM data, join on store only
        df = df.merge(
            crm_latest[join_cols],
            on=store_col,
            how='left'
        )

    return df


def _compute_crm_lags(
    crm_df: pd.DataFrame,
    store_col: str,
    crm_feature_cols: List[str]
) -> pd.DataFrame:
    """
    Compute lag features for CRM demographics (optional, weak FE signal).

    Parameters
    ----------
    crm_df : pd.DataFrame
        CRM data
    store_col : str
        Store column
    crm_feature_cols : List[str]
        CRM feature columns

    Returns
    -------
    pd.DataFrame
        CRM data with lag features added
    """
    df = crm_df.copy()

    # Lags: 1, 4 weeks
    for feature_col in crm_feature_cols:
        if feature_col not in df.columns:
            continue

        for lag_weeks in [1, 4]:
            df = compute_lag(
                df,
                group_cols=[store_col],
                date_col='week_date',
                value_col=feature_col,
                lag_weeks=lag_weeks
            )

    return df


def _compute_crm_rolls(
    crm_df: pd.DataFrame,
    store_col: str,
    crm_feature_cols: List[str]
) -> pd.DataFrame:
    """
    Compute rolling mean features for CRM demographics (optional, weak FE signal).

    Parameters
    ----------
    crm_df : pd.DataFrame
        CRM data
    store_col : str
        Store column
    crm_feature_cols : List[str]
        CRM feature columns

    Returns
    -------
    pd.DataFrame
        CRM data with rolling mean features added
    """
    df = crm_df.copy()

    # Rolling means: 4, 8 weeks
    for feature_col in crm_feature_cols:
        if feature_col not in df.columns:
            continue

        for window_weeks in [4, 8]:
            df = compute_rolling_mean(
                df,
                group_cols=[store_col],
                date_col='week_date',
                value_col=feature_col,
                window_weeks=window_weeks,
                winsorize=False  # Don't winsorize percentages
            )

    return df


def get_top_crm_features() -> dict:
    """
    Get list of top CRM features by target based on driver screening results.

    Returns
    -------
    dict
        Mapping of target → list of (feature, pooled_correlation) tuples

    Examples
    --------
    >>> top_features = get_top_crm_features()
    >>> web_aov_features = top_features['WEB_AOV']
    >>> print(web_aov_features[:3])
    [('crm_dwelling_single_family_pct', 0.21),
     ('crm_owner_renter_owner_pct', 0.21),
     ...]
    """
    return {
        'WEB_AOV': [
            ('crm_dwelling_single_family_dwelling_unit_pct', 0.21),
            ('crm_owner_renter_owner_pct', 0.21),
        ],
        'WEB_Sales': [
            ('crm_income_150k_plus_pct', 0.20),
            ('crm_education_college_pct', 0.18),
        ],
        'BM_Conversion': [
            ('crm_age_55_64_pct', 0.14),
        ],
        'BM_Sales': [
            # Most CRM features have ρ = 0.15-0.18 for B&M sales
            ('crm_owner_renter_owner_pct', 0.18),
            ('crm_income_150k_plus_pct', 0.17),
        ]
    }


def validate_crm_correlations(
    df: pd.DataFrame,
    tolerance: float = 0.02
) -> pd.DataFrame:
    """
    Validate CRM feature correlations against driver screening benchmarks.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table with CRM features and targets
    tolerance : float, default=0.02
        Acceptable difference from benchmarks

    Returns
    -------
    pd.DataFrame
        Validation results

    Examples
    --------
    >>> results = validate_crm_correlations(canonical_df)
    >>> failures = results[results['status'] == 'FAIL']
    >>> if len(failures) > 0:
    ...     print("CRM correlations outside tolerance:")
    ...     print(failures)
    """
    top_features = get_top_crm_features()

    validation_results = []

    # Target column mapping
    target_map = {
        'WEB_AOV': 'label_log_aov',
        'WEB_Sales': 'label_log_sales',
        'BM_Conversion': 'label_logit_conversion',
        'BM_Sales': 'label_log_sales'
    }

    channel_map = {
        'WEB_AOV': 'WEB',
        'WEB_Sales': 'WEB',
        'BM_Conversion': 'B&M',
        'BM_Sales': 'B&M'
    }

    for target_key, feature_list in top_features.items():
        target_col = target_map[target_key]
        channel = channel_map[target_key]

        # Filter to relevant channel
        if 'channel' in df.columns:
            df_channel = df[df['channel'] == channel]
        else:
            df_channel = df

        for feature_name, expected_corr in feature_list:
            if feature_name not in df_channel.columns:
                validation_results.append({
                    'feature': feature_name,
                    'target': target_key,
                    'expected_corr': expected_corr,
                    'actual_corr': np.nan,
                    'difference': np.nan,
                    'status': 'MISSING'
                })
                continue

            if target_col not in df_channel.columns:
                validation_results.append({
                    'feature': feature_name,
                    'target': target_key,
                    'expected_corr': expected_corr,
                    'actual_corr': np.nan,
                    'difference': np.nan,
                    'status': 'TARGET_MISSING'
                })
                continue

            # Compute actual correlation
            valid_mask = (
                df_channel[feature_name].notna() &
                df_channel[target_col].notna()
            )

            if valid_mask.sum() < 10:
                validation_results.append({
                    'feature': feature_name,
                    'target': target_key,
                    'expected_corr': expected_corr,
                    'actual_corr': np.nan,
                    'difference': np.nan,
                    'status': 'INSUFFICIENT_DATA'
                })
                continue

            actual_corr = df_channel.loc[valid_mask, feature_name].corr(
                df_channel.loc[valid_mask, target_col]
            )

            difference = abs(actual_corr - expected_corr)
            status = 'PASS' if difference <= tolerance else 'FAIL'

            validation_results.append({
                'feature': feature_name,
                'target': target_key,
                'expected_corr': expected_corr,
                'actual_corr': actual_corr,
                'difference': difference,
                'status': status,
                'n_obs': valid_mask.sum()
            })

    return pd.DataFrame(validation_results)
