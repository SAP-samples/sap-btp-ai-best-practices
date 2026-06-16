"""
Feature Validation Module

Functions to validate feature engineering results against driver screening benchmarks,
detect leakage, verify coverage, and ensure channel-specific logic is correct.

Author: EPIC 4 Feature Engineering
Status: Phase 1 - Core Infrastructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings


def validate_feature_correlations(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    benchmarks_dict: Dict[str, Dict[str, float]],
    threshold: float = 0.02,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Validate that computed feature correlations match driver screening benchmarks.

    Compares actual feature correlations against empirical benchmarks from driver screening.
    Flags features where |actual - expected| > threshold, indicating potential implementation issues.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature columns
    targets_df : pd.DataFrame
        Dataframe with target columns (label_log_sales, label_log_aov, label_logit_conversion)
    benchmarks_dict : Dict[str, Dict[str, float]]
        Expected correlations in format:
        {
            'feature_name': {
                'target_name': expected_correlation,
                'correlation_type': 'FE' or 'Pooled'
            },
            ...
        }
    threshold : float, default=0.02
        Maximum acceptable difference between actual and expected correlation
    method : str, default='pearson'
        Correlation method: 'pearson' or 'spearman'

    Returns
    -------
    pd.DataFrame
        Validation results with columns:
        - feature: Feature name
        - target: Target name
        - expected_corr: Expected correlation from benchmarks
        - actual_corr: Computed correlation
        - difference: abs(actual - expected)
        - status: 'PASS' or 'FAIL'

    Examples
    --------
    >>> benchmarks = {
    ...     'dma_seasonal_weight': {
    ...         'label_log_sales_bm': 0.42,
    ...         'correlation_type': 'FE'
    ...     },
    ...     'log_sales_roll_mean_13': {
    ...         'label_log_sales_bm': 0.82,
    ...         'correlation_type': 'Pooled'
    ...     }
    ... }
    >>> results = validate_feature_correlations(features_df, targets_df, benchmarks)
    >>> failures = results[results['status'] == 'FAIL']
    """
    validation_results = []

    for feature_name, benchmark_info in benchmarks_dict.items():
        # Skip if feature doesn't exist
        if feature_name not in features_df.columns:
            warnings.warn(f"Feature '{feature_name}' not found in features_df, skipping validation")
            continue

        # Extract benchmark correlations
        for key, expected_corr in benchmark_info.items():
            if key == 'correlation_type':
                continue  # Skip metadata field

            target_name = key

            # Skip if target doesn't exist
            if target_name not in targets_df.columns:
                warnings.warn(f"Target '{target_name}' not found in targets_df, skipping validation")
                continue

            # Compute actual correlation
            # Remove NaN pairs for fair comparison
            valid_mask = features_df[feature_name].notna() & targets_df[target_name].notna()
            if valid_mask.sum() < 10:
                warnings.warn(
                    f"Too few valid observations ({valid_mask.sum()}) for {feature_name} vs {target_name}"
                )
                continue

            if method == 'pearson':
                actual_corr = features_df.loc[valid_mask, feature_name].corr(
                    targets_df.loc[valid_mask, target_name]
                )
            elif method == 'spearman':
                actual_corr = features_df.loc[valid_mask, feature_name].corr(
                    targets_df.loc[valid_mask, target_name],
                    method='spearman'
                )
            else:
                raise ValueError(f"Invalid method '{method}', use 'pearson' or 'spearman'")

            # Compute difference and status
            difference = abs(actual_corr - expected_corr)
            status = 'PASS' if difference <= threshold else 'FAIL'

            validation_results.append({
                'feature': feature_name,
                'target': target_name,
                'expected_corr': expected_corr,
                'actual_corr': actual_corr,
                'difference': difference,
                'status': status,
                'n_obs': valid_mask.sum()
            })

    return pd.DataFrame(validation_results)


def validate_feature_coverage(
    features_df: pd.DataFrame,
    expected_features_list: List[str],
    max_nan_rate: float = 0.05,
    channel_col: Optional[str] = None,
    channel_specific_features: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Validate that all expected features are present with acceptable NaN rates.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature columns
    expected_features_list : List[str]
        List of feature names that should be present
    max_nan_rate : float, default=0.05
        Maximum acceptable NaN rate (5%)
    channel_col : Optional[str]
        Name of channel column (e.g., 'channel')
        If provided, checks NaN rates separately by channel
    channel_specific_features : Optional[Dict[str, List[str]]]
        Dictionary mapping channel → list of features expected only for that channel
        Example: {'WEB': ['allocated_web_traffic_roll_mean_4'], 'B&M': ['ConversionRate_roll_mean_13']}

    Returns
    -------
    pd.DataFrame
        Coverage results with columns:
        - feature: Feature name
        - present: Boolean, whether feature exists
        - nan_rate: NaN rate (0 to 1)
        - channel: Channel name (if channel_col provided, otherwise 'All')
        - status: 'PASS', 'HIGH_NAN', or 'MISSING'

    Examples
    --------
    >>> expected_features = ['log_sales_lag_1', 'log_sales_roll_mean_13', 'dma_seasonal_weight']
    >>> results = validate_feature_coverage(features_df, expected_features, max_nan_rate=0.05)
    >>> issues = results[results['status'] != 'PASS']

    >>> # With channel-specific validation
    >>> channel_features = {
    ...     'WEB': ['allocated_web_traffic_roll_mean_4'],
    ...     'B&M': ['ConversionRate_roll_mean_13']
    ... }
    >>> results = validate_feature_coverage(
    ...     features_df,
    ...     expected_features,
    ...     channel_col='channel',
    ...     channel_specific_features=channel_features
    ... )
    """
    coverage_results = []

    # If channel column provided, validate by channel
    if channel_col is not None and channel_col in features_df.columns:
        channels = features_df[channel_col].unique()
    else:
        channels = ['All']

    for channel in channels:
        # Filter to channel if applicable
        if channel == 'All':
            df_channel = features_df
        else:
            df_channel = features_df[features_df[channel_col] == channel]

        # Determine which features to check for this channel
        if channel_specific_features is not None and channel != 'All':
            # Check general features + channel-specific features
            general_features = [f for f in expected_features_list
                               if not any(f in chan_feats
                                         for chan_feats in channel_specific_features.values())]
            channel_features_list = channel_specific_features.get(channel, [])
            features_to_check = general_features + channel_features_list
        else:
            features_to_check = expected_features_list

        for feature_name in features_to_check:
            # Check presence
            present = feature_name in df_channel.columns

            if not present:
                coverage_results.append({
                    'feature': feature_name,
                    'present': False,
                    'nan_rate': np.nan,
                    'channel': channel,
                    'status': 'MISSING'
                })
                continue

            # Compute NaN rate
            nan_rate = df_channel[feature_name].isna().mean()

            # Determine status
            # For channel-specific features, allow 100% NaN in other channels
            if (channel_specific_features is not None and
                channel != 'All' and
                feature_name not in channel_specific_features.get(channel, [])):
                # Feature is specific to another channel, expect 100% NaN
                expected_nan_rate = 1.0
                status = 'PASS' if nan_rate >= 0.95 else 'FAIL_SHOULD_BE_NAN'
            else:
                # Feature should have low NaN rate
                if nan_rate <= max_nan_rate:
                    status = 'PASS'
                else:
                    status = 'HIGH_NAN'

            coverage_results.append({
                'feature': feature_name,
                'present': True,
                'nan_rate': nan_rate,
                'channel': channel,
                'status': status,
                'n_rows': len(df_channel),
                'n_nans': df_channel[feature_name].isna().sum()
            })

    return pd.DataFrame(coverage_results)


def validate_leakage_prevention(
    features_df: pd.DataFrame,
    origin_col: str,
    target_col: Optional[str] = None,
    lag_roll_features: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Ensure no features leak future information across origin_week_date boundaries.

    CRITICAL VALIDATION: For features observed at t0 (lags, rolling means), all data
    used to compute the feature must be ≤ origin_week_date. This function validates
    that lag/rolling features are identical across all horizons for the same origin week.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature columns and origin_week_date
    origin_col : str
        Name of origin week column (e.g., 'origin_week_date')
    target_col : Optional[str]
        Name of target week column (e.g., 'target_week_date')
        If provided, checks that target_col > origin_col
    lag_roll_features : Optional[List[str]]
        List of lag/rolling feature names to check
        If None, auto-detects features with 'lag_' or 'roll_' in name
    verbose : bool, default=True
        If True, prints detailed warnings for each violation

    Returns
    -------
    Dict[str, any]
        Validation results:
        - 'status': 'PASS' or 'FAIL'
        - 'n_violations': Number of features with leakage
        - 'violating_features': List of feature names with leakage
        - 'details': DataFrame with per-feature details

    Examples
    --------
    >>> results = validate_leakage_prevention(
    ...     canonical_df,
    ...     origin_col='origin_week_date',
    ...     target_col='target_week_date'
    ... )
    >>> if results['status'] == 'FAIL':
    ...     print(f"LEAKAGE DETECTED in {results['n_violations']} features!")
    ...     print(results['violating_features'])
    """
    # Auto-detect lag/roll features if not provided
    if lag_roll_features is None:
        lag_roll_features = [col for col in features_df.columns
                            if 'lag_' in col.lower() or 'roll_' in col.lower() or 'roll' in col.lower()]

    if len(lag_roll_features) == 0:
        warnings.warn("No lag/rolling features detected, skipping leakage validation")
        return {
            'status': 'SKIP',
            'n_violations': 0,
            'violating_features': [],
            'details': pd.DataFrame()
        }

    # Check 1: Validate origin_col < target_col if target_col provided
    if target_col is not None and target_col in features_df.columns:
        invalid_dates = features_df[features_df[origin_col] >= features_df[target_col]]
        if len(invalid_dates) > 0:
            warnings.warn(
                f"Found {len(invalid_dates)} rows where {origin_col} >= {target_col}! "
                "This indicates a data generation issue."
            )

    # Check 2: Lag/rolling features should be identical across horizons for same origin
    # Group by store/channel/origin and check if feature values vary by horizon
    group_cols = [col for col in features_df.columns
                  if col in ['profit_center_nbr', 'store_id', 'channel', origin_col]]

    if len(group_cols) == 0:
        warnings.warn(f"Could not find grouping columns, using only {origin_col}")
        group_cols = [origin_col]

    violation_details = []

    for feature in lag_roll_features:
        if feature not in features_df.columns:
            continue

        # Check if feature varies by horizon within same origin group
        # If feature is properly computed at t0, it should be constant for all horizons
        unique_counts = features_df.groupby(group_cols)[feature].nunique()

        # Allow for slight floating point errors
        max_unique = unique_counts.max()

        if max_unique > 1:
            # Feature varies by horizon - LEAKAGE DETECTED
            pct_varying = (unique_counts > 1).mean()

            if verbose:
                print(f"⚠️  LEAKAGE WARNING: '{feature}' varies by horizon for {pct_varying:.1%} of groups")

            violation_details.append({
                'feature': feature,
                'max_unique_values': max_unique,
                'pct_groups_varying': pct_varying,
                'status': 'FAIL'
            })
        else:
            violation_details.append({
                'feature': feature,
                'max_unique_values': max_unique,
                'pct_groups_varying': 0.0,
                'status': 'PASS'
            })

    details_df = pd.DataFrame(violation_details)
    violating_features = details_df[details_df['status'] == 'FAIL']['feature'].tolist()

    overall_status = 'PASS' if len(violating_features) == 0 else 'FAIL'

    return {
        'status': overall_status,
        'n_violations': len(violating_features),
        'violating_features': violating_features,
        'details': details_df
    }


def validate_channel_specific_features(
    features_df: pd.DataFrame,
    channel_col: str,
    channel_feature_map: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Ensure channel-specific features are properly NaN for other channels.

    For example, WEB-only features (allocated_web_traffic) should be 100% NaN for B&M rows,
    and B&M-only features (ConversionRate for stores with traffic data) should be NaN for WEB rows.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature columns and channel column
    channel_col : str
        Name of channel column (e.g., 'channel')
    channel_feature_map : Dict[str, List[str]]
        Mapping of channel → features that should ONLY exist for that channel
        Example:
        {
            'WEB': ['allocated_web_traffic_roll_mean_4', 'allocated_web_traffic_lag_1'],
            'B&M': ['ConversionRate_roll_mean_13', 'logit_conversion']
        }

    Returns
    -------
    pd.DataFrame
        Validation results with columns:
        - feature: Feature name
        - expected_channel: Channel where feature should exist
        - other_channel: Channel where feature should be NaN
        - nan_rate_in_other_channel: NaN rate in other channel (should be ~1.0)
        - status: 'PASS' or 'FAIL'

    Examples
    --------
    >>> channel_map = {
    ...     'WEB': ['allocated_web_traffic_roll_mean_4'],
    ...     'B&M': ['ConversionRate_roll_mean_13']
    ... }
    >>> results = validate_channel_specific_features(df, 'channel', channel_map)
    >>> failures = results[results['status'] == 'FAIL']
    """
    validation_results = []

    channels = features_df[channel_col].unique()

    for expected_channel, feature_list in channel_feature_map.items():
        # Get other channels
        other_channels = [ch for ch in channels if ch != expected_channel]

        for feature_name in feature_list:
            if feature_name not in features_df.columns:
                warnings.warn(f"Feature '{feature_name}' not found in features_df")
                continue

            for other_channel in other_channels:
                # Check NaN rate in other channel
                other_channel_df = features_df[features_df[channel_col] == other_channel]

                if len(other_channel_df) == 0:
                    continue

                nan_rate = other_channel_df[feature_name].isna().mean()

                # Expect ~100% NaN in other channel (allow for 95%+ to be safe)
                status = 'PASS' if nan_rate >= 0.95 else 'FAIL'

                validation_results.append({
                    'feature': feature_name,
                    'expected_channel': expected_channel,
                    'other_channel': other_channel,
                    'nan_rate_in_other_channel': nan_rate,
                    'n_rows_other_channel': len(other_channel_df),
                    'n_nans_other_channel': other_channel_df[feature_name].isna().sum(),
                    'status': status
                })

    return pd.DataFrame(validation_results)


def validate_feature_ranges(
    features_df: pd.DataFrame,
    feature_range_map: Dict[str, Tuple[float, float]],
    allow_nan: bool = True
) -> pd.DataFrame:
    """
    Validate that features fall within expected ranges.

    Useful for catching data quality issues like:
    - Negative values for inherently positive features (e.g., sales, AOV)
    - Out-of-bound percentages (should be [0, 1])
    - Unrealistic values (e.g., conversion rate > 1)

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature columns
    feature_range_map : Dict[str, Tuple[float, float]]
        Mapping of feature → (min_value, max_value)
        Example:
        {
            'pct_omni_channel_roll_mean_4': (0.0, 1.0),
            'log_sales_lag_1': (-10.0, 15.0)
        }
    allow_nan : bool, default=True
        If True, NaN values are allowed and don't count as violations

    Returns
    -------
    pd.DataFrame
        Validation results with columns:
        - feature: Feature name
        - expected_min: Expected minimum value
        - expected_max: Expected maximum value
        - actual_min: Actual minimum value (excluding NaN if allow_nan=True)
        - actual_max: Actual maximum value (excluding NaN if allow_nan=True)
        - n_below_min: Count of values < expected_min
        - n_above_max: Count of values > expected_max
        - status: 'PASS' or 'FAIL'

    Examples
    --------
    >>> range_map = {
    ...     'pct_omni_channel_roll_mean_4': (0.0, 1.0),
    ...     'ConversionRate_roll_mean_13': (0.0, 1.0)
    ... }
    >>> results = validate_feature_ranges(df, range_map)
    >>> violations = results[results['status'] == 'FAIL']
    """
    validation_results = []

    for feature_name, (expected_min, expected_max) in feature_range_map.items():
        if feature_name not in features_df.columns:
            warnings.warn(f"Feature '{feature_name}' not found in features_df")
            continue

        feature_series = features_df[feature_name]

        if allow_nan:
            feature_series = feature_series.dropna()

        if len(feature_series) == 0:
            warnings.warn(f"Feature '{feature_name}' is all NaN, skipping range validation")
            continue

        actual_min = feature_series.min()
        actual_max = feature_series.max()

        n_below_min = (feature_series < expected_min).sum()
        n_above_max = (feature_series > expected_max).sum()

        status = 'PASS' if (n_below_min == 0 and n_above_max == 0) else 'FAIL'

        validation_results.append({
            'feature': feature_name,
            'expected_min': expected_min,
            'expected_max': expected_max,
            'actual_min': actual_min,
            'actual_max': actual_max,
            'n_below_min': n_below_min,
            'n_above_max': n_above_max,
            'n_valid_obs': len(feature_series),
            'status': status
        })

    return pd.DataFrame(validation_results)
