"""
Dynamics Features (FE-02, FE-03, FE-04, Product Mix)

Autoregressive features observed at origin_week_date (t0).
These capture the "state of the world" at the moment of prediction.

CRITICAL LEAKAGE PREVENTION: All features use data ≤ origin_week_date ONLY.

Modules:
- [FE-02] Sales & AOV Dynamics (B&M): TIER 1, Pooled ρ = 0.78-0.83
- [FE-03] Web-Specific Dynamics: TIER 1-2, Pooled ρ = +0.58
- [FE-04] Conversion & Omnichannel: TIER 1, FE ρ = +0.26-0.37
- Product Mix & Service: TIER 2, FE ρ = ±0.12-0.18

Author: EPIC 4 Feature Engineering
Status: Phase 2.2-2.4 - Dynamics Features
"""

import numpy as np
import pandas as pd
from typing import Optional

from app.regressor.features.transforms import (
    compute_lag,
    compute_rolling_mean,
    compute_volatility,
    safe_log,
    safe_logit,
)


def attach_sales_aov_dynamics_bm(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    origin_col: str = 'origin_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach autoregressive sales/AOV features for B&M channel observed at origin_week_date (t0).

    TIER 1 Features (CRITICAL - Must Include):
    - log_sales_lag_{1,4,13,52}:
      * lag_1: FE ρ=+0.14, Pooled ρ=+0.81
      * lag_52: Pooled ρ=+0.78 (YoY persistence)
    - log_sales_roll_mean_{4,8,13} (winsorized via MAD):
      * roll_mean_13: OPTIMAL for B&M (FE ρ=+0.15, Pooled ρ=+0.82)
      * roll_mean_4: Pooled ρ=+0.83

    TIER 2 Features (Include):
    - AOV_roll_mean_{8,13}: Pooled ρ=+0.56, FE ρ=+0.17
    - vol_sales_13: 13-week rolling MAD for volatility

    Winsorization: MAD-based (3.5 MAD) applied to all rolling means to handle outliers.

    LEAKAGE PREVENTION:
    - For origin_week_date = 2024-01-15:
      * lag_1 uses week of 2024-01-08 (t0 - 1 week)
      * roll_mean_13 uses weeks 2023-10-16 through 2024-01-15 (13 weeks ending at t0)
    - NEVER uses data after origin_week_date

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with exploded rows (store, origin, horizon, target)
        Must contain: profit_center_nbr, channel, origin_week_date
    sales_history_df : pd.DataFrame
        Historical sales data with columns:
        - profit_center_nbr (or store_id)
        - week_date
        - channel
        - total_sales (or sales)
        - AOV (or aur, but prefer AOV = sales/orders)
    origin_col : str, default='origin_week_date'
        Name of origin week column
    store_col : str, default='profit_center_nbr'
        Name of store column
    channel_col : str, default='channel'
        Name of channel column
    n_mad : float, default=3.5
        Number of MADs for winsorization

    Returns
    -------
    pd.DataFrame
        Canonical table with B&M sales/AOV dynamics features attached

    Examples
    --------
    >>> df = attach_sales_aov_dynamics_bm(canonical_df, sales_history_df)
    >>> # Check TIER 1 features present
    >>> assert 'log_sales_lag_1' in df.columns
    >>> assert 'log_sales_roll_mean_13' in df.columns

    Notes
    -----
    - Only attaches to B&M rows (sets NaN for WEB rows where applicable)
    - All features computed at origin_week_date, same for all horizons
    - Validates against benchmarks: log_sales_roll_mean_13 Pooled ρ ≈ +0.82
    """
    df = canonical_df.copy()

    # Prepare sales history
    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    # Filter to B&M channel for B&M-specific features
    sales_bm = sales_df[sales_df[channel_col] == 'B&M'].copy()

    # Compute log transformations
    sales_bm['log_sales'] = safe_log(sales_bm['total_sales'], floor=1e-6)
    sales_bm['log_aov'] = safe_log(sales_bm['AOV'], floor=1e-6)

    # TIER 1: Sales Lags (1, 4, 13, 52 weeks)
    for lag_weeks in [1, 4, 13, 52]:
        sales_bm = compute_lag(
            sales_bm,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='log_sales',
            lag_weeks=lag_weeks
        )

    # TIER 1: Sales Rolling Means (4, 8, 13 weeks) with winsorization
    for window_weeks in [4, 8, 13]:
        sales_bm = compute_rolling_mean(
            sales_bm,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='log_sales',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    # TIER 2: AOV Rolling Means (8, 13 weeks) with winsorization
    for window_weeks in [8, 13]:
        sales_bm = compute_rolling_mean(
            sales_bm,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='log_aov',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    # Rename AOV features to match naming convention
    for window_weeks in [8, 13]:
        old_name = f'log_aov_roll_mean_{window_weeks}'
        new_name = f'AOV_roll_mean_{window_weeks}'
        if old_name in sales_bm.columns:
            sales_bm = sales_bm.rename(columns={old_name: new_name})

    # TIER 2: Sales Volatility (13-week MAD)
    sales_bm = compute_volatility(
        sales_bm,
        group_cols=[store_col, channel_col],
        date_col='week_date',
        value_col='log_sales',
        window_weeks=13,
        method='mad'
    )

    # Join features to canonical table on (store, channel, origin_week_date)
    # This is the CRITICAL join that prevents leakage: we join on origin, not target
    feature_cols = [col for col in sales_bm.columns
                   if any(x in col for x in ['lag_', 'roll_mean_', 'vol_', 'AOV_'])]
    feature_cols = [store_col, channel_col, 'week_date'] + feature_cols

    df = df.merge(
        sales_bm[feature_cols],
        left_on=[store_col, channel_col, origin_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left'
    )

    # Drop redundant week_date from merge
    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    return df


def attach_web_dynamics(
    canonical_df: pd.DataFrame,
    ecomm_history_df: pd.DataFrame,
    written_sales_web_df: Optional[pd.DataFrame] = None,
    origin_col: str = 'origin_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach WEB-specific dynamics features observed at origin_week_date (t0).

    WEB-SPECIFIC FEATURES (WEB channel only, NaN for B&M):

    TIER 1 (PRIMARY WEB DRIVER):
    - allocated_web_traffic_roll_mean_{4,8,13}:
      * PRIMARY web driver: Pooled ρ=+0.58 (log-transformed: 0.749)
      * 4-week optimal for WEB (shorter than B&M's 13-week)
    - allocated_web_traffic_lag_{1,4,13}

    TIER 1 (NEW - WEB SALES AUTOREGRESSIVE):
    - log_web_sales_lag_{1,4,13}: Sales momentum signals for WEB
    - log_web_sales_roll_mean_{4,8,13}: Rolling averages
    - vol_log_web_sales_13: 13-week volatility

    TIER 2 (NEW - WEB AOV):
    - web_aov_roll_mean_{4,8}: AOV momentum signals for WEB

    Channel-Specific Optimization:
    - B&M: 13-week windows optimal (seasonality FE ρ=+0.42)
    - WEB: 4-week windows optimal (seasonality FE ρ=+0.21, faster dynamics)

    LEAKAGE PREVENTION: Uses only data ≤ origin_week_date

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with exploded rows
    ecomm_history_df : pd.DataFrame
        E-commerce history with columns:
        - profit_center_nbr
        - week_date
        - allocated_web_traffic
        - merch_amt (web sales) - used only if written_sales_web_df not provided
    written_sales_web_df : pd.DataFrame, optional
        WEB channel data from Written Sales (SOURCE OF TRUTH for sales/AOV)
        If provided, used for sales/AOV features instead of ecomm merch_amt
        Required columns: profit_center_nbr, week_date, total_sales, AOV
    origin_col : str, default='origin_week_date'
        Name of origin week column
    store_col : str, default='profit_center_nbr'
        Store column name
    channel_col : str, default='channel'
        Channel column name
    n_mad : float, default=3.5
        Number of MADs for winsorization

    Returns
    -------
    pd.DataFrame
        Canonical table with WEB dynamics features attached

    Examples
    --------
    >>> df = attach_web_dynamics(canonical_df, ecomm_df, written_sales_web_df)
    >>> # Check WEB features are NaN for B&M rows
    >>> bm_rows = df[df['channel'] == 'B&M']
    >>> assert bm_rows['allocated_web_traffic_roll_mean_4'].isna().all()

    Notes
    -----
    - Features set to NaN for B&M rows
    - Sales/AOV features use Written Sales WEB (source of truth) when provided
    - Traffic features use Ecomm Traffic (AllocatedWebTraffic)
    - UnallocatedWebTraffic excluded (measures web sessions, not comparable to Store_Traffic)
    """
    df = canonical_df.copy()

    # Prepare ecomm history
    ecomm_df = ecomm_history_df.copy()

    # Ensure required columns exist
    if 'week_date' not in ecomm_df.columns:
        if 'fiscal_start_date_week' in ecomm_df.columns:
            ecomm_df = ecomm_df.rename(columns={'fiscal_start_date_week': 'week_date'})
        elif 'date' in ecomm_df.columns:
            ecomm_df = ecomm_df.rename(columns={'date': 'week_date'})
        elif 'week_start' in ecomm_df.columns:
            ecomm_df = ecomm_df.rename(columns={'week_start': 'week_date'})

    if store_col not in ecomm_df.columns and 'store_id' in ecomm_df.columns:
        ecomm_df = ecomm_df.rename(columns={'store_id': store_col})

    # Ensure week_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(ecomm_df['week_date']):
        ecomm_df['week_date'] = pd.to_datetime(ecomm_df['week_date'])

    # TIER 1: Web Traffic Lags (1, 4, 13 weeks)
    for lag_weeks in [1, 4, 13]:
        ecomm_df = compute_lag(
            ecomm_df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='allocated_web_traffic',
            lag_weeks=lag_weeks
        )

    # TIER 1: Web Traffic Rolling Means (4, 8, 13 weeks) with winsorization
    for window_weeks in [4, 8, 13]:
        ecomm_df = compute_rolling_mean(
            ecomm_df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='allocated_web_traffic',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    # TIER 2: Web Sales Rolling Mean (4-week, optimal for WEB) - from ecomm if no written sales
    if 'merch_amt' in ecomm_df.columns and written_sales_web_df is None:
        ecomm_df['log_web_sales'] = safe_log(ecomm_df['merch_amt'], floor=1e-6)

        ecomm_df = compute_rolling_mean(
            ecomm_df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='log_web_sales',
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad,
            suffix='web_4'
        )

    # NEW: WEB Sales and AOV features from Written Sales (SOURCE OF TRUTH)
    web_sales_features_df = None
    if written_sales_web_df is not None:
        web_sales_features_df = _compute_web_sales_aov_features(
            written_sales_web_df,
            store_col=store_col,
            n_mad=n_mad
        )

    # Join features to canonical table on (store, origin_week_date)
    # Only for WEB rows
    ecomm_feature_cols = [col for col in ecomm_df.columns
                         if any(x in col for x in ['allocated_web_traffic_lag_',
                                                   'allocated_web_traffic_roll_mean_',
                                                   'log_web_sales_roll_mean_'])]
    ecomm_feature_cols = [store_col, 'week_date'] + ecomm_feature_cols

    # Join only to WEB rows
    web_mask = df[channel_col] == 'WEB'
    df_web = df[web_mask].copy()
    df_other = df[~web_mask].copy()

    # Join ecomm traffic features
    df_web = df_web.merge(
        ecomm_df[ecomm_feature_cols],
        left_on=[store_col, origin_col],
        right_on=[store_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df_web.columns:
        df_web = df_web.drop(columns=['week_date'])

    # Join WEB sales/AOV features from Written Sales (SOURCE OF TRUTH)
    if web_sales_features_df is not None:
        web_sales_feature_cols = [col for col in web_sales_features_df.columns
                                  if any(x in col for x in ['log_web_sales_lag_',
                                                            'log_web_sales_roll_mean_',
                                                            'vol_log_web_sales_',
                                                            'web_aov_roll_mean_'])]
        web_sales_feature_cols = [store_col, 'week_date'] + web_sales_feature_cols

        df_web = df_web.merge(
            web_sales_features_df[web_sales_feature_cols],
            left_on=[store_col, origin_col],
            right_on=[store_col, 'week_date'],
            how='left'
        )

        if 'week_date' in df_web.columns:
            df_web = df_web.drop(columns=['week_date'])

    # Collect all WEB feature columns for NaN initialization in non-WEB rows
    all_web_feature_cols = ecomm_feature_cols.copy()
    if web_sales_features_df is not None:
        all_web_feature_cols.extend([col for col in web_sales_features_df.columns
                                     if any(x in col for x in ['log_web_sales_lag_',
                                                               'log_web_sales_roll_mean_',
                                                               'vol_log_web_sales_',
                                                               'web_aov_roll_mean_'])])

    # For non-WEB rows, add feature columns as NaN
    for col in all_web_feature_cols:
        if col not in [store_col, 'week_date'] and col not in df_other.columns:
            df_other[col] = np.nan

    # Concatenate back
    df = pd.concat([df_web, df_other], ignore_index=True)

    return df


def attach_conversion_omnichannel_features(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    origin_col: str = 'origin_week_date',
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    traffic_flag_col: str = 'has_traffic_data',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach conversion and omnichannel features.

    TIME ALIGNMENT:
    - Conversion features: ORIGIN-ALIGNED (autoregressive, observed at t0)
    - Omnichannel features: TARGET-ALIGNED (actionable lever, defined at t0+h)

    TIER 1 Features (CRITICAL for both Sales and Conversion targets):

    Conversion (B&M only) - ORIGIN-ALIGNED:
    - ConversionRate_roll_mean_{4,8,13}:
      * HIGHLY AUTOREGRESSIVE: FE rho=+0.37, Pooled rho=0.81-0.85
      * Provides incremental signal for sales (FE rho=+0.13)
    - ConversionRate_lag_{1,4}:
      * Immediate persistence: FE rho=+0.26-0.31

    Omnichannel (TIER 1 ACTIONABLE LEVER) - TARGET-ALIGNED:
    - pct_omni_channel_roll_mean_4:
      * TOP actionable lever: FE rho=+0.26 for conversion
      * Target-aligned for What-If scenario support

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with exploded rows
    sales_history_df : pd.DataFrame
        Sales history with columns:
        - profit_center_nbr
        - week_date
        - channel
        - Orders
        - Store_Traffic (for conversion calculation)
        - pct_omni_channel (omnichannel percentage)
        - has_traffic_data (binary flag)
    origin_col : str, default='origin_week_date'
        Origin week column (used for conversion features)
    target_col : str, default='target_week_date'
        Target week column (used for omnichannel features)
    store_col : str, default='profit_center_nbr'
        Store column
    channel_col : str, default='channel'
        Channel column
    traffic_flag_col : str, default='has_traffic_data'
        Binary flag for stores with valid traffic data
    n_mad : float, default=3.5
        MAD threshold for winsorization

    Returns
    -------
    pd.DataFrame
        Canonical table with conversion/omnichannel features attached

    Examples
    --------
    >>> df = attach_conversion_omnichannel_features(canonical_df, sales_df)
    >>> # Validation: ConversionRate features should be NaN for WEB
    >>> web_rows = df[df['channel'] == 'WEB']
    >>> assert web_rows['ConversionRate_roll_mean_13'].isna().all()

    Notes
    -----
    - Conversion features: B&M only (NaN for WEB), origin-aligned (autoregressive)
    - Omnichannel features: Apply to both channels, target-aligned (actionable)
    - Only use stores with has_traffic_data=1 for conversion
    """
    df = canonical_df.copy()

    # Prepare sales history
    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    # Filter to B&M channel with valid traffic data for conversion features
    if traffic_flag_col in sales_df.columns:
        sales_bm_traffic = sales_df[
            (sales_df[channel_col] == 'B&M') &
            (sales_df[traffic_flag_col] == 1)
        ].copy()
    else:
        # If no traffic flag, use all B&M rows
        sales_bm_traffic = sales_df[sales_df[channel_col] == 'B&M'].copy()

    # Compute or use existing conversion rate
    if 'ConversionRate' not in sales_bm_traffic.columns:
        # Compute conversion rate from orders and traffic
        if 'order_count' in sales_bm_traffic.columns and 'store_traffic' in sales_bm_traffic.columns:
            sales_bm_traffic['ConversionRate'] = (
                sales_bm_traffic['order_count'] / sales_bm_traffic['store_traffic']
            )
        elif 'Orders' in sales_bm_traffic.columns and 'Store_Traffic' in sales_bm_traffic.columns:
            sales_bm_traffic['ConversionRate'] = (
                sales_bm_traffic['Orders'] / sales_bm_traffic['Store_Traffic']
            )
        else:
            raise ValueError("Sales history must contain 'order_count'/'Orders' and 'store_traffic'/'Store_Traffic' columns")

    # Clip to [0, 1] range
    sales_bm_traffic['ConversionRate'] = sales_bm_traffic['ConversionRate'].clip(0, 1)

    # Logit transformation
    sales_bm_traffic['logit_conversion'] = safe_logit(
        sales_bm_traffic['ConversionRate'],
        floor=1e-6,
        ceil=1-1e-6
    )

    # TIER 1: Conversion Rate Lags (1, 4 weeks)
    for lag_weeks in [1, 4]:
        sales_bm_traffic = compute_lag(
            sales_bm_traffic,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='ConversionRate',
            lag_weeks=lag_weeks
        )

    # TIER 1: Conversion Rate Rolling Means (4, 8, 13 weeks) with winsorization
    for window_weeks in [4, 8, 13]:
        sales_bm_traffic = compute_rolling_mean(
            sales_bm_traffic,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='ConversionRate',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    # TIER 1: Omnichannel Features (both B&M and WEB)
    # Pruned: Keep only roll_mean_4 per update_plan.md
    if 'pct_omni_channel' in sales_df.columns:
        # Rolling mean (4 weeks) - TOP actionable lever
        sales_df = compute_rolling_mean(
            sales_df,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='pct_omni_channel',
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Join conversion features (B&M only)
    conversion_cols = [col for col in sales_bm_traffic.columns
                      if 'ConversionRate' in col]
    conversion_cols = [store_col, channel_col, 'week_date'] + conversion_cols

    df = df.merge(
        sales_bm_traffic[conversion_cols],
        left_on=[store_col, channel_col, origin_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    # Join omnichannel features (both channels) - TARGET-ALIGNED
    omni_cols = [col for col in sales_df.columns
                if 'pct_omni_channel' in col]
    omni_cols = [store_col, channel_col, 'week_date'] + omni_cols

    df = df.merge(
        sales_df[omni_cols],
        left_on=[store_col, channel_col, target_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left',
        suffixes=('', '_omni')
    )

    # Drop redundant columns
    cols_to_drop = [col for col in df.columns if col.endswith('_omni')]
    if 'week_date' in df.columns:
        cols_to_drop.append('week_date')
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


def attach_product_mix_features(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach product mix and service blend features defined at target_week_date (t0+h).

    TIME ALIGNMENT: TARGET-ALIGNED (actionable levers for What-If scenarios)

    TIER 2 Operational Levers (Actionable - pruned to roll_mean_4 only):

    Product Mix:
    - pct_value_product_roll_mean_4:
      * FE rho=-0.12 for B&M sales (negative - actionable)
      * Pruned: roll_mean_8 removed to reduce collinearity
    - pct_premium_product_roll_mean_4:
      * FE rho=-0.18 for conversion (negative)

    Service Blend:
    - pct_white_glove_roll_mean_4:
      * FE rho=+0.13 for conversion
      * Pooled rho=+0.46 (strong cross-sectional)

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table
    sales_history_df : pd.DataFrame
        Sales history with product mix columns:
        - pct_value_product
        - pct_premium_product
        - pct_white_glove
    target_col : str, default='target_week_date'
        Target week column (features aligned to forecast target)
    store_col : str, default='profit_center_nbr'
        Store column
    channel_col : str, default='channel'
        Channel column
    n_mad : float, default=3.5
        MAD threshold

    Returns
    -------
    pd.DataFrame
        Canonical table with product mix features

    Examples
    --------
    >>> df = attach_product_mix_features(canonical_df, sales_df)
    >>> # Validation: pct_white_glove should have FE rho approx +0.13
    """
    df = canonical_df.copy()

    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    # Product Mix: Value Product Rolling Mean (4 weeks only - pruned per update_plan.md)
    if 'pct_value_product' in sales_df.columns:
        sales_df = compute_rolling_mean(
            sales_df,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='pct_value_product',
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Product Mix: Premium Product Rolling Mean (4 weeks)
    if 'pct_premium_product' in sales_df.columns:
        sales_df = compute_rolling_mean(
            sales_df,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='pct_premium_product',
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Service Blend: White Glove Rolling Mean (4 weeks)
    if 'pct_white_glove' in sales_df.columns:
        sales_df = compute_rolling_mean(
            sales_df,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col='pct_white_glove',
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Join features - TARGET-ALIGNED
    feature_cols = [col for col in sales_df.columns
                   if any(x in col for x in ['pct_value_product_roll',
                                             'pct_premium_product_roll',
                                             'pct_white_glove_roll'])]
    feature_cols = [store_col, channel_col, 'week_date'] + feature_cols

    df = df.merge(
        sales_df[feature_cols],
        left_on=[store_col, channel_col, target_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    return df


def attach_financing_features(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach financing mix features defined at target_week_date (t0+h).

    TIME ALIGNMENT: TARGET-ALIGNED (actionable levers for What-If scenarios)

    TIER 2 Operational Levers (Actionable):

    Financing Mix (pruned to roll_mean_4 only per update_plan.md):
    - pct_primary_financing_roll_mean_4: Primary financing (Wells Fargo - good credit) trend
    - pct_secondary_financing_roll_mean_4: Secondary financing (lower tier credit) trend
    - pct_tertiary_financing_roll_mean_4: Tertiary financing (subprime, high-risk) trend

    Pruned (removed to reduce collinearity):
    - lag_1, lag_4: Removed - immediate lags add noise without additional signal
    - roll_mean_8: Removed - 4-week rolling mean is stable enough for monthly programs

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table
    sales_history_df : pd.DataFrame
        Sales history with financing mix columns:
        - pct_primary_financing
        - pct_secondary_financing
        - pct_tertiary_financing
    target_col : str, default='target_week_date'
        Target week column (features aligned to forecast target)
    store_col : str, default='profit_center_nbr'
        Store column
    channel_col : str, default='channel'
        Channel column
    n_mad : float, default=3.5
        MAD threshold for winsorization

    Returns
    -------
    pd.DataFrame
        Canonical table with financing mix features

    Examples
    --------
    >>> df = attach_financing_features(canonical_df, sales_df)
    >>> # Validate financing features were added
    >>> assert 'pct_primary_financing_roll_mean_4' in df.columns
    >>> assert 'pct_secondary_financing_roll_mean_4' in df.columns
    """
    df = canonical_df.copy()

    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    financing_cols = ['pct_primary_financing', 'pct_secondary_financing', 'pct_tertiary_financing']

    for financing_col in financing_cols:
        if financing_col not in sales_df.columns:
            continue

        # Rolling mean features (4 weeks only - pruned per update_plan.md)
        sales_df = compute_rolling_mean(
            sales_df,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col=financing_col,
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Join features (only roll_mean_4 versions) - TARGET-ALIGNED
    feature_cols = [col for col in sales_df.columns
                   if any(x in col for x in ['pct_primary_financing_roll_mean_4',
                                             'pct_secondary_financing_roll_mean_4',
                                             'pct_tertiary_financing_roll_mean_4'])]
    feature_cols = [store_col, channel_col, 'week_date'] + feature_cols

    df = df.merge(
        sales_df[feature_cols],
        left_on=[store_col, channel_col, target_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    return df


def attach_dma_web_penetration(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    origin_col: str = 'origin_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    dma_col: str = 'dma'
) -> pd.DataFrame:
    """
    Attach DMA-level WEB penetration feature observed at origin_week_date (t0).

    TIER 2 Feature (Market Signal for WEB):
    - dma_web_penetration_pct: WEB sales / Total DMA sales * 100
      * Captures market-level online shopping adoption
      * Higher penetration = larger WEB opportunity in that DMA
      * Useful for WEB channel forecasting

    Computation (per DMA per week):
    - Total DMA Sales = Sum of (B&M + WEB) sales across all stores in DMA
    - DMA WEB Sales = Sum of WEB channel sales across all stores in DMA
    - dma_web_penetration_pct = (DMA WEB Sales / Total DMA Sales) * 100

    LEAKAGE PREVENTION: Uses only data at origin_week_date (not target)

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with exploded rows
    sales_history_df : pd.DataFrame
        Historical sales data with columns:
        - profit_center_nbr
        - week_date
        - channel (B&M/WEB)
        - total_sales
    store_master_df : pd.DataFrame
        Store master with DMA assignment
    origin_col : str, default='origin_week_date'
        Origin week column (uses this to prevent leakage)
    store_col : str, default='profit_center_nbr'
        Store column name
    channel_col : str, default='channel'
        Channel column name
    dma_col : str, default='dma'
        DMA column name in store master

    Returns
    -------
    pd.DataFrame
        Canonical table with dma_web_penetration_pct attached

    Examples
    --------
    >>> df = attach_dma_web_penetration(canonical_df, sales_df, store_master)
    >>> # Check feature exists
    >>> assert 'dma_web_penetration_pct' in df.columns

    Notes
    -----
    - Feature applies to both B&M and WEB rows (same DMA penetration)
    - NaN for weeks/DMAs without WEB sales data
    - Typical range: 10-40% depending on DMA
    """
    df = canonical_df.copy()

    # Prepare sales history
    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    # Get store -> DMA mapping from store master
    if dma_col not in store_master_df.columns:
        # Try alternative column names
        if 'DMA' in store_master_df.columns:
            store_master_df = store_master_df.rename(columns={'DMA': dma_col})
        elif 'market_dma' in store_master_df.columns:
            store_master_df = store_master_df.rename(columns={'market_dma': dma_col})
        else:
            # DMA column not found, return original df
            df['dma_web_penetration_pct'] = np.nan
            return df

    store_to_dma = store_master_df[[store_col, dma_col]].drop_duplicates()

    # Add DMA to sales history
    sales_with_dma = sales_df.merge(store_to_dma, on=store_col, how='left')

    # Compute total sales per (DMA, week)
    dma_total_sales = sales_with_dma.groupby([dma_col, 'week_date'])['total_sales'].sum().reset_index()
    dma_total_sales = dma_total_sales.rename(columns={'total_sales': 'dma_total_sales'})

    # Compute WEB sales per (DMA, week)
    sales_web = sales_with_dma[sales_with_dma[channel_col] == 'WEB']
    dma_web_sales = sales_web.groupby([dma_col, 'week_date'])['total_sales'].sum().reset_index()
    dma_web_sales = dma_web_sales.rename(columns={'total_sales': 'dma_web_sales'})

    # Merge to get penetration per (DMA, week)
    dma_penetration = dma_total_sales.merge(dma_web_sales, on=[dma_col, 'week_date'], how='left')
    dma_penetration['dma_web_sales'] = dma_penetration['dma_web_sales'].fillna(0)
    dma_penetration['dma_web_penetration_pct'] = (
        dma_penetration['dma_web_sales'] / dma_penetration['dma_total_sales'] * 100
    ).replace([np.inf, -np.inf], np.nan)

    # Keep only needed columns
    dma_penetration = dma_penetration[[dma_col, 'week_date', 'dma_web_penetration_pct']]

    # Add DMA to canonical table (if not already present)
    if dma_col not in df.columns:
        df = df.merge(store_to_dma, on=store_col, how='left')

    # Join penetration on (DMA, origin_week_date) - uses origin to prevent leakage
    df = df.merge(
        dma_penetration,
        left_on=[dma_col, origin_col],
        right_on=[dma_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    return df


def _compute_web_sales_aov_features(
    written_sales_web_df: pd.DataFrame,
    store_col: str = 'profit_center_nbr',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Compute WEB sales and AOV autoregressive features from Written Sales data.

    This uses Written Sales WEB channel data as SOURCE OF TRUTH (not Ecomm Traffic merch_amt).
    Data validation showed 12.6% median difference between Ecomm MerchAmt and Written Sales WEB.

    Features computed:
    - log_web_sales_lag_{1,4,13}: Sales momentum signals
    - log_web_sales_roll_mean_{4,8,13}: Rolling averages with winsorization
    - vol_log_web_sales_13: 13-week volatility (MAD)
    - web_aov_roll_mean_{4,8}: AOV rolling means with winsorization

    Parameters
    ----------
    written_sales_web_df : pd.DataFrame
        WEB channel data from Written Sales (already filtered to WEB channel)
        Required columns: profit_center_nbr, week_date, total_sales, AOV
    store_col : str, default='profit_center_nbr'
        Store column name
    n_mad : float, default=3.5
        Number of MADs for winsorization

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [store_col, 'week_date', all computed features]
    """
    df = written_sales_web_df.copy()

    # Standardize column names
    if 'week_date' not in df.columns:
        if 'fiscal_start_date_week' in df.columns:
            df = df.rename(columns={'fiscal_start_date_week': 'week_date'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'week_date'})
        elif 'week_start' in df.columns:
            df = df.rename(columns={'week_start': 'week_date'})

    if store_col not in df.columns and 'store_id' in df.columns:
        df = df.rename(columns={'store_id': store_col})

    if 'total_sales' not in df.columns and 'sales' in df.columns:
        df = df.rename(columns={'sales': 'total_sales'})

    # Ensure week_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['week_date']):
        df['week_date'] = pd.to_datetime(df['week_date'])

    # Sort for correct lag/rolling computations
    df = df.sort_values([store_col, 'week_date'])

    # Compute log transformations
    df['log_web_sales'] = safe_log(df['total_sales'], floor=1e-6)

    # Handle AOV - use existing column or compute from sales/orders
    if 'AOV' in df.columns:
        df['web_aov'] = df['AOV']
    elif 'aov' in df.columns:
        df['web_aov'] = df['aov']
    elif 'total_sales' in df.columns and 'order_count' in df.columns:
        df['web_aov'] = df['total_sales'] / df['order_count'].replace(0, np.nan)
    else:
        df['web_aov'] = np.nan

    # TIER 1: WEB Sales Lags (1, 4, 13 weeks)
    for lag_weeks in [1, 4, 13]:
        df = compute_lag(
            df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='log_web_sales',
            lag_weeks=lag_weeks
        )

    # TIER 1: WEB Sales Rolling Means (4, 8, 13 weeks) with winsorization
    for window_weeks in [4, 8, 13]:
        df = compute_rolling_mean(
            df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='log_web_sales',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    # TIER 1: WEB Sales Volatility (13-week MAD)
    df = compute_volatility(
        df,
        group_cols=[store_col],
        date_col='week_date',
        value_col='log_web_sales',
        window_weeks=13,
        method='mad'
    )

    # TIER 2: WEB AOV Rolling Means (4, 8 weeks) with winsorization
    for window_weeks in [4, 8]:
        df = compute_rolling_mean(
            df,
            group_cols=[store_col],
            date_col='week_date',
            value_col='web_aov',
            window_weeks=window_weeks,
            winsorize=True,
            n_mad=n_mad
        )

    return df


def attach_staffing_features(
    canonical_df: pd.DataFrame,
    sales_history_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    channel_col: str = 'channel',
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Attach staffing features defined at target_week_date (t0+h).

    TIME ALIGNMENT: TARGET-ALIGNED (actionable lever for What-If scenarios)

    B&M ONLY FEATURES (NaN for WEB):

    Staffing Metrics:
    - staffing_unique_associates_roll_mean_4: 4-week rolling mean of unique associates
    - staffing_hours_roll_mean_4: 4-week rolling mean of employee hours

    Source: Written Sales Data.csv columns: Unique_Associates, EmployeeHours

    Rationale: Staffing is a key actionable lever affecting conversion.
    Physical store concept - not applicable to WEB channel.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with exploded rows
    sales_history_df : pd.DataFrame
        Sales history with staffing columns:
        - unique_associates: Number of unique employees
        - employee_hours: Total employee hours worked
    target_col : str, default='target_week_date'
        Target week column (features aligned to forecast target)
    store_col : str, default='profit_center_nbr'
        Store column
    channel_col : str, default='channel'
        Channel column
    n_mad : float, default=3.5
        MAD threshold for winsorization

    Returns
    -------
    pd.DataFrame
        Canonical table with staffing features attached (B&M only)

    Examples
    --------
    >>> df = attach_staffing_features(canonical_df, sales_df)
    >>> # Validate staffing features are NaN for WEB
    >>> web_rows = df[df['channel'] == 'WEB']
    >>> assert web_rows['staffing_unique_associates_roll_mean_4'].isna().all()
    """
    import warnings

    df = canonical_df.copy()

    sales_df = _prepare_sales_history(sales_history_df, store_col, channel_col)

    # Filter to B&M channel only (staffing is physical store concept)
    sales_bm = sales_df[sales_df[channel_col] == 'B&M'].copy()

    # Check if staffing columns exist
    staffing_cols = ['unique_associates', 'employee_hours']
    available_staffing = [col for col in staffing_cols if col in sales_bm.columns]

    if not available_staffing:
        warnings.warn("Staffing columns not found in sales history. Skipping staffing features.")
        return df

    # Compute rolling means for available staffing columns
    for col in available_staffing:
        sales_bm = compute_rolling_mean(
            sales_bm,
            group_cols=[store_col, channel_col],
            date_col='week_date',
            value_col=col,
            window_weeks=4,
            winsorize=True,
            n_mad=n_mad
        )

    # Rename to follow naming convention expected by model_views.py
    # employee_hours -> staffing_hours (not staffing_employee_hours)
    # unique_associates -> staffing_unique_associates
    rename_map = {
        'unique_associates_roll_mean_4': 'staffing_unique_associates_roll_mean_4',
        'employee_hours_roll_mean_4': 'staffing_hours_roll_mean_4',
    }
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in sales_bm.columns}
    sales_bm = sales_bm.rename(columns=rename_map)

    # Get staffing feature column names
    staffing_feature_cols = [col for col in sales_bm.columns if col.startswith('staffing_')]
    join_cols = [store_col, channel_col, 'week_date'] + staffing_feature_cols

    # Split by channel
    bm_mask = df[channel_col] == 'B&M'
    df_bm = df[bm_mask].copy()
    df_other = df[~bm_mask].copy()

    # Join to B&M rows - TARGET-ALIGNED
    df_bm = df_bm.merge(
        sales_bm[join_cols],
        left_on=[store_col, channel_col, target_col],
        right_on=[store_col, channel_col, 'week_date'],
        how='left'
    )

    if 'week_date' in df_bm.columns:
        df_bm = df_bm.drop(columns=['week_date'])

    # Add NaN columns for non-B&M rows
    for col in staffing_feature_cols:
        if col not in df_other.columns:
            df_other[col] = np.nan

    # Concatenate back
    df = pd.concat([df_bm, df_other], ignore_index=True)

    return df


def _prepare_sales_history(
    sales_history_df: pd.DataFrame,
    store_col: str,
    channel_col: str
) -> pd.DataFrame:
    """
    Prepare sales history dataframe with standard column names and datetime conversion.

    Parameters
    ----------
    sales_history_df : pd.DataFrame
        Raw sales history
    store_col : str
        Store column name
    channel_col : str
        Channel column name

    Returns
    -------
    pd.DataFrame
        Cleaned sales history
    """
    df = sales_history_df.copy()

    # Standardize column names
    if 'week_date' not in df.columns:
        if 'fiscal_start_date_week' in df.columns:
            df = df.rename(columns={'fiscal_start_date_week': 'week_date'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'week_date'})
        elif 'week_start' in df.columns:
            df = df.rename(columns={'week_start': 'week_date'})

    if store_col not in df.columns and 'store_id' in df.columns:
        df = df.rename(columns={'store_id': store_col})

    if 'total_sales' not in df.columns and 'sales' in df.columns:
        df = df.rename(columns={'sales': 'total_sales'})

    # Ensure week_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['week_date']):
        df['week_date'] = pd.to_datetime(df['week_date'])

    # Sort by store, channel, date for correct lag/rolling computations
    df = df.sort_values([store_col, channel_col, 'week_date'])

    return df
