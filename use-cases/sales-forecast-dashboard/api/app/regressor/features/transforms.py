"""
Feature Transform Utilities

Reusable functions for computing lags, rolling means, winsorization, and cyclical encoding.
These utilities are the building blocks for all feature engineering tasks in EPIC 4.

Author: EPIC 4 Feature Engineering
Status: Phase 1 - Core Infrastructure
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union


def compute_lag(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    value_col: str,
    lag_weeks: int,
    suffix: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute lag features for time series data.

    CRITICAL LEAKAGE PREVENTION: This function computes lags using ONLY historical data.
    For a row at origin_week_date, lag_1 uses data from origin_week_date - 1 week.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    group_cols : List[str]
        Columns to group by (e.g., ['profit_center_nbr', 'channel'])
    date_col : str
        Name of the date column (must be datetime)
    value_col : str
        Name of the value column to lag
    lag_weeks : int
        Number of weeks to lag (must be positive)
    suffix : Optional[str]
        Optional suffix for the output column name
        If None, uses f"{value_col}_lag_{lag_weeks}"

    Returns
    -------
    pd.DataFrame
        Input dataframe with new lag column added

    Examples
    --------
    >>> df = compute_lag(df, ['store_id'], 'week_date', 'log_sales', lag_weeks=1)
    >>> # Creates column: log_sales_lag_1

    >>> df = compute_lag(df, ['store_id', 'channel'], 'week_date', 'sales', lag_weeks=52, suffix='yoy')
    >>> # Creates column: sales_lag_yoy
    """
    if lag_weeks <= 0:
        raise ValueError(f"lag_weeks must be positive, got {lag_weeks}")

    # Create output column name
    if suffix is not None:
        lag_col = f"{value_col}_lag_{suffix}"
    else:
        lag_col = f"{value_col}_lag_{lag_weeks}"

    # Sort by group and date to ensure correct ordering
    df = df.sort_values(group_cols + [date_col])

    # Compute lag using groupby shift
    df[lag_col] = df.groupby(group_cols)[value_col].shift(lag_weeks)

    return df


def compute_rolling_mean(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    value_col: str,
    window_weeks: int,
    suffix: Optional[str] = None,
    winsorize: bool = False,
    n_mad: float = 3.5
) -> pd.DataFrame:
    """
    Compute rolling mean features for time series data.

    CRITICAL LEAKAGE PREVENTION: For a row at origin_week_date with window=13,
    the rolling mean uses data from (origin_week_date - 12 weeks) through origin_week_date,
    i.e., the 13 weeks ending AT the origin week (inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    group_cols : List[str]
        Columns to group by (e.g., ['profit_center_nbr', 'channel'])
    date_col : str
        Name of the date column (must be datetime)
    value_col : str
        Name of the value column to compute rolling mean
    window_weeks : int
        Size of the rolling window in weeks (must be positive)
    suffix : Optional[str]
        Optional suffix for the output column name
        If None, uses f"{value_col}_roll_mean_{window_weeks}"
    winsorize : bool, default=False
        If True, applies MAD-based winsorization before computing rolling mean
    n_mad : float, default=3.5
        Number of MADs for winsorization threshold (only used if winsorize=True)

    Returns
    -------
    pd.DataFrame
        Input dataframe with new rolling mean column added

    Examples
    --------
    >>> df = compute_rolling_mean(df, ['store_id'], 'week_date', 'log_sales', window_weeks=13)
    >>> # Creates column: log_sales_roll_mean_13

    >>> df = compute_rolling_mean(
    ...     df, ['store_id', 'channel'], 'week_date', 'sales',
    ...     window_weeks=4, winsorize=True, n_mad=3.5
    ... )
    >>> # Creates winsorized column: sales_roll_mean_4
    """
    if window_weeks <= 0:
        raise ValueError(f"window_weeks must be positive, got {window_weeks}")

    # Create output column name
    if suffix is not None:
        roll_col = f"{value_col}_roll_mean_{suffix}"
    else:
        roll_col = f"{value_col}_roll_mean_{window_weeks}"

    # Sort by group and date to ensure correct ordering
    df = df.sort_values(group_cols + [date_col])

    # Apply winsorization if requested
    if winsorize:
        # Create temporary winsorized column
        temp_col = f"_temp_winsorized_{value_col}"
        df[temp_col] = df.groupby(group_cols)[value_col].transform(
            lambda x: winsorize_mad(x, n_mad=n_mad)
        )
        compute_col = temp_col
    else:
        compute_col = value_col

    # Compute rolling mean using groupby rolling
    # min_periods=1 ensures we get a value even for the first few weeks
    df[roll_col] = df.groupby(group_cols)[compute_col].transform(
        lambda x: x.rolling(window=window_weeks, min_periods=1).mean()
    )

    # Clean up temporary column if created
    if winsorize:
        df = df.drop(columns=[temp_col])

    return df


def winsorize_mad(
    series: pd.Series,
    n_mad: float = 3.5
) -> pd.Series:
    """
    Apply MAD-based winsorization to clip extreme values.

    Winsorization replaces extreme values with threshold values rather than removing them.
    This is more robust than standard deviation-based methods for skewed distributions.

    Formula:
        lower_bound = median - n_mad * MAD
        upper_bound = median + n_mad * MAD
        MAD = median(|x - median(x)|)

    Parameters
    ----------
    series : pd.Series
        Input series to winsorize
    n_mad : float, default=3.5
        Number of MADs (Median Absolute Deviations) for threshold
        3.5 MAD ≈ 3 standard deviations for normal distributions

    Returns
    -------
    pd.Series
        Winsorized series with extreme values clipped

    Examples
    --------
    >>> sales = pd.Series([100, 105, 110, 108, 1000])  # 1000 is an outlier
    >>> winsorized = winsorize_mad(sales, n_mad=3.5)
    >>> # 1000 will be clipped to approximately median + 3.5*MAD

    Notes
    -----
    - NaN values are preserved and not included in MAD calculation
    - If all values are NaN, returns the input series unchanged
    - More robust than standard deviation for heavy-tailed distributions
    """
    # Handle edge case: all NaN
    if series.isna().all():
        return series

    # Compute median and MAD
    median = series.median()
    mad = (series - median).abs().median()

    # Handle edge case: MAD = 0 (all values identical)
    if mad == 0:
        return series

    # Compute bounds
    lower_bound = median - n_mad * mad
    upper_bound = median + n_mad * mad

    # Clip values
    return series.clip(lower=lower_bound, upper=upper_bound)


def encode_cyclical(
    week_of_year: Union[pd.Series, np.ndarray, int],
    period: int = 52
) -> tuple:
    """
    Encode cyclical features using sin/cos transformation.

    This encoding preserves the cyclical nature of calendar features (e.g., week 52 is close to week 1).
    Linear encoding would treat week 1 and week 52 as maximally different.

    Formula:
        sin_component = sin(2π * week_of_year / period)
        cos_component = cos(2π * week_of_year / period)

    Parameters
    ----------
    week_of_year : Union[pd.Series, np.ndarray, int]
        Week of year values (1-52 or 1-53)
    period : int, default=52
        Period of the cycle (52 for standard years, 53 for leap years)

    Returns
    -------
    tuple of (sin_woy, cos_woy)
        Two arrays/series with sin and cos components

    Examples
    --------
    >>> sin_woy, cos_woy = encode_cyclical(df['week_of_year'], period=52)
    >>> df['sin_woy'] = sin_woy
    >>> df['cos_woy'] = cos_woy

    >>> # For a single value
    >>> sin_val, cos_val = encode_cyclical(26, period=52)  # Week 26 (mid-year)

    Notes
    -----
    - Week 1 and Week 52 will have similar (sin, cos) pairs, preserving cyclical proximity
    - The two components together uniquely identify each week of the year
    - NaN values in input produce NaN in output
    """
    # Convert to numpy for computation
    if isinstance(week_of_year, pd.Series):
        woy_array = week_of_year.values
        return_series = True
    elif isinstance(week_of_year, (int, float)):
        woy_array = np.array([week_of_year])
        return_series = False
    else:
        woy_array = np.array(week_of_year)
        return_series = False

    # Compute sin and cos components
    # Normalize to [0, 2π] range
    angle = 2 * np.pi * woy_array / period
    sin_component = np.sin(angle)
    cos_component = np.cos(angle)

    # Return in original format
    if isinstance(week_of_year, pd.Series):
        return pd.Series(sin_component, index=week_of_year.index), \
               pd.Series(cos_component, index=week_of_year.index)
    elif isinstance(week_of_year, (int, float)):
        return float(sin_component[0]), float(cos_component[0])
    else:
        return sin_component, cos_component


def safe_log(
    x: Union[pd.Series, np.ndarray, float],
    floor: float = 1e-6
) -> Union[pd.Series, np.ndarray, float]:
    """
    Compute log transformation with floor to prevent log(0) or log(negative).

    Formula:
        safe_log(x) = log(max(x, floor))

    Parameters
    ----------
    x : Union[pd.Series, np.ndarray, float]
        Input values to transform
    floor : float, default=1e-6
        Minimum value before taking log (prevents -inf)

    Returns
    -------
    Union[pd.Series, np.ndarray, float]
        Log-transformed values (same type as input)

    Examples
    --------
    >>> log_sales = safe_log(sales_df['sales'], floor=1e-6)
    >>> # Handles sales=0 by using log(1e-6) instead of log(0)=-inf

    Notes
    -----
    - Preserves NaN values
    - Input type determines output type (Series → Series, array → array, float → float)
    """
    # Apply floor
    if isinstance(x, pd.Series):
        x_floored = x.clip(lower=floor)
    else:
        x_floored = np.maximum(x, floor)

    # Compute log
    return np.log(x_floored)


def safe_logit(
    p: Union[pd.Series, np.ndarray, float],
    floor: float = 1e-6,
    ceil: float = 1 - 1e-6
) -> Union[pd.Series, np.ndarray, float]:
    """
    Compute logit transformation with floor/ceiling to prevent logit(0) or logit(1).

    Formula:
        safe_logit(p) = log(p_clipped / (1 - p_clipped))
        where p_clipped = clip(p, floor, ceil)

    Parameters
    ----------
    p : Union[pd.Series, np.ndarray, float]
        Input probability values (should be in [0, 1])
    floor : float, default=1e-6
        Minimum probability (prevents -inf)
    ceil : float, default=1-1e-6
        Maximum probability (prevents +inf)

    Returns
    -------
    Union[pd.Series, np.ndarray, float]
        Logit-transformed values (same type as input)

    Examples
    --------
    >>> logit_conversion = safe_logit(df['conversion_rate'], floor=1e-6, ceil=1-1e-6)
    >>> # Handles conversion=0 or conversion=1 safely

    Notes
    -----
    - Preserves NaN values
    - Input type determines output type
    - Inverse transformation: p = expit(logit_p) = 1 / (1 + exp(-logit_p))
    """
    # Apply floor and ceiling
    if isinstance(p, pd.Series):
        p_clipped = p.clip(lower=floor, upper=ceil)
    else:
        p_clipped = np.clip(p, floor, ceil)

    # Compute logit: log(p / (1-p))
    return np.log(p_clipped / (1 - p_clipped))


def compute_volatility(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    value_col: str,
    window_weeks: int,
    method: str = 'mad',
    suffix: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute rolling volatility (dispersion measure) for time series data.

    Volatility captures the variability/uncertainty in recent observations.
    MAD (Median Absolute Deviation) is more robust than standard deviation for outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    group_cols : List[str]
        Columns to group by (e.g., ['profit_center_nbr', 'channel'])
    date_col : str
        Name of the date column (must be datetime)
    value_col : str
        Name of the value column to compute volatility
    window_weeks : int
        Size of the rolling window in weeks
    method : str, default='mad'
        Volatility method: 'mad' (median absolute deviation) or 'std' (standard deviation)
    suffix : Optional[str]
        Optional suffix for the output column name
        If None, uses f"vol_{value_col}_{window_weeks}"

    Returns
    -------
    pd.DataFrame
        Input dataframe with new volatility column added

    Examples
    --------
    >>> df = compute_volatility(df, ['store_id'], 'week_date', 'log_sales', window_weeks=13)
    >>> # Creates column: vol_log_sales_13 (using MAD)

    >>> df = compute_volatility(df, ['store_id'], 'week_date', 'sales', window_weeks=13, method='std')
    >>> # Creates column: vol_sales_13 (using standard deviation)
    """
    if window_weeks <= 0:
        raise ValueError(f"window_weeks must be positive, got {window_weeks}")

    if method not in ['mad', 'std']:
        raise ValueError(f"method must be 'mad' or 'std', got {method}")

    # Create output column name
    if suffix is not None:
        vol_col = f"vol_{value_col}_{suffix}"
    else:
        vol_col = f"vol_{value_col}_{window_weeks}"

    # Sort by group and date to ensure correct ordering
    df = df.sort_values(group_cols + [date_col])

    # Compute rolling volatility
    if method == 'mad':
        # MAD = median(|x - median(x)|)
        def rolling_mad(x):
            return x.rolling(window=window_weeks, min_periods=1).apply(
                lambda vals: (vals - vals.median()).abs().median(),
                raw=False
            )
        df[vol_col] = df.groupby(group_cols)[value_col].transform(rolling_mad)
    else:  # method == 'std'
        df[vol_col] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window_weeks, min_periods=1).std()
        )

    return df
