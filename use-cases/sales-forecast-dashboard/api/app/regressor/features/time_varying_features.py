"""
Time-Varying Known-in-Advance Features (FE-01)

Features that depend on target_week_date (t0+h) but are deterministic/knowable at forecast time.
Includes seasonality, calendar, and holiday features.

TIER 1 Features:
- dma_seasonal_weight: FE ρ=+0.42 (B&M), +0.21 (WEB)
- Cyclical week-of-year encoding
- Holiday indicators and proximity features

Author: EPIC 4 Feature Engineering
Status: Phase 2.1 - Time-Varying Features
"""

import numpy as np
import pandas as pd
from typing import Optional

from app.regressor.features.transforms import encode_cyclical


def attach_time_varying_features(
    exploded_df: pd.DataFrame,
    dma_seasonality_df: pd.DataFrame,
    holiday_calendar_df: pd.DataFrame,
    web_seasonality_df: Optional[pd.DataFrame] = None,
    date_col: str = 'target_week_date',
    dma_col: str = 'dma',
    horizon_col: str = 'horizon'
) -> pd.DataFrame:
    """
    Attach time-varying features known in advance at target_week_date (t0 + h).

    These features vary by horizon because they depend on target_week_date, but they are
    all deterministic and knowable at forecast time (no autoregressive components).

    TIER 1 Features:
    - dma_seasonal_weight_{h}: DMA-specific seasonality using N-1 logic (B&M)
    - web_seasonal_weight_{h}: Global WEB seasonality using N-1 logic (WEB)

    Calendar Features:
    - woy_{h}: Week of year (1-52/53)
    - sin_woy_{h}, cos_woy_{h}: Cyclical encoding of week-of-year
    - fiscal_year_{h}, fiscal_period_{h}: Fiscal calendar

    TIER 1-3 Holiday Features:
    - is_holiday_{h}: Binary indicator for major holidays
    - is_black_friday_window_{h}: ±1 week around Black Friday
    - is_xmas_window_{h}: ±2 weeks around Christmas
    - weeks_to_holiday: Continuous distance to next major holiday (TIER 3, FE ρ=+0.10)
    - is_pre_holiday_1wk_{h}: 1 week before major holiday (TIER 3, FE ρ=-0.09)
    - is_pre_holiday_2wk_{h}: 2 weeks before major holiday (TIER 3, FE ρ=-0.07)
    - is_pre_holiday_3wk_{h}: 3 weeks before major holiday (TIER 3, FE ρ=-0.04)

    Horizon Feature:
    - horizon: Forecast horizon (1-52), included as feature per engineering instructions

    Parameters
    ----------
    exploded_df : pd.DataFrame
        Canonical table with (store, origin, horizon, target_week_date) rows
    dma_seasonality_df : pd.DataFrame
        DMA seasonality weights (B&M) with columns: dma, fiscal_year, fiscal_week, weight
    holiday_calendar_df : pd.DataFrame
        Holiday calendar
    web_seasonality_df : Optional[pd.DataFrame]
        Global WEB seasonality weights with columns: fiscal_year, fiscal_week, weight
    date_col : str
        Name of the target date column
    dma_col : str
        Name of the DMA column
    horizon_col : str
        Name of the horizon column

    Returns
    -------
    pd.DataFrame
        Input dataframe with time-varying features attached
    """
    df = exploded_df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Standardize DMA column name
    if dma_col not in df.columns:
        if 'dma_id' in df.columns:
            dma_col = 'dma_id'
        elif 'dma' in df.columns:
            dma_col = 'dma'
        else:
            raise ValueError(f"DMA column '{dma_col}' not found in dataframe")

    # 1. Extract calendar features from target_week_date
    df = _attach_calendar_features(df, date_col)

    # 2. Join Seasonality (B&M and WEB)
    df = _attach_seasonality(df, dma_seasonality_df, web_seasonality_df, date_col, dma_col)

    # 3. Attach holiday features
    df = _attach_holiday_features(df, holiday_calendar_df, date_col)

    # 4. Ensure horizon is present as a feature
    if horizon_col not in df.columns:
        raise ValueError(f"Horizon column '{horizon_col}' not found in dataframe")

    return df


def _attach_calendar_features(
    df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Attach calendar-derived features from target_week_date."""
    # Week of year (1-52 or 1-53)
    df['woy'] = df[date_col].dt.isocalendar().week

    # Cyclical encoding
    df['sin_woy'], df['cos_woy'] = encode_cyclical(df['woy'], period=52)

    # Month and quarter
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter

    # Fiscal calendar (assuming fiscal year aligns with calendar year)
    df['fiscal_year'] = df[date_col].dt.year

    # Fiscal period (13 periods per year = ~4 weeks each)
    df['fiscal_period'] = ((df['woy'] - 1) // 4) + 1
    df['fiscal_period'] = df['fiscal_period'].clip(upper=13)  # Cap at 13

    return df


def _attach_seasonality(
    df: pd.DataFrame,
    dma_seasonality_df: pd.DataFrame,
    web_seasonality_df: Optional[pd.DataFrame],
    date_col: str,
    dma_col: str
) -> pd.DataFrame:
    """
    Join seasonality weights using N-1 logic on (dma, fiscal_year, woy).
    Apply B&M weights to B&M channel, WEB weights to WEB channel.
    """
    # Ensure woy and fiscal_year exist
    if 'woy' not in df.columns:
        df['woy'] = df[date_col].dt.isocalendar().week
    if 'fiscal_year' not in df.columns:
        df['fiscal_year'] = df[date_col].dt.year

    # --- 1. B&M Seasonality ---
    bm_seas = dma_seasonality_df.copy()
    
    # Normalize columns
    if dma_col not in bm_seas.columns:
        if 'market_city' in bm_seas.columns:
            bm_seas = bm_seas.rename(columns={'market_city': dma_col})
    
    if 'week_of_year' not in bm_seas.columns and 'fiscal_week' in bm_seas.columns:
        bm_seas = bm_seas.rename(columns={'fiscal_week': 'week_of_year'})
        
    if 'seasonal_weight' not in bm_seas.columns and 'weight' in bm_seas.columns:
        bm_seas = bm_seas.rename(columns={'weight': 'seasonal_weight'})

    # Merge B&M seasonality
    # Key: dma, fiscal_year, week_of_year
    df = df.merge(
        bm_seas[[dma_col, 'fiscal_year', 'week_of_year', 'seasonal_weight']],
        left_on=[dma_col, 'fiscal_year', 'woy'],
        right_on=[dma_col, 'fiscal_year', 'week_of_year'],
        how='left'
    )
    df = df.rename(columns={'seasonal_weight': 'dma_seasonal_weight'})
    if 'week_of_year' in df.columns:
        df = df.drop(columns=['week_of_year'])

    # --- 2. WEB Seasonality ---
    if web_seasonality_df is not None:
        web_seas = web_seasonality_df.copy()
        
        # Normalize columns
        if dma_col not in web_seas.columns:
            if 'market_city' in web_seas.columns:
                web_seas = web_seas.rename(columns={'market_city': dma_col})
        
        if 'week_of_year' not in web_seas.columns and 'fiscal_week' in web_seas.columns:
            web_seas = web_seas.rename(columns={'fiscal_week': 'week_of_year'})
            
        if 'weight' in web_seas.columns:
            web_seas = web_seas.rename(columns={'weight': 'web_seasonal_weight'})
            
        # Merge WEB seasonality (DMA-specific)
        # Key: dma, fiscal_year, week_of_year
        df = df.merge(
            web_seas[[dma_col, 'fiscal_year', 'week_of_year', 'web_seasonal_weight']],
            left_on=[dma_col, 'fiscal_year', 'woy'],
            right_on=[dma_col, 'fiscal_year', 'week_of_year'],
            how='left'
        )
        if 'week_of_year' in df.columns:
            df = df.drop(columns=['week_of_year'])
    else:
        df['web_seasonal_weight'] = np.nan

    # --- 3. Apply Logic ---
    # If channel is B&M, dma_seasonal_weight is valid.
    # If channel is WEB, use web_seasonal_weight as primary, or fallback to dma_seasonal_weight if needed.
    # However, typically we want a single 'seasonal_weight' feature or keep them separate.
    # The model B uses 'dma_seasonal_weight'. 
    # Decision: 
    # - For B&M rows: keep dma_seasonal_weight
    # - For WEB rows: fill dma_seasonal_weight with web_seasonal_weight (so the model sees the correct curve in the same column)
    # - Also keep web_seasonal_weight as a separate feature?
    
    if 'channel' in df.columns:
        is_web = df['channel'] == 'WEB'
        
        # For WEB rows, overwrite dma_seasonal_weight with web_seasonal_weight if available
        # This allows the same feature name to carry the correct signal for both channels
        if 'web_seasonal_weight' in df.columns:
             df.loc[is_web, 'dma_seasonal_weight'] = df.loc[is_web, 'web_seasonal_weight']

    return df


def _attach_holiday_features(
    df: pd.DataFrame,
    holiday_calendar_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """
    Attach holiday features based on target_week_date.

    TIER 1-3 Features:
    - is_holiday: Binary indicator (major holidays)
    - is_black_friday_window: ±1 week around Black Friday
    - is_xmas_window: ±2 weeks around Christmas
    - weeks_to_holiday: Continuous distance to next major holiday (TIER 3, FE ρ=+0.10)
    - is_pre_holiday_{1,2,3}wk: Pre-holiday indicators (TIER 3)
    """
    # Ensure holiday calendar date column is datetime
    holiday_df = holiday_calendar_df.copy()

    # Standardize date column name to 'week_date'
    if 'week_date' not in holiday_df.columns:
        if 'date' in holiday_df.columns:
            holiday_df = holiday_df.rename(columns={'date': 'week_date'})
        elif 'week_start' in holiday_df.columns:
            holiday_df = holiday_df.rename(columns={'week_start': 'week_date'})
        elif 'fiscal_start_date_week' in holiday_df.columns:
            holiday_df = holiday_df.rename(columns={'fiscal_start_date_week': 'week_date'})
        else:
            raise ValueError("Holiday calendar must have 'week_date', 'date', 'week_start', or 'fiscal_start_date_week' column")

    if not pd.api.types.is_datetime64_any_dtype(holiday_df['week_date']):
        holiday_df['week_date'] = pd.to_datetime(holiday_df['week_date'])

    # Join is_holiday indicator
    if 'is_holiday' in holiday_df.columns:
        df = df.merge(
            holiday_df[['week_date', 'is_holiday']],
            left_on=date_col,
            right_on='week_date',
            how='left'
        )
        df = df.drop(columns=['week_date'])
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    else:
        # If is_holiday not provided, create from holiday_name
        if 'holiday_name' in holiday_df.columns:
            holiday_df['is_holiday'] = (holiday_df['holiday_name'].notna() &
                                       (holiday_df['holiday_name'] != '')).astype(int)
            df = df.merge(
                holiday_df[['week_date', 'is_holiday']],
                left_on=date_col,
                right_on='week_date',
                how='left'
            )
            df = df.drop(columns=['week_date'])
            df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        else:
            # No holiday data available
            df['is_holiday'] = 0

    # Create Black Friday and Christmas window indicators
    df = _create_holiday_windows(df, holiday_df, date_col)

    # Create weeks_to_holiday (continuous distance to next major holiday)
    df = _create_weeks_to_holiday(df, holiday_df, date_col)

    # Create pre-holiday indicators
    df = _create_pre_holiday_indicators(df, holiday_df, date_col)

    return df


def _create_holiday_windows(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Create holiday window indicators (Black Friday ±1 week, Christmas ±2 weeks)."""
    # Identify anchor column to use for window calculations (actual holiday date if available)
    anchor_col = 'holiday_anchor_date' if 'holiday_anchor_date' in holiday_df.columns else 'week_date'

    # Identify Black Friday and Christmas dates from holiday names
    if 'holiday_name' not in holiday_df.columns:
        # Can't create specific windows without holiday names
        df['is_black_friday_window'] = 0
        df['is_xmas_window'] = 0
        return df

    # Extract Black Friday dates
    black_friday_dates = pd.to_datetime(holiday_df[
        holiday_df['holiday_name'].str.contains('Black Friday', case=False, na=False)
    ][anchor_col].dropna().unique())

    # Extract Christmas dates (use anchor to keep windows tighter to the day)
    christmas_dates = pd.to_datetime(holiday_df[
        holiday_df['holiday_name'].str.contains('Christmas', case=False, na=False)
    ][anchor_col].dropna().unique())

    # Create Black Friday window (±1 week = within 7 days)
    df['is_black_friday_window'] = 0
    for bf_date in black_friday_dates:
        days_diff = (df[date_col] - bf_date).dt.days.abs()
        df.loc[days_diff <= 7, 'is_black_friday_window'] = 1

    # Create Christmas window (±2 weeks = within 14 days)
    df['is_xmas_window'] = 0
    for xmas_date in christmas_dates:
        days_diff = (df[date_col] - xmas_date).dt.days.abs()
        df.loc[days_diff <= 14, 'is_xmas_window'] = 1

    return df


def _create_weeks_to_holiday(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Create weeks_to_holiday: continuous distance to next major holiday."""
    # Use anchor dates if available for more precise distance calculation
    anchor_col = 'holiday_anchor_date' if 'holiday_anchor_date' in holiday_df.columns else 'week_date'

    # Get dates of major holidays
    major_holidays = holiday_df[holiday_df['is_holiday'] == 1][anchor_col].dropna().unique()
    major_holidays = pd.Series(pd.to_datetime(major_holidays)).sort_values()

    if len(major_holidays) == 0:
        df['weeks_to_holiday'] = np.nan
        return df

    # For each row, find next major holiday
    def compute_weeks_to_next_holiday(target_date):
        future_holidays = major_holidays[major_holidays > target_date]
        if len(future_holidays) == 0:
            # Wrap around to next year's first holiday (approximate)
            return np.nan
        
        next_holiday = future_holidays.iloc[0]
        weeks_diff = (next_holiday - target_date).days / 7.0
        return weeks_diff

    df['weeks_to_holiday'] = df[date_col].apply(compute_weeks_to_next_holiday)

    return df


def _create_pre_holiday_indicators(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Create pre-holiday indicators (1, 2, 3 weeks before major holidays)."""
    # Use anchor dates if available for tighter windows
    anchor_col = 'holiday_anchor_date' if 'holiday_anchor_date' in holiday_df.columns else 'week_date'

    # Get dates of major holidays
    major_holidays = holiday_df[holiday_df['is_holiday'] == 1][anchor_col].dropna().unique()
    major_holidays = pd.to_datetime(major_holidays)

    # Initialize indicators
    df['is_pre_holiday_1wk'] = 0
    df['is_pre_holiday_2wk'] = 0
    df['is_pre_holiday_3wk'] = 0

    # For each major holiday, mark weeks before it
    for holiday_date in major_holidays:
        # 1 week before (5-11 days before to account for week boundaries)
        days_before = (holiday_date - df[date_col]).dt.days
        df.loc[(days_before >= 5) & (days_before <= 11), 'is_pre_holiday_1wk'] = 1

        # 2 weeks before (12-18 days before)
        df.loc[(days_before >= 12) & (days_before <= 18), 'is_pre_holiday_2wk'] = 1

        # 3 weeks before (19-25 days before)
        df.loc[(days_before >= 19) & (days_before <= 25), 'is_pre_holiday_3wk'] = 1

    return df
