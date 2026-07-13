"""
DMA Awareness & Consideration Features (FE-07)

Market-level brand awareness and consideration metrics from YouGov.
Data is at market-group level (39 aggregated markets), not individual stores.
Uses 4-week rolling mean to smooth weekly volatility in survey data.

TIER 1 for WEB (top-15 driver), TIER 2 for B&M:
- brand_awareness_dma_roll_mean_4: Pooled ρ=+0.31 (WEB), +0.18 (B&M), +0.21 (Conversion)

TIER 3 (optional):
- brand_consideration_dma_roll_mean_4: Pooled ρ<=+0.17

Coverage: 96.7% of operational stores via YOUGOV_DMA_MAP cascading.
11 unmapped stores use sister-DMA fallback.

Author: EPIC 4 Feature Engineering
Status: Phase 2.7 - Awareness Features
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings


def attach_awareness_features(
    canonical_df: pd.DataFrame,
    awareness_df: pd.DataFrame,
    yougov_map_df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    target_col: str = 'target_week_date',
    store_col: str = 'profit_center_nbr',
    include_consideration: bool = True
) -> pd.DataFrame:
    """
    Attach DMA-level brand awareness and consideration features (4-week rolling mean).

    TIME ALIGNMENT: TARGET-ALIGNED (market signals at forecast target for What-If scenarios)

    Uses 4-week rolling mean to smooth weekly volatility in survey data.

    TIER 1 for WEB (top-15 driver), TIER 2 for B&M:
    - brand_awareness_dma_roll_mean_4:
      * WEB Sales: Pooled rho=+0.31 (CRITICAL - top 15 driver)
      * B&M Sales: Pooled rho=+0.18 (moderate cross-sectional)
      * B&M Conversion: Pooled rho=+0.21

    TIER 3 (optional):
    - brand_consideration_dma_roll_mean_4:
      * Pooled rho<=+0.17 (weaker than awareness)

    Cascading Mechanism:
    1. Map store's market_city -> Market via YOUGOV_DMA_MAP
    2. Join awareness at (Market, target_col) granularity (target_week_date)
    3. For 11 unmapped stores (CHAMPAIGN/SPRINGFIELD, SPARTANBURG/ASHEVILLE, etc.):
       - Apply sister-DMA fallback to "Single DMAs" groupings
    4. Forward-fill weekly gaps

    Data Structure: 7.4k rows = 39 aggregate markets x ~190 weeks
    Coverage: 96.7% of operational stores

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with (store, origin, horizon, target_week_date)
    awareness_df : pd.DataFrame
        YouGov awareness data with columns:
        - Market: Market grouping name (39 unique markets)
        - week_date: Week ending date
        - Awareness: Brand awareness score (0-100 or 0-1)
        - Consideration: Brand consideration score (optional)
    yougov_map_df : pd.DataFrame
        YOUGOV_DMA_MAP table (from Master Tables workbook, sheet 25):
        - market_city: Store's market city (from store master)
        - Market: Aggregate market grouping for awareness lookup
    store_master_df : pd.DataFrame
        Store master with market_city column
    target_col : str, default='target_week_date'
        Date column to use for awareness lookup (target-aligned for What-If scenarios)
    store_col : str, default='profit_center_nbr'
        Store column name
    include_consideration : bool, default=True
        Whether to include brand_consideration (TIER 3, optional)

    Returns
    -------
    pd.DataFrame
        Canonical table with awareness/consideration features

    Examples
    --------
    >>> from app.regressor.data_ingestion.awareness import load_awareness_data
    >>> from app.regressor.io_utils import load_store_master
    >>>
    >>> awareness_df = load_awareness_data()
    >>> yougov_map = load_yougov_dma_map()
    >>> store_master = load_store_master()
    >>>
    >>> df = attach_awareness_features(
    ...     canonical_df,
    ...     awareness_df,
    ...     yougov_map,
    ...     store_master
    ... )
    >>> # Validate coverage
    >>> coverage = df['brand_awareness_dma_roll_mean_4'].notna().mean()
    >>> assert coverage >= 0.967, f"Coverage {coverage:.1%} < 96.7%"

    Notes
    -----
    - Awareness is measured at market-group level (not individual stores)
    - Data is weekly (not monthly), forward-fill gaps
    - FE correlations approx 0 (treat as static/cross-sectional feature)
    - For unmapped stores, use sister-DMA fallback
    """
    df = canonical_df.copy()

    # Ensure target_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[target_col]):
        df[target_col] = pd.to_datetime(df[target_col])

    # Step 1: Map stores to markets via YOUGOV_DMA_MAP
    df = _map_stores_to_markets(
        df,
        store_master_df,
        yougov_map_df,
        store_col
    )

    # Step 2: Prepare awareness data (forward-fill gaps)
    awareness_clean = _prepare_awareness_data(awareness_df)

    # Step 3: Join awareness on (Market, target_week_date)
    df = _join_awareness(
        df,
        awareness_clean,
        target_col,
        include_consideration
    )

    # Step 4: Apply sister-DMA fallback for unmapped stores
    df = _apply_sister_dma_fallback(
        df,
        awareness_clean,
        store_master_df,
        target_col,
        store_col,
        include_consideration
    )

    return df


def _map_stores_to_markets(
    df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    yougov_map_df: pd.DataFrame,
    store_col: str
) -> pd.DataFrame:
    """
    Map stores to awareness markets via market_city → YOUGOV_DMA_MAP.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table
    store_master_df : pd.DataFrame
        Store master with market_city column
    yougov_map_df : pd.DataFrame
        YOUGOV_DMA_MAP with (market_city, Market) mapping
    store_col : str
        Store column name

    Returns
    -------
    pd.DataFrame
        Dataframe with 'Market' column added
    """
    # Extract store → market_city mapping
    if 'market_city' not in store_master_df.columns:
        # Try alternatives
        if 'Market_City' in store_master_df.columns:
            store_master_df = store_master_df.rename(columns={'Market_City': 'market_city'})
        elif 'city' in store_master_df.columns:
            store_master_df = store_master_df.rename(columns={'city': 'market_city'})
        else:
            warnings.warn("Store master missing 'market_city' column, awareness mapping will fail")
            df['Market'] = np.nan
            return df

    store_to_market_city = store_master_df[[store_col, 'market_city']].drop_duplicates()

    # FIX: Normalize market_city to uppercase for case-insensitive join
    # Store Master uses UPPERCASE (e.g., "ALBANY"), YOUGOV_DMA_MAP uses Title Case (e.g., "Albany")
    store_to_market_city = store_to_market_city.copy()
    store_to_market_city['market_city'] = store_to_market_city['market_city'].str.upper()

    # Join store → market_city
    df = df.merge(
        store_to_market_city,
        on=store_col,
        how='left'
    )

    # Join market_city → Market via YOUGOV_DMA_MAP
    # Normalize mapping columns from lower to proper case
    if 'market' in yougov_map_df.columns and 'Market' not in yougov_map_df.columns:
        yougov_map_df = yougov_map_df.rename(columns={'market': 'Market'})

    if 'market_city' in yougov_map_df.columns and 'Market' in yougov_map_df.columns:
        # FIX: Normalize YOUGOV_DMA_MAP market_city to uppercase for case-insensitive join
        yougov_map_normalized = yougov_map_df[['market_city', 'Market']].copy()
        yougov_map_normalized['market_city'] = yougov_map_normalized['market_city'].str.upper()
        df = df.merge(
            yougov_map_normalized,
            on='market_city',
            how='left'
        )
    else:
        warnings.warn("YOUGOV_DMA_MAP missing required columns (market_city, Market)")
        df['Market'] = np.nan

    return df


def _prepare_awareness_data(
    awareness_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare awareness data: standardize columns, forward-fill gaps.

    Parameters
    ----------
    awareness_df : pd.DataFrame
        Raw awareness data

    Returns
    -------
    pd.DataFrame
        Cleaned awareness data
    """
    df = awareness_df.copy()

    # Standardize column names
    if 'Market' not in df.columns and 'market' in df.columns:
        df = df.rename(columns={'market': 'Market'})

    if 'week_date' not in df.columns and 'week_start' in df.columns:
        df = df.rename(columns={'week_start': 'week_date'})
    if 'week_date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'week_date'})
    elif 'week_date' not in df.columns and 'Week_Date' in df.columns:
        df = df.rename(columns={'Week_Date': 'week_date'})

    # Normalize awareness/consideration column names (handle mixed casing)
    rename_map = {}
    for src, dst in [
        ('Awareness', 'brand_awareness'),
        ('awareness', 'brand_awareness'),
        ('Consideration', 'brand_consideration'),
        ('consideration', 'brand_consideration'),
    ]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure week_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['week_date']):
        df['week_date'] = pd.to_datetime(df['week_date'])

    # Forward-fill gaps within each Market
    # First, create complete date range per Market
    markets = df['Market'].unique()
    date_range = pd.date_range(df['week_date'].min(), df['week_date'].max(), freq='W-MON')

    # Create complete grid
    complete_grid = pd.MultiIndex.from_product(
        [markets, date_range],
        names=['Market', 'week_date']
    ).to_frame(index=False)

    # Merge with actual data
    df = complete_grid.merge(
        df,
        on=['Market', 'week_date'],
        how='left'
    )

    # Forward-fill within each Market
    df = df.sort_values(['Market', 'week_date'])
    df['brand_awareness'] = df.groupby('Market')['brand_awareness'].ffill()

    if 'brand_consideration' in df.columns:
        df['brand_consideration'] = df.groupby('Market')['brand_consideration'].ffill()

    # Compute 4-week rolling mean for more stable features (reduces week-to-week volatility)
    df['brand_awareness_roll_mean_4'] = (
        df.groupby('Market')['brand_awareness']
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    if 'brand_consideration' in df.columns:
        df['brand_consideration_roll_mean_4'] = (
            df.groupby('Market')['brand_consideration']
            .transform(lambda x: x.rolling(4, min_periods=1).mean())
        )

    return df


def _join_awareness(
    df: pd.DataFrame,
    awareness_df: pd.DataFrame,
    target_col: str,
    include_consideration: bool
) -> pd.DataFrame:
    """
    Join awareness scores on (Market, target_col) so we only use values known at that date.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table with Market column
    awareness_df : pd.DataFrame
        Awareness data with (Market, week_date, brand_awareness)
    target_col : str
        Date column to use for lookup (e.g., origin_week_date)
    include_consideration : bool
        Include consideration score

    Returns
    -------
    pd.DataFrame
        Dataframe with awareness scores
    """
    # Select columns to join - use 4-week rolling mean for stability
    join_cols = ['Market', 'week_date', 'brand_awareness_roll_mean_4']
    if include_consideration and 'brand_consideration_roll_mean_4' in awareness_df.columns:
        join_cols.append('brand_consideration_roll_mean_4')

    # Join on (Market, target_col)
    df = df.merge(
        awareness_df[join_cols],
        left_on=['Market', target_col],
        right_on=['Market', 'week_date'],
        how='left'
    )

    # Drop redundant week_date
    if 'week_date' in df.columns:
        df = df.drop(columns=['week_date'])

    # Rename to standard feature names (rolling mean versions for stability)
    if 'brand_awareness_roll_mean_4' in df.columns:
        df = df.rename(columns={'brand_awareness_roll_mean_4': 'brand_awareness_dma_roll_mean_4'})

    if 'brand_consideration_roll_mean_4' in df.columns:
        df = df.rename(columns={'brand_consideration_roll_mean_4': 'brand_consideration_dma_roll_mean_4'})

    return df


def _apply_sister_dma_fallback(
    df: pd.DataFrame,
    awareness_df: pd.DataFrame,
    store_master_df: pd.DataFrame,
    target_col: str,
    store_col: str,
    include_consideration: bool
) -> pd.DataFrame:
    """
    Apply sister-DMA fallback for stores without awareness mapping.

    For 11 unmapped stores (CHAMPAIGN/SPRINGFIELD, SPARTANBURG/ASHEVILLE, etc.),
    map to appropriate "Single DMAs" regional groupings.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table
    awareness_df : pd.DataFrame
        Awareness data
    store_master_df : pd.DataFrame
        Store master with DMA info
    target_col : str
        Date column used for awareness lookup (aligns with target_col passed to _join_awareness)
    store_col : str
        Store column
    include_consideration : bool
        Include consideration

    Returns
    -------
    pd.DataFrame
        Dataframe with fallback awareness filled
    """
    # Identify stores missing awareness (now using rolling mean feature)
    missing_awareness = df['brand_awareness_dma_roll_mean_4'].isna()

    if missing_awareness.sum() == 0:
        return df  # All stores have awareness

    # For unmapped stores, try to use "Single DMAs" groupings from awareness data
    # This is a simplified fallback - in production, would use proper sister-DMA logic

    # Get unique markets from awareness data
    available_markets = awareness_df['Market'].unique()

    # Find "Single DMAs" or regional groupings
    single_dma_markets = [m for m in available_markets if 'Single' in str(m) or 'Other' in str(m)]

    if len(single_dma_markets) > 0:
        # Use first available single DMA grouping as fallback
        fallback_market = single_dma_markets[0]

        warnings.warn(
            f"Applying fallback awareness using '{fallback_market}' for "
            f"{missing_awareness.sum()} rows with missing awareness data"
        )

        # Get fallback awareness values (use rolling mean version)
        fallback_awareness = awareness_df[
            awareness_df['Market'] == fallback_market
        ][['week_date', 'brand_awareness_roll_mean_4']].rename(
            columns={'brand_awareness_roll_mean_4': 'fallback_awareness'}
        )

        # Merge fallback values
        df = df.merge(
            fallback_awareness,
            left_on=target_col,
            right_on='week_date',
            how='left',
            suffixes=('', '_fallback')
        )

        # Fill missing awareness with fallback
        if 'fallback_awareness' in df.columns:
            df.loc[
                df['brand_awareness_dma_roll_mean_4'].isna(),
                'brand_awareness_dma_roll_mean_4'
            ] = df.loc[
                df['brand_awareness_dma_roll_mean_4'].isna(),
                'fallback_awareness'
            ]

            df = df.drop(columns=['fallback_awareness', 'week_date'], errors='ignore')

    return df


def validate_awareness_coverage(
    df: pd.DataFrame,
    store_col: str = 'profit_center_nbr',
    min_coverage: float = 0.967
) -> dict:
    """
    Validate awareness coverage meets 96.7% threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical table with awareness features
    store_col : str, default='profit_center_nbr'
        Store column
    min_coverage : float, default=0.967
        Minimum required coverage (96.7%)

    Returns
    -------
    dict
        Validation results:
        - coverage: Actual coverage rate
        - status: 'PASS' or 'FAIL'
        - n_stores_total: Total stores
        - n_stores_with_awareness: Stores with awareness data
        - n_stores_missing: Stores without awareness data
        - missing_stores: List of store IDs without awareness

    Examples
    --------
    >>> results = validate_awareness_coverage(canonical_df)
    >>> print(f"Coverage: {results['coverage']:.1%}")
    >>> if results['status'] == 'FAIL':
    ...     print(f"Missing stores: {results['missing_stores']}")
    """
    # Unique stores
    stores = df[[store_col]].drop_duplicates()
    n_total = len(stores)

    # Stores with awareness (check at store level, not row level)
    stores_with_awareness = df[df['brand_awareness_dma_roll_mean_4'].notna()][[store_col]].drop_duplicates()
    n_with_awareness = len(stores_with_awareness)

    # Coverage rate
    coverage = n_with_awareness / n_total if n_total > 0 else 0

    # Status
    status = 'PASS' if coverage >= min_coverage else 'FAIL'

    # Missing stores
    all_stores = set(stores[store_col])
    stores_with_data = set(stores_with_awareness[store_col])
    missing_stores = list(all_stores - stores_with_data)

    return {
        'coverage': coverage,
        'status': status,
        'n_stores_total': n_total,
        'n_stores_with_awareness': n_with_awareness,
        'n_stores_missing': len(missing_stores),
        'missing_stores': missing_stores
    }
