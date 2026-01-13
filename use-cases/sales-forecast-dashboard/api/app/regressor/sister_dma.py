from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from .io_utils import load_store_master, load_written_sales, load_market_region_map
from .geo import haversine_miles, compute_dma_centroids
from .paths import ensure_artifacts_subdir


def _dma_history_years() -> pd.DataFrame:
    """Compute number of distinct fiscal years of sales available per DMA (market_city)."""
    sales = load_written_sales()[["profit_center_nbr", "fiscal_year"]].dropna()  # drop rows with missing fiscal_year
    stores = load_store_master()[["profit_center_nbr", "market_city", "market"]]  # load store master data - get market_city and market for each store
    df = sales.merge(stores, on="profit_center_nbr", how="left").dropna(subset=["market_city"])  # merge sales and stores data on profit_center_nbr - Adds market_city to the sales dataframe.
    years = df.groupby(["market_city"])['fiscal_year'].nunique().rename('years_of_history').reset_index()  # group by market_city and count the number of unique fiscal_years - get the number of years of history for each market_city
    # Attach market -> region mapping via MARKET sheet
    market_region = load_market_region_map()
    # We need region at DMA; aggregate by most common region among markets present
    stores_mkt = stores.merge(market_region, on="market", how="left")
    dma_region = (
        stores_mkt.groupby('market_city')['region'].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan).reset_index()
    )  # group by market_city and get the most common region among markets present
    out = years.merge(dma_region, on='market_city', how='left')  # merge the number of years of history with the region for each market_city
    return out


def build_sister_dma_map(min_years: int = 3) -> pd.DataFrame:
    """Create a sister-DMA mapping for DMAs with insufficient history. This history value comes from the seasonality baseline requirement.
    1. Seasonality baseline requires at least 3 years of history for each DMA.
    2. DMAs with less than 3 years cannot compute their own seasonality
    3. We need to map these DMAs to a sister DMA with at least 3 years of history

    Strategy:
    - Identify DMAs with < min_years of history in Written Sales Data.
    - For each, choose nearest DMA (by centroid) with >= min_years, preferring same region.
    - Fallback to nearest overall if no same-region candidate exists.

    Returns columns: market_city, sister_market_city, distance_miles, same_region (bool), reason.
    """
    hist = _dma_history_years()  # get the number of years of history for each market_city
    cent = compute_dma_centroids()  # compute the centroids for each DMA: latitude and longitude of the centroid of each DMA 

    # merge the number of years of history with the centroids for each DMA
    data = hist.merge(cent, on='market_city', how='left').rename(columns={'latitude':'lat', 'longitude':'lon'})
    candidates = data[data['years_of_history'] >= min_years].dropna(subset=['lat','lon']).copy() # get the DMAs with at least 3 years of history
    targets = data[data['years_of_history'] < min_years].dropna(subset=['lat','lon']).copy() # get the DMAs with less than 3 years of history

    rows = []
    for _, t in targets.iterrows():
        pool = candidates.copy()
        # Prefer same region first - if the region of the target DMA is the same as the region of the candidate DMA, set the region_match to True
        pool['region_match'] = (pool['region'] == t['region'])
        if pool['region_match'].any():
            pool = pool[pool['region_match']]
            reason = f"nearest_in_region={t['region']}" # if the region of the target DMA is the same as the region of the candidate DMA, set the reason to the region of the target DMA
        else:
            reason = "nearest_overall" # if the region of the target DMA is not the same as the region of the candidate DMA, set the reason to the nearest overall

        # Compute distances to all candidates in pool
        pool['distance_miles'] = pool.apply(lambda r: haversine_miles(t['lat'], t['lon'], r['lat'], r['lon']), axis=1) # compute the distance between the target DMA and the candidate DMA
        best = pool.sort_values('distance_miles').head(1)
        if best.empty:
            continue # if there is no candidate DMA with at least 3 years of history in the pool, skip the target DMA
        b = best.iloc[0]
        rows.append({
            'market_city': t['market_city'],
            'sister_market_city': b['market_city'],
            'distance_miles': float(b['distance_miles']),
            'same_region': bool(b.get('region_match', False)),
            'reason': reason,
            'target_years': int(t['years_of_history']),
            'candidate_years': int(b['years_of_history']),
        })

    return pd.DataFrame(rows) # return the dataframe with the sister DMA mapping


def save_sister_dma_map() -> str:
    df = build_sister_dma_map()
    artifacts = ensure_artifacts_subdir('seasonality')
    fp = artifacts / 'sister_dma_map.csv'
    df.to_csv(fp, index=False)
    return str(fp)
