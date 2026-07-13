from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import pandas as pd

from .io_utils import load_store_master, load_market_region_map
from .paths import ensure_artifacts_subdir


def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points in miles.

    Supports both scalar inputs and pandas Series (vectorized).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array-like
        Latitude and longitude coordinates in degrees

    Returns
    -------
    float or array-like
        Distance in miles
    """
    R_km = 6371  # Earth's mean radius in km
    R_miles = R_km * 0.621371

    # Use numpy functions for vectorization support
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R_miles * c


def compute_store_neighbors() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise distances and per-store neighbor summaries.

    Returns:
    - pairs: (store_i, store_j, distance_miles)
    - summary: one row per store with features: min_distance, count_10mi, count_20mi, count_30mi
    """
    stores = load_store_master()
    # Filter to retail/outlet stores (exclude DC and rows with missing coords)
    stores = stores[(stores["location_type"].astype(str).str.lower() != "dc")]
    stores = stores.dropna(subset=["latitude", "longitude", "profit_center_nbr"]).copy()
    stores["profit_center_nbr"] = stores["profit_center_nbr"].astype(int)

    coords = stores[["profit_center_nbr", "latitude", "longitude", "market_city", "market"]].reset_index(drop=True)
    n = len(coords)  # number of stores

    # Compute full pairwise distances (upper triangle) efficiently
    pairs = []
    arr = coords[["latitude", "longitude"]].to_numpy()
    ids = coords["profit_center_nbr"].to_numpy()
    for i in range(n): # iterate over all stores
        lat1, lon1 = arr[i]
        for j in range(i + 1, n): # iterate over all stores except the current store
            lat2, lon2 = arr[j]
            d = haversine_miles(lat1, lon1, lat2, lon2) # compute the distance between the two stores
            pairs.append((int(ids[i]), int(ids[j]), d))
    pairs_df = pd.DataFrame(pairs, columns=["store_i", "store_j", "distance_miles"]) if pairs else pd.DataFrame(columns=["store_i", "store_j", "distance_miles"]) 

    # Build neighbor summary per store
    if not pairs_df.empty:
        stacked = pd.concat(
            [
                pairs_df.rename(columns={"store_i": "store", "store_j": "neighbor"}),
                pairs_df.rename(columns={"store_j": "store", "store_i": "neighbor"}),
            ],
            ignore_index=True,
        )
        agg = (
            stacked.groupby("store")["distance_miles"]
            .agg(
                min_distance_miles="min",
                count_10mi=lambda s: (s <= 10).sum(),
                count_20mi=lambda s: (s <= 20).sum(),
                count_30mi=lambda s: (s <= 30).sum(),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(columns=["store", "min_distance_miles", "count_10mi", "count_20mi", "count_30mi"])

    summary = coords.merge(agg, left_on="profit_center_nbr", right_on="store", how="left").drop(columns=["store"]) 
    return pairs_df, summary


def save_store_neighbors() -> Tuple[str, str]:
    pairs_df, summary = compute_store_neighbors()
    artifacts = ensure_artifacts_subdir("geo")
    p1 = artifacts / "store_pairs_distances.csv"
    p2 = artifacts / "store_neighbor_summary.csv"
    pairs_df.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    return str(p1), str(p2)


def compute_dma_centroids() -> pd.DataFrame:
    """Compute centroids for each DMA (market_city) as mean lat/lon of its stores."""
    stores = load_store_master()
    sub = stores.dropna(subset=["latitude", "longitude"]).copy()
    cent = (
        sub.groupby(["market_city"])[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    return cent


    # plotting helpers removed to avoid extra dependencies in prototype runtime
