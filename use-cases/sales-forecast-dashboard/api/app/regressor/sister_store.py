from __future__ import annotations

import pandas as pd

from app.regressor.geo import haversine_miles
from app.regressor.io_utils import load_store_master, load_written_sales, load_market_region_map
from app.regressor.paths import ensure_artifacts_subdir


def _store_history_years() -> pd.DataFrame:
    """Compute number of distinct fiscal years of sales available per store."""
    sales = load_written_sales()[["profit_center_nbr", "fiscal_year"]].dropna(subset=["profit_center_nbr", "fiscal_year"])
    years = sales.groupby("profit_center_nbr")["fiscal_year"].nunique().rename("years_of_history").reset_index()
    return years


def build_sister_store_map(min_years: int = 3) -> pd.DataFrame:
    """
    Map stores with insufficient history to the closest store that meets the history threshold.

    Preference hierarchy when picking a sister store:
    1) Same DMA/market_city if candidates exist.
    2) Same region (via MARKET sheet) if candidates exist.
    3) Nearest overall.

    Returns columns: profit_center_nbr, sister_profit_center_nbr, distance_miles,
    same_market_city (bool), same_region (bool), reason, target_years, candidate_years.
    """
    history = _store_history_years()
    stores = load_store_master()[["profit_center_nbr", "market_city", "market", "latitude", "longitude"]]
    market_region = load_market_region_map()
    stores = stores.merge(market_region, on="market", how="left")

    data = history.merge(stores, on="profit_center_nbr", how="left")
    data = data.dropna(subset=["latitude", "longitude"])

    candidates = data[data["years_of_history"] >= min_years].copy()
    targets = data[data["years_of_history"] < min_years].copy()

    rows = []
    for _, t in targets.iterrows():
        pool = candidates[candidates["profit_center_nbr"] != t["profit_center_nbr"]].copy()
        if pool.empty:
            continue

        pool["same_market_city"] = pool["market_city"] == t["market_city"]
        pool["same_region"] = pool["region"] == t["region"]

        # Select preference tier
        if pool["same_market_city"].any():
            pool_tier = pool[pool["same_market_city"]]
            reason = f"nearest_market_city={t['market_city']}"
        elif pool["same_region"].any():
            pool_tier = pool[pool["same_region"]]
            reason = f"nearest_in_region={t['region']}"
        else:
            pool_tier = pool
            reason = "nearest_overall"

        pool_tier = pool_tier.copy()
        pool_tier["distance_miles"] = pool_tier.apply(
            lambda r: haversine_miles(t["latitude"], t["longitude"], r["latitude"], r["longitude"]), axis=1
        )
        best = pool_tier.sort_values("distance_miles").head(1)
        if best.empty:
            continue
        b = best.iloc[0]
        rows.append(
            {
                "profit_center_nbr": t["profit_center_nbr"],
                "sister_profit_center_nbr": b["profit_center_nbr"],
                "distance_miles": float(b["distance_miles"]),
                "same_market_city": bool(b["same_market_city"]),
                "same_region": bool(b["same_region"]),
                "reason": reason,
                "target_years": int(t["years_of_history"]),
                "candidate_years": int(b["years_of_history"]),
            }
        )

    return pd.DataFrame(rows)


def save_sister_store_map(filename: str = "sister_store_map.csv") -> str:
    df = build_sister_store_map()
    artifacts = ensure_artifacts_subdir("seasonality")
    fp = artifacts / filename
    df.to_csv(fp, index=False)
    return str(fp)
