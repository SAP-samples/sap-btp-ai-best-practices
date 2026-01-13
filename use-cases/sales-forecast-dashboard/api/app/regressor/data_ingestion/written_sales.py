from __future__ import annotations

import pandas as pd

from app.regressor.io_utils import load_written_sales as _load_written_sales


def compute_traffic_missingness_flag(
    df: pd.DataFrame,
    *,
    null_threshold: float = 0.2,
    min_weeks: int = 8,
    traffic_col: str = "store_traffic",
    store_col: str = "profit_center_nbr",
) -> pd.DataFrame:
    """
    Compute a per-store `has_traffic_data` flag based on missing traffic share.

    Rules:
    - Calculate missing_rate = (# nulls / total weeks) per store.
    - Flag is 1 if missing_rate <= null_threshold AND non-null weeks >= min_weeks.
    - Else flag is 0. Applies to Conversion training exclusion logic (ENGINEERING_INSTRUCTIONS).
    """
    if df.empty:
        return pd.DataFrame(columns=[store_col, "has_traffic_data"])

    grp = df.groupby(store_col)
    total = grp[traffic_col].size()
    missing = grp[traffic_col].apply(lambda s: s.isna().sum())
    non_null = grp[traffic_col].apply(lambda s: s.notna().sum())

    miss_rate = (missing / total).fillna(1.0)
    flag = (miss_rate <= null_threshold) & (non_null >= min_weeks)
    flag = flag.astype(int).rename("has_traffic_data")
    return flag.reset_index()


def load_written_sales_with_flags(
    null_threshold: float = 0.2,
    min_weeks: int = 8,
) -> pd.DataFrame:
    """
    Load Written Sales and append `has_traffic_data` per store.

    Parameters:
        null_threshold: Max allowed share of missing traffic weeks before flagging out.
        min_weeks: Minimum non-null traffic weeks required to be considered valid.
    """
    df = _load_written_sales()
    flag_df = compute_traffic_missingness_flag(df, null_threshold=null_threshold, min_weeks=min_weeks)
    return df.merge(flag_df, on="profit_center_nbr", how="left")
