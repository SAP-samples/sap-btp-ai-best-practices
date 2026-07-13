from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from .io_utils import load_store_master, load_written_sales, load_ecomm_traffic
from .sister_dma import build_sister_dma_map
from .paths import ensure_artifacts_subdir

# Prophet-based DMA components are intentionally omitted per EPIC 2 scope.


def _moving_average_cyclic(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple cyclic moving average smoothing (wrap-around)."""
    # If the window is 1 or less, no smoothing is needed; return the original values.
    if window <= 1:
        return values
    n = len(values)
    pad = window // 2  # Number of points to pad on each side for wrap-around effect.
    # Extend values on both ends to enable cyclic (wrap-around) smoothing.
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    # Create the moving average kernel (equal weights).
    kernel = np.ones(window) / window
    # Perform convolution with the kernel. 'valid' mode gives back n values, sliding across the original length.
    sm = np.convolve(ext, kernel, mode="valid")
    # After convolution, we get back a smoothed array of length n (original length).
    return sm


def _compute_dma_seasonality_from_data(
    df: pd.DataFrame,
    value_col: str,
    smoothing_window: int,
    fold_week53_to1: bool,
    apply_sister_fallback: bool,
    min_years_required: int,
    dma_col: str = "market_city"
) -> pd.DataFrame:
    """Core logic to compute DMA seasonality using N-1 logic."""
    all_years = sorted(df['fiscal_year'].unique())
    if not all_years:
        return pd.DataFrame()
        
    results = []
    # Generate weights for each year Y in dataset + 1 future year
    target_years = all_years + [max(all_years) + 1]
    
    for target_year in target_years:
        # Use data strictly before target_year
        history = df[df['fiscal_year'] < target_year].copy()
        
        # Fallback: If no history, use all available data (prevents empty starts)
        if history.empty:
            history = df.copy()
            
        if history.empty:
            continue
            
        # Aggregate to DMA-week
        grp = (
            history.groupby([dma_col, "fiscal_year", "fiscal_week"], dropna=False)[value_col].sum().reset_index()
        )

        # Compute yearly totals per DMA
        totals = grp.groupby([dma_col, "fiscal_year"], dropna=False)[value_col].sum().rename("total_year").reset_index()
        grp = grp.merge(totals, on=[dma_col, "fiscal_year"], how="left")
        grp["share"] = np.where(grp["total_year"] > 0, grp[value_col] / grp["total_year"], 0.0)

        if fold_week53_to1:
            grp["fiscal_week_fold"] = grp["fiscal_week"].where(grp["fiscal_week"] != 53, 1)
        else:
            grp["fiscal_week_fold"] = grp["fiscal_week"]

        # Average share across years
        avg = (
            grp.groupby([dma_col, "fiscal_week_fold"], dropna=False)["share"].mean().reset_index()
        )
        avg = avg.rename(columns={"fiscal_week_fold": "fiscal_week"})

        # Keep only 52 weeks
        avg = avg[avg["fiscal_week"].between(1, 52)]

        # Apply smoothing per DMA
        def smooth_group(sub: pd.DataFrame) -> pd.DataFrame:
            sub = sub.sort_values("fiscal_week").copy()
            vals = sub["share"].to_numpy()
            sm = _moving_average_cyclic(vals, smoothing_window) if len(vals) == 52 else vals
            sm = sm / sm.sum() if sm.sum() > 0 else sm
            sub["weight"] = sm
            return sub[[dma_col, "fiscal_week", "weight"]]

        out = avg.groupby(dma_col, as_index=False, group_keys=False).apply(smooth_group)
        out = out.sort_values([dma_col, "fiscal_week"]).reset_index(drop=True)

        # Apply sister-DMA fallback if requested
        if apply_sister_fallback:
            sister_map = build_sister_dma_map(min_years=min_years_required)
            if not sister_map.empty:
                targets = sister_map["market_city"].unique().tolist()
                # Remove existing weights for targets to overwrite with sister weights
                out = out[~out[dma_col].isin(targets)].copy()
                
                # Prepare relation: Map sister_market_city -> market_city
                # sister_map has [market_city (target), sister_market_city (source)]
                rel = (
                    out.rename(columns={dma_col: "sister_market_city"})
                    .merge(sister_map[["market_city", "sister_market_city"]], on="sister_market_city", how="right")
                )
                # Rename back to target market_city
                # rel now has [fiscal_week, weight, sister_market_city, market_city]
                # We want [dma_col, fiscal_week, weight]
                
                # Ensure columns match
                rel = rel.rename(columns={"market_city": dma_col})
                rel = rel[[dma_col, "fiscal_week", "weight"]].dropna(subset=["weight"])
                
                out = pd.concat([out, rel], ignore_index=True)
                
        out['fiscal_year'] = target_year
        results.append(out)
        
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values([dma_col, "fiscal_year", "fiscal_week"]).reset_index(drop=True)
    return final_df


def compute_dma_seasonality(
    smoothing_window: int = 3,
    fold_week53_to1: bool = True,
    min_years_required: int = 3,
    apply_sister_fallback: bool = True,
    channel_filter: str = 'B&M'
) -> pd.DataFrame:
    """Recompute DMA seasonal weights using N-1 logic.
    
    Method: 
    - For each target year Y, use data from years < Y.
    - Filter for specific channel (default B&M).
    """
    sales = load_written_sales()
    
    if channel_filter == 'B&M':
        sales = sales[sales['channel'] != 'WEB'].copy()
    elif channel_filter == 'WEB':
        sales = sales[sales['channel'] == 'WEB'].copy()
        
    stores = load_store_master()[["profit_center_nbr", "market_city"]].copy()
    df = sales.merge(stores, on="profit_center_nbr", how="left")
    
    return _compute_dma_seasonality_from_data(
        df,
        value_col="total_sales",
        dma_col="market_city",
        smoothing_window=smoothing_window,
        fold_week53_to1=fold_week53_to1,
        apply_sister_fallback=apply_sister_fallback,
        min_years_required=min_years_required
    )


def compute_web_seasonality(
    smoothing_window: int = 3,
    fold_week53_to1: bool = True,
    min_years_required: int = 3,
    apply_sister_fallback: bool = True
) -> pd.DataFrame:
    """Compute WEB seasonal weights per DMA using N-1 logic.
    
    Uses Ecomm Traffic (merch_amt) as authoritative source.
    Attributed to DMA via 'market_city' column in source.
    """
    ecomm = load_ecomm_traffic()
    ecomm["fiscal_year"] = ecomm["fiscal_start_date_week"].dt.year
    ecomm["fiscal_week"] = ecomm["fiscal_start_date_week"].dt.isocalendar().week
    
    # Use 'market_city' from ecomm data directly
    
    return _compute_dma_seasonality_from_data(
        ecomm,
        value_col="merch_amt",
        dma_col="market_city",
        smoothing_window=smoothing_window,
        fold_week53_to1=fold_week53_to1,
        apply_sister_fallback=apply_sister_fallback,
        min_years_required=min_years_required
    )


def save_dma_seasonality_csv(path: Optional[str] = None) -> str:
    """Compute and save DMA seasonality to artifacts directory. Returns the file path."""
    out = compute_dma_seasonality()
    artifacts = ensure_artifacts_subdir("seasonality")
    fp = artifacts / (path or "dma_seasonality.csv")
    out.to_csv(fp, index=False)
    return str(fp)
