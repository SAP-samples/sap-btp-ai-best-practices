from __future__ import annotations

import pandas as pd

from app.regressor.io_utils import load_awareness_consideration as _load_awareness_consideration
from app.regressor.io_utils import load_yougov_dma_map as _load_yougov_dma_map


def load_awareness_with_mapping() -> pd.DataFrame:
    """
    Load weekly awareness/consideration and attach YouGov DMA/market metadata.

    Returns columns:
    - market, week_start, awareness, consideration (as available)
    - yougov_dma, region, market_city (if present in mapping)
    """
    aw = _load_awareness_consideration()
    mapping = _load_yougov_dma_map()

    if aw.empty:
        # Preserve expected columns even when no data
        base_cols = ["market", "week_start", "awareness", "consideration"]
        map_cols = ["yougov_dma", "region", "market_city"]
        return pd.DataFrame(columns=base_cols + map_cols)

    mapping = mapping.drop_duplicates(subset=["market"])
    merged = aw.merge(mapping, on="market", how="left")
    return merged


# Backward-compatible alias expected by feature pipeline
def load_awareness_data() -> pd.DataFrame:
    return load_awareness_with_mapping()
