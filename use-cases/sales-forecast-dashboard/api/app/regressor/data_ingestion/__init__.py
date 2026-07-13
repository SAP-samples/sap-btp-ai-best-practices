"""
Data ingestion entrypoints.
- Written sales loader with traffic missingness flag.
- Ecomm traffic loader.
- Awareness/consideration weekly feeds (DMA/market mapped).
- CRM demographics (static + time-varying mix).
- Store/master data helpers.
"""

from .written_sales import load_written_sales_with_flags, compute_traffic_missingness_flag
from .ecomm_traffic import load_ecomm_traffic
from .awareness import load_awareness_with_mapping
from .crm_mix import load_demographics_with_typing
from .store_master import load_store_master, load_market_region_map, load_yougov_dma_map

__all__ = [
    "load_written_sales_with_flags",
    "compute_traffic_missingness_flag",
    "load_ecomm_traffic",
    "load_awareness_with_mapping",
    "load_demographics_with_typing",
    "load_store_master",
    "load_market_region_map",
    "load_yougov_dma_map",
]
