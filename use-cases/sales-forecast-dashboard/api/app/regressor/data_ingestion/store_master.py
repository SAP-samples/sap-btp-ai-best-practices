from app.regressor.io_utils import (
    load_store_master as _load_store_master,
    load_market_region_map as _load_market_region_map,
    load_yougov_dma_map as _load_yougov_dma_map,
)


def load_store_master():
    """Wrapper around prototype loader; returns standardized store master."""
    return _load_store_master()


def load_market_region_map():
    """Wrapper for Market -> Region mapping."""
    return _load_market_region_map()


def load_yougov_dma_map():
    """Wrapper for YouGov DMA mapping."""
    return _load_yougov_dma_map()

