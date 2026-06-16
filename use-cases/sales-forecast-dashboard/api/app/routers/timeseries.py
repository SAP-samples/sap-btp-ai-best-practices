"""Timeseries API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..security import get_api_key
from ..models.forecast import StoreTimeseries, DMATimeseries
from ..services.data_service import get_store_timeseries, get_dma_timeseries

router = APIRouter(
    prefix="/timeseries",
    tags=["timeseries"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/store/{store_id}", response_model=StoreTimeseries)
async def get_store_timeseries_data(
    store_id: int,
    channel: Optional[str] = Query(
        "B&M",
        description="Channel: B&M or WEB. Defaults to B&M."
    )
):
    """
    Get timeseries data for a specific store including SHAP features.

    Args:
        store_id: The store ID
        channel: Channel name (B&M or WEB). Defaults to B&M.
    """
    timeseries = get_store_timeseries(store_id, channel=channel)
    if not timeseries:
        raise HTTPException(
            status_code=404,
            detail=f"Timeseries data for store {store_id} (channel={channel}) not found"
        )
    return timeseries


@router.get("/dma/{dma_name:path}", response_model=DMATimeseries)
async def get_dma_timeseries_data(dma_name: str):
    """Get aggregated timeseries data for a DMA."""
    timeseries = get_dma_timeseries(dma_name)
    if not timeseries:
        raise HTTPException(
            status_code=404,
            detail=f"Timeseries data for DMA '{dma_name}' not found"
        )
    return timeseries
