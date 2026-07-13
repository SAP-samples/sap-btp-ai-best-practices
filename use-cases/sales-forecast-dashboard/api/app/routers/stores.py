"""Stores API routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ..security import get_api_key
from ..models.forecast import Store, StoresResponse
from ..services.data_service import get_stores_data, get_store_by_id

router = APIRouter(
    prefix="/stores",
    tags=["stores"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/", response_model=StoresResponse)
async def get_all_stores(
    dma: Optional[str] = Query(None, description="Filter by DMA name"),
    include_non_comp: bool = Query(
        False,
        description="Include non-comp stores (default: comp stores only)"
    )
):
    """
    Get all stores, optionally filtered by DMA.

    By default, only comp stores (open 60+ weeks) are returned.
    Use include_non_comp=true to include all stores.
    """
    stores = get_stores_data(dma=dma, include_non_comp=include_non_comp)
    return StoresResponse(stores=stores, total=len(stores))


@router.get("/{store_id}", response_model=Store)
async def get_store(store_id: int):
    """Get specific store by ID."""
    store = get_store_by_id(store_id)
    if not store:
        raise HTTPException(status_code=404, detail=f"Store with ID {store_id} not found")
    return store
