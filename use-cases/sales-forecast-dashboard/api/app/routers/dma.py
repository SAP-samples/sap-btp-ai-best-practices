"""DMA (Designated Market Area) API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query

from ..security import get_api_key
from ..models.forecast import DMA, DMASummaryResponse
from ..services.data_service import get_dma_data, get_dma_by_name

router = APIRouter(
    prefix="/dma",
    tags=["dma"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/", response_model=DMASummaryResponse)
async def get_all_dmas(
    include_non_comp: bool = Query(
        False,
        description="Include non-comp stores in aggregation (default: comp stores only)"
    )
):
    """
    Get summary data for all DMAs.

    By default, DMA aggregates are computed from comp stores only.
    Use include_non_comp=true to include all stores in aggregation.

    Note: DMA summary is pre-computed during data generation. This parameter
    is reserved for future use when real-time aggregation is supported.
    """
    # Note: include_non_comp is accepted for API consistency but currently
    # has no effect because DMA summary is pre-aggregated from comp stores
    # during data generation. To include non-comp stores, regenerate data
    # with comp_stores_only=False.
    dmas = get_dma_data()
    return DMASummaryResponse(dmas=dmas, total=len(dmas))


@router.get("/{dma_name}", response_model=DMA)
async def get_dma(dma_name: str):
    """Get specific DMA by name."""
    dma = get_dma_by_name(dma_name)
    if not dma:
        raise HTTPException(status_code=404, detail=f"DMA '{dma_name}' not found")
    return dma
