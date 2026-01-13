"""Admin API routes for maintenance operations."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from ..security import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_api_key)],
)


class RegenerateResponse(BaseModel):
    """Response model for data regeneration."""
    status: str
    message: str
    details: Optional[dict] = None


class RegenerateStatusResponse(BaseModel):
    """Response model for regeneration status check."""
    is_running: bool
    last_run: Optional[str] = None
    last_result: Optional[dict] = None


# Track regeneration state
_regeneration_state = {
    "is_running": False,
    "last_run": None,
    "last_result": None,
}


def _run_regeneration(skip_timeseries: bool = False):
    """Background task to run data regeneration."""
    global _regeneration_state

    from datetime import datetime
    from ..scripts.regenerate_dashboard_data import regenerate_data

    _regeneration_state["is_running"] = True
    _regeneration_state["last_run"] = datetime.now().isoformat()

    try:
        logger.info("Starting data regeneration from API call...")
        result = regenerate_data(
            skip_timeseries=skip_timeseries,
            verbose=True
        )
        _regeneration_state["last_result"] = result
        logger.info(f"Data regeneration completed: {result}")
    except Exception as e:
        logger.error(f"Data regeneration failed: {e}")
        _regeneration_state["last_result"] = {"status": "error", "error": str(e)}
    finally:
        _regeneration_state["is_running"] = False


@router.post("/regenerate-data", response_model=RegenerateResponse)
async def regenerate_dashboard_data(
    background_tasks: BackgroundTasks,
    skip_timeseries: bool = Query(
        False,
        description="Skip timeseries file generation (faster, for testing)"
    ),
    sync: bool = Query(
        False,
        description="Run synchronously instead of in background (blocks until complete)"
    ),
):
    """
    Regenerate all dashboard data from SAP HANA.

    This endpoint triggers the data regeneration script that:
    - Loads 2025 and 2024 predictions from HANA
    - Computes YoY sales changes for traffic light colors
    - Generates stores.json, dma_summary.json, and timeseries files

    By default, runs in the background and returns immediately.
    Use sync=true to wait for completion (may timeout for large datasets).
    """
    global _regeneration_state

    # Check if already running
    if _regeneration_state["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Data regeneration is already in progress. Check /admin/regenerate-status for updates."
        )

    if sync:
        # Run synchronously (blocking)
        logger.info("Running data regeneration synchronously...")
        _run_regeneration(skip_timeseries=skip_timeseries)

        result = _regeneration_state["last_result"]
        if result and result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return RegenerateResponse(
            status="completed",
            message="Data regeneration completed successfully",
            details=result
        )
    else:
        # Run in background
        background_tasks.add_task(_run_regeneration, skip_timeseries)

        return RegenerateResponse(
            status="started",
            message="Data regeneration started in background. Check /admin/regenerate-status for progress.",
            details={"skip_timeseries": skip_timeseries}
        )


@router.get("/regenerate-status", response_model=RegenerateStatusResponse)
async def get_regeneration_status():
    """
    Check the status of data regeneration.

    Returns whether a regeneration is currently running,
    when the last regeneration was started, and its result.
    """
    return RegenerateStatusResponse(
        is_running=_regeneration_state["is_running"],
        last_run=_regeneration_state["last_run"],
        last_result=_regeneration_state["last_result"]
    )
