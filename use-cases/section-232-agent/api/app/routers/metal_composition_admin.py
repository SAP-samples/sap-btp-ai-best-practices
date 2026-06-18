"""Administrative FastAPI router for metal composition data setup."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..models.metal_composition import ClassificationResetResponse, GCCTrackerHanaRefreshResponse
from ..security import get_api_key
from ..services.metal_composition.config import MetalCompositionSettings, get_settings
from ..services.metal_composition.hana_refresh import (
    GCCTrackerHanaRefreshError,
    GCCTrackerWorkbookLoadError,
    refresh_metal_composition_hana,
)
from ..services.metal_composition.service import get_metal_composition_service
from ..services.metal_composition.workbook_format import (
    is_supported_gcc_tracker_workbook,
    supported_gcc_tracker_workbook_description,
)


logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])
_REFRESH_LOCK = Lock()


def _safe_upload_filename(filename: str) -> str:
    basename = Path(filename or "gcc-tracker.xlsb").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return cleaned or "gcc-tracker.xlsb"


def _store_uploaded_workbook(
    *,
    settings: MetalCompositionSettings,
    filename: str,
    content: bytes,
) -> Path:
    upload_dir = settings.cache_dir / "gcc_tracker_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{uuid.uuid4().hex}_{_safe_upload_filename(filename)}"
    stored_path.write_bytes(content)
    return stored_path


def invalidate_metal_composition_service_cache() -> None:
    """Invalidate the singleton service so future requests reload HANA-backed data."""

    cache_clear = getattr(get_metal_composition_service, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def reset_classification_state_after_tracker_refresh() -> ClassificationResetResponse:
    """Clear classification state that is indexed by the previous GCC tracker dataset.

    Returns:
        ClassificationResetResponse: Counts for saved snapshots cleared and active
        classification jobs cancelled.
    """

    return get_metal_composition_service().reset_classifications()


@router.post("/gcc-tracker/refresh-hana", response_model=GCCTrackerHanaRefreshResponse)
async def refresh_gcc_tracker_hana(
    file: UploadFile = File(...),
    source_path: Optional[str] = Form(None),
) -> GCCTrackerHanaRefreshResponse:
    """Upload a GCC Tracker workbook and refresh the configured HANA serving table."""

    filename = file.filename or "gcc-tracker.xlsb"
    if not is_supported_gcc_tracker_workbook(filename):
        detail = (
            "Uploaded GCC Tracker file must be a "
            f"{supported_gcc_tracker_workbook_description()} workbook."
        )
        raise HTTPException(status_code=422, detail=detail)
    if not _REFRESH_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A GCC Tracker HANA refresh is already running.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded GCC Tracker file is empty.")

        settings = get_settings()
        try:
            stored_path = await asyncio.to_thread(
                _store_uploaded_workbook,
                settings=settings,
                filename=filename,
                content=content,
            )
            refresh_payload = await asyncio.to_thread(
                refresh_metal_composition_hana,
                stored_path,
                settings=settings,
            )
            reset_payload = await asyncio.to_thread(reset_classification_state_after_tracker_refresh)
        except (FileNotFoundError, GCCTrackerWorkbookLoadError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GCCTrackerHanaRefreshError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            logger.exception("Failed to store uploaded GCC Tracker workbook")
            raise HTTPException(status_code=500, detail=f"Failed to store uploaded GCC Tracker workbook: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - keep admin API errors bounded
            logger.exception("Unexpected GCC Tracker HANA refresh failure")
            raise HTTPException(status_code=500, detail="Failed to refresh GCC Tracker HANA table.") from exc

        invalidate_metal_composition_service_cache()
        return GCCTrackerHanaRefreshResponse(
            status="completed",
            uploaded_filename=filename,
            uploaded_size_bytes=len(content),
            source_path=source_path,
            stored_workbook_path=str(stored_path),
            sheet_name=str(refresh_payload["sheet_name"]),
            hana_schema=str(refresh_payload["hana_schema"]),
            hana_table=str(refresh_payload["hana_table"]),
            source_row_count=int(refresh_payload["source_row_count"]),
            prepared_row_count=int(refresh_payload["prepared_row_count"]),
            refresh_result=dict(refresh_payload.get("refresh_result") or {}),
            cleared_classification_count=int(reset_payload.cleared_classification_count),
            cancelled_job_count=int(reset_payload.cancelled_job_count),
            service_cache_invalidated=True,
        )
    finally:
        _REFRESH_LOCK.release()
