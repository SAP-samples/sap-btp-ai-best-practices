"""
Optimizer API Router

Endpoints for the credit optimizer: process lifecycle, configuration,
execution, results, and downloads.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response

from ..models.optimizer import (
    CreateProcessResponse,
    InvoiceListResponse,
    LimitsConfig,
    OptimizationResults,
    ProcessDetail,
    ProcessStatus,
    ProcessSummary,
    RulesConfig,
    SolverSettings,
)
from ..optimizer.model.limits_import import LimitsImportError, import_limits_payload
from ..security import get_api_key
from ..services.optimizer.process_manager import ProcessManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimizer",
    tags=["optimizer"],
    dependencies=[Depends(get_api_key)],
)

# Lazy singleton
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager


# ---------------------------------------------------------------------------
# Process Lifecycle
# ---------------------------------------------------------------------------


@router.post("/processes", response_model=CreateProcessResponse)
async def create_process(
    file: UploadFile = File(..., description="Offer or extraction Excel file (.xlsx)"),
    cohort: Optional[str] = Form(None, description="Cohort target date"),
    sheet_name: str = Form("SAPUI5 Export", description="Sheet name in workbook"),
):
    """Create a new optimization process by uploading an input Excel file."""
    mgr = get_process_manager()
    try:
        content = await file.read()
        record = mgr.create_process(
            file_content=content,
            filename=file.filename or "extraction.xlsx",
            cohort=cohort,
            sheet_name=sheet_name,
        )
        return CreateProcessResponse(
            process_id=record["id"],
            status=ProcessStatus(record["status"]),
            created_at=record["created_at"],
            extraction_filename=record["extraction_filename"],
            available_cohorts=record.get("available_cohorts", []),
        )
    except Exception as exc:
        logger.exception("Failed to create process")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/processes", response_model=List[ProcessSummary])
async def list_processes(
    process_status: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List optimization processes."""
    mgr = get_process_manager()
    records = mgr.list_processes(status=process_status, limit=limit, offset=offset)
    return [
        ProcessSummary(
            process_id=r["id"],
            status=ProcessStatus(r["status"]),
            created_at=r.get("created_at"),
            started_at=r.get("started_at"),
            completed_at=r.get("completed_at"),
            extraction_filename=r.get("extraction_filename"),
            cohort=r.get("cohort"),
            planning_mode=r.get("planning_mode"),
            source_profile=r.get("source_profile"),
            candidate_count=r.get("candidate_count"),
            selected_count=r.get("selected_count"),
            excluded_count=r.get("excluded_count"),
            selected_amount=r.get("selected_amount"),
            candidate_amount=r.get("candidate_amount"),
            optimizer_status=r.get("optimizer_status"),
            error_message=r.get("error_message"),
        )
        for r in records
    ]


@router.get("/processes/{process_id}", response_model=ProcessDetail)
async def get_process(process_id: str):
    """Get full detail for a process."""
    mgr = get_process_manager()
    r = mgr.get_process(process_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Process not found")

    return ProcessDetail(
        process_id=r["id"],
        status=ProcessStatus(r["status"]),
        created_at=r.get("created_at"),
        started_at=r.get("started_at"),
        completed_at=r.get("completed_at"),
        extraction_filename=r.get("extraction_filename"),
        cohort=r.get("cohort"),
        cohort_match_granularity=r.get("cohort_match_granularity"),
        sheet_name=r.get("sheet_name"),
        release_event=r.get("release_event"),
        release_event_mode=r.get("release_event_mode"),
        planning_mode=r.get("planning_mode"),
        planning_start_date=r.get("planning_start_date"),
        horizon_weeks=r.get("horizon_weeks"),
        attempt_cap=r.get("attempt_cap"),
        source_profile=r.get("source_profile"),
        lifecycle_input_path=r.get("lifecycle_input_path"),
        solver_settings=SolverSettings(
            max_time_seconds=r.get("solver_max_time_seconds", 60),
            random_seed=r.get("solver_random_seed", 0),
            num_search_workers=r.get("solver_num_search_workers", 1),
        ),
        candidate_count=r.get("candidate_count"),
        selected_count=r.get("selected_count"),
        excluded_count=r.get("excluded_count"),
        candidate_amount=r.get("candidate_amount"),
        selected_amount=r.get("selected_amount"),
        optimizer_status=r.get("optimizer_status"),
        error_message=r.get("error_message"),
    )


@router.delete("/processes/{process_id}")
async def delete_process(process_id: str):
    """Delete a process and its directory."""
    mgr = get_process_manager()
    deleted = mgr.delete_process(process_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Process not found")
    return {"success": True, "process_id": process_id}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@router.get("/processes/{process_id}/limits")
async def get_limits(process_id: str) -> Dict[str, Any]:
    """Get the limits configuration for a process."""
    mgr = get_process_manager()
    try:
        return mgr.get_limits(process_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


@router.put("/processes/{process_id}/limits")
async def update_limits(process_id: str, limits: LimitsConfig) -> Dict[str, Any]:
    """Update the limits configuration for a process."""
    mgr = get_process_manager()
    try:
        return mgr.update_limits(process_id, limits.model_dump())
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


@router.post("/processes/{process_id}/limits/import")
async def import_limits(
    process_id: str,
    file: UploadFile = File(..., description="Limits file (.xlsx/.xls/.yaml/.yml/.json)"),
) -> Dict[str, Any]:
    """Import manual limits payload from Excel/YAML/JSON without persisting it."""
    mgr = get_process_manager()
    try:
        existing_limits = mgr.get_limits(process_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")

    try:
        content = await file.read()
        limits_payload, import_summary = import_limits_payload(
            content,
            filename=file.filename or "",
            existing_limits=existing_limits,
        )
        return {
            "limits_payload": limits_payload,
            "import_summary": import_summary,
        }
    except LimitsImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to import limits file")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/processes/{process_id}/rules")
async def get_rules(process_id: str) -> Dict[str, Any]:
    """Get the rules configuration for a process."""
    mgr = get_process_manager()
    try:
        return mgr.get_rules(process_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


@router.put("/processes/{process_id}/rules")
async def update_rules(process_id: str, rules: RulesConfig) -> Dict[str, Any]:
    """Update the rules configuration for a process."""
    mgr = get_process_manager()
    try:
        return mgr.update_rules(process_id, rules.model_dump(exclude_none=True))
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


@router.get("/processes/{process_id}/params")
async def get_params(process_id: str) -> Dict[str, Any]:
    """Get solver/process parameters."""
    mgr = get_process_manager()
    r = mgr.get_process(process_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Process not found")
    return {
        "cohort": r.get("cohort"),
        "cohort_match_granularity": r.get("cohort_match_granularity"),
        "sheet_name": r.get("sheet_name"),
        "release_event": r.get("release_event"),
        "release_event_mode": r.get("release_event_mode"),
        "planning_mode": r.get("planning_mode", "single_week"),
        "planning_start_date": r.get("planning_start_date"),
        "horizon_weeks": r.get("horizon_weeks", 8),
        "attempt_cap": r.get("attempt_cap", 1),
        "source_profile": r.get("source_profile", "extraction_file"),
        "lifecycle_input_path": r.get("lifecycle_input_path"),
        "solver_max_time_seconds": r.get("solver_max_time_seconds", 60),
        "solver_random_seed": r.get("solver_random_seed", 0),
        "solver_num_search_workers": r.get("solver_num_search_workers", 1),
    }


@router.put("/processes/{process_id}/params")
async def update_params(process_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update solver/process parameters."""
    mgr = get_process_manager()
    try:
        record = mgr.update_params(process_id, params)
        if record is None:
            raise HTTPException(status_code=404, detail="Process not found")
        return record
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


@router.get("/processes/{process_id}/cohorts")
async def get_cohorts(process_id: str) -> List[Dict[str, Any]]:
    """List available cohorts detected from the extraction Excel."""
    mgr = get_process_manager()
    try:
        return mgr.get_cohorts(process_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Process not found")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@router.post("/processes/{process_id}/run")
async def run_optimization(
    process_id: str,
    background_tasks: BackgroundTasks,
    run_params: Optional[Dict[str, Any]] = Body(default=None),
):
    """Start optimization as a background task."""
    mgr = get_process_manager()
    try:
        if run_params:
            updates: Dict[str, Any] = {}
            if run_params.get("planning_mode") in {"single_week", "multi_week"}:
                updates["planning_mode"] = run_params["planning_mode"]
            if "planning_start_date" in run_params:
                updates["planning_start_date"] = run_params.get("planning_start_date")
            if "horizon_weeks" in run_params:
                updates["horizon_weeks"] = run_params.get("horizon_weeks")
            if "attempt_cap" in run_params:
                updates["attempt_cap"] = run_params.get("attempt_cap")
            if updates:
                mgr.update_params(process_id, updates)
        record = mgr.start_optimization(process_id)
        background_tasks.add_task(mgr.run_in_background, process_id)
        return {"process_id": process_id, "status": record["status"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/processes/{process_id}/status")
async def get_status(process_id: str) -> Dict[str, Any]:
    """Poll execution status."""
    mgr = get_process_manager()
    r = mgr.get_process(process_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Process not found")
    progress = None
    raw_progress = r.get("progress_json")
    if raw_progress:
        try:
            progress = raw_progress if isinstance(raw_progress, dict) else json.loads(raw_progress)
        except Exception:
            progress = None

    return {
        "process_id": r["id"],
        "status": r["status"],
        "started_at": r.get("started_at"),
        "error_message": r.get("error_message"),
        "candidate_count": r.get("candidate_count"),
        "selected_count": r.get("selected_count"),
        "excluded_count": r.get("excluded_count"),
        "progress": progress,
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@router.get("/processes/{process_id}/results")
async def get_results(process_id: str) -> Dict[str, Any]:
    """Get full optimization results (run_metadata)."""
    mgr = get_process_manager()
    results = mgr.get_results(process_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Results not found. Process may not have completed.")
    return results


@router.get("/processes/{process_id}/results/selected")
async def get_selected_invoices(
    process_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Get paginated selected invoices."""
    mgr = get_process_manager()
    return mgr.get_invoices(process_id, invoice_type="selected", limit=limit, offset=offset)


@router.get("/processes/{process_id}/results/excluded")
async def get_excluded_invoices(
    process_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    stage: Optional[str] = Query(None, description="Filter by excluded_stage: 'rule' or 'optimizer'"),
    reason: Optional[str] = Query(None, description="Filter by excluded_reason substring"),
) -> Dict[str, Any]:
    """Get paginated excluded invoices with optional filters."""
    mgr = get_process_manager()
    return mgr.get_invoices(
        process_id,
        invoice_type="excluded",
        limit=limit,
        offset=offset,
        stage_filter=stage,
        reason_filter=reason,
    )


@router.get("/processes/{process_id}/results/weekly-plan")
async def get_weekly_plan(
    process_id: str,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Get paginated weekly planning rows (multi-week mode)."""
    mgr = get_process_manager()
    return mgr.get_invoices(process_id, invoice_type="weekly_plan", limit=limit, offset=offset)


@router.get("/processes/{process_id}/results/exposure")
async def get_weekly_exposure(
    process_id: str,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    entity_type: Optional[str] = Query(None, description="Filter entity type: facility/customer/group"),
) -> Dict[str, Any]:
    """Get paginated weekly exposure rows (multi-week mode)."""
    mgr = get_process_manager()
    return mgr.get_weekly_exposure(
        process_id,
        limit=limit,
        offset=offset,
        entity_type=entity_type,
    )


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def _file_response(path, media_type: str, filename: str) -> FileResponse:
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=filename,
    )


@router.get("/processes/{process_id}/download/selected")
async def download_selected(process_id: str):
    """Download selected.xlsx."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "selected")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_response(
        path,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "selected.xlsx",
    )


@router.get("/processes/{process_id}/download/excluded")
async def download_excluded(process_id: str):
    """Download excluded.xlsx."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "excluded")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_response(
        path,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "excluded.xlsx",
    )


@router.get("/processes/{process_id}/download/summary")
async def download_summary(process_id: str):
    """Download run_summary.md."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "summary")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_response(path, "text/markdown", "run_summary.md")


@router.get("/processes/{process_id}/download/weekly-plan")
async def download_weekly_plan(process_id: str):
    """Download weekly_plan.xlsx."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "weekly_plan")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_response(
        path,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "weekly_plan.xlsx",
    )


@router.get("/processes/{process_id}/download/weekly-exposure")
async def download_weekly_exposure(process_id: str):
    """Download weekly_exposure.xlsx."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "weekly_exposure")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_response(
        path,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "weekly_exposure.xlsx",
    )


@router.get("/processes/{process_id}/download/pdf")
async def download_pdf(process_id: str):
    """Download run_summary.pdf (generated on demand)."""
    mgr = get_process_manager()
    path = mgr.get_file_path(process_id, "pdf")
    if path is None:
        raise HTTPException(status_code=404, detail="File not found or results not available")
    return _file_response(path, "application/pdf", "run_summary.pdf")


@router.get("/processes/{process_id}/download/report")
async def download_report_zip(process_id: str):
    """Download ZIP bundle with all output files."""
    mgr = get_process_manager()
    zip_path = mgr.generate_report_zip(process_id)
    if zip_path is None:
        raise HTTPException(status_code=404, detail="No output files available")
    return _file_response(zip_path, "application/zip", "optimizer_report.zip")
