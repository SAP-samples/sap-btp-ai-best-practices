"""Jobs Management API Router"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
import logging
from pathlib import Path
import zipfile
import io

from ..config import settings
from ..models.jobs import JobType, JobStatus, Job
# Import removed - JobStatusResponse not in responses.py
from ..services import job_manager
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["jobs"]
)


# Health check endpoint - must be defined before /{job_id} to avoid path conflict
@router.get("/health")
async def health_check():
    """Check jobs service health"""
    return {
        "status": "healthy",
        "service": "jobs",
        "version": settings.API_VERSION
    }


@router.get("/")
async def list_jobs(
    profile_id: Optional[str] = Query(None, description="Filter by profile ID"),
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return")
):
    """
    List recent jobs with optional filters.
    
    Returns jobs sorted by creation time (newest first).
    """
    try:
        jobs = job_manager.list_jobs(
            profile_id=profile_id,
            job_type=job_type,
            status=status,
            limit=limit
        )
        
        return [
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "error_details": job.error_details,
                "result_summary": job.result_summary if job.status == JobStatus.COMPLETED else None,
                "result_location": job.result_location if job.status == JobStatus.COMPLETED else None,
                "result_size_bytes": job.result_size_bytes if job.status == JobStatus.COMPLETED else None,
                "download_url": f"/api/jobs/{job.job_id}/download" if job.status == JobStatus.COMPLETED else None
            }
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Get detailed information about a specific job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at
    }
    
    if job.status == JobStatus.FAILED:
        response["error"] = job.error
        response["error_details"] = job.error_details
    
    if job.status == JobStatus.COMPLETED:
        response["result_summary"] = job.result_summary
        response["result_location"] = job.result_location
        response["result_size_bytes"] = job.result_size_bytes
        response["download_url"] = f"/api/jobs/{job_id}/download"
    
    return response


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'"
        )
    
    # Update job status to cancelled
    job_manager.update_job_status(job_id, JobStatus.CANCELLED)
    
    return {
        "status": "success",
        "message": f"Job '{job_id}' cancelled",
        "job_id": job_id
    }


@router.get("/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the result file for a completed job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    if not job.result_location:
        raise HTTPException(
            status_code=404,
            detail="No result file available for this job"
        )
    
    result_path = Path(job.result_location)
    
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Result file no longer exists"
        )
    
    return FileResponse(
        path=result_path,
        filename=result_path.name,
        media_type="text/csv" if result_path.suffix == ".csv" else "application/json"
    )


@router.get("/{job_id}/download/vendor-evaluation")
async def download_vendor_evaluation_csv(job_id: str):
    """Download the vendor evaluation CSV for a pipeline job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.job_type != JobType.PIPELINE:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available for pipeline jobs"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    csv_path = file_manager.get_result_file_path(job_id, "vendor_evaluation.csv")
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Vendor evaluation CSV not found"
        )
    
    return FileResponse(
        path=csv_path,
        filename="vendor_evaluation.csv",
        media_type="text/csv"
    )


@router.get("/{job_id}/download/optimization")
async def download_optimization_csv(job_id: str):
    """Download the optimization allocation CSV for a pipeline job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.job_type != JobType.PIPELINE:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available for pipeline jobs"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    csv_path = file_manager.get_result_file_path(job_id, "optimization_allocation.csv")
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Optimization allocation CSV not found"
        )
    
    return FileResponse(
        path=csv_path,
        filename="optimization_allocation.csv",
        media_type="text/csv"
    )


@router.get("/{job_id}/download/comparison")
async def download_comparison_csv(job_id: str):
    """Download the comparison CSV for a pipeline job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.job_type != JobType.PIPELINE:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available for pipeline jobs"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    csv_path = file_manager.get_result_file_path(job_id, "comparison.csv")
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Comparison CSV not found"
        )
    
    return FileResponse(
        path=csv_path,
        filename="comparison.csv",
        media_type="text/csv"
    )


@router.get("/{job_id}/download/all")
async def download_all_results(job_id: str):
    """Download all pipeline results as a ZIP archive"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.job_type != JobType.PIPELINE:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available for pipeline jobs"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add pipeline summary JSON
        summary_path = file_manager.get_result_file_path(job_id, "pipeline_summary.json")
        if summary_path.exists():
            zip_file.write(summary_path, "pipeline_summary.json")
        
        # Add vendor evaluation CSV
        vendor_csv = file_manager.get_result_file_path(job_id, "vendor_evaluation.csv")
        if vendor_csv.exists():
            zip_file.write(vendor_csv, "vendor_evaluation.csv")
        
        # Add optimization CSV
        opt_csv = file_manager.get_result_file_path(job_id, "optimization_allocation.csv")
        if opt_csv.exists():
            zip_file.write(opt_csv, "optimization_allocation.csv")
        
        # Add comparison CSV
        comp_csv = file_manager.get_result_file_path(job_id, "comparison.csv")
        if comp_csv.exists():
            zip_file.write(comp_csv, "comparison.csv")
    
    # Prepare ZIP for download
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.getvalue()),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=pipeline_results_{job_id}.zip"
        }
    )


@router.post("/cleanup")
async def cleanup_expired_jobs():
    """Clean up expired jobs and their results"""
    try:
        count = job_manager.clean_expired_jobs()
        return {
            "status": "success",
            "message": f"Cleaned up {count} expired jobs",
            "jobs_cleaned": count
        }
    except Exception as e:
        logger.error(f"Failed to clean up jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))