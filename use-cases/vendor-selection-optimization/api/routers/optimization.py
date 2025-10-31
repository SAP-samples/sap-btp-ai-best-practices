"""Optimization API Router"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import logging

from ..config import settings
from ..models.requests import (
    RunPipelineRequest,
    EvaluateVendorsRequest,
    OptimizeAllocationRequest,
    ComparePoliciesRequest
)
from ..models.responses import (
    AsyncJobResponse,
    InlineEvaluationResponse,
    OptimizationSummaryResponse,
    SummaryComparisonResponse
)
from ..models.jobs import JobType, JobStatus
from ..services import (
    vendor_evaluator,
    optimizer,
    comparator,
    pipeline_runner,
    job_manager
)
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["optimization"]
)


# Dependency to validate profile exists
def validate_profile(profile_id: str) -> str:
    """Validate that profile exists"""
    if not file_manager.profile_exists(profile_id):
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{profile_id}' not found"
        )
    return profile_id


@router.post("/pipeline", response_model=AsyncJobResponse)
async def run_complete_pipeline(
    request: RunPipelineRequest,
    background_tasks: BackgroundTasks,
    profile_id: str = Depends(validate_profile)
) -> AsyncJobResponse:
    """
    Run the complete procurement optimization pipeline.
    
    This endpoint orchestrates the full optimization workflow:
    1. Vendor evaluation with tariff impact
    2. Procurement optimization using linear programming
    3. Policy comparison between historical and optimized allocation
    
    The operation always runs asynchronously due to computational complexity.
    """
    try:
        # Create job
        job = job_manager.create_job(
            job_type=JobType.PIPELINE,
            profile_id=profile_id,
            request_params=request.model_dump(),
            estimated_duration=300  # 5 minutes estimate
        )
        
        # Schedule pipeline execution
        background_tasks.add_task(
            pipeline_runner.run_pipeline,
            profile_id,
            request,
            job.job_id
        )
        
        return AsyncJobResponse(
            status="accepted",
            job_id=job.job_id,
            estimated_duration_seconds=300,
            result_endpoints=job_manager.get_job_result_endpoints(job.job_id)
        )
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-vendors")
async def evaluate_vendors(
    request: EvaluateVendorsRequest,
    background_tasks: BackgroundTasks,
    profile_id: str = Depends(validate_profile)
):
    """
    Evaluate and rank vendors based on multiple cost factors.
    
    This endpoint calculates effective costs including:
    - Base price
    - Tariff impact
    - Logistics costs
    - Lead time holding costs
    - Risk factors
    
    Returns inline for small datasets, async for large ones.
    """
    try:
        # Check if should run async
        if vendor_evaluator.should_run_async(profile_id, request.filters):
            # Create job
            job = job_manager.create_job(
                job_type=JobType.EVALUATION,
                profile_id=profile_id,
                request_params=request.model_dump(),
                estimated_duration=60
            )
            
            # Schedule async execution
            background_tasks.add_task(
                vendor_evaluator.run_evaluation_async,
                profile_id,
                request,
                job.job_id
            )
            
            return AsyncJobResponse(
                status="accepted",
                job_id=job.job_id,
                estimated_duration_seconds=60,
                result_endpoints=job_manager.get_job_result_endpoints(job.job_id)
            )
        else:
            # Run synchronously for small datasets
            results = vendor_evaluator.run_evaluation(profile_id, request)
            return InlineEvaluationResponse(**results)
            
    except Exception as e:
        logger.error(f"Failed to evaluate vendors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-allocation", response_model=AsyncJobResponse)
async def optimize_procurement_allocation(
    request: OptimizeAllocationRequest,
    background_tasks: BackgroundTasks,
    profile_id: str = Depends(validate_profile)
) -> AsyncJobResponse:
    """
    Optimize procurement allocation using linear programming.
    
    This endpoint finds the optimal supplier allocation that:
    - Minimizes total effective cost
    - Meets all material demands
    - Respects supplier capacity constraints
    - Enforces multi-supplier requirements
    
    Always runs asynchronously due to computational complexity.
    """
    try:
        # Create job
        job = job_manager.create_job(
            job_type=JobType.OPTIMIZATION,
            profile_id=profile_id,
            request_params=request.model_dump(),
            estimated_duration=120
        )
        
        # Schedule optimization
        background_tasks.add_task(
            optimizer.run_optimization,
            profile_id,
            request,
            job.job_id
        )
        
        return AsyncJobResponse(
            status="accepted",
            job_id=job.job_id,
            estimated_duration_seconds=120,
            result_endpoints=job_manager.get_job_result_endpoints(job.job_id)
        )
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-policies", response_model=AsyncJobResponse)
async def compare_procurement_policies(
    request: ComparePoliciesRequest,
    background_tasks: BackgroundTasks,
    profile_id: str = Depends(validate_profile)
) -> AsyncJobResponse:
    """
    Compare historical procurement with optimized allocation.
    
    This endpoint analyzes:
    - Cost savings potential
    - Allocation changes by supplier/material
    - Economic impact breakdown by cost component
    
    Always runs asynchronously due to data processing requirements.
    """
    try:
        # Create job
        job = job_manager.create_job(
            job_type=JobType.COMPARISON,
            profile_id=profile_id,
            request_params=request.model_dump(),
            estimated_duration=60
        )
        
        # Schedule comparison
        background_tasks.add_task(
            comparator.run_comparison,
            profile_id,
            request,
            job.job_id
        )
        
        return AsyncJobResponse(
            status="accepted",
            job_id=job.job_id,
            estimated_duration_seconds=60,
            result_endpoints=job_manager.get_job_result_endpoints(job.job_id)
        )
        
    except Exception as e:
        logger.error(f"Failed to start comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Job Status Endpoints
@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of an async job"""
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
        response["download_url"] = f"/api/optimize/jobs/{job_id}/download"
    
    return response


@router.get("/jobs/{job_id}/download")
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
    
    from pathlib import Path
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


# Result-specific endpoints for different job types
@router.get("/jobs/{job_id}/summary")
async def get_job_summary(job_id: str):
    """Get summary results for optimization, comparison, or pipeline jobs"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    if job.job_type == JobType.OPTIMIZATION:
        summary = optimizer.get_optimization_summary(job_id)
        if summary:
            return OptimizationSummaryResponse(**summary)
    elif job.job_type == JobType.COMPARISON:
        summary = comparator.get_comparison_summary(job_id)
        if summary:
            return SummaryComparisonResponse(**summary)
    elif job.job_type == JobType.PIPELINE:
        summary = pipeline_runner.get_pipeline_summary(job_id)
        if summary:
            return summary
    
    return job.result_summary or {"message": "No summary available"}


@router.get("/jobs/{job_id}/allocations")
async def get_optimization_allocations(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    material_filter: Optional[str] = Query(None, description="Filter by material ID"),
    supplier_filter: Optional[str] = Query(None, description="Filter by supplier ID")
):
    """Get paginated allocation details for an optimization job"""
    job = job_manager.get_job(job_id)
    if not job or job.job_type != JobType.OPTIMIZATION:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization job '{job_id}' not found"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    results = optimizer.get_allocation_details(
        job_id=job_id,
        page=page,
        page_size=page_size,
        material_filter=material_filter,
        supplier_filter=supplier_filter
    )
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No allocation details available"
        )
    
    return results


@router.get("/jobs/{job_id}/comparisons")
async def get_comparison_details(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    supplier_filter: Optional[str] = Query(None, description="Filter by supplier ID"),
    material_filter: Optional[str] = Query(None, description="Filter by material ID"),
    change_type: Optional[str] = Query(None, enum=["increased", "decreased", "new", "removed"], 
                                       description="Filter by change type")
):
    """Get paginated comparison details for a comparison job"""
    job = job_manager.get_job(job_id)
    if not job or job.job_type != JobType.COMPARISON:
        raise HTTPException(
            status_code=404,
            detail=f"Comparison job '{job_id}' not found"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    results = comparator.get_comparison_details(
        job_id=job_id,
        page=page,
        page_size=page_size,
        supplier_filter=supplier_filter,
        material_filter=material_filter,
        change_type=change_type
    )
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No comparison details available"
        )
    
    return results


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check optimization service health"""
    return {
        "status": "healthy",
        "service": "optimization",
        "version": settings.API_VERSION
    }