"""Job Management Service for tracking async operations"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..config import settings
from ..models.jobs import Job, JobStatus, JobType, JobResult

logger = logging.getLogger(__name__)


class JobManager:
    """Manages job creation, updates, and retrieval"""
    
    def __init__(self):
        self.storage_path = settings.JOB_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = settings.JOB_TTL_HOURS
        
    def _get_job_file_path(self, job_id: str) -> Path:
        """Get the file path for a job"""
        return self.storage_path / f"{job_id}.json"
    
    def _save_job(self, job: Job) -> None:
        """Save job to storage"""
        job_file = self._get_job_file_path(job.job_id)
        job_data = job.model_dump_json(indent=2)
        job_file.write_text(job_data)
        
    def _load_job(self, job_id: str) -> Optional[Job]:
        """Load job from storage"""
        job_file = self._get_job_file_path(job_id)
        if not job_file.exists():
            return None
        
        try:
            job_data = json.loads(job_file.read_text())
            # Convert string dates back to datetime objects
            for date_field in ['created_at', 'updated_at', 'started_at', 'completed_at']:
                if job_data.get(date_field):
                    job_data[date_field] = datetime.fromisoformat(job_data[date_field])
            return Job(**job_data)
        except Exception as e:
            logger.error(f"Error loading job {job_id}: {e}")
            return None
    
    def create_job(
        self,
        job_type: JobType,
        profile_id: str,
        request_params: Dict[str, Any],
        estimated_duration: int = 60
    ) -> Job:
        """Create a new job"""
        job_id = f"{job_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            profile_id=profile_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            request_params=request_params
        )
        
        self._save_job(job)
        logger.info(f"Created job {job_id} of type {job_type.value}")
        
        return job
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """Update job status"""
        job = self._load_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for status update")
            return None
        
        job.status = status
        job.updated_at = datetime.now()
        
        if progress is not None:
            job.progress = progress
            
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.now()
            
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = datetime.now()
            if job.started_at:
                job.execution_time_seconds = (job.completed_at - job.started_at).total_seconds()
                
        if error:
            job.error = error
            job.error_details = error_details or {}
        
        self._save_job(job)
        logger.info(f"Updated job {job_id} status to {status.value}")
        
        return job
    
    def update_job_result(
        self,
        job_id: str,
        result_location: str,
        result_size_bytes: int,
        result_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """Update job with result information"""
        job = self._load_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for result update")
            return None
        
        job.result_location = result_location
        job.result_size_bytes = result_size_bytes
        job.result_summary = result_summary or {}
        job.updated_at = datetime.now()
        
        self._save_job(job)
        logger.info(f"Updated job {job_id} with result location: {result_location}")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        return self._load_job(job_id)
    
    def get_job_result_endpoints(self, job_id: str) -> Dict[str, str]:
        """Get result endpoints for a job"""
        base_path = f"/api/jobs/{job_id}"
        endpoints = {
            "status": f"{base_path}/status",
            "download": f"{base_path}/download"
        }
        
        job = self._load_job(job_id)
        if job and job.job_type == JobType.EVALUATION:
            endpoints["results"] = f"{base_path}/results"
        elif job and job.job_type == JobType.PIPELINE:
            endpoints["vendor_evaluation_csv"] = f"{base_path}/download/vendor-evaluation"
            endpoints["optimization_csv"] = f"{base_path}/download/optimization"
            endpoints["comparison_csv"] = f"{base_path}/download/comparison"
            endpoints["all_results_zip"] = f"{base_path}/download/all"
        elif job and job.job_type == JobType.OPTIMIZATION:
            endpoints["summary"] = f"{base_path}/summary"
            endpoints["allocations"] = f"{base_path}/allocations"
        elif job and job.job_type == JobType.COMPARISON:
            endpoints["summary"] = f"{base_path}/summary"
            
        return endpoints
    
    def clean_expired_jobs(self) -> int:
        """Clean up expired jobs"""
        count = 0
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        
        for job_file in self.storage_path.glob("*.json"):
            try:
                job = self._load_job(job_file.stem)
                if job and job.created_at < cutoff_time:
                    # Delete job file and any associated result files
                    job_file.unlink()
                    if job.result_location:
                        result_path = Path(job.result_location)
                        if result_path.exists():
                            result_path.unlink()
                    count += 1
                    logger.info(f"Cleaned up expired job: {job.job_id}")
            except Exception as e:
                logger.error(f"Error cleaning up job file {job_file}: {e}")
                
        return count
    
    def list_jobs(
        self,
        profile_id: Optional[str] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filters"""
        jobs = []
        
        for job_file in sorted(self.storage_path.glob("*.json"), reverse=True)[:limit * 2]:
            job = self._load_job(job_file.stem)
            if not job:
                continue
                
            # Apply filters
            if profile_id and job.profile_id != profile_id:
                continue
            if job_type and job.job_type != job_type:
                continue
            if status and job.status != status:
                continue
                
            jobs.append(job)
            
            if len(jobs) >= limit:
                break
                
        return jobs


# Global job manager instance
job_manager = JobManager()