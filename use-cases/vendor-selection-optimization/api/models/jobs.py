"""Job Models for Procurement Assistant API"""

from typing import Dict, List, Optional, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class JobType(str, Enum):
    """Types of jobs"""
    PIPELINE = "pipeline"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    COMPARISON = "comparison"


class JobStatus(str, Enum):
    """Job status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Job model for tracking async operations"""
    job_id: str
    job_type: JobType
    status: JobStatus
    profile_id: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0, le=100)
    
    # Request parameters
    request_params: Dict[str, Any]
    
    # Result information
    result_location: Optional[str] = None
    result_size_bytes: Optional[int] = None
    result_summary: Optional[Dict[str, Any]] = None
    
    # Error information
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    execution_time_seconds: Optional[float] = None
    
    def to_status_response(self) -> Dict[str, Any]:
        """Convert to status response format"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


class JobResult(BaseModel):
    """Job result model"""
    job_id: str
    job_type: JobType
    status: JobStatus
    result_type: Literal["summary", "full", "download"]
    data: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None