"""Common Pydantic models used across multiple endpoints."""

from pydantic import BaseModel
from typing import Optional, Any, Dict
import time


class ErrorResponse(BaseModel):
    """Standard error response model.

    Attributes:
        error: Human-readable error message
        code: Optional error code for programmatic handling
        details: Optional additional error details
    """

    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Service health status ("healthy", "degraded", "unhealthy")
        timestamp: Current Unix timestamp
        service: Service identifier
        version: Optional service version
    """

    status: str
    timestamp: float
    service: str
    version: Optional[str] = None

    @classmethod
    def healthy(cls, service: str, version: Optional[str] = None) -> "HealthResponse":
        """Create a healthy response."""
        return cls(
            status="healthy", timestamp=time.time(), service=service, version=version
        )


class PaginationResponse(BaseModel):
    """Standard pagination metadata.

    Attributes:
        page: Current page number (1-based)
        size: Number of items per page
        total: Total number of items
        pages: Total number of pages
    """

    page: int
    size: int
    total: int
    pages: int
