# Models package for organizing domain-specific Pydantic models
# This allows for better organization as the application grows

# Re-export models for convenience
from .common import ErrorResponse, HealthResponse

__all__ = ["ErrorResponse", "HealthResponse"]
