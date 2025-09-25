# Models package for organizing domain-specific Pydantic models
# This allows for better organization as the application grows

# Re-export models for convenience
from .common import ErrorResponse, HealthResponse
from .match import MatchRequest, MatchResponse, MatchPerClient

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "MatchRequest",
    "MatchResponse",
    "MatchPerClient",
]
