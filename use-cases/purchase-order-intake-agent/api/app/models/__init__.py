"""Pydantic request/response models, grouped by domain."""

from .common import ErrorResponse, HealthResponse

__all__ = ["ErrorResponse", "HealthResponse"]
