"""Safe public errors for transient drawing-feasibility assessments."""

from __future__ import annotations

from app.ecn.errors import EcnImpactError, SourceValidationError


class DrawingValidationError(SourceValidationError):
    """Return a safe HTTP-400 validation error for one drawing upload."""


class DrawingUploadTooLargeError(EcnImpactError):
    """Return a drawing-specific safe HTTP-413 error without source content."""

    def __init__(self, maximum_bytes: int) -> None:
        """Create the configured-size-limit error."""
        super().__init__(
            "DRAWING_FILE_TOO_LARGE",
            "Drawing file exceeds the configured size limit",
            status_code=413,
            details={"maximum_bytes": maximum_bytes},
        )


class DrawingServiceUnavailableError(EcnImpactError):
    """Return safe availability information without exposing provider details."""

    def __init__(self, code: str = "DRAWING_MODEL_UNAVAILABLE") -> None:
        """Create a generic unavailable response."""
        super().__init__(
            code,
            "Drawing feasibility analysis is not configured or currently unavailable",
            status_code=503,
        )


class DrawingAnalysisError(EcnImpactError):
    """Return a safe assessment failure without emitting drawing or model content."""

    def __init__(
        self,
        code: str = "DRAWING_ANALYSIS_FAILED",
        *,
        status_code: int = 502,
    ) -> None:
        """Create a generic assessment failure with a safe status code."""
        super().__init__(
            code,
            "Drawing feasibility analysis could not be completed",
            status_code=status_code,
        )
