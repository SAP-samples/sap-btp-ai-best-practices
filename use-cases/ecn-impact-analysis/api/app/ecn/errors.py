"""Safe typed error hierarchy for the ECN impact pipeline.

All errors carry a ``code`` (a stable machine-readable slug), a
``public_message`` (safe for API responses), an HTTP ``status_code``, and
optional structured ``details``.

Sensitive source content must never be embedded in error messages or
``details``; callers are responsible for keeping raw content out of these
objects before raising.
"""

from typing import Literal

from app.ecn.models import JsonValue


class EcnImpactError(Exception):
    """Expose a stable safe error without embedding source content.

    This is the root of the ECN impact error hierarchy.  All sub-classes
    specialize the HTTP status code or add invariant checks.

    Args:
        code: A stable, machine-readable slug identifying the error kind
            (e.g. ``"ARTIFACT_TOO_LARGE"``).
        public_message: A human-readable message that is safe to surface in
            API responses and logs.
        status_code: The HTTP status code that best describes the error.
        details: Optional structured key/value details for diagnostic use;
            must not contain source content.
    """

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        status_code: int,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        """Initialize the safe public error fields documented by this class."""
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.status_code = status_code
        self.details = details


class SourceValidationError(EcnImpactError):
    """Represent malformed or unacceptable source content with HTTP status 400.

    The ``status_code`` parameter is typed as ``Literal[400]`` and enforced
    at runtime so that callers cannot accidentally use a different status code
    for validation errors.

    Args:
        code: A stable error code slug.
        public_message: A human-readable message safe for API responses.
        status_code: Must be ``400``; any other value raises ``ValueError``.
        details: Optional structured diagnostic details.

    Raises:
        ValueError: If ``status_code`` is not ``400``.
    """

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        status_code: Literal[400] = 400,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        """Initialize a source-validation error and enforce HTTP status 400."""
        if status_code != 400:
            raise ValueError("source validation errors must use HTTP status 400")
        super().__init__(code, public_message, status_code=400, details=details)


class StartupConfigurationError(EcnImpactError):
    """Represent missing credentials or an incompatible persistence contract.

    Raised during application startup when required environment variables,
    database credentials, or schema versions are absent or incompatible.
    """


class PersistenceStateError(EcnImpactError):
    """Represent an immutable/state-machine conflict with HTTP status 409.

    Raised when a caller attempts to modify an immutable record or to
    transition a job through an invalid state.
    """


class PersistenceNotFoundError(EcnImpactError):
    """Represent a missing repository object with HTTP status 404.

    Raised when a requested upload, artifact, ECN row, or analysis job
    does not exist in the persistence layer.
    """


class UploadTooLargeError(EcnImpactError):
    """Represent a byte stream that crosses its configured upload limit.

    The ``status_code`` is fixed to ``413`` and enforced at runtime so that
    callers cannot accidentally use a different status for size-limit errors.

    Args:
        code: Must be ``"UPLOAD_TOO_LARGE"``.
        public_message: A human-readable message safe for API responses.
        status_code: Must be ``413``; any other value raises ``ValueError``.
        details: Optional structured diagnostic details (e.g. maximum_bytes).

    Raises:
        ValueError: If ``status_code`` is not ``413``.
    """

    def __init__(
        self,
        code: Literal["UPLOAD_TOO_LARGE"] = "UPLOAD_TOO_LARGE",
        public_message: str = "Uploaded file exceeds its configured size limit",
        *,
        status_code: Literal[413] = 413,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        """Initialize an upload-size error and enforce HTTP status 413."""
        if status_code != 413:
            raise ValueError("upload size errors must use HTTP status 413")
        super().__init__(code, public_message, status_code=413, details=details)
