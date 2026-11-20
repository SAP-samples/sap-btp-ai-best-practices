"""Safe provider-failure classification and bounded retry metadata extraction."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass


PROVIDER_ERROR_CODES = frozenset(
    {
        "PROVIDER_AUTH_FAILED",
        "PROVIDER_FORBIDDEN",
        "MODEL_NOT_AVAILABLE",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_SERVER_ERROR",
        "PROVIDER_RESPONSE_INVALID",
    }
)
"""Allowlisted error categories that may cross the provider boundary."""


@dataclass(frozen=True)
class ProviderFailure:
    """Carry only bounded, non-secret diagnostics from one provider exception.

    Args:
        error_code: Stable application error category.
        http_status: Safe HTTP status when the SDK exposes one.
        provider_code: Bounded machine code returned by the provider.
        request_id: Bounded request identifier suitable for support correlation.
        retry_after_seconds: Parsed Retry-After delay when it is a positive integer.
        retryable: Whether policy permits another attempt for this failure.
    """

    error_code: str
    http_status: int | None
    provider_code: str | None
    request_id: str | None
    retry_after_seconds: int | None
    retryable: bool


def _bounded(value: object, maximum: int = 200) -> str | None:
    """Return trimmed metadata only when it fits its persistence boundary."""
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered[:maximum]


def _response_for_exception(error: BaseException) -> object | None:
    """Return the SDK response object without inspecting exception text."""
    return getattr(error, "response", None)


def _status_code(response: object | None) -> int | None:
    """Read an integer HTTP status from a response mapping or object."""
    if response is None:
        return None
    value = (
        response.get("status_code")
        if isinstance(response, Mapping)
        else getattr(response, "status_code", None)
    )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _headers(response: object | None) -> Mapping[str, object]:
    """Return case-normalizable response headers or an empty mapping."""
    if response is None:
        return {}
    value = (
        response.get("headers")
        if isinstance(response, Mapping)
        else getattr(response, "headers", None)
    )
    return value if isinstance(value, Mapping) else {}


def _header(headers: Mapping[str, object], name: str) -> object | None:
    """Read one response header case-insensitively."""
    expected = name.casefold()
    return next(
        (value for key, value in headers.items() if str(key).casefold() == expected),
        None,
    )


def _provider_code(response: object | None) -> str | None:
    """Extract only the machine error code from a JSON response body."""
    if response is None:
        return None
    try:
        payload = (
            response.get("body")
            if isinstance(response, Mapping)
            else response.json()
        )
    except Exception:  # noqa: BLE001 - malformed provider bodies are ignored.
        return None
    if not isinstance(payload, Mapping):
        return None
    error_payload = payload.get("error")
    if isinstance(error_payload, Mapping):
        return _bounded(error_payload.get("code"), 120)
    return _bounded(payload.get("code"), 120)


def _retry_after_seconds(headers: Mapping[str, object]) -> int | None:
    """Parse a bounded integer Retry-After value and ignore HTTP dates."""
    value = _header(headers, "retry-after")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 86_400 else None


def classify_provider_failure(error: BaseException) -> ProviderFailure:
    """Classify one exception without retaining its message, URL, or body.

    Args:
        error: Provider or transport exception raised by the native client.

    Returns:
        An allowlisted error category with safe correlation metadata.
    """
    response = _response_for_exception(error)
    status = _status_code(response)
    headers = _headers(response)
    provider_code = _provider_code(response)
    lowered_code = provider_code.casefold() if provider_code else ""

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)) or "timeout" in type(
        error
    ).__name__.casefold():
        error_code = "PROVIDER_TIMEOUT"
    elif status == 401:
        error_code = "PROVIDER_AUTH_FAILED"
    elif status == 403:
        error_code = "PROVIDER_FORBIDDEN"
    elif status == 404 or (
        status == 400
        and any(value in lowered_code for value in ("deployment", "model", "notfound"))
    ):
        error_code = "MODEL_NOT_AVAILABLE"
    elif status == 429:
        error_code = "PROVIDER_RATE_LIMITED"
    elif status is not None and 500 <= status <= 599:
        error_code = "PROVIDER_SERVER_ERROR"
    else:
        error_code = "PROVIDER_RESPONSE_INVALID"

    retryable = error_code in {
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_SERVER_ERROR",
    }
    return ProviderFailure(
        error_code=error_code,
        http_status=status,
        provider_code=provider_code,
        request_id=_bounded(
            _header(headers, "x-request-id")
            or _header(headers, "x-correlation-id"),
            200,
        ),
        retry_after_seconds=_retry_after_seconds(headers),
        retryable=retryable,
    )
