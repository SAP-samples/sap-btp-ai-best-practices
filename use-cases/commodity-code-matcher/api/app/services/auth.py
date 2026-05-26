"""Authentication helpers for FastAPI extraction endpoints."""

from __future__ import annotations

import os
from hmac import compare_digest

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Validate the optional deployment API key for protected routes.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header.

    Raises:
        HTTPException: Returns 401 when the configured key is missing or wrong,
            and 503 when production is configured without an ``API_KEY``.
    """

    configured_key = os.getenv("API_KEY", "").strip()
    app_env = os.getenv("APP_ENV", "").strip().lower()

    if not configured_key:
        if app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured.",
            )
        return

    if not x_api_key or not compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
