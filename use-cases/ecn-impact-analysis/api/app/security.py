"""Constant-time API-key authentication for protected business routes."""

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str | None = Security(API_KEY_HEADER)) -> str:
    """Validate `X-API-Key` in constant time and return the accepted key."""
    configured_key = os.getenv("API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )

    if api_key_header is not None and secrets.compare_digest(
        api_key_header, configured_key
    ):
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )
