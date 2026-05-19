"""Common Pydantic models shared across API endpoints."""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    timestamp: float
    service: str
    version: Optional[str] = None

    @classmethod
    def healthy(cls, service: str, version: Optional[str] = None) -> "HealthResponse":
        return cls(status="healthy", timestamp=time.time(), service=service, version=version)

