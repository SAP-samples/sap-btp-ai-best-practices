from pydantic import BaseModel
from typing import Optional, Any, Dict
import time


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    service: str
    version: Optional[str] = None

    @classmethod
    def healthy(cls, service: str, version: Optional[str] = None) -> "HealthResponse":
        return cls(status="healthy", timestamp=time.time(), service=service, version=version)
