"""Pydantic models for BPMN generation workflow."""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class BPMNGenerationResponse(BaseModel):
    """Response payload returned after requesting BPMN generation."""

    bpmn_xml: str
    provider: str
    model: str
    success: bool
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
