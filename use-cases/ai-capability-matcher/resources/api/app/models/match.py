"""
Pydantic models for CSV-to-CSV matching API.

These models define the request and response payloads for the comparator
endpoint that matches client products to AI catalog products using embeddings
and optionally LLM-based reasoning.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    """Request payload for the matching endpoint.

    The frontend UI is expected to read the CSVs and send rows as dictionaries,
    along with which columns to use for text generation and a display column
    for result naming.
    """

    ai_rows: List[Dict[str, Any]] = Field(..., description="List of AI catalog rows (dicts).")
    client_rows: List[Dict[str, Any]] = Field(..., description="List of client rows (dicts).")
    selected_ai_columns: List[str] = Field(..., description="Columns from AI catalog used to build text.")
    selected_client_columns: List[str] = Field(..., description="Columns from client data used to build text.")
    matching_column: Optional[str] = Field(
        default=None, description="Column in the AI catalog to show as the product name in results."
    )
    num_matches: int = Field(5, ge=1, le=10, description="Number of ranked matches to return per client row.")
    batch_size: int = Field(5, ge=1, le=10, description="Batch size for LLM calls.")
    use_llm: bool = Field(True, description="If true, request LLM reasoning and ranking.")
    batch_system_prompt: Optional[str] = Field(
        default=None, description="Optional custom system prompt for LLM batch matching."
    )


class MatchPerClient(BaseModel):
    """Results for a single client row, flattened for easy merging in UI."""

    results: Dict[str, Optional[str]]


class MatchResponse(BaseModel):
    """Response payload for the matching endpoint."""

    success: bool
    message: Optional[str] = None
    model: Optional[str] = Field(default=None, description="Optional LLM model used (if any).")
    result_columns: List[str] = Field(
        ..., description="Column headers included in each per-row result dictionary."
    )
    matches: List[MatchPerClient] = Field(
        ..., description="Per-client results aligned to the order of client_rows in the request."
    )


