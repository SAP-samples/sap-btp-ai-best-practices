"""Request/response models for the extraction endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionOptions(BaseModel):
    """Configuration mirrors the CLI flags in ``doc_extraction.main``."""

    llm_verify: bool = True
    llm_model: Optional[str] = None
    llm_min_confidence: float = 0.6
    top_k: int = 5
    merge_headers: bool = False
    output_name: Optional[str] = None

    embedding_model: Optional[str] = None
    show_preview: bool = False


class ExtractionResponse(BaseModel):
    """API response containing extraction metadata and previews."""

    output_path: str = Field(..., description="Location of the generated Excel workbook")
    download_path: str = Field(..., description="Relative path used to download the Excel file")
    output_exists: bool = Field(..., description="Whether the output file is present on disk")
    file_count: int = Field(..., description="Number of PDFs processed")
    llm_verify: bool
    top_k: int
    runtime_seconds: float
    reference_data_version: str = Field(..., description="Validated version of the HANA reference dataset")

    headers_preview: List[Dict[str, Any]]
    line_items_preview: List[Dict[str, Any]]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
