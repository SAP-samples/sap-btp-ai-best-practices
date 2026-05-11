"""Routes that expose the doc_extraction pipeline via HTTP."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..models.extraction import ExtractionOptions, ExtractionResponse
from ..services.extraction_service import ExtractionConfig, resolve_download_path, run_extraction_pipeline

logger = logging.getLogger(__name__)
_DEFAULT_OPTIONS = ExtractionOptions()

router = APIRouter()


@router.get("/defaults", response_model=ExtractionOptions)
async def get_defaults() -> ExtractionOptions:
    """Expose the default configuration so the UI can prime form fields."""

    return ExtractionOptions()  # type: ignore[arg-type]


@router.post(
    "/run",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def run_extraction(
    files: List[UploadFile] = File(..., description="One or more PDF documents"),
    llm_verify: bool = Form(True, description="Run LLM verification after embeddings"),
    llm_model: Optional[str] = Form(None, description="Override the verification LLM identifier"),
    llm_min_confidence: float = Form(0.6, description="Minimum confidence threshold for LLM verification"),
    top_k: int = Form(5, description="Number of codes to keep per line item"),
    merge_headers: bool = Form(False, description="Merge extracted headers into line items before embeddings"),
    output_name: Optional[str] = Form(None, description="Optional output workbook name (without extension)"),
    embedding_model: Optional[str] = Form(None, description="Embedding model override"),
    show_preview: bool = Form(False, description="Display preview in embedding step (used in notebooks)"),
) -> ExtractionResponse:
    """Execute the full extraction + embedding pipeline for uploaded PDFs."""

    try:
        config = ExtractionConfig(
            llm_verify=llm_verify,
            llm_model=llm_model,
            llm_min_confidence=llm_min_confidence,
            top_k=top_k,
            merge_headers=merge_headers,
            output_name=output_name,
            embedding_model=embedding_model,
            show_preview=show_preview,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Invalid extraction configuration")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        result_payload = await run_extraction_pipeline(upload_files=files, config=config)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during extraction")
        error_message = f"Extraction failed: {exc}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message) from exc

    return ExtractionResponse(**result_payload)


@router.get("/download", response_class=FileResponse)
async def download_output(path: str):
    """Download the generated Excel file by its relative path."""

    try:
        resolved = resolve_download_path(path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path.")

    return FileResponse(resolved, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
