"""Routes that expose the doc_extraction pipeline via HTTP."""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from ..models.extraction import ExtractionJobStatusResponse, ExtractionJobSubmitResponse, ExtractionOptions
from ..services.auth import require_api_key
from ..services.extraction_jobs import (
    ExtractionJobManager,
    HanaExtractionJobRepository,
    JobFilePayload,
    JobNotFoundError,
    JobResultNotReadyError,
    JobStorageError,
    QueueFullError,
)
from ..services.extraction_service import ExtractionConfig

logger = logging.getLogger(__name__)
_DEFAULT_OPTIONS = ExtractionOptions()
_JOB_MANAGER: ExtractionJobManager | None = None
_JOB_MANAGER_LOCK = threading.Lock()

router = APIRouter(dependencies=[Depends(require_api_key)])


def get_job_manager() -> ExtractionJobManager:
    """Return the process-wide in-process extraction job manager.

    Returns:
        A lazily initialized manager backed by HANA job tables.
    """

    global _JOB_MANAGER
    if _JOB_MANAGER is not None:
        return _JOB_MANAGER
    with _JOB_MANAGER_LOCK:
        if _JOB_MANAGER is None:
            repository = HanaExtractionJobRepository()
            _JOB_MANAGER = ExtractionJobManager(
                repository=repository,
                max_workers=int(os.getenv("EXTRACTION_JOB_WORKERS", "1")),
                max_queued_jobs=int(os.getenv("EXTRACTION_MAX_QUEUED_JOBS", "20")),
            )
    return _JOB_MANAGER


async def _read_uploads(files: List[UploadFile]) -> list[JobFilePayload]:
    """Read and validate uploaded PDF files for job submission.

    Args:
        files: FastAPI upload objects from the multipart request.

    Returns:
        Job file payloads containing file names, MIME types, and bytes.

    Raises:
        HTTPException: Returns 400 when files are missing or not PDFs.
    """

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one PDF must be uploaded.")

    payloads: list[JobFilePayload] = []
    for upload in files:
        filename = upload.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type for '{filename}'. Only PDF is allowed.",
            )
        payloads.append(
            JobFilePayload(
                filename=filename,
                content_type=upload.content_type or "application/pdf",
                content=await upload.read(),
            )
        )
    return payloads


@router.get("/defaults", response_model=ExtractionOptions)
async def get_defaults() -> ExtractionOptions:
    """Expose the default configuration so the UI can prime form fields."""

    return ExtractionOptions()  # type: ignore[arg-type]


@router.post(
    "/run",
    response_model=ExtractionJobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
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
) -> ExtractionJobSubmitResponse:
    """Submit an extraction + embedding job for uploaded PDFs."""

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
        payloads = await _read_uploads(files)
        return get_job_manager().submit(files=payloads, config=config)
    except QueueFullError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except JobStorageError as exc:
        logger.exception("Extraction job storage unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during extraction job submission")
        error_message = f"Extraction job submission failed: {exc}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message) from exc


@router.get("/jobs/{job_id}", response_model=ExtractionJobStatusResponse)
async def get_job_status(job_id: str) -> ExtractionJobStatusResponse:
    """Return current status and result metadata for an extraction job."""

    try:
        return get_job_manager().get_status(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    except JobStorageError as exc:
        logger.exception("Extraction job storage unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.get("/jobs/{job_id}/download")
async def download_job_output(job_id: str) -> Response:
    """Download the generated Excel workbook for a completed extraction job."""

    try:
        result = get_job_manager().get_result_file(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    except JobResultNotReadyError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job result is not ready.")
    except JobStorageError as exc:
        logger.exception("Extraction job storage unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    headers = {"Content-Disposition": f'attachment; filename="{result.filename}"'}
    return Response(content=result.content, media_type=result.content_type, headers=headers)
