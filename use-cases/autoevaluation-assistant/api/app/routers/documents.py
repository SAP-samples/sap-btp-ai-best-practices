"""API routes for Document Manager corpus upload and file management."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.models.documents import (
    DocumentDeleteResponse,
    DocumentIngestionJobResponse,
    DocumentListResponse,
)
from app.routers.ai_review import get_ai_review_repository
from app.security import get_api_key
from app.services.document_corpus import document_upload_payload, validate_document_file_name


router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("", response_model=DocumentIngestionJobResponse)
async def upload_documents(
    assessment_id: Annotated[str, Form()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    files: Annotated[list[UploadFile], File()],
) -> DocumentIngestionJobResponse:
    """Upload documents into the assessment corpus and create an ingestion job.

    Inputs:
        assessment_id: Assessment whose corpus receives the uploaded files.
        files: PDF, DOCX, XLSX, XLSM, or EML documents.
        repository: AI review repository dependency.

    Outputs:
        DocumentIngestionJobResponse: Created ingestion job state.
    """
    documents: list[dict[str, Any]] = []
    for upload in files:
        try:
            file_name = validate_document_file_name(upload.filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        documents.append(
            document_upload_payload(
                file_name=file_name,
                content_type=upload.content_type,
                content=await upload.read(),
            )
        )
    try:
        return repository.create_document_ingestion_job(
            assessment_id=assessment_id,
            documents=documents,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentListResponse:
    """List corpus documents for one assessment.

    Inputs:
        assessment_id: Assessment whose corpus should be listed.
        repository: AI review repository dependency.

    Outputs:
        DocumentListResponse: Document metadata and readiness counts.
    """
    return repository.get_document_list(assessment_id)


@router.get("/ingestion-jobs/{job_id}", response_model=DocumentIngestionJobResponse)
async def get_document_ingestion_job(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentIngestionJobResponse:
    """Return current ingestion progress for one upload job.

    Inputs:
        job_id: Document ingestion job identifier.
        repository: AI review repository dependency.

    Outputs:
        DocumentIngestionJobResponse: Current ingestion progress.
    """
    try:
        return repository.get_document_ingestion_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> Response:
    """Download the original blob for one corpus document.

    Inputs:
        document_id: Corpus document identifier.
        assessment_id: Assessment that owns the document.
        repository: AI review repository dependency.

    Outputs:
        Response: Binary document response with attachment headers.
    """
    try:
        document = repository.get_document_download(
            assessment_id=assessment_id,
            document_id=document_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=document.content,
        media_type=document.content_type,
        headers={"Content-Disposition": f'attachment; filename="{document.file_name}"'},
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentDeleteResponse:
    """Delete a corpus document, extracted text, and embedded chunks.

    Inputs:
        document_id: Corpus document identifier.
        assessment_id: Assessment that owns the document.
        repository: AI review repository dependency.

    Outputs:
        DocumentDeleteResponse: Delete status.
    """
    deleted = repository.delete_document(
        assessment_id=assessment_id,
        document_id=document_id,
    )
    return DocumentDeleteResponse(document_id=document_id, deleted=deleted)
