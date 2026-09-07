"""API routes for global admin documents used by Joule RAG."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
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
async def upload_admin_documents(
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    files: Annotated[list[UploadFile], File()],
) -> DocumentIngestionJobResponse:
    """Upload documents into the global admin corpus and create a job.

    Inputs:
        files: PDF, DOCX, XLSX, XLSM, or EML documents.
        repository: AI review repository dependency.

    Outputs:
        DocumentIngestionJobResponse: Created admin ingestion job state.
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
        return repository.create_admin_document_ingestion_job(documents=documents)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=DocumentListResponse)
async def list_admin_documents(
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentListResponse:
    """List global admin corpus documents.

    Inputs:
        repository: AI review repository dependency.

    Outputs:
        DocumentListResponse: Admin document metadata and readiness counts.
    """
    return repository.get_admin_document_list()


@router.get("/ingestion-jobs/{job_id}", response_model=DocumentIngestionJobResponse)
async def get_admin_document_ingestion_job(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentIngestionJobResponse:
    """Return current ingestion progress for one admin upload job.

    Inputs:
        job_id: Admin document ingestion job identifier.
        repository: AI review repository dependency.

    Outputs:
        DocumentIngestionJobResponse: Current ingestion progress.
    """
    try:
        return repository.get_admin_document_ingestion_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{document_id}/download")
async def download_admin_document(
    document_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> Response:
    """Download the original blob for one admin corpus document.

    Inputs:
        document_id: Admin corpus document identifier.
        repository: AI review repository dependency.

    Outputs:
        Response: Binary document response with attachment headers.
    """
    try:
        document = repository.get_admin_document_download(document_id=document_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=document.content,
        media_type=document.content_type,
        headers={"Content-Disposition": f'attachment; filename="{document.file_name}"'},
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_admin_document(
    document_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> DocumentDeleteResponse:
    """Delete an admin corpus document and its extracted/vector rows.

    Inputs:
        document_id: Admin corpus document identifier.
        repository: AI review repository dependency.

    Outputs:
        DocumentDeleteResponse: Delete status.
    """
    deleted = repository.delete_admin_document(document_id=document_id)
    return DocumentDeleteResponse(document_id=document_id, deleted=deleted)
