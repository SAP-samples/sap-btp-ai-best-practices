"""Models for the Document Manager HANA-backed corpus."""

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """One uploaded document shown in the Document Manager.

    Inputs:
        Field values loaded from the document corpus tables.

    Outputs:
        API-safe metadata for uploaded documents without blob content.
    """

    document_id: str
    assessment_id: str
    file_name: str
    content_type: str
    content_hash: str
    status: str
    chunk_count: int = 0
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class DocumentListResponse(BaseModel):
    """Documents and aggregate corpus readiness for one assessment.

    Inputs:
        Documents returned by the repository for one assessment.

    Outputs:
        List payload used by the Document Manager and Assessment pages.
    """

    assessment_id: str
    documents: list[DocumentSummary] = Field(default_factory=list)
    indexed_document_count: int = 0
    total_document_count: int = 0
    indexed_chunk_count: int = 0


class DocumentIngestionJobResponse(BaseModel):
    """Created or polled document ingestion job state.

    Inputs:
        Job fields persisted by the repository and worker.

    Outputs:
        Status payload used by upload and polling clients.
    """

    job_id: str
    assessment_id: str
    status: str
    document_count: int
    processed_document_count: int = 0
    indexed_chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None


class DocumentDownload(BaseModel):
    """Internal document download payload.

    Inputs:
        Repository row containing a file blob and metadata.

    Outputs:
        Binary content plus response headers for FastAPI routes.
    """

    document_id: str
    file_name: str
    content_type: str
    content: bytes


class DocumentDeleteResponse(BaseModel):
    """Response returned after a document delete request.

    Inputs:
        Delete result computed by the repository.

    Outputs:
        Client-visible deletion status.
    """

    document_id: str
    deleted: bool
