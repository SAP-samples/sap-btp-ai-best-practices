"""Admin document corpus persistence for the in-memory repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from app.models.ai_review import (
    BatchReviewJobResponse,
    QuestionReviewResult,
    ReviewJobResponse,
    ReviewJobStatusResponse,
    ReviewResetResponse,
    ReviewTaskStatus,
)
from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.models.documents import (
    DocumentDownload,
    DocumentIngestionJobResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.models.language import DEFAULT_LANGUAGE, normalize_language
from app.services.document_extractors import ExtractedDocument

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    document_content_hash,
)

class MemoryAdminDocumentsMixin:
    """Store and search shared admin document corpus rows in memory."""

    def create_admin_document_ingestion_job(
        self,
        documents: list[dict[str, Any]],
    ) -> DocumentIngestionJobResponse:
        """Store global admin documents and create one ingestion job."""
        if not documents:
            raise ValueError("Admin document ingestion requires at least one document.")
        job_id = f"admin-doc-job-{uuid4()}"
        existing_by_hash = {
            document["content_hash"]: document_id
            for document_id, document in self.admin_documents.items()
        }
        linked_document_ids: list[str] = []
        for payload in documents:
            content = _document_payload_bytes(payload.get("content", b""))
            content_hash = document_content_hash(content)
            document_id = existing_by_hash.get(content_hash)
            if document_id is None:
                document_id = f"admin-document-{uuid4()}"
                self.admin_documents[document_id] = {
                    "document_id": document_id,
                    "assessment_id": ADMIN_DOCUMENT_CORPUS_ID,
                    "content_hash": content_hash,
                    "file_name": payload["file_name"],
                    "content_type": payload.get(
                        "content_type",
                        "application/octet-stream",
                    ),
                    "content": content,
                    "status": "pending",
                    "error_message": None,
                    "created_at": str(time.time()),
                    "updated_at": str(time.time()),
                }
                existing_by_hash[content_hash] = document_id
            else:
                self.admin_documents[document_id]["status"] = "pending"
                self.admin_documents[document_id]["error_message"] = None
                self.admin_documents[document_id]["updated_at"] = str(time.time())
            if document_id not in linked_document_ids:
                linked_document_ids.append(document_id)
        self.admin_document_ingestion_jobs[job_id] = {
            "job_id": job_id,
            "assessment_id": ADMIN_DOCUMENT_CORPUS_ID,
            "status": "pending",
            "lease_owner": None,
            "document_count": len(linked_document_ids),
            "processed_document_count": 0,
            "indexed_chunk_count": 0,
            "error_code": None,
            "error_message": None,
        }
        self.admin_document_ingestion_job_documents[job_id] = linked_document_ids
        return DocumentIngestionJobResponse(
            job_id=job_id,
            assessment_id=ADMIN_DOCUMENT_CORPUS_ID,
            status="pending",
            document_count=len(linked_document_ids),
        )

    def list_admin_documents(self) -> list[DocumentSummary]:
        """Return global admin document metadata."""
        summaries: list[DocumentSummary] = []
        for document in self.admin_documents.values():
            chunk_count = sum(
                1
                for chunk in self.admin_document_chunks
                if chunk.get("document_id") == document["document_id"]
            )
            summaries.append(
                DocumentSummary(
                    document_id=document["document_id"],
                    assessment_id=ADMIN_DOCUMENT_CORPUS_ID,
                    file_name=document["file_name"],
                    content_type=document["content_type"],
                    content_hash=document["content_hash"],
                    status=document["status"],
                    error_message=document.get("error_message"),
                    chunk_count=chunk_count,
                    created_at=document.get("created_at"),
                    updated_at=document.get("updated_at"),
                )
            )
        return summaries

    def get_admin_document_list(self) -> DocumentListResponse:
        """Return admin document metadata plus aggregate readiness counts."""
        documents = self.list_admin_documents()
        return DocumentListResponse(
            assessment_id=ADMIN_DOCUMENT_CORPUS_ID,
            documents=documents,
            indexed_document_count=sum(1 for document in documents if document.status == "indexed"),
            total_document_count=len(documents),
            indexed_chunk_count=sum(document.chunk_count for document in documents),
        )

    def get_admin_document_ingestion_job_status(
        self,
        job_id: str,
    ) -> DocumentIngestionJobResponse:
        """Return current status and progress counts for an admin ingestion job."""
        if job_id not in self.admin_document_ingestion_jobs:
            raise KeyError(f"Unknown admin document ingestion job ID: {job_id}")
        job = self.admin_document_ingestion_jobs[job_id]
        return DocumentIngestionJobResponse(
            job_id=job["job_id"],
            assessment_id=ADMIN_DOCUMENT_CORPUS_ID,
            status=job["status"],
            document_count=job["document_count"],
            processed_document_count=job.get("processed_document_count", 0),
            indexed_chunk_count=job.get("indexed_chunk_count", 0),
            error_code=job.get("error_code"),
            error_message=job.get("error_message"),
        )

    def get_admin_document_download(self, document_id: str) -> DocumentDownload:
        """Return original admin document bytes and download metadata."""
        document = self.admin_documents.get(document_id)
        if document is None:
            raise KeyError(f"Unknown admin document ID: {document_id}")
        return DocumentDownload(
            document_id=document_id,
            file_name=document["file_name"],
            content_type=document["content_type"],
            content=_document_payload_bytes(document["content"]),
        )

    def delete_admin_document(self, document_id: str) -> bool:
        """Delete one admin document and derived retrieval payloads."""
        document = self.admin_documents.get(document_id)
        if document is None:
            return False
        self.admin_documents.pop(document_id, None)
        self.admin_document_extractions.pop(document_id, None)
        self.admin_document_extracted_blocks.pop(document_id, None)
        self.admin_document_chunks = [
            chunk
            for chunk in self.admin_document_chunks
            if chunk.get("document_id") != document_id
        ]
        for job_id, document_ids in list(
            self.admin_document_ingestion_job_documents.items()
        ):
            self.admin_document_ingestion_job_documents[job_id] = [
                current_id for current_id in document_ids if current_id != document_id
            ]
        return True

    def lease_next_admin_document_ingestion_job(
        self,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """Lease one pending admin document ingestion job."""
        for job in self.admin_document_ingestion_jobs.values():
            if job["status"] != "pending":
                continue
            job["status"] = "extracting_documents"
            job["lease_owner"] = worker_id
            documents = [
                deepcopy(self.admin_documents[document_id])
                for document_id in self.admin_document_ingestion_job_documents.get(
                    job["job_id"],
                    [],
                )
            ]
            return {**deepcopy(job), "documents": documents}
        return None

    def update_admin_document_ingestion_progress(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        processed_document_count: int | None = None,
        indexed_chunk_count: int | None = None,
    ) -> None:
        """Persist extraction or embedding progress for an admin ingestion job."""
        job = self.admin_document_ingestion_jobs[job_id]
        if job.get("lease_owner") != worker_id:
            return
        job["status"] = status
        if processed_document_count is not None:
            job["processed_document_count"] = processed_document_count
        if indexed_chunk_count is not None:
            job["indexed_chunk_count"] = indexed_chunk_count

    def save_admin_document_extraction(
        self,
        document_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Persist extraction summary and source blocks for one admin document."""
        extraction_id = f"admin-document-extraction-{uuid4()}"
        self.admin_document_extractions[document_id] = [
            {
                "extraction_id": extraction_id,
                "document_id": document_id,
                "file_name": extracted.file_name,
                "document_type": extracted.document_type,
                "estimated_tokens": estimated_tokens,
                "warnings": deepcopy(warnings),
            }
        ]
        self.admin_document_extracted_blocks[document_id] = [
            {
                "extraction_id": extraction_id,
                "document_id": document_id,
                "block_id": block.block_id,
                "block_type": block.block_type,
                "text": block.text,
            }
            for block in extracted.blocks
        ]
        return extraction_id

    def save_admin_document_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Replace indexed chunks for one admin document."""
        remaining = [
            chunk
            for chunk in self.admin_document_chunks
            if chunk.get("document_id") != document_id
        ]
        normalized_chunks = []
        for chunk in chunks:
            normalized_chunks.append(
                {
                    **deepcopy(chunk),
                    "document_id": document_id,
                    "similarity_score": chunk.get("similarity_score", 1.0),
                }
            )
        self.admin_document_chunks = remaining + normalized_chunks

    def complete_admin_document_ingestion_job(
        self,
        job_id: str,
        worker_id: str,
        indexed_chunk_count: int,
    ) -> None:
        """Mark an admin ingestion job completed and linked documents indexed."""
        job = self.admin_document_ingestion_jobs[job_id]
        if job.get("lease_owner") != worker_id:
            return
        job["status"] = "completed"
        job["processed_document_count"] = job["document_count"]
        job["indexed_chunk_count"] = indexed_chunk_count
        for document_id in self.admin_document_ingestion_job_documents.get(job_id, []):
            if document_id in self.admin_documents:
                self.admin_documents[document_id]["status"] = "indexed"
                self.admin_documents[document_id]["updated_at"] = str(time.time())

    def save_admin_document_ingestion_failure(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark an admin ingestion job and linked documents failed."""
        job = self.admin_document_ingestion_jobs[job_id]
        if job.get("lease_owner") != worker_id:
            return
        job["status"] = "failed"
        job["error_code"] = error_code
        job["error_message"] = error_message
        for document_id in self.admin_document_ingestion_job_documents.get(job_id, []):
            if document_id in self.admin_documents:
                self.admin_documents[document_id]["status"] = "failed"
                self.admin_documents[document_id]["error_message"] = error_message

    def search_admin_document_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return highest-similarity embedded chunks from the admin corpus."""
        _ = query_embedding
        return deepcopy(self.admin_document_chunks)[: max(1, min(int(top_k), 20))]
