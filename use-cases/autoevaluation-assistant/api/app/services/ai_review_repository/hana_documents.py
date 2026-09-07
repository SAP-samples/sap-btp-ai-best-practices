"""Assessment document corpus persistence for the HANA AI review repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from sqlalchemy import text

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
from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, table_name_from_create_statement
from app.services.joule_knowledge_importer import vector_to_json

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    _row_batches,
    document_content_hash,
)

class HanaDocumentsMixin:
    """Store and search assessment-scoped document corpus rows."""

    def create_document_ingestion_job(
        self,
        assessment_id: str,
        documents: list[dict[str, Any]],
    ) -> DocumentIngestionJobResponse:
        """Store uploaded corpus documents and create one ingestion job.

        Inputs:
            assessment_id: Assessment whose corpus receives the documents.
            documents: Uploaded file payloads with file_name, content_type, and
                content bytes.

        Outputs:
            DocumentIngestionJobResponse: Pending ingestion job summary.

        Raises:
            ValueError: Raised when no documents are supplied.
        """
        if not documents:
            raise ValueError("Document ingestion requires at least one document.")

        job_id = f"document-ingestion-job-{uuid4()}"
        linked_document_ids: list[str] = []
        self.session.execute(
            text(
                "insert into ai_document_ingestion_jobs "
                "(job_id, assessment_id, status) "
                "values (:job_id, :assessment_id, 'pending')"
            ),
            {"job_id": job_id, "assessment_id": assessment_id},
        )
        for document in documents:
            content = _document_payload_bytes(document.get("content", b""))
            content_hash = document_content_hash(content)
            existing_row = self.session.execute(
                text(
                    "select top 1 document_id "
                    "from ai_documents "
                    "where assessment_id = :assessment_id "
                    "and content_hash = :content_hash "
                    "order by created_at"
                ),
                {
                    "assessment_id": assessment_id,
                    "content_hash": content_hash,
                },
            ).mappings().first()
            if existing_row is None:
                document_id = f"document-{uuid4()}"
                self.session.execute(
                    text(
                        "insert into ai_documents "
                        "(document_id, assessment_id, content_hash, file_name, "
                        "content_type, content_blob, status) "
                        "values (:document_id, :assessment_id, :content_hash, "
                        ":file_name, :content_type, :content_blob, 'pending')"
                    ),
                    {
                        "document_id": document_id,
                        "assessment_id": assessment_id,
                        "content_hash": content_hash,
                        "file_name": document["file_name"],
                        "content_type": document.get(
                            "content_type",
                            "application/octet-stream",
                        ),
                        "content_blob": content,
                    },
                )
            else:
                document_id = str(existing_row["document_id"])
                self.session.execute(
                    text(
                        "update ai_documents "
                        "set status = 'pending', error_message = null, "
                        "updated_at = current_timestamp "
                        "where document_id = :document_id"
                    ),
                    {"document_id": document_id},
                )
            if document_id in linked_document_ids:
                continue
            linked_document_ids.append(document_id)
            self.session.execute(
                text(
                    "insert into ai_document_ingestion_job_documents "
                    "(link_id, job_id, document_id) "
                    "values (:link_id, :job_id, :document_id)"
                ),
                {
                    "link_id": f"document-ingestion-link-{uuid4()}",
                    "job_id": job_id,
                    "document_id": document_id,
                },
            )
        self.session.execute(
            text(
                "update ai_document_ingestion_jobs "
                "set document_count = :document_count "
                "where job_id = :job_id"
            ),
            {"job_id": job_id, "document_count": len(linked_document_ids)},
        )
        return DocumentIngestionJobResponse(
            job_id=job_id,
            assessment_id=assessment_id,
            status="pending",
            document_count=len(linked_document_ids),
        )

    def list_documents(self, assessment_id: str) -> list[DocumentSummary]:
        """Return corpus document metadata for one assessment.

        Inputs:
            assessment_id: Assessment whose uploaded documents should be listed.

        Outputs:
            list[DocumentSummary]: Ordered document metadata with chunk counts.
        """
        rows = self.session.execute(
            text(
                "select d.document_id, d.assessment_id, d.file_name, "
                "d.content_type, d.content_hash, d.status, d.error_message, "
                "d.created_at, d.updated_at, "
                "coalesce(c.chunk_count, 0) as chunk_count "
                "from ai_documents d "
                "left join ("
                "  select document_id, count(*) as chunk_count "
                "  from ai_document_chunks "
                "  group by document_id"
                ") c on c.document_id = d.document_id "
                "where d.assessment_id = :assessment_id "
                "order by d.created_at"
            ),
            {"assessment_id": assessment_id},
        ).mappings().all()
        return [
            DocumentSummary(
                document_id=str(row["document_id"]),
                assessment_id=str(row["assessment_id"]),
                file_name=str(row["file_name"]),
                content_type=str(row["content_type"]),
                content_hash=str(row["content_hash"]),
                status=str(row["status"]),
                error_message=row["error_message"],
                chunk_count=int(row["chunk_count"] or 0),
                created_at=str(row["created_at"]) if row["created_at"] else None,
                updated_at=str(row["updated_at"]) if row["updated_at"] else None,
            )
            for row in rows
        ]

    def get_document_list(self, assessment_id: str) -> DocumentListResponse:
        """Return document metadata plus aggregate readiness counts.

        Inputs:
            assessment_id: Assessment whose corpus should be summarized.

        Outputs:
            DocumentListResponse: Documents and indexed corpus counts.
        """
        documents = self.list_documents(assessment_id)
        return DocumentListResponse(
            assessment_id=assessment_id,
            documents=documents,
            indexed_document_count=sum(1 for document in documents if document.status == "indexed"),
            total_document_count=len(documents),
            indexed_chunk_count=sum(document.chunk_count for document in documents),
        )

    def get_document_ingestion_job_status(
        self,
        job_id: str,
    ) -> DocumentIngestionJobResponse:
        """Return current progress for one document ingestion job.

        Inputs:
            job_id: Ingestion job identifier.

        Outputs:
            DocumentIngestionJobResponse: Current persisted job state.

        Raises:
            KeyError: Raised when the job does not exist.
        """
        row = self.session.execute(
            text(
                "select job_id, assessment_id, status, document_count, "
                "processed_document_count, indexed_chunk_count, error_code, "
                "error_message "
                "from ai_document_ingestion_jobs "
                "where job_id = :job_id"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if row is None:
            raise KeyError(f"Unknown document ingestion job ID: {job_id}")
        return DocumentIngestionJobResponse(
            job_id=str(row["job_id"]),
            assessment_id=str(row["assessment_id"]),
            status=str(row["status"]),
            document_count=int(row["document_count"] or 0),
            processed_document_count=int(row["processed_document_count"] or 0),
            indexed_chunk_count=int(row["indexed_chunk_count"] or 0),
            error_code=row["error_code"],
            error_message=row["error_message"],
        )

    def get_document_download(
        self,
        assessment_id: str,
        document_id: str,
    ) -> DocumentDownload:
        """Return original document bytes and download metadata.

        Inputs:
            assessment_id: Assessment that owns the document.
            document_id: Document identifier.

        Outputs:
            DocumentDownload: Original content and metadata.

        Raises:
            KeyError: Raised when no matching document exists.
        """
        row = self.session.execute(
            text(
                "select document_id, file_name, content_type, content_blob as content "
                "from ai_documents "
                "where assessment_id = :assessment_id and document_id = :document_id"
            ),
            {"assessment_id": assessment_id, "document_id": document_id},
        ).mappings().first()
        if row is None:
            raise KeyError(f"Unknown document ID: {document_id}")
        return DocumentDownload(
            document_id=str(row["document_id"]),
            file_name=str(row["file_name"]),
            content_type=str(row["content_type"]),
            content=_document_payload_bytes(row["content"]),
        )

    def delete_document(self, assessment_id: str, document_id: str) -> bool:
        """Delete one corpus document and derived retrieval payloads.

        Inputs:
            assessment_id: Assessment that owns the document.
            document_id: Document identifier to delete.

        Outputs:
            bool: True when a document row was deleted.
        """
        exists = self.session.execute(
            text(
                "select document_id from ai_documents "
                "where assessment_id = :assessment_id and document_id = :document_id"
            ),
            {"assessment_id": assessment_id, "document_id": document_id},
        ).mappings().first()
        if exists is None:
            return False

        extraction_filter = (
            "select extraction_id from ai_document_extractions "
            "where document_id = :document_id"
        )
        self.session.execute(
            text(
                "delete from ai_document_extracted_blocks "
                f"where extraction_id in ({extraction_filter})"
            ),
            {"document_id": document_id},
        )
        self.session.execute(
            text("delete from ai_document_chunks where document_id = :document_id"),
            {"document_id": document_id},
        )
        self.session.execute(
            text("delete from ai_document_extractions where document_id = :document_id"),
            {"document_id": document_id},
        )
        self.session.execute(
            text(
                "delete from ai_document_ingestion_job_documents "
                "where document_id = :document_id"
            ),
            {"document_id": document_id},
        )
        self.session.execute(
            text(
                "delete from ai_documents "
                "where assessment_id = :assessment_id and document_id = :document_id"
            ),
            {"assessment_id": assessment_id, "document_id": document_id},
        )
        return True

    def lease_next_document_ingestion_job(
        self,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """Lease one pending or expired document ingestion job.

        Inputs:
            worker_id: Identifier for the active worker process.

        Outputs:
            dict[str, Any] | None: Job row with linked documents, or None.
        """
        row = self.session.execute(
            text(
                "select top 1 job_id, assessment_id, status "
                "from ai_document_ingestion_jobs "
                "where status = 'pending' "
                "or (status in ('extracting_documents', 'embedding_documents') "
                "and lease_expires_at <= current_timestamp) "
                "order by created_at"
            )
        ).mappings().first()
        if row is None:
            return None
        lease_result = self.session.execute(
            text(
                "update ai_document_ingestion_jobs "
                "set status = 'extracting_documents', lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_timestamp, 3600), "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and (status = 'pending' "
                "or (status in ('extracting_documents', 'embedding_documents') "
                "and lease_expires_at <= current_timestamp))"
            ),
            {"worker_id": worker_id, "job_id": row["job_id"]},
        )
        if lease_result.rowcount == 0:
            return None
        documents = self.session.execute(
            text(
                "select d.document_id, d.file_name, d.content_type, "
                "d.content_hash, d.content_blob as content "
                "from ai_document_ingestion_job_documents l "
                "join ai_documents d on d.document_id = l.document_id "
                "where l.job_id = :job_id "
                "order by l.created_at"
            ),
            {"job_id": row["job_id"]},
        ).mappings().all()
        return {
            "job_id": row["job_id"],
            "assessment_id": row["assessment_id"],
            "status": "extracting_documents",
            "lease_owner": worker_id,
            "documents": [dict(document) for document in documents],
        }

    def update_document_ingestion_progress(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        processed_document_count: int | None = None,
        indexed_chunk_count: int | None = None,
    ) -> None:
        """Persist extraction or embedding progress for an ingestion job.

        Inputs:
            job_id: Ingestion job identifier.
            worker_id: Worker that holds the active lease.
            status: New ingestion phase status.
            processed_document_count: Optional document progress count.
            indexed_chunk_count: Optional indexed chunk progress count.

        Outputs:
            None. Matching leased job rows are updated.
        """
        self.session.execute(
            text(
                "update ai_document_ingestion_jobs "
                "set status = :status, "
                "processed_document_count = coalesce(:processed_document_count, processed_document_count), "
                "indexed_chunk_count = coalesce(:indexed_chunk_count, indexed_chunk_count), "
                "updated_at = current_timestamp "
                "where job_id = :job_id and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "status": status,
                "processed_document_count": processed_document_count,
                "indexed_chunk_count": indexed_chunk_count,
            },
        )

    def save_document_extraction(
        self,
        document_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Persist extraction summary and source text blocks for one document.

        Inputs:
            document_id: Corpus document identifier.
            extracted: Extracted document payload.
            estimated_tokens: Estimated token count for all blocks.
            warnings: Non-fatal extraction warnings.

        Outputs:
            str: Generated extraction identifier.
        """
        extraction_id = f"document-extraction-{uuid4()}"
        self.session.execute(
            text("delete from ai_document_chunks where document_id = :document_id"),
            {"document_id": document_id},
        )
        old_extractions = self.session.execute(
            text(
                "select extraction_id from ai_document_extractions "
                "where document_id = :document_id"
            ),
            {"document_id": document_id},
        ).mappings().all()
        for row in old_extractions:
            self.session.execute(
                text(
                    "delete from ai_document_extracted_blocks "
                    "where extraction_id = :extraction_id"
                ),
                {"extraction_id": row["extraction_id"]},
            )
        self.session.execute(
            text("delete from ai_document_extractions where document_id = :document_id"),
            {"document_id": document_id},
        )
        self.session.execute(
            text(
                "insert into ai_document_extractions "
                "(extraction_id, document_id, file_name, document_type, "
                "page_count, extracted_block_count, total_characters, "
                "estimated_tokens, quality_json, warnings_json) "
                "values (:extraction_id, :document_id, :file_name, "
                ":document_type, :page_count, :extracted_block_count, "
                ":total_characters, :estimated_tokens, :quality_json, "
                ":warnings_json)"
            ),
            {
                "extraction_id": extraction_id,
                "document_id": document_id,
                "file_name": extracted.file_name,
                "document_type": extracted.document_type,
                "page_count": extracted.metadata.get("page_count"),
                "extracted_block_count": len(extracted.blocks),
                "total_characters": sum(len(block.text) for block in extracted.blocks),
                "estimated_tokens": estimated_tokens,
                "quality_json": json.dumps(
                    extracted.metadata.get("quality", {}),
                    ensure_ascii=False,
                ),
                "warnings_json": json.dumps(warnings, ensure_ascii=False),
            },
        )
        block_rows = [
            {
                "block_row_id": f"document-block-{uuid4()}",
                "extraction_id": extraction_id,
                "document_id": document_id,
                "block_id": block.block_id,
                "block_type": block.block_type,
                "text": block.text,
                "location_json": json.dumps(
                    {
                        "section_label": block.section_label,
                        "sheet_name": block.sheet_name,
                        "table_name": block.table_name,
                        "page": block.page,
                        "row_start": block.row_start,
                        "row_end": block.row_end,
                    },
                    ensure_ascii=False,
                ),
            }
            for block in extracted.blocks
        ]
        if block_rows:
            self.session.execute(
                text(
                    "insert into ai_document_extracted_blocks "
                    "(block_row_id, extraction_id, document_id, block_id, "
                    "block_type, text, location_json) "
                    "values (:block_row_id, :extraction_id, :document_id, "
                    ":block_id, :block_type, :text, :location_json)"
                ),
                block_rows,
            )
        return extraction_id

    def save_document_chunks(
        self,
        assessment_id: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Replace indexed chunks for one document with embedded chunk rows.

        Inputs:
            assessment_id: Assessment that owns the document.
            document_id: Corpus document identifier.
            chunks: Embedded chunk dictionaries.

        Outputs:
            None. Existing chunks for the document are replaced.
        """
        self.session.execute(
            text("delete from ai_document_chunks where document_id = :document_id"),
            {"document_id": document_id},
        )
        rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "assessment_id": assessment_id,
                "document_id": document_id,
                "file_name": chunk["file_name"],
                "document_type": chunk["document_type"],
                "source_block_ids_json": json.dumps(
                    chunk.get("source_block_ids", []),
                    ensure_ascii=False,
                ),
                "location_json": json.dumps(
                    chunk.get("location_json", {}),
                    ensure_ascii=False,
                ),
                "chunk_text": chunk["chunk_text"],
                "estimated_tokens": chunk["estimated_tokens"],
                "embedding_model": chunk["embedding_model"],
                "embedding": vector_to_json(chunk.get("embedding", [])),
                "content_hash": chunk["content_hash"],
                "chunk_kind": chunk.get("chunk_kind", "raw"),
            }
            for chunk in chunks
        ]
        for batch in _row_batches(rows):
            self.session.execute(
                text(
                    "insert into ai_document_chunks "
                    "(chunk_id, assessment_id, document_id, file_name, "
                    "document_type, source_block_ids_json, location_json, "
                    "chunk_text, estimated_tokens, embedding_model, embedding, "
                    "content_hash, chunk_kind) "
                    "values (:chunk_id, :assessment_id, :document_id, "
                    ":file_name, :document_type, :source_block_ids_json, "
                    ":location_json, :chunk_text, :estimated_tokens, "
                    ":embedding_model, to_real_vector(:embedding), "
                    ":content_hash, :chunk_kind)"
                ),
                batch,
            )

    def complete_document_ingestion_job(
        self,
        job_id: str,
        worker_id: str,
        indexed_chunk_count: int,
    ) -> None:
        """Mark an ingestion job completed and linked documents indexed.

        Inputs:
            job_id: Ingestion job identifier.
            worker_id: Worker that holds the active lease.
            indexed_chunk_count: Number of chunks indexed by the worker.

        Outputs:
            None. Job and document rows are updated.
        """
        self.session.execute(
            text(
                "update ai_document_ingestion_jobs "
                "set status = 'completed', indexed_chunk_count = :indexed_chunk_count, "
                "processed_document_count = document_count, updated_at = current_timestamp "
                "where job_id = :job_id and lease_owner = :worker_id"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "indexed_chunk_count": indexed_chunk_count,
            },
        )
        self.session.execute(
            text(
                "update ai_documents set status = 'indexed', error_message = null, "
                "updated_at = current_timestamp "
                "where document_id in ("
                "select document_id from ai_document_ingestion_job_documents "
                "where job_id = :job_id)"
            ),
            {"job_id": job_id},
        )

    def save_document_ingestion_failure(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark an ingestion job and linked documents failed.

        Inputs:
            job_id: Ingestion job identifier.
            worker_id: Worker that holds the active lease.
            error_code: Compact failure code.
            error_message: Human-readable failure detail.

        Outputs:
            None. Job and document rows are updated.
        """
        self.session.execute(
            text(
                "update ai_document_ingestion_jobs "
                "set status = 'failed', error_code = :error_code, "
                "error_message = :error_message, updated_at = current_timestamp "
                "where job_id = :job_id and lease_owner = :worker_id"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "error_code": error_code,
                "error_message": error_message,
            },
        )
        self.session.execute(
            text(
                "update ai_documents set status = 'failed', "
                "error_message = :error_message, updated_at = current_timestamp "
                "where document_id in ("
                "select document_id from ai_document_ingestion_job_documents "
                "where job_id = :job_id)"
            ),
            {"job_id": job_id, "error_message": error_message},
        )

    def search_document_chunks(
        self,
        assessment_id: str,
        query_embedding: list[float],
        top_k: int = 8,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return highest-similarity embedded chunks for one assessment corpus.

        Inputs:
            assessment_id: Assessment corpus to search.
            query_embedding: Query vector from the embedding model.
            top_k: Maximum chunk rows to return.
            limit: Optional alias used by tests and generic callers.

        Outputs:
            list[dict[str, Any]]: Retrieved chunk rows with similarity scores.
        """
        row_limit = int(limit or top_k)
        query_vector = vector_to_json(query_embedding)
        rows = self.session.execute(
            text(
                f"select top {row_limit} chunk_id, document_id as attachment_id, "
                "document_id, file_name, document_type, source_block_ids_json, "
                "location_json, chunk_text, "
                "cosine_similarity(embedding, to_real_vector(:query_vector)) as similarity_score "
                "from ai_document_chunks "
                "where assessment_id = :assessment_id and embedding is not null "
                "order by similarity_score desc"
            ),
            {"assessment_id": assessment_id, "query_vector": query_vector},
        ).mappings().all()
        return [
            {
                "chunk_id": row["chunk_id"],
                "attachment_id": row["attachment_id"],
                "document_id": row["document_id"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "source_block_ids_json": row["source_block_ids_json"],
                "chunk_text": row["chunk_text"],
                "location_json": (
                    json.loads(row["location_json"])
                    if isinstance(row["location_json"], str)
                    else dict(row["location_json"])
                ),
                "similarity_score": float(row["similarity_score"]),
            }
            for row in rows
        ]
