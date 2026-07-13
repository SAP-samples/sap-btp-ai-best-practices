"""Internal classification job manager for metal composition workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

from app.models.metal_composition import (
    ClassificationJobStatusResponse,
    ClassificationJobSubmissionResponse,
    ClassificationResetResponse,
    ClassificationStatsResponse,
    DocumentMode,
    MissingDocumentsBatchItem,
    MissingDocumentsConfirmationDetail,
)

from .classification_jobs import (
    ClassificationJobItemSeed,
    ClassificationJobType,
    PersistedClassificationJob,
    SUPERSEDED_ERROR_MESSAGE,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .service import MetalCompositionService, ResolvedItemContext


class MetalCompositionJobService:
    """Own classification job submission, execution, and reset flows."""

    def __init__(self, service: MetalCompositionService) -> None:
        self.service = service

    @staticmethod
    def _job_submission_response(job: PersistedClassificationJob) -> ClassificationJobSubmissionResponse:
        return ClassificationJobSubmissionResponse(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            submitted_at=job.submitted_at,
            total_count=job.total_count,
            completed_count=job.completed_count,
            failed_count=job.failed_count,
        )

    @staticmethod
    def _job_status_response(job: PersistedClassificationJob) -> ClassificationJobStatusResponse:
        return ClassificationJobStatusResponse(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            total_count=job.total_count,
            completed_count=job.completed_count,
            failed_count=job.failed_count,
            error_message=job.error_message,
        )

    def _raise_missing_documents_for_contexts(
        self,
        contexts: Sequence[ResolvedItemContext],
        docs_statuses: Sequence[str],
    ) -> None:
        from .service import MissingDocumentsConfirmationRequiredError, PDF_REQUIRED_BLOCK_MESSAGE

        raise MissingDocumentsConfirmationRequiredError(
            MissingDocumentsConfirmationDetail(
                message=PDF_REQUIRED_BLOCK_MESSAGE,
                items=[
                    MissingDocumentsBatchItem(
                        item_id=context.item_id,
                        product_code=context.product_code,
                        priority=context.priority,
                        docs_status=docs_status,
                    )
                    for context, docs_status in zip(contexts, docs_statuses)
                ],
            )
        )

    def _supersede_active_classifications(
        self,
        contexts: Sequence[ResolvedItemContext],
    ) -> None:
        self.service.classification_job_store.supersede_active_items(
            {(context.item_id, context.dataset_scope) for context in contexts},
            error_message=SUPERSEDED_ERROR_MESSAGE,
        )

    def submit_classify_item_job(
        self,
        item_id: str,
        *,
        job_type: ClassificationJobType = "single",
    ) -> ClassificationJobSubmissionResponse:
        return self.submit_predict_item_job(item_id, document_mode="text_only", job_type=job_type)

    def submit_predict_item_job(
        self,
        item_id: str,
        *,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
        job_type: ClassificationJobType = "single",
    ) -> ClassificationJobSubmissionResponse:
        """Submit one Material Master item for background prediction.

        Inputs:
            item_id: Material Master item id to classify.
            document_mode: Whether PDFs are ignored or required as additional evidence.
            include_token_usage: API-only diagnostic flag for recording model token metadata.
            job_type: Persisted job type.

        Expected output:
            A job submission response for polling.
        """

        service = self.service
        context = service._resolve_item_context(item_id)
        has_documents, docs_status = service._context_document_state(context)
        if service._classification_requires_documents(context, document_mode=document_mode) and not has_documents:
            self._raise_missing_documents_for_contexts([context], [docs_status])
        self._supersede_active_classifications([context])
        job = service.classification_job_store.submit_job(
            job_type=job_type,
            items=[
                ClassificationJobItemSeed(
                    item_id=context.item_id,
                    dataset_scope=context.dataset_scope,
                    position=0,
                    document_mode=document_mode,
                    include_token_usage=include_token_usage,
                )
            ],
        )
        return self._job_submission_response(job)

    def submit_classify_items_job(
        self,
        item_ids: Sequence[str],
        *,
        confirm_missing_documents: bool = False,
        job_type: ClassificationJobType = "batch",
    ) -> ClassificationJobSubmissionResponse:
        del confirm_missing_documents
        return self.submit_predict_items_job(item_ids, document_mode="text_only", job_type=job_type)

    def submit_predict_items_job(
        self,
        item_ids: Sequence[str],
        *,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
        job_type: ClassificationJobType = "batch",
    ) -> ClassificationJobSubmissionResponse:
        """Submit multiple Material Master items for background prediction.

        Inputs:
            item_ids: Material Master item ids to classify.
            document_mode: Batch-wide document evidence mode.
            include_token_usage: API-only diagnostic flag for recording model token metadata.
            job_type: Persisted job type.

        Expected output:
            A job submission response for polling.
        """

        service = self.service
        if not item_ids:
            job = service.classification_job_store.submit_job(job_type=job_type, items=[])
            return self._job_submission_response(job)

        contexts: List[ResolvedItemContext] = []
        missing_contexts: List[ResolvedItemContext] = []
        missing_docs_statuses: List[str] = []
        for item_id in item_ids:
            context = service._resolve_item_context(item_id)
            has_documents, docs_status = service._context_document_state(context)
            if service._classification_requires_documents(context, document_mode=document_mode) and not has_documents:
                missing_contexts.append(context)
                missing_docs_statuses.append(docs_status)
            contexts.append(context)

        if missing_contexts:
            self._raise_missing_documents_for_contexts(missing_contexts, missing_docs_statuses)

        self._supersede_active_classifications(contexts)
        job = service.classification_job_store.submit_job(
            job_type=job_type,
            items=[
                ClassificationJobItemSeed(
                    item_id=context.item_id,
                    dataset_scope=context.dataset_scope,
                    position=index,
                    document_mode=document_mode,
                    include_token_usage=include_token_usage,
                )
                for index, context in enumerate(contexts)
            ],
        )
        return self._job_submission_response(job)

    def get_classification_job(self, job_id: str) -> ClassificationJobStatusResponse:
        job = self.service.classification_job_store.get_job(job_id)
        if job is None:
            raise KeyError(f"classification job {job_id} not found")
        return self._job_status_response(job)

    def claim_next_classification_job(self, worker_id: str) -> Optional[PersistedClassificationJob]:
        return self.service.classification_job_store.claim_next_queued_job(worker_id=worker_id)

    def process_claimed_classification_job(self, job_id: str) -> None:
        service = self.service
        logger.info("process_claimed_classification_job: starting job %s", job_id)
        job = service.classification_job_store.get_job(job_id)
        if job is None:
            raise KeyError(f"classification job {job_id} not found")
        job_items = [
            item
            for item in service.classification_job_store.get_job_items(job_id)
            if item.status in {"queued", "running"}
        ]
        if not job_items:
            logger.info("process_claimed_classification_job: job %s has no active items left to process", job_id)
            return

        service.classification_job_store.mark_job_items_running(job_id)
        try:
            item_ids = [item.item_id for item in job_items]
            logger.info(
                "process_claimed_classification_job: job %s has %d items (path=%s)",
                job_id,
                len(item_ids),
                "single" if len(item_ids) == 1 else "batch",
            )
            if len(item_ids) == 1:
                results = [
                    service.classify_item(
                        item_ids[0],
                        owner_job_id=job_id,
                        document_mode=job_items[0].document_mode,
                        include_token_usage=job_items[0].include_token_usage,
                    )
                ]
            else:
                document_modes = {item.document_mode for item in job_items}
                include_token_usage_values = {item.include_token_usage for item in job_items}
                if len(document_modes) == 1 and len(include_token_usage_values) == 1:
                    results = list(
                        service.classify_items(
                            item_ids,
                            owner_job_id=job_id,
                            document_mode=job_items[0].document_mode,
                            include_token_usage=job_items[0].include_token_usage,
                        ).results
                    )
                else:
                    results = [
                        service.classify_item(
                            item.item_id,
                            owner_job_id=job_id,
                            document_mode=item.document_mode,
                            include_token_usage=item.include_token_usage,
                        )
                        for item in job_items
                    ]

            logger.info(
                "process_claimed_classification_job: job %s finished with %d results",
                job_id,
                len(results),
            )
            for item, result in zip(job_items, results):
                owns_item = service.classification_job_store.is_item_owned_by_job(
                    item.item_id,
                    item.dataset_scope,
                    job_id,
                )
                if not owns_item:
                    item_status = "failed"
                    error_message = SUPERSEDED_ERROR_MESSAGE
                    last_classified_at = None
                else:
                    item_status = "completed" if result.result.status == "completed" else "failed"
                    error_message = result.result.error if item_status == "failed" else None
                    last_classified_at = result.last_classified_at
                service.classification_job_store.record_job_item_result(
                    job_id,
                    position=item.position,
                    status=item_status,
                    error_message=error_message,
                    last_classified_at=last_classified_at,
                )
        except Exception as exc:
            logger.exception("process_claimed_classification_job: job %s failed", job_id)
            service.classification_job_store.fail_job(job_id, error_message=str(exc))

    def drain_classification_jobs(self, worker_id: str = "test-worker") -> int:
        processed = 0
        while True:
            job = self.claim_next_classification_job(worker_id)
            if job is None:
                break
            self.process_claimed_classification_job(job.job_id)
            processed += 1
        return processed

    def get_classification_stats(self) -> ClassificationStatsResponse:
        saved_classification_count, latest_classified_at = self.service.ui_state_store.get_classification_stats()
        return ClassificationStatsResponse(
            saved_classification_count=saved_classification_count,
            latest_classified_at=latest_classified_at,
        )

    def reset_classifications(self) -> ClassificationResetResponse:
        cleared_classification_count = self.service.ui_state_store.clear_classification_snapshots()
        cancelled_job_count = self.service.classification_job_store.cancel_all_active_jobs()
        return ClassificationResetResponse(
            cleared_classification_count=cleared_classification_count,
            cancelled_job_count=cancelled_job_count,
        )
