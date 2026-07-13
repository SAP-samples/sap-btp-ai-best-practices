"""Internal classification execution manager for metal composition workflows."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from app.models.metal_composition import (
    DocumentMode,
    ItemClassificationResponse,
    ItemClassifyBatchResponse,
    MetalCompositionAgentOutputs,
    MetalCompositionResponse,
    ResolutionMode,
)

from .workflow import DiagramPayload
from .timing import finish_timing, utc_now_iso

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .service import MetalCompositionPredictInput, MetalCompositionService


@dataclass(frozen=True)
class MetalCompositionBatchRequestItem:
    """Internal item payload used by the batch classification executor.

    Inputs:
        product_code: GCC product code from the resolved item context.
        selected_source_row_id: GCC source row id selected for this item.
        resolution_mode: Current source-resolution mode; only ``auto`` is supported.
        document_mode: Whether PDFs should be included as classification evidence.
        include_token_usage: Whether provider token usage should be captured.

    Expected output:
        Immutable batch-executor input converted into ``MetalCompositionPredictInput``.
    """

    product_code: str
    selected_source_row_id: Optional[int] = None
    resolution_mode: ResolutionMode = "auto"
    document_mode: DocumentMode = "text_only"
    include_token_usage: bool = False


class MetalCompositionClassificationService:
    """Own workflow-backed prediction and saved classification execution."""

    def __init__(self, service: MetalCompositionService) -> None:
        self.service = service

    def predict(self, request: MetalCompositionPredictInput) -> MetalCompositionResponse:
        service = self.service
        request_started_perf = perf_counter()
        request_started_at = utc_now_iso()
        cold_start = service._consume_cold_start_flag()
        phases: dict = {}
        try:
            resolve_started_perf = perf_counter()
            resolve_started_at = utc_now_iso()
            resolve_substeps: dict = {}
            resolution = service._resolve_predict_record(
                request,
                resolve_substeps=resolve_substeps,
            )
            phases["resolve_source"] = finish_timing(
                resolve_started_perf,
                resolve_started_at,
                details={
                    "cache_source": service.cache_source or service.serving_store.metadata.get("source"),
                    "resolution_mode": request.resolution_mode,
                    **(
                        {"record_origin": resolution.record_origin}
                        if resolution.record_origin is not None
                        else {}
                    ),
                },
                substeps=resolve_substeps,
            )
            if resolution.record is None and not resolution.candidates:
                return service._response_with_timing(
                    response_payload={
                        "status": "not_found",
                        "product_code": request.product_code,
                        "document_mode": request.document_mode,
                        "error": "No completed GCC record matched the provided product code.",
                    },
                    existing_timing=None,
                    phases=phases,
                    request_started_perf=request_started_perf,
                    request_started_at=request_started_at,
                    cold_start=cold_start,
                )
            if resolution.record is None:
                return service._response_with_timing(
                    response_payload={
                        "status": "needs_disambiguation",
                        "product_code": request.product_code,
                        "document_mode": request.document_mode,
                        "candidates": resolution.candidates,
                        "error": resolution.error,
                    },
                    existing_timing=None,
                    phases=phases,
                    request_started_perf=request_started_perf,
                    request_started_at=request_started_at,
                    cold_start=cold_start,
                )
            record = resolution.record
            effective_composition_mode = service._resolve_effective_composition_mode(
                request,
                record_origin=resolution.record_origin,
                source_row_id=record.source_row_id,
            )
            document_mode = request.document_mode
            documents_required = (
                effective_composition_mode != "gcc_tracker"
                or document_mode == "with_documents"
            )
            if documents_required and not request.diagram_payloads:
                return service._response_with_timing(
                    response_payload={
                        "status": "failed",
                        "product_code": request.product_code,
                        "document_mode": document_mode,
                        "error": "At least one PDF file is required.",
                    },
                    existing_timing=None,
                    phases=phases,
                    request_started_perf=request_started_perf,
                    request_started_at=request_started_at,
                    cold_start=cold_start,
                )
            gcc_tracker_composition = (
                service._build_gcc_tracker_composition(
                    source_row_id=record.source_row_id,
                    total_weight_grams=record.summary.total_weight_gram,
                )
                if effective_composition_mode == "gcc_tracker"
                else None
            )

            workflow_started_perf = perf_counter()
            workflow_started_at = utc_now_iso()
            workflow_runner = service._get_workflow_runner()
            workflow_result = workflow_runner.run(
                product_code=request.product_code,
                source_row_id=record.source_row_id,
                source_summary=record.summary.model_dump(),
                source_row=record.source_row,
                composition_mode=effective_composition_mode,
                document_mode=document_mode,
                diagram_payloads=request.diagram_payloads,
                gcc_tracker_composition=gcc_tracker_composition,
                include_token_usage=bool(request.include_token_usage),
            )
            phases["workflow_call"] = finish_timing(workflow_started_perf, workflow_started_at)

            return service._response_with_timing(
                response_payload={
                    "status": "completed",
                    "product_code": request.product_code,
                    "document_mode": document_mode,
                    "selected_source": record.summary,
                    "final_composition": workflow_result.get("final_composition"),
                    "hts_classification": workflow_result.get("hts_classification"),
                    "section_232_assessment": workflow_result.get("section_232_assessment"),
                    "token_usage": workflow_result.get("token_usage"),
                    "agent_outputs": MetalCompositionAgentOutputs(
                        diagram=workflow_result.get("diagram_output", {}),
                        hts_fact_profile=workflow_result.get("hts_fact_profile", {}),
                        hana_tree_search=workflow_result.get("hana_tree_search_output", {}),
                        hts_resolution=workflow_result.get("hts_resolution_output", {}),
                        section_232_reasoner=workflow_result.get("section_232_reasoner_output", {}),
                    ),
                },
                existing_timing=workflow_result.get("timing"),
                phases=phases,
                request_started_perf=request_started_perf,
                request_started_at=request_started_at,
                cold_start=cold_start,
            )
        except Exception as exc:
            return service._response_with_timing(
                response_payload={
                    "status": "failed",
                    "product_code": request.product_code,
                    "document_mode": request.document_mode,
                    "error": str(exc),
                },
                existing_timing=None,
                phases=phases,
                request_started_perf=request_started_perf,
                request_started_at=request_started_at,
                cold_start=cold_start,
            )

    def predict_batch(
        self,
        items: List[MetalCompositionBatchRequestItem],
        *,
        diagram_map: Optional[Dict[int, List[DiagramPayload]]] = None,
    ) -> List[MetalCompositionResponse]:
        if not items:
            return []

        diagram_map = dict(diagram_map or {})
        max_workers = max(1, min(self.service.settings.batch_max_concurrency, len(items)))
        results: List[Optional[MetalCompositionResponse]] = [None] * len(items)

        def run_single(index: int, item: MetalCompositionBatchRequestItem) -> MetalCompositionResponse:
            request = self.service._build_predict_input_for_context  # placate linters when TYPE_CHECKING is false
            del request
            predict_input = self._build_predict_input(index, item, diagram_map)
            return self.predict(predict_input)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(run_single, index, item): index
                for index, item in enumerate(items)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = MetalCompositionResponse(
                        status="failed",
                        product_code=items[index].product_code,
                        error=str(exc),
                    )

        return [result for result in results if result is not None]

    def classify_item(
        self,
        item_id: str,
        *,
        owner_job_id: Optional[str] = None,
        document_mode: str = "text_only",
        include_token_usage: bool = False,
    ) -> ItemClassificationResponse:
        service = self.service
        context = service._resolve_item_context(item_id)
        has_documents, docs_status = service._context_document_state(context)
        if service._classification_requires_documents(context, document_mode=document_mode) and not has_documents:
            result = service._build_blocked_result(
                context=context,
                docs_status=docs_status,
                document_mode=document_mode,
            )
        else:
            predict_input = service._build_predict_input_for_context(
                context,
                document_mode=document_mode,
                include_token_usage=include_token_usage,
            )
            result = self.predict(predict_input)
        if owner_job_id is not None and not service.classification_job_store.is_item_owned_by_job(
            context.item_id,
            context.dataset_scope,
            owner_job_id,
        ):
            logger.info(
                "classify_item: skipping snapshot for superseded item %s owned by newer job",
                context.item_id,
            )
            return ItemClassificationResponse(
                item_id=context.item_id,
                last_classified_at=None,
                result=result,
            )
        snapshot = service.ui_state_store.save_classification_snapshot(
            context.item_id,
            dataset_scope=context.dataset_scope,
            result=result,
        )
        return ItemClassificationResponse(
            item_id=context.item_id,
            last_classified_at=snapshot.last_classified_at,
            result=result,
        )

    def classify_items(
        self,
        item_ids: Sequence[str],
        *,
        confirm_missing_documents: bool = False,
        owner_job_id: Optional[str] = None,
        document_mode: str = "text_only",
        include_token_usage: bool = False,
    ) -> ItemClassifyBatchResponse:
        del confirm_missing_documents
        service = self.service
        if not item_ids:
            return ItemClassifyBatchResponse(total=0, completed=0, failed=0, results=[])

        resolved_contexts: Dict[int, object] = {}
        errors: Dict[int, ItemClassificationResponse] = {}
        missing_contexts: List[object] = []
        missing_docs_statuses: List[str] = []
        batch_items: List[MetalCompositionBatchRequestItem] = []
        diagram_map: Dict[int, List[DiagramPayload]] = {}
        batch_indexes: Dict[int, int] = {}

        for original_index, item_id in enumerate(item_ids):
            try:
                context = service._resolve_item_context(item_id)
                has_documents, docs_status = service._context_document_state(context)
                if service._classification_requires_documents(context, document_mode=document_mode) and not has_documents:
                    missing_contexts.append(context)
                    missing_docs_statuses.append(docs_status)
                resolved_contexts[original_index] = context
            except Exception as exc:
                errors[original_index] = ItemClassificationResponse(
                    item_id=item_id,
                    result=MetalCompositionResponse(
                        status="failed",
                        product_code=item_id,
                        error=str(exc),
                    ),
                )

        if missing_contexts:
            service._raise_missing_documents_for_contexts(missing_contexts, missing_docs_statuses)

        for original_index, item_id in enumerate(item_ids):
            context = resolved_contexts.get(original_index)
            if context is None:
                continue
            try:
                resolved_contexts[original_index] = context
                batch_item, diagram_payloads = service._build_batch_request_for_context(
                    context,
                    document_mode=document_mode,
                    include_token_usage=include_token_usage,
                )
                batch_indexes[len(batch_items)] = original_index
                batch_items.append(batch_item)
                if diagram_payloads:
                    diagram_map[len(batch_items) - 1] = diagram_payloads
            except Exception as exc:
                errors[original_index] = ItemClassificationResponse(
                    item_id=item_id,
                    result=MetalCompositionResponse(
                        status="failed",
                        product_code=item_id,
                        error=str(exc),
                    ),
                )

        batch_results = self.predict_batch(batch_items, diagram_map=diagram_map) if batch_items else []
        ordered_results: List[Optional[ItemClassificationResponse]] = [None] * len(item_ids)

        for batch_index, result in enumerate(batch_results):
            original_index = batch_indexes[batch_index]
            context = resolved_contexts[original_index]
            if owner_job_id is not None and not service.classification_job_store.is_item_owned_by_job(
                context.item_id,
                context.dataset_scope,
                owner_job_id,
            ):
                snapshot = None
            else:
                snapshot = service.ui_state_store.save_classification_snapshot(
                    context.item_id,
                    dataset_scope=context.dataset_scope,
                    result=result,
                )
            ordered_results[original_index] = ItemClassificationResponse(
                item_id=context.item_id,
                last_classified_at=snapshot.last_classified_at if snapshot is not None else None,
                result=result,
            )

        for original_index, error_result in errors.items():
            ordered_results[original_index] = error_result

        final_results = [result for result in ordered_results if result is not None]
        completed = sum(1 for result in final_results if result.result.status == "completed")
        failed = sum(1 for result in final_results if result.result.status == "failed")
        return ItemClassifyBatchResponse(
            total=len(final_results),
            completed=completed,
            failed=failed,
            results=final_results,
        )

    def _build_predict_input(
        self,
        index: int,
        item: MetalCompositionBatchRequestItem,
        diagram_map: Dict[int, List[DiagramPayload]],
    ) -> MetalCompositionPredictInput:
        from .service import MetalCompositionPredictInput

        return MetalCompositionPredictInput(
            product_code=item.product_code,
            selected_source_row_id=item.selected_source_row_id,
            resolution_mode=item.resolution_mode,
            document_mode=item.document_mode,
            diagram_payloads=list(diagram_map.get(index, [])),
            include_token_usage=item.include_token_usage,
        )
