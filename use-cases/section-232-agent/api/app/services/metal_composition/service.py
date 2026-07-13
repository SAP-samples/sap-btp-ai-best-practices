"""Service layer for the metal composition API and UI workflows."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

from app.models.metal_composition import (
    BlockingReason,
    ClassificationJobConflictDetail,
    ClassificationJobRef,
    ClassificationJobStatusResponse,
    ClassificationJobSubmissionResponse,
    ClassificationResetResponse,
    ClassificationStatsResponse,
    HTSCatalogSourceDeleteResponse,
    HTSCatalogSourceFileSummary,
    HTSCatalogSourceListResponse,
    HTSCatalogSourceUploadResponse,
    HTSCatalogSummaryResponse,
    ItemClassificationResponse,
    ItemClassifyBatchResponse,
    MetalCompositionAppSettings,
    MetalCompositionAgentOutputs,
    MetalCompositionCandidate,
    MetalCompositionFacetOption,
    MissingDocumentsBatchItem,
    MissingDocumentsConfirmationDetail,
    MetalCompositionItemDetail,
    MetalCompositionItemListResponse,
    MetalCompositionItemSummary,
    MetalCompositionResponse,
    CompositionMode,
    DocumentMode,
    ResolutionMode,
    Section232CancelDraftBatchResponse,
    Section232ClassificationResponse,
    Section232DraftBatchProcessResponse,
    Section232DraftBatchSummary,
    Section232DraftRuleItem,
    Section232DraftRuleBulkReviewResponse,
    Section232DraftRuleDeleteResponse,
    Section232DraftRuleListResponse,
    Section232DraftRuleReviewResponse,
    Section232DirectClassificationRequest,
    Section232EligibleCodeDetail,
    Section232EligibleCodeDetailListResponse,
    Section232EligibleCodeListResponse,
    Section232PublishDraftBatchResponse,
    Section232PublishedRuleDeleteResponse,
    Section232ResetResponse,
    Section232Assessment,
    Section232RulesetSummaryResponse,
    Section232ReviewRow,
    Section232ReviewWorkspaceResponse,
    Section232SourceDocumentMetadata,
    Section232SourceListResponse,
    Section232SourceSummary,
    Section232SourceUpdateMode,
    Section232SourceUploadResponse,
    HTSCandidate,
)

from .config import MetalCompositionSettings, get_settings
from .classification_jobs import (
    ClassificationJobItemSeed,
    ClassificationJobStore,
    ClassificationJobType,
    InMemoryClassificationJobStore,
    PersistedClassificationJob,
    SUPERSEDED_ERROR_MESSAGE,
)
from .classification_service import MetalCompositionClassificationService
from .classification_service import MetalCompositionBatchRequestItem
from .documents import (
    MetalCompositionDocumentStore,
)
from .item_service import MetalCompositionItemService
from .job_service import MetalCompositionJobService
from .report_export import build_classification_report_pdf
from .section_232_service import MetalCompositionSection232Service
from .hts_catalog import (
    CHAPTER_TITLES,
    HanaHTSCatalogResolver,
    canonicalize_hts_code,
    compile_hts_catalog_frame,
    compile_hts_code_map_frame,
    refresh_hts_catalog_tables,
)
from .hts_catalog_sources import (
    CHAPTER_FILENAME_RE,
    HTSCatalogSourceStore,
    InMemoryHTSCatalogSourceStore,
    decode_catalog_upload,
    normalize_hts_catalog_filename,
    validate_catalog_source_text,
)
from .section_232_sources import (
    InMemorySection232SourceStore,
    Section232SourceStore,
    collect_section_232_match_evidence,
    extract_section_232_pdf,
)
from .section_232_rule_interpreter import interpret_section_232_batch
from .section_232_rules_engine import (
    build_skipped_weight_override,
    build_section_232_ruleset_assessment,
    build_section_232_ruleset_reasoner_output,
    evaluate_section_232_ruleset,
    legal_coverage_established,
)
from .section_232_rulesets import (
    InMemorySection232RulesetStore,
    PersistedSection232RulesetStore,
    section_232_rule_overlay_identity,
)
from .serving_store import ResolvedMaterialRecord, ServingStore, load_serving_store, normalize_lookup_value
from .store_protocols import (
    ClassificationJobStoreProtocol,
    HTSCatalogSourceStoreProtocol,
    Section232RulesetStoreProtocol,
    Section232SourceStoreProtocol,
    UIStateStoreProtocol,
)
from .timing import finish_timing, rank_timings, utc_now_iso
from .ui_state import (
    InMemoryMetalCompositionUIStateStore,
    MetalCompositionUIStateStore,
    StoredDocumentReference,
)
from .workflow import DiagramPayload, MetalCompositionWorkflowRunner
from .workflow.normalize import _code_digit_count
from app.utils.hana import HANAConnection


PRODUCT_CODE_COLUMN = "Product code"
PRIORITY_COLUMN = "Priority"
PRIORITY_DETAIL_COLUMN = "Priority.1"
BUSINESS_SEGMENT_COLUMN = "Business Segment"
SITE_COLUMN = "Site"
PN_COLUMN = "PN Revised/ Standardized"
PART_DESCRIPTION_COLUMN = "Part description"
NEW_PART_DESCRIPTION_COLUMN = "New Part Description"
TOTAL_WEIGHT_COLUMN = "Total Weight (Gram)"
MATERIAL_CONTENT_METHOD_COLUMN = "Material Content Method"
MATERIAL_IDENTIFIED_COLUMN = "MaterialIdentified"
DATE_STARTED_COLUMN = "Date Started"
DATE_COMPLETED_COLUMN = "Date Completed"
SIGNATURE_COLUMNS = [
    "source_row_id",
    "normalized_product_code",
    PRIORITY_COLUMN,
    BUSINESS_SEGMENT_COLUMN,
    SITE_COLUMN,
    PRODUCT_CODE_COLUMN,
    PN_COLUMN,
    PART_DESCRIPTION_COLUMN,
    NEW_PART_DESCRIPTION_COLUMN,
    PRIORITY_DETAIL_COLUMN,
    MATERIAL_CONTENT_METHOD_COLUMN,
    MATERIAL_IDENTIFIED_COLUMN,
    TOTAL_WEIGHT_COLUMN,
    DATE_STARTED_COLUMN,
    DATE_COMPLETED_COLUMN,
]
PDF_REQUIRED_BLOCK_MESSAGE = (
    "Classification is blocked because this item does not have an uploaded PDF. "
    "Upload a PDF to extract engineering clues for classification."
)
SECTION_232_REVIEW_PAGE_SIZE = 100


def _section_232_publication_window_overlaps(left: Any, right: Any) -> bool:
    left_start = date.fromisoformat(left.effective_from) if getattr(left, "effective_from", None) else date.min
    left_end = date.fromisoformat(left.effective_to) if left.effective_to else date.max
    right_start = date.fromisoformat(right.effective_from) if getattr(right, "effective_from", None) else date.min
    right_end = date.fromisoformat(right.effective_to) if right.effective_to else date.max
    return left_start <= right_end and right_start <= left_end


def _section_232_publication_window_bounds(candidate: Any) -> tuple[date, date]:
    start = date.fromisoformat(candidate.effective_from) if getattr(candidate, "effective_from", None) else date.min
    end = date.fromisoformat(candidate.effective_to) if candidate.effective_to else date.max
    return start, end


def _section_232_publication_slot_identity(candidate: Any) -> tuple[str]:
    code, _scope = section_232_rule_overlay_identity(candidate)
    return (code,)


def _section_232_material_rule_identity(candidate: Any) -> tuple[str, str, str, str]:
    code, scope = section_232_rule_overlay_identity(candidate)
    return (code, scope, candidate.rule_type, candidate.coverage_effect)


def _section_232_candidate_processed_at(
    candidate: Any,
    *,
    batch_created_at_by_id: Dict[str, str],
) -> str:
    explicit_processed_at = str(getattr(candidate, "processed_at", "") or "").strip()
    if explicit_processed_at:
        return explicit_processed_at
    batch_id = str(getattr(candidate, "batch_id", "") or "").strip()
    return str(batch_created_at_by_id.get(batch_id, "") or "").strip()


def _section_232_merge_overlapping_rule_candidates(candidates: Sequence[Any]) -> list[Any]:
    grouped_candidates: dict[tuple[str, str, str, str], list[Any]] = {}
    for candidate in candidates:
        grouped_candidates.setdefault(_section_232_material_rule_identity(candidate), []).append(candidate)

    merged_candidates: list[Any] = []
    for group in grouped_candidates.values():
        ordered_group = sorted(
            group,
            key=lambda candidate: (
                *_section_232_publication_window_bounds(candidate),
                str(getattr(candidate, "candidate_id", "") or ""),
            ),
        )
        current = ordered_group[0]
        for candidate in ordered_group[1:]:
            if not _section_232_publication_window_overlaps(current, candidate):
                merged_candidates.append(current)
                current = candidate
                continue

            current_start, current_end = _section_232_publication_window_bounds(current)
            next_start, next_end = _section_232_publication_window_bounds(candidate)
            merged_start = min(current_start, next_start)
            merged_end = max(current_end, next_end)
            if current_start == merged_start and current_end == merged_end:
                representative = current
                secondary = candidate
            elif next_start == merged_start and next_end == merged_end:
                representative = candidate
                secondary = current
            else:
                representative = candidate
                secondary = current

            current = dataclass_replace(
                representative,
                effective_from=merged_start.isoformat(),
                effective_to=None if merged_end == date.max else merged_end.isoformat(),
                source_document_ids=list(
                    dict.fromkeys(
                        [
                            *list(representative.source_document_ids or []),
                            *list(secondary.source_document_ids or []),
                        ]
                    )
                ),
                source_pages=list(
                    dict.fromkeys(
                        [
                            *list(representative.source_pages or []),
                            *list(secondary.source_pages or []),
                        ]
                    )
                ),
                source_excerpt=representative.source_excerpt or secondary.source_excerpt,
                interpreter_confidence=max(
                    float(getattr(representative, "interpreter_confidence", 0.0) or 0.0),
                    float(getattr(secondary, "interpreter_confidence", 0.0) or 0.0),
                ),
                catalog_match_found=bool(
                    getattr(representative, "catalog_match_found", False)
                    or getattr(secondary, "catalog_match_found", False)
                ),
                review_decision="accepted",
            )
        merged_candidates.append(current)
    return merged_candidates


def _section_232_normalized_code(value: Any) -> str:
    return canonicalize_hts_code(getattr(value, "hts_code", value))


def _section_232_group_candidates_by_code(candidates: Sequence[Any]) -> dict[str, list[Any]]:
    grouped_candidates: dict[str, list[Any]] = {}
    for candidate in candidates:
        normalized_code = _section_232_normalized_code(candidate)
        if not normalized_code:
            continue
        grouped_candidates.setdefault(normalized_code, []).append(candidate)
    return grouped_candidates


def _section_232_group_source_uploaded_at(
    candidates: Sequence[Any],
    *,
    source_by_id: Dict[str, Any],
) -> str:
    uploaded_at_values = sorted(
        {
            str(getattr(source_by_id.get(source_id), "uploaded_at", "") or "").strip()
            for candidate in candidates
            for source_id in list(getattr(candidate, "source_document_ids", []) or [])
            if str(getattr(source_by_id.get(source_id), "uploaded_at", "") or "").strip()
        }
    )
    return uploaded_at_values[-1] if uploaded_at_values else ""


def _section_232_group_batch_created_at(
    candidates: Sequence[Any],
    *,
    batch_created_at_by_id: Dict[str, str],
) -> str:
    created_at_values = sorted(
        {
            str(batch_created_at_by_id.get(str(getattr(candidate, "batch_id", "") or ""), "") or "").strip()
            for candidate in candidates
            if str(batch_created_at_by_id.get(str(getattr(candidate, "batch_id", "") or ""), "") or "").strip()
        }
    )
    return created_at_values[-1] if created_at_values else ""


def _section_232_group_precedence_key(
    candidates: Sequence[Any],
    *,
    source_by_id: Dict[str, Any],
    batch_created_at_by_id: Dict[str, str],
) -> tuple[str, str]:
    return (
        _section_232_group_source_uploaded_at(candidates, source_by_id=source_by_id),
        _section_232_group_batch_created_at(candidates, batch_created_at_by_id=batch_created_at_by_id),
    )


@dataclass(frozen=True)
class MetalCompositionPredictInput:
    product_code: str
    selected_source_row_id: Optional[int] = None
    resolution_mode: ResolutionMode = "auto"
    composition_mode: Optional[CompositionMode] = None
    document_mode: DocumentMode = "text_only"
    diagram_payloads: List[DiagramPayload] = field(default_factory=list)
    include_token_usage: bool = False


@dataclass(frozen=True)
class ResolvedItemContext:
    item_id: str
    item_type: Literal["mm"]
    dataset_scope: str
    product_code: str
    source_row_id: Optional[int] = None
    priority: Optional[str] = None
    priority_detail: Optional[str] = None
    business_segment: Optional[str] = None
    site: Optional[str] = None
    pn_revised_standardized: Optional[str] = None
    part_description: Optional[str] = None
    new_part_description: Optional[str] = None
    total_weight_gram: Optional[float] = None
    material_content_method: Optional[str] = None
    material_identified: Optional[str] = None
    date_started: Optional[str] = None
    date_completed: Optional[str] = None


class MissingDocumentsConfirmationRequiredError(Exception):
    def __init__(self, detail: MissingDocumentsConfirmationDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


class ClassificationAlreadyRunningError(Exception):
    def __init__(self, detail: ClassificationJobConflictDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


class ExportReportUnavailableError(Exception):
    """Raised when a PDF export is requested without a completed classification."""


@dataclass(frozen=True)
class ItemSearchMatch:
    item: MetalCompositionItemSummary
    score: float
    match_basis: Literal["product_code", "description"]


@dataclass(frozen=True)
class PredictResolutionResult:
    record: Optional[ResolvedMaterialRecord]
    candidates: List[MetalCompositionCandidate] = field(default_factory=list)
    record_origin: Optional[Literal["mm"]] = None
    error: Optional[str] = None


class _Section232WorkflowRuntimeStore:
    def __init__(self, source_store: object, ruleset_store: object) -> None:
        self._source_store = source_store
        self.ruleset_store = ruleset_store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source_store, name)


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_filter(value: Optional[str], query: Optional[str]) -> bool:
    if not query:
        return True
    return query.lower() in (value or "").lower()


def _matches_facet(value: Optional[str], selected: Optional[str]) -> bool:
    if not selected:
        return True
    return (value or "").strip().lower() == selected.strip().lower()


def _matches_optional_boolean(value: bool, selected: Optional[bool]) -> bool:
    if selected is None:
        return True
    return value is selected


def _normalize_document_mode(value: object) -> DocumentMode:
    """Normalize a document evidence mode value for classification requests.

    Inputs:
        value: Raw mode value supplied by API, Joule, UI, or internal job metadata.

    Expected output:
        ``text_only`` or ``with_documents``.
    """

    text = str(value or "text_only").strip().lower()
    if text in {"", "text_only"}:
        return "text_only"
    if text == "with_documents":
        return "with_documents"
    raise ValueError("document_mode must be 'text_only' or 'with_documents'.")


class MetalCompositionService:
    """Resolve source rows and execute the LangGraph workflow."""

    def __init__(
        self,
        *,
        serving_store: Optional[ServingStore] = None,
        workflow_runner: Optional[MetalCompositionWorkflowRunner] = None,
        settings: Optional[MetalCompositionSettings] = None,
        ui_state_store: Optional[UIStateStoreProtocol] = None,
        classification_job_store: Optional[ClassificationJobStoreProtocol] = None,
        section_232_source_store: Optional[Section232SourceStoreProtocol] = None,
        section_232_ruleset_store: Optional[Section232RulesetStoreProtocol] = None,
        hts_catalog_source_store: Optional[HTSCatalogSourceStoreProtocol] = None,
        startup_timing: Optional[dict] = None,
        cache_source: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.startup_timing = dict(startup_timing or {})
        self.cache_source = cache_source
        self.workflow_runner = workflow_runner
        self.serving_store = serving_store
        self.section_232_source_store = section_232_source_store
        self.section_232_ruleset_store = section_232_ruleset_store
        self.hts_catalog_source_store = hts_catalog_source_store
        self._first_request_lock = Lock()
        self._first_request_pending = True

        if self.serving_store is None:
            load_result = load_serving_store(self.settings)
            self.serving_store = load_result.store
            self.startup_timing = load_result.startup_timing
            self.cache_source = load_result.loaded_from

        if self.section_232_source_store is None:
            self.section_232_source_store = Section232SourceStore(self.settings)

        if self.section_232_ruleset_store is None:
            if isinstance(self.section_232_source_store, InMemorySection232SourceStore):
                self.section_232_ruleset_store = InMemorySection232RulesetStore()
            else:
                self.section_232_ruleset_store = PersistedSection232RulesetStore(
                    self.settings,
                    connection=getattr(self.section_232_source_store, "connection", None),
                )

        if self.hts_catalog_source_store is None:
            try:
                self.hts_catalog_source_store = HTSCatalogSourceStore(self.settings)
            except Exception:
                self.hts_catalog_source_store = InMemoryHTSCatalogSourceStore(self.settings)

        if self.workflow_runner is None:
            self.workflow_runner = MetalCompositionWorkflowRunner(
                self.settings,
                section_232_source_store=self._build_workflow_section_232_runtime_store(),
            )
        else:
            setattr(
                self.workflow_runner,
                "section_232_source_store",
                self._build_workflow_section_232_runtime_store(),
            )

        self.dataset_signature = self._compute_dataset_signature()
        self.document_store = MetalCompositionDocumentStore(self.settings.uploaded_document_root)
        self.ui_state_store = ui_state_store or MetalCompositionUIStateStore(self.settings)
        if classification_job_store is None:
            if isinstance(self.ui_state_store, InMemoryMetalCompositionUIStateStore):
                classification_job_store = InMemoryClassificationJobStore()
            else:
                classification_job_store = ClassificationJobStore(self.settings)
        self.classification_job_store = classification_job_store
        self.item_service = MetalCompositionItemService(self)
        self.classification_service = MetalCompositionClassificationService(self)
        self.job_service = MetalCompositionJobService(self)
        self.section_232_service = MetalCompositionSection232Service(self)

    def _compute_dataset_signature(self) -> str:
        # Only hash the stable source rows. Runtime load metadata can change across
        # restarts (for example HANA vs snapshot fallback) even when the Material Master data is
        # identical, and should not invalidate saved classifications.
        source_df = self.serving_store.source_df.reindex(columns=SIGNATURE_COLUMNS).copy()
        source_df = source_df.sort_values(by=["source_row_id"], kind="stable").reset_index(drop=True)
        signature_payload = {
            "signature_version": 1,
            "rows": source_df.fillna("").to_csv(index=False),
        }
        return hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _get_workflow_runner(self) -> MetalCompositionWorkflowRunner:
        if self.workflow_runner is None:
            raise RuntimeError("workflow_runner was not initialized")
        return self.workflow_runner

    def _get_workflow_llm(self) -> Any:
        workflow_runner = self._get_workflow_runner()
        llm = getattr(workflow_runner, "llm", None)
        if llm is None:
            workflow_runner = MetalCompositionWorkflowRunner(
                self.settings,
                hts_catalog_resolver=getattr(workflow_runner, "hts_catalog_resolver", None),
                section_232_source_store=self._build_workflow_section_232_runtime_store(),
            )
            self.workflow_runner = workflow_runner
            llm = workflow_runner.llm
        return llm

    def _build_workflow_section_232_runtime_store(self) -> object:
        return _Section232WorkflowRuntimeStore(
            source_store=self.section_232_source_store,
            ruleset_store=self.section_232_ruleset_store,
        )

    def get_app_settings(self) -> MetalCompositionAppSettings:
        return self.item_service.get_app_settings()

    def update_app_settings(
        self,
        *,
        use_material_master_metal_composition: bool,
    ) -> MetalCompositionAppSettings:
        return self.item_service.update_app_settings(
            use_material_master_metal_composition=use_material_master_metal_composition,
        )

    def _resolve_effective_composition_mode(
        self,
        request: MetalCompositionPredictInput,
        *,
        record_origin: Optional[Literal["mm"]],
        source_row_id: Optional[int],
    ) -> CompositionMode:
        if record_origin != "mm" or source_row_id is None or int(source_row_id) < 0:
            return "diagram_manual"
        requested_mode = request.composition_mode
        if requested_mode in {"diagram_manual", "material_master"}:
            return requested_mode
        return (
            "material_master"
            if self.get_app_settings().use_material_master_metal_composition
            else "diagram_manual"
        )

    def _composition_mode_for_context(self, context: ResolvedItemContext) -> CompositionMode:
        if context.item_type != "mm" or context.source_row_id is None:
            return "diagram_manual"
        return (
            "material_master"
            if self.get_app_settings().use_material_master_metal_composition
            else "diagram_manual"
        )

    def _build_material_master_composition(
        self,
        *,
        source_row_id: int,
        total_weight_grams: Optional[float],
    ) -> Dict[str, Any]:
        metal_profile = self.serving_store.get_material_master_metal_profile(int(source_row_id))
        top_level_grams = {
            metal: float(value or 0.0)
            for metal, value in metal_profile.top_level_grams.items()
        }
        steel_subtype_grams = {
            subtype: float(value or 0.0)
            for subtype, value in metal_profile.steel_subtype_grams.items()
        }
        estimated_total_metal_grams = float(sum(top_level_grams.values()))
        provenance_notes = [
            "Material Master composition mode was used.",
            "Missing prepared Material Master gram values were normalized to zero.",
        ]
        return {
            "is_metal_item": estimated_total_metal_grams > 0.0,
            "total_weight_grams": total_weight_grams,
            "estimated_total_metal_grams": estimated_total_metal_grams,
            "top_level_grams": top_level_grams,
            "steel_subtype_grams": steel_subtype_grams,
            "confidence": 1.0,
            "reasoning": (
                "Metal composition was sourced directly from Material Master prepared gram columns. "
                "Blank or missing Material Master gram fields were normalized to 0 g."
            ),
            "provenance": {
                "dominant_source": "material_master",
                "top_level_sources": {
                    metal: ("material_master" if float(weight_grams or 0.0) > 0.0 else "none")
                    for metal, weight_grams in top_level_grams.items()
                },
                "steel_subtype_sources": {
                    subtype: ("material_master" if float(weight_grams or 0.0) > 0.0 else "none")
                    for subtype, weight_grams in steel_subtype_grams.items()
                },
                "needs_human_review": False,
                "notes": provenance_notes,
            },
        }

    @staticmethod
    def _build_blocking_reason(*, docs_status: str) -> BlockingReason:
        return BlockingReason(
            code="pdf_required_blocked",
            message=PDF_REQUIRED_BLOCK_MESSAGE,
            docs_status=docs_status,
        )

    def _build_blocked_result(
        self,
        *,
        context: ResolvedItemContext,
        docs_status: str,
        document_mode: DocumentMode = "text_only",
    ) -> MetalCompositionResponse:
        blocking_reason = self._build_blocking_reason(docs_status=docs_status)
        return MetalCompositionResponse(
            status="blocked",
            product_code=context.product_code,
            document_mode=document_mode,
            blocking_reason=blocking_reason,
            error=blocking_reason.message,
        )

    def _consume_cold_start_flag(self) -> bool:
        with self._first_request_lock:
            cold_start = self._first_request_pending
            self._first_request_pending = False
            return cold_start

    def _finalize_timing(
        self,
        *,
        existing_timing: Optional[dict],
        phases: dict,
        request_started_perf: float,
        request_started_at: str,
        cold_start: bool,
    ) -> dict:
        timing = dict(existing_timing or {})
        merged_phases = dict(timing.get("phases") or {})
        merged_phases.update(phases)
        ranked_phases = rank_timings(merged_phases)
        total_timing = finish_timing(request_started_perf, request_started_at)
        timing["phases"] = {
            **merged_phases,
            "request_total": total_timing,
        }
        timing["summary"] = {
            **dict(timing.get("summary") or {}),
            "total_request_duration_ms": float(total_timing["duration_ms"]),
            "slowest_recorded_phase": ranked_phases[0]["name"] if ranked_phases else None,
            "ranked_phases": ranked_phases,
            "cold_start": cold_start,
            "cache_source": self.cache_source or self.serving_store.metadata.get("source"),
        }
        if self.startup_timing:
            timing["startup"] = dict(self.startup_timing)
        return timing

    def _response_with_timing(
        self,
        *,
        response_payload: dict,
        existing_timing: Optional[dict],
        phases: dict,
        request_started_perf: float,
        request_started_at: str,
        cold_start: bool,
    ) -> MetalCompositionResponse:
        assembly_started_perf = perf_counter()
        assembly_started_at = utc_now_iso()
        response = MetalCompositionResponse(**response_payload)
        phases["response_assembly"] = finish_timing(assembly_started_perf, assembly_started_at)
        response.timing = self._finalize_timing(
            existing_timing=existing_timing,
            phases=phases,
            request_started_perf=request_started_perf,
            request_started_at=request_started_at,
            cold_start=cold_start,
        )
        return response

    def _resolve_predict_record(
        self,
        request: MetalCompositionPredictInput,
        *,
        resolve_substeps: dict,
    ) -> PredictResolutionResult:
        lookup_started_perf = perf_counter()
        lookup_started_at = utc_now_iso()
        candidates = self.serving_store.lookup_candidates(request.product_code)
        resolve_substeps["candidate_lookup"] = finish_timing(
            lookup_started_perf,
            lookup_started_at,
        )
        if len(candidates) == 1:
            record_started_perf = perf_counter()
            record_started_at = utc_now_iso()
            record = self.serving_store.get_resolved_record(
                product_code=request.product_code,
                source_row_id=candidates[0].source_row_id,
            )
            resolve_substeps["resolved_record_build"] = finish_timing(
                record_started_perf,
                record_started_at,
            )
            return PredictResolutionResult(record=record, candidates=candidates, record_origin="mm")

        if (
            request.selected_source_row_id is not None
            and any(candidate.source_row_id == int(request.selected_source_row_id) for candidate in candidates)
        ):
            record_started_perf = perf_counter()
            record_started_at = utc_now_iso()
            record = self.serving_store.get_resolved_record(
                product_code=request.product_code,
                source_row_id=int(request.selected_source_row_id),
            )
            resolve_substeps["resolved_record_build"] = finish_timing(
                record_started_perf,
                record_started_at,
            )
            return PredictResolutionResult(record=record, candidates=candidates, record_origin="mm")

        if not candidates:
            return PredictResolutionResult(record=None, candidates=[])

        return PredictResolutionResult(
            record=None,
            candidates=candidates,
            error=(
                "selected_source_row_id did not match any candidate for this product code."
                if request.selected_source_row_id is not None
                else None
            ),
        )

    def predict(self, request: MetalCompositionPredictInput) -> MetalCompositionResponse:
        return self.classification_service.predict(request)

    def predict_batch(
        self,
        items: List[MetalCompositionBatchRequestItem],
        *,
        diagram_map: Optional[Dict[int, List[DiagramPayload]]] = None,
    ) -> List[MetalCompositionResponse]:
        return self.classification_service.predict_batch(items, diagram_map=diagram_map)

    def list_items(
        self,
        *,
        priority: Optional[str] = None,
        business_segment: Optional[str] = None,
        product_code: Optional[str] = None,
        pn_revised_standardized: Optional[str] = None,
        new_part_description: Optional[str] = None,
        part_description: Optional[str] = None,
        has_documents: Optional[bool] = None,
        is_classified: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> MetalCompositionItemListResponse:
        return self.item_service.list_items(
            priority=priority,
            business_segment=business_segment,
            product_code=product_code,
            pn_revised_standardized=pn_revised_standardized,
            new_part_description=new_part_description,
            part_description=part_description,
            has_documents=has_documents,
            is_classified=is_classified,
            limit=limit,
            offset=offset,
        )

    def search_items_by_product_code(self, product_code: str) -> List[ItemSearchMatch]:
        return self.item_service.search_items_by_product_code(product_code)

    def search_items_by_description(self, query: str, *, limit: int = 5) -> List[ItemSearchMatch]:
        return self.item_service.search_items_by_description(query, limit=limit)

    def _get_hts_catalog_resolver_for_admin(self) -> HanaHTSCatalogResolver:
        workflow_runner = self._get_workflow_runner()
        resolver = getattr(workflow_runner, "hts_catalog_resolver", None)
        if resolver is None:
            resolver = HanaHTSCatalogResolver(settings=self.settings)
            setattr(workflow_runner, "hts_catalog_resolver", resolver)
        return resolver

    def _get_existing_hts_catalog_resolver_for_admin(self) -> Optional[HanaHTSCatalogResolver]:
        return getattr(self._get_workflow_runner(), "hts_catalog_resolver", None)

    def _set_hts_catalog_resolver_for_admin(
        self,
        *,
        catalog_frame: pd.DataFrame,
        code_map_frame: pd.DataFrame,
    ) -> None:
        workflow_runner = self._get_workflow_runner()
        setattr(
            workflow_runner,
            "hts_catalog_resolver",
            HanaHTSCatalogResolver(
                settings=self.settings,
                catalog_frame=catalog_frame,
                code_map_frame=code_map_frame,
            ),
        )

    def _build_hts_catalog_summary(self) -> HTSCatalogSummaryResponse:
        managed_sources = list(self.hts_catalog_source_store.list_sources())
        refresh_state = self.hts_catalog_source_store.get_refresh_state()
        chapter_numbers = sorted(
            source.chapter_number
            for source in managed_sources
            if source.chapter_number is not None
        )
        has_code_map = any(source.filename == "hts_code_map.csv" for source in managed_sources)
        catalog_row_count = int(refresh_state.catalog_row_count or 0)
        code_map_row_count = int(refresh_state.code_map_row_count or 0)
        try:
            resolver = self._get_hts_catalog_resolver_for_admin()
            catalog_row_count = int(len(resolver.catalog_frame))
            code_map_row_count = int(len(resolver.code_map_frame))
        except Exception:
            pass

        return HTSCatalogSummaryResponse(
            managed_file_count=len(managed_sources),
            managed_chapter_file_count=len(chapter_numbers),
            loaded_chapters=chapter_numbers,
            has_code_map=has_code_map,
            last_refresh_status=refresh_state.last_refresh_status,
            last_refresh_at=refresh_state.last_refresh_at,
            last_refresh_error=refresh_state.last_refresh_error,
            catalog_row_count=catalog_row_count,
            code_map_row_count=code_map_row_count,
        )

    def _lookup_hts_catalog_description(self, hts_code: str) -> Optional[str]:
        try:
            resolver = self._get_hts_catalog_resolver_for_admin()
        except Exception:
            return None

        row = resolver.catalog_by_code.get(canonicalize_hts_code(hts_code)) or {}
        return (
            str(row.get("path_description") or "").strip()
            or str(row.get("description") or "").strip()
            or None
        )

    @staticmethod
    def _section_232_missing_catalog_match() -> dict[str, object]:
        return {
            "catalog_match_type": "missing",
            "catalog_representative_code": None,
            "catalog_family_match_count": 0,
            "catalog_description": None,
            "catalog_warning": "HTS code not found in managed catalog",
        }

    def _resolve_section_232_catalog_matches(
        self,
        legal_hts_codes: Sequence[str],
    ) -> Dict[str, dict[str, object]]:
        normalized_codes = [
            canonicalize_hts_code(code)
            for code in legal_hts_codes
            if canonicalize_hts_code(code)
        ]
        normalized_codes = list(dict.fromkeys(normalized_codes))
        if not normalized_codes:
            return {}

        resolver = self._get_existing_hts_catalog_resolver_for_admin()
        catalog_frame = getattr(resolver, "catalog_frame", None) if resolver is not None else None
        if catalog_frame is not None and not getattr(catalog_frame, "empty", True):
            return self._resolve_section_232_catalog_matches_from_frame(
                normalized_codes,
                catalog_frame=catalog_frame,
            )
        return self._resolve_section_232_catalog_matches_from_hana(normalized_codes)

    def _resolve_section_232_catalog_matches_from_frame(
        self,
        normalized_codes: Sequence[str],
        *,
        catalog_frame: pd.DataFrame,
    ) -> Dict[str, dict[str, object]]:
        frame = catalog_frame.copy()
        frame["__normalized_code"] = frame["code"].map(canonicalize_hts_code)
        frame["__digit_count"] = frame["__normalized_code"].map(_code_digit_count)
        results: Dict[str, dict[str, object]] = {}
        for normalized_hts_code in normalized_codes:
            results[normalized_hts_code] = self._resolve_section_232_catalog_match_from_rows(
                normalized_hts_code,
                exact_rows=frame.loc[frame["__normalized_code"] == normalized_hts_code],
                family_rows=frame.loc[frame["__normalized_code"].map(lambda code: code.startswith(normalized_hts_code))],
            )
        return results

    def _resolve_section_232_catalog_matches_from_hana(
        self,
        normalized_codes: Sequence[str],
    ) -> Dict[str, dict[str, object]]:
        results = {
            normalized_code: self._section_232_missing_catalog_match()
            for normalized_code in normalized_codes
        }
        connection = HANAConnection()
        schema = self.settings.hts_hana_schema or None
        table_name = self.settings.hts_catalog_hana_table

        def _quote_identifier(identifier: str) -> str:
            return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'

        qualified_table = (
            f'{_quote_identifier(schema)}.{_quote_identifier(table_name)}'
            if schema
            else _quote_identifier(table_name)
        )
        selected_columns = '"CODE", "RAW_CODE", "DESCRIPTION", "PATH_DESCRIPTION"'
        try:
            if not connection.table_exists(table_name, schema=schema):
                return results
            exact_placeholders = ", ".join("?" for _ in normalized_codes)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {selected_columns}
                    FROM {qualified_table}
                    WHERE "CODE" IN ({exact_placeholders})
                    """,
                    list(normalized_codes),
                )
                exact_rows = cursor.fetchall()
                exact_columns = [str(description[0]).lower() for description in (cursor.description or [])]
            exact_records = [
                {exact_columns[index]: value for index, value in enumerate(row)}
                for row in exact_rows
            ]

            prefix_placeholders = " OR ".join('"CODE" LIKE ?' for _ in normalized_codes)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {selected_columns}
                    FROM {qualified_table}
                    WHERE {prefix_placeholders}
                    """,
                    [f"{normalized_code}%" for normalized_code in normalized_codes],
                )
                family_rows = cursor.fetchall()
                family_columns = [str(description[0]).lower() for description in (cursor.description or [])]
            family_records = [
                {family_columns[index]: value for index, value in enumerate(row)}
                for row in family_rows
            ]
        except Exception:
            return results

        exact_by_code: Dict[str, List[Dict[str, object]]] = {}
        for row in exact_records:
            normalized_code = canonicalize_hts_code(row.get("code"))
            exact_by_code.setdefault(normalized_code, []).append(row)

        family_by_code: Dict[str, List[Dict[str, object]]] = {normalized_code: [] for normalized_code in normalized_codes}
        for row in family_records:
            row_code = canonicalize_hts_code(row.get("code"))
            for normalized_code in normalized_codes:
                if row_code.startswith(normalized_code):
                    family_by_code[normalized_code].append(row)

        for normalized_code in normalized_codes:
            results[normalized_code] = self._resolve_section_232_catalog_match_from_rows(
                normalized_code,
                exact_rows=exact_by_code.get(normalized_code, []),
                family_rows=family_by_code.get(normalized_code, []),
            )
        return results

    def _resolve_section_232_catalog_match_from_rows(
        self,
        normalized_hts_code: str,
        *,
        exact_rows: Any,
        family_rows: Any,
    ) -> dict[str, object]:
        missing_result = self._section_232_missing_catalog_match()
        if not normalized_hts_code:
            return missing_result

        exact_records = (
            exact_rows.to_dict(orient="records")
            if hasattr(exact_rows, "to_dict")
            else list(exact_rows or [])
        )
        if exact_records:
            representative_row = exact_records[0]
            representative_code = self._format_section_232_catalog_code(
                representative_row.get("raw_code") or representative_row.get("code") or ""
            )
            description = (
                str(representative_row.get("path_description") or "").strip()
                or str(representative_row.get("description") or "").strip()
                or None
            )
            return {
                "catalog_match_type": "exact",
                "catalog_representative_code": representative_code or normalized_hts_code,
                "catalog_family_match_count": 1,
                "catalog_description": description,
                "catalog_warning": None,
            }

        family_records = (
            family_rows.to_dict(orient="records")
            if hasattr(family_rows, "to_dict")
            else list(family_rows or [])
        )
        if not family_records:
            return missing_result

        prepared_family_records = []
        for row in family_records:
            candidate_code = canonicalize_hts_code(row.get("code"))
            if not candidate_code.startswith(normalized_hts_code):
                continue
            prepared_family_records.append(
                {
                    **row,
                    "__normalized_code": candidate_code,
                    "__digit_count": _code_digit_count(candidate_code),
                }
            )
        if not prepared_family_records:
            return missing_result

        prepared_family_records.sort(
            key=lambda row: (
                -int(row.get("__digit_count") or 0),
                str(row.get("__normalized_code") or ""),
            )
        )
        representative_row = prepared_family_records[0]
        representative_code = self._format_section_232_catalog_code(
            representative_row.get("raw_code") or representative_row.get("code") or ""
        )
        description = (
            str(representative_row.get("path_description") or "").strip()
            or str(representative_row.get("description") or "").strip()
            or None
        )
        return {
            "catalog_match_type": "family",
            "catalog_representative_code": representative_code or None,
            "catalog_family_match_count": int(len(prepared_family_records)),
            "catalog_description": description,
            "catalog_warning": None,
        }

    def _resolve_section_232_catalog_match(self, legal_hts_code: str) -> dict[str, object]:
        normalized_hts_code = canonicalize_hts_code(legal_hts_code)
        if not normalized_hts_code:
            return self._section_232_missing_catalog_match()
        return self._resolve_section_232_catalog_matches([normalized_hts_code]).get(
            normalized_hts_code,
            self._section_232_missing_catalog_match(),
        )

    @staticmethod
    def _format_section_232_catalog_code(value: object) -> str:
        normalized = canonicalize_hts_code(value)
        digits = "".join(char for char in normalized if char.isdigit()) or "".join(
            char for char in str(value) if char.isdigit()
        )
        if len(digits) == 10:
            return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}.{digits[8:10]}"
        if len(digits) == 8:
            return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}"
        if len(digits) == 6:
            return f"{digits[:4]}.{digits[4:6]}"
        return normalized or str(value).strip()

    def _list_active_section_232_eligible_codes(self) -> List[str]:
        return self.section_232_ruleset_store.list_active_eligible_codes(on_date=date.today())

    def _build_section_232_batch_map(self) -> Dict[str, Any]:
        return {
            batch.batch_id: batch
            for batch in self.section_232_ruleset_store.list_draft_batches()
        }

    def _build_section_232_batch_created_at_map(self) -> Dict[str, str]:
        return {
            batch_id: str(getattr(batch, "created_at", "") or "")
            for batch_id, batch in self._build_section_232_batch_map().items()
        }

    def _build_section_232_source_filename_map(self, batch_ids: Sequence[str]) -> Dict[str, str]:
        if not batch_ids:
            return {}
        batch_id_set = set(batch_ids)
        source_filename_by_id: Dict[str, str] = {}
        for batch in self.section_232_ruleset_store.list_draft_batches():
            if batch.batch_id not in batch_id_set:
                continue
            for source_id, filename in zip(batch.source_ids, batch.source_filenames):
                if source_id and filename and source_id not in source_filename_by_id:
                    source_filename_by_id[source_id] = filename
        return source_filename_by_id

    def _build_section_232_source_map(
        self,
        *,
        source_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        selected_ids = {str(value) for value in source_ids or [] if str(value)}
        source_by_id: Dict[str, Any] = {}
        for source in self.section_232_source_store.list_sources():
            if selected_ids and source.source_id not in selected_ids:
                continue
            source_by_id[source.source_id] = source
        return source_by_id

    def _build_section_232_source_documents(
        self,
        *,
        source_document_ids: Sequence[str],
        source_filename_by_id: Dict[str, str],
        source_by_id: Dict[str, Any],
    ) -> List[Section232SourceDocumentMetadata]:
        documents: List[Section232SourceDocumentMetadata] = []
        seen_source_ids: set[str] = set()
        for raw_source_id in source_document_ids or []:
            source_id = str(raw_source_id or "").strip()
            if not source_id or source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            source = source_by_id.get(source_id)
            filename = (
                str(getattr(source, "filename", "") or "").strip()
                or str(source_filename_by_id.get(source_id, "") or "").strip()
                or None
            )
            uploaded_at = str(getattr(source, "uploaded_at", "") or "").strip() or None
            documents.append(
                Section232SourceDocumentMetadata(
                    source_id=source_id,
                    filename=filename,
                    uploaded_at=uploaded_at,
                )
            )
        return documents

    def _build_section_232_draft_rule_item(
        self,
        candidate: Any,
        *,
        source_filename_by_id: Optional[Dict[str, str]] = None,
        source_by_id: Optional[Dict[str, Any]] = None,
        batch_created_at_by_id: Optional[Dict[str, str]] = None,
        catalog_match: Optional[Dict[str, object]] = None,
    ) -> Section232DraftRuleItem:
        resolved_catalog_match = dict(catalog_match or self._resolve_section_232_catalog_match(candidate.hts_code))
        description = resolved_catalog_match["catalog_description"]
        catalog_match_found = resolved_catalog_match["catalog_match_type"] != "missing"
        resolved_source_by_id = dict(source_by_id or {})
        resolved_source_filename_by_id = dict(source_filename_by_id or {})
        resolved_batch_created_at_by_id = dict(batch_created_at_by_id or {})
        if not resolved_source_by_id:
            resolved_source_by_id = self._build_section_232_source_map(
                source_ids=list(candidate.source_document_ids or []),
            )
        for source_id, source in resolved_source_by_id.items():
            filename = str(getattr(source, "filename", "") or "").strip()
            if source_id and filename and source_id not in resolved_source_filename_by_id:
                resolved_source_filename_by_id[source_id] = filename
        if not resolved_source_filename_by_id:
            resolved_source_filename_by_id = {
                source_id: source.filename
                for source_id, source in resolved_source_by_id.items()
            }
        if not resolved_batch_created_at_by_id:
            resolved_batch_created_at_by_id = self._build_section_232_batch_created_at_map()
        match_evidence = collect_section_232_match_evidence(
            list(resolved_source_by_id.values()),
            hts_code=candidate.hts_code,
            source_ids=list(candidate.source_document_ids or []),
            source_pages=list(candidate.source_pages or []),
        )
        source_documents = self._build_section_232_source_documents(
            source_document_ids=list(candidate.source_document_ids or []),
            source_filename_by_id=resolved_source_filename_by_id,
            source_by_id=resolved_source_by_id,
        )
        source_uploaded_at = _section_232_group_source_uploaded_at(
            [candidate],
            source_by_id=resolved_source_by_id,
        ) or None
        return Section232DraftRuleItem(
            candidate_id=candidate.candidate_id,
            legal_hts_code=candidate.hts_code,
            hts_code=candidate.hts_code,
            description=description,
            catalog_match_type=str(resolved_catalog_match["catalog_match_type"]),
            catalog_representative_code=resolved_catalog_match["catalog_representative_code"],
            catalog_family_match_count=int(resolved_catalog_match["catalog_family_match_count"]),
            rule_type=candidate.rule_type,
            coverage_effect=candidate.coverage_effect,
            effective_from=candidate.effective_from,
            effective_to=candidate.effective_to,
            rate_text=getattr(candidate, "rate_text", None),
            metal_scope=candidate.metal_scope,
            source_document_ids=list(candidate.source_document_ids),
            source_filenames=[
                resolved_source_filename_by_id[source_id]
                for source_id in candidate.source_document_ids
                if source_id in resolved_source_filename_by_id
            ],
            source_documents=source_documents,
            source_uploaded_at=source_uploaded_at,
            source_pages=list(candidate.source_pages),
            source_excerpt=candidate.source_excerpt,
            match_evidence=match_evidence,
            interpreter_confidence=candidate.interpreter_confidence,
            catalog_match_found=catalog_match_found,
            catalog_warning=resolved_catalog_match["catalog_warning"],
            candidate_quality=str(getattr(candidate, "candidate_quality", "normal") or "normal"),
            candidate_flags=list(getattr(candidate, "candidate_flags", []) or []),
            processed_at=(
                str(getattr(candidate, "processed_at", "") or "").strip()
                or resolved_batch_created_at_by_id.get(str(getattr(candidate, "batch_id", "") or ""), None)
            ),
            review_decision=candidate.review_decision,
        )

    def _build_section_232_draft_batch_summary(self, batch: Any) -> Section232DraftBatchSummary:
        stats = self.section_232_ruleset_store.get_draft_batch_stats(batch_id=batch.batch_id)
        return Section232DraftBatchSummary(
            batch_id=batch.batch_id,
            status=batch.status,
            source_count=len(batch.source_ids),
            source_filenames=list(batch.source_filenames),
            rule_candidate_count=stats.total,
            pending_count=stats.pending_count,
            accepted_count=stats.accepted_count,
            rejected_count=stats.rejected_count,
            warning_count=stats.warning_count,
            created_at=batch.created_at,
        )

    def _build_section_232_review_row(
        self,
        candidate: Any,
        *,
        source_filename_by_id: Optional[Dict[str, str]] = None,
        source_by_id: Optional[Dict[str, Any]] = None,
        batch_created_at_by_id: Optional[Dict[str, str]] = None,
        catalog_match: Optional[Dict[str, object]] = None,
    ) -> Section232ReviewRow:
        return Section232ReviewRow.model_validate(
            self._build_section_232_draft_rule_item(
                candidate,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
                batch_created_at_by_id=batch_created_at_by_id,
                catalog_match=catalog_match,
            ).model_dump(mode="python")
        )

    def _build_section_232_review_rows_from_candidates(
        self,
        candidates: Sequence[Any],
        *,
        source_filename_by_id: Optional[Dict[str, str]] = None,
        source_by_id: Optional[Dict[str, Any]] = None,
        batch_created_at_by_id: Optional[Dict[str, str]] = None,
    ) -> List[Section232ReviewRow]:
        candidate_list = list(candidates)
        catalog_matches = self._resolve_section_232_catalog_matches(
            [str(getattr(candidate, "hts_code", "") or "") for candidate in candidate_list]
        )
        ordered_candidates = self._sort_section_232_review_candidates(candidate_list, catalog_matches)
        return [
            self._build_section_232_review_row(
                candidate,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
                batch_created_at_by_id=batch_created_at_by_id,
                catalog_match=catalog_matches.get(canonicalize_hts_code(candidate.hts_code)),
            )
            for candidate in ordered_candidates
        ]

    def _build_section_232_draft_rule_items_from_candidates(
        self,
        candidates: Sequence[Any],
        *,
        source_filename_by_id: Optional[Dict[str, str]] = None,
        source_by_id: Optional[Dict[str, Any]] = None,
        batch_created_at_by_id: Optional[Dict[str, str]] = None,
    ) -> List[Section232DraftRuleItem]:
        candidate_list = list(candidates)
        catalog_matches = self._resolve_section_232_catalog_matches(
            [str(getattr(candidate, "hts_code", "") or "") for candidate in candidate_list]
        )
        ordered_candidates = self._sort_section_232_review_candidates(candidate_list, catalog_matches)
        return [
            self._build_section_232_draft_rule_item(
                candidate,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
                batch_created_at_by_id=batch_created_at_by_id,
                catalog_match=catalog_matches.get(canonicalize_hts_code(candidate.hts_code)),
            )
            for candidate in ordered_candidates
        ]

    @staticmethod
    def _section_232_hts_sort_key(hts_code: str) -> tuple[Any, ...]:
        normalized_code = canonicalize_hts_code(hts_code)
        if not normalized_code:
            return (float("inf"), float("inf"), float("inf"), str(hts_code or ""))
        numeric_parts: List[int] = []
        for token in normalized_code.split("."):
            try:
                numeric_parts.append(int(token))
            except ValueError:
                numeric_parts.append(int(1e12))
        padded_parts = (numeric_parts + [-1, -1, -1])[:3]
        return (*padded_parts, normalized_code)

    def _sort_section_232_review_rows(self, rows: Sequence[Any]) -> List[Any]:
        return sorted(
            list(rows),
            key=lambda row: (
                0 if str(getattr(row, "catalog_match_type", "") or "").strip().lower() == "missing" else 1,
                self._section_232_hts_sort_key(getattr(row, "legal_hts_code", "")),
                str(getattr(row, "candidate_id", "") or ""),
            ),
        )

    def _sort_section_232_review_candidates(
        self,
        candidates: Sequence[Any],
        catalog_matches: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> List[Any]:
        catalog_matches = catalog_matches or {}
        return sorted(
            list(candidates),
            key=lambda candidate: (
                0
                if str(
                    (
                        catalog_matches.get(canonicalize_hts_code(getattr(candidate, "hts_code", ""))) or {}
                    ).get("catalog_match_type", "missing")
                )
                .strip()
                .lower()
                == "missing"
                else 1,
                self._section_232_hts_sort_key(str(getattr(candidate, "hts_code", "") or "")),
                str(getattr(candidate, "candidate_id", "") or ""),
            ),
        )

    def get_section_232_ruleset_summary(self) -> Section232RulesetSummaryResponse:
        return self.section_232_service.get_section_232_ruleset_summary()

    def get_section_232_review_workspace(
        self,
        *,
        batch_id: str | None = None,
        version: str | None = None,
        hts_query: str | None = None,
        limit: int = SECTION_232_REVIEW_PAGE_SIZE,
        offset: int = 0,
    ) -> Section232ReviewWorkspaceResponse:
        return self.section_232_service.get_section_232_review_workspace(
            batch_id=batch_id,
            version=version,
            hts_query=hts_query,
            limit=limit,
            offset=offset,
        )

    def _get_pending_section_232_draft_batch(self, batch_id: str) -> Any:
        normalized_batch_id = str(batch_id or "").strip()
        batch = next(
            (
                item
                for item in self.section_232_ruleset_store.list_draft_batches(status="pending_review")
                if item.batch_id == normalized_batch_id
            ),
            None,
        )
        if batch is None:
            raise KeyError(f"draft batch {normalized_batch_id} not found")
        return batch

    def process_section_232_draft_batch(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
        include_token_usage: bool = False,
    ) -> Section232DraftBatchProcessResponse:
        return self.section_232_service.process_section_232_draft_batch(
            uploads=uploads,
            review_page_size=SECTION_232_REVIEW_PAGE_SIZE,
            include_token_usage=include_token_usage,
        )

    def cancel_section_232_draft_batch(self, batch_id: str) -> Section232CancelDraftBatchResponse:
        return self.section_232_service.cancel_section_232_draft_batch(batch_id)

    def list_section_232_draft_rules(
        self,
        *,
        batch_id: str,
        limit: int = SECTION_232_REVIEW_PAGE_SIZE,
        offset: int = 0,
    ) -> Section232DraftRuleListResponse:
        return self.section_232_service.list_section_232_draft_rules(
            batch_id=batch_id,
            limit=limit,
            offset=offset,
        )

    def review_section_232_draft_rule(
        self,
        *,
        batch_id: str,
        candidate_id: str,
        decision: str,
    ) -> Section232DraftRuleReviewResponse:
        return self.section_232_service.review_section_232_draft_rule(
            batch_id=batch_id,
            candidate_id=candidate_id,
            decision=decision,
        )

    def review_section_232_draft_rules(
        self,
        *,
        batch_id: str,
        candidate_ids: Sequence[str],
        selection_mode: str = "explicit",
        excluded_candidate_ids: Sequence[str] = (),
        decision: str,
    ) -> Section232DraftRuleBulkReviewResponse:
        return self.section_232_service.review_section_232_draft_rules(
            batch_id=batch_id,
            candidate_ids=candidate_ids,
            selection_mode=selection_mode,
            excluded_candidate_ids=excluded_candidate_ids,
            decision=decision,
        )

    def delete_section_232_draft_hts_code(
        self,
        *,
        batch_id: str,
        hts_code: str,
    ) -> Section232DraftRuleDeleteResponse:
        return self.section_232_service.delete_section_232_draft_hts_code(
            batch_id=batch_id,
            hts_code=hts_code,
        )

    def _ruleset_store_connection(self) -> Any | None:
        ruleset_connection = getattr(self.section_232_ruleset_store, "connection", None)
        if ruleset_connection is None or not callable(getattr(ruleset_connection, "transaction", None)):
            return None
        return ruleset_connection

    def _build_section_232_publish_snapshot(
        self,
        *,
        accepted_candidates: Sequence[Any],
    ) -> List[Any]:
        publication_slots_by_identity: dict[tuple[str, str], list[Any]] = {}
        for candidate in accepted_candidates:
            publication_slot = _section_232_publication_slot_identity(candidate)
            publication_slots_by_identity.setdefault(publication_slot, []).append(candidate)

        duplicate_publication_slots = {
            publication_slot[0]
            for publication_slot, candidates_for_slot in publication_slots_by_identity.items()
            if len(candidates_for_slot) > 1
        }
        if duplicate_publication_slots:
            duplicate_hts_codes = ", ".join(sorted(duplicate_publication_slots))
            raise ValueError(
                "draft batch has duplicate normalized publication slots for HTS codes "
                f"{duplicate_hts_codes} and cannot be published"
            )

        active_rules = [
            dataclass_replace(candidate, review_decision="accepted")
            for candidate in self.section_232_ruleset_store.list_active_rules()
        ]
        batch_created_at_by_id = self._build_section_232_batch_created_at_map()

        selected_by_slot: dict[tuple[str, str], Any] = {
            _section_232_publication_slot_identity(candidate): candidate
            for candidate in active_rules
        }

        for candidate in accepted_candidates:
            slot = _section_232_publication_slot_identity(candidate)
            code = slot[0]
            incoming_processed_at = _section_232_candidate_processed_at(
                candidate,
                batch_created_at_by_id=batch_created_at_by_id,
            )
            delete_override = self.section_232_ruleset_store.get_delete_override(code)
            if delete_override is not None and incoming_processed_at and incoming_processed_at <= delete_override.deleted_at:
                continue

            existing = selected_by_slot.get(slot)
            if existing is None:
                selected_by_slot[slot] = candidate
                continue

            existing_processed_at = _section_232_candidate_processed_at(
                existing,
                batch_created_at_by_id=batch_created_at_by_id,
            )
            if incoming_processed_at > existing_processed_at:
                selected_by_slot[slot] = candidate

        merged_candidates = list(selected_by_slot.values())
        merged_candidates.sort(
            key=lambda candidate: (
                _section_232_candidate_processed_at(candidate, batch_created_at_by_id=batch_created_at_by_id),
                _section_232_normalized_code(candidate),
                section_232_rule_overlay_identity(candidate)[1],
            )
        )
        return merged_candidates

    def publish_section_232_draft_batch(
        self,
        *,
        batch_id: str,
        published_by: str,
    ) -> Section232PublishDraftBatchResponse:
        return self.section_232_service.publish_section_232_draft_batch(
            batch_id=batch_id,
            published_by=published_by,
        )

    def delete_section_232_published_hts_code(
        self,
        *,
        hts_code: str,
        published_by: str,
    ) -> Section232PublishedRuleDeleteResponse:
        return self.section_232_service.delete_section_232_published_hts_code(
            hts_code=hts_code,
            published_by=published_by,
        )

    def classify_section_232(
        self,
        payload: Section232DirectClassificationRequest,
    ) -> Section232ClassificationResponse:
        return self.section_232_service.classify_section_232(payload)

    def list_hts_catalog_sources(self) -> HTSCatalogSourceListResponse:
        return self.section_232_service.list_hts_catalog_sources()

    def list_section_232_sources(self) -> Section232SourceListResponse:
        return self.section_232_service.list_section_232_sources()

    def list_section_232_eligible_hts_codes(self) -> Section232EligibleCodeListResponse:
        return self.section_232_service.list_section_232_eligible_hts_codes()

    def list_section_232_eligible_hts_code_details(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Section232EligibleCodeDetailListResponse:
        return self.section_232_service.list_section_232_eligible_hts_code_details(
            query=query,
            limit=limit,
            offset=offset,
        )

    def upload_section_232_sources(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
        update_mode: Section232SourceUpdateMode = "append",
    ) -> Section232SourceUploadResponse:
        return self.section_232_service.upload_section_232_sources(
            uploads=uploads,
            update_mode=update_mode,
        )

    def upload_section_232_source(
        self,
        *,
        filename: str,
        content: bytes,
        update_mode: Section232SourceUpdateMode = "append",
    ) -> Section232SourceUploadResponse:
        return self.section_232_service.upload_section_232_source(
            filename=filename,
            content=content,
            update_mode=update_mode,
        )

    def upload_hts_catalog_sources(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
    ) -> HTSCatalogSourceUploadResponse:
        return self.section_232_service.upload_hts_catalog_sources(uploads=uploads)

    def delete_hts_catalog_source(
        self,
        *,
        filename: str,
    ) -> HTSCatalogSourceDeleteResponse:
        return self.section_232_service.delete_hts_catalog_source(filename=filename)

    def _refresh_hts_catalog_source_texts(
        self,
        *,
        source_texts: Dict[str, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        with TemporaryDirectory(prefix="managed-hts-catalog-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for filename, content_text in source_texts.items():
                (tmp_dir / filename).write_text(content_text, encoding="utf-8")
            catalog_frame = compile_hts_catalog_frame(csv_dir=tmp_dir)
            code_map_frame = compile_hts_code_map_frame(code_map_path=tmp_dir / "hts_code_map.csv")
            refresh_result = refresh_hts_catalog_tables(
                settings=self.settings,
                csv_dir=tmp_dir,
                code_map_path=tmp_dir / "hts_code_map.csv",
            )
        return catalog_frame, code_map_frame, refresh_result

    def get_classification_stats(self) -> ClassificationStatsResponse:
        return self.job_service.get_classification_stats()

    def reset_classifications(self) -> ClassificationResetResponse:
        return self.job_service.reset_classifications()

    def _shared_section_232_connection(self) -> Any | None:
        source_connection = getattr(self.section_232_source_store, "connection", None)
        ruleset_connection = getattr(self.section_232_ruleset_store, "connection", None)
        if source_connection is None or source_connection is not ruleset_connection:
            return None
        if not callable(getattr(source_connection, "transaction", None)):
            return None
        return source_connection

    def reset_section_232_data(self) -> Section232ResetResponse:
        return self.section_232_service.reset_section_232_data()

    def get_item_detail(self, item_id: str) -> MetalCompositionItemDetail:
        return self.item_service.get_item_detail(item_id)

    def export_classification_report(self, item_id: str) -> tuple[str, bytes]:
        detail = self.get_item_detail(item_id)
        classification = detail.latest_classification
        if classification is None:
            raise ExportReportUnavailableError("A completed classification is required before exporting a report.")
        if classification.status != "completed":
            raise ExportReportUnavailableError("Only completed classifications can be exported as a report.")

        safe_product_code = re.sub(r"[^A-Za-z0-9._-]+", "-", detail.product_code or "classification-report").strip("-")
        filename = f"{safe_product_code or 'classification-report'}-classification-report.pdf"
        return filename, build_classification_report_pdf(detail)

    def replace_item_documents(
        self,
        item_id: str,
        *,
        document_paths: Sequence[str],
    ) -> MetalCompositionItemDetail:
        return self.item_service.replace_item_documents(item_id, document_paths=document_paths)

    def upload_item_document(
        self,
        item_id: str,
        *,
        filename: str,
        content: bytes,
    ) -> MetalCompositionItemDetail:
        return self.item_service.upload_item_document(
            item_id,
            filename=filename,
            content=content,
        )

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

    @staticmethod
    def _job_ref(job: PersistedClassificationJob) -> ClassificationJobRef:
        return ClassificationJobRef(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
        )

    def _raise_missing_documents_for_contexts(
        self,
        contexts: Sequence[ResolvedItemContext],
        docs_statuses: Sequence[str],
    ) -> None:
        self.job_service._raise_missing_documents_for_contexts(contexts, docs_statuses)

    def _supersede_active_classifications(
        self,
        contexts: Sequence[ResolvedItemContext],
    ) -> None:
        self.job_service._supersede_active_classifications(contexts)

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
            item_id: Material Master item id in the form ``mm:<source_row_id>``.
            document_mode: Whether assigned PDFs should be ignored or used as extra evidence.
            include_token_usage: API-only diagnostic flag for recording model token metadata.
            job_type: Classification job type recorded for the async worker.

        Expected output:
            Background job submission metadata for polling.
        """

        return self.job_service.submit_predict_item_job(
            item_id,
            document_mode=_normalize_document_mode(document_mode),
            include_token_usage=include_token_usage,
            job_type=job_type,
        )

    def submit_classify_items_job(
        self,
        item_ids: Sequence[str],
        *,
        confirm_missing_documents: bool = False,
        job_type: ClassificationJobType = "batch",
    ) -> ClassificationJobSubmissionResponse:
        return self.submit_predict_items_job(
            item_ids,
            document_mode="text_only",
            job_type=job_type,
        )

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
            item_ids: Material Master item ids in request order.
            document_mode: Batch-wide document evidence mode.
            include_token_usage: API-only diagnostic flag for recording model token metadata.
            job_type: Classification job type recorded for the async worker.

        Expected output:
            Background job submission metadata for polling.
        """

        return self.job_service.submit_predict_items_job(
            item_ids,
            document_mode=_normalize_document_mode(document_mode),
            include_token_usage=include_token_usage,
            job_type=job_type,
        )

    def get_classification_job(self, job_id: str) -> ClassificationJobStatusResponse:
        return self.job_service.get_classification_job(job_id)

    def claim_next_classification_job(self, worker_id: str) -> Optional[PersistedClassificationJob]:
        return self.job_service.claim_next_classification_job(worker_id)

    def process_claimed_classification_job(self, job_id: str) -> None:
        self.job_service.process_claimed_classification_job(job_id)

    def drain_classification_jobs(self, worker_id: str = "test-worker") -> int:
        return self.job_service.drain_classification_jobs(worker_id)

    def classify_item(
        self,
        item_id: str,
        *,
        owner_job_id: Optional[str] = None,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
    ) -> ItemClassificationResponse:
        return self.classification_service.classify_item(
            item_id,
            owner_job_id=owner_job_id,
            document_mode=_normalize_document_mode(document_mode),
            include_token_usage=include_token_usage,
        )

    def classify_items(
        self,
        item_ids: Sequence[str],
        *,
        confirm_missing_documents: bool = False,
        owner_job_id: Optional[str] = None,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
    ) -> ItemClassifyBatchResponse:
        return self.classification_service.classify_items(
            item_ids,
            confirm_missing_documents=confirm_missing_documents,
            owner_job_id=owner_job_id,
            document_mode=_normalize_document_mode(document_mode),
            include_token_usage=include_token_usage,
        )

    def _list_base_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for row in self.serving_store.source_df.to_dict("records"):
            source_row_id = int(row["source_row_id"])
            items.append(
                {
                    "item_id": f"mm:{source_row_id}",
                    "item_type": "mm",
                    "dataset_scope": self.dataset_signature,
                    "source_row_id": source_row_id,
                    "product_code": _optional_text(row.get(PRODUCT_CODE_COLUMN)),
                    "priority": _optional_text(row.get(PRIORITY_COLUMN)),
                    "business_segment": _optional_text(row.get(BUSINESS_SEGMENT_COLUMN)),
                    "pn_revised_standardized": _optional_text(row.get(PN_COLUMN)),
                    "new_part_description": _optional_text(row.get(NEW_PART_DESCRIPTION_COLUMN)),
                    "part_description": _optional_text(row.get(PART_DESCRIPTION_COLUMN)),
                    "site": _optional_text(row.get(SITE_COLUMN)),
                    "total_weight_gram": _optional_float(row.get(TOTAL_WEIGHT_COLUMN)),
                }
            )

        return items

    def _build_facets(self, items: Sequence[Dict[str, Any]]) -> Dict[str, List[MetalCompositionFacetOption]]:
        def count_values(field_name: str) -> List[MetalCompositionFacetOption]:
            counts: Dict[str, int] = {}
            for item in items:
                value = item.get(field_name)
                if not value:
                    continue
                counts[str(value)] = counts.get(str(value), 0) + 1
            return [
                MetalCompositionFacetOption(value=value, count=count)
                for value, count in sorted(counts.items(), key=lambda entry: entry[0].lower())
            ]

        return {
            "priority": count_values("priority"),
            "business_segment": count_values("business_segment"),
        }

    def _build_item_summaries(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        assignment_map: Optional[Dict[tuple[str, str], List[StoredDocumentReference]]] = None,
        snapshot_map: Optional[Dict[tuple[str, str], Any]] = None,
        has_documents_cache: Optional[Dict[tuple[str, str], bool]] = None,
    ) -> List[MetalCompositionItemSummary]:
        if not items:
            return []

        assignment_map = assignment_map or self.ui_state_store.get_document_assignment_map(
            [(item["item_id"], item["dataset_scope"]) for item in items]
        )
        snapshot_map = snapshot_map or self.ui_state_store.get_classification_snapshot_map(
            [(item["item_id"], item["dataset_scope"]) for item in items]
        )
        has_documents_cache = has_documents_cache or {}

        summaries: List[MetalCompositionItemSummary] = []
        for item in items:
            key = (item["item_id"], item["dataset_scope"])
            assigned_documents = self.document_store.references_from_stored(assignment_map.get(key, []))
            snapshot = snapshot_map.get(key)
            docs_status = self._list_docs_status(item, assigned_documents)
            summaries.append(
                MetalCompositionItemSummary(
                    item_id=item["item_id"],
                    item_type=item["item_type"],
                    source_row_id=item["source_row_id"],
                    product_code=item["product_code"] or "",
                    priority=item["priority"],
                    business_segment=item["business_segment"],
                    pn_revised_standardized=item["pn_revised_standardized"],
                    new_part_description=item["new_part_description"],
                    part_description=item["part_description"],
                    site=item["site"],
                    total_weight_gram=item["total_weight_gram"],
                    has_documents=has_documents_cache.get(
                        key,
                        docs_status.startswith("Assigned"),
                    ),
                    docs_status=docs_status,
                    is_classified=snapshot is not None,
                    classification_status=snapshot.status if snapshot is not None else "not_classified",
                    last_classified_at=snapshot.last_classified_at if snapshot is not None else None,
                )
            )
        return summaries

    def _item_has_documents(
        self,
        item: Dict[str, Any],
        *,
        assigned_documents: Sequence[Any],
    ) -> bool:
        if assigned_documents:
            return True
        return False

    def _list_docs_status(self, item: Dict[str, Any], assigned_documents: Sequence[Any]) -> str:
        del item
        if assigned_documents:
            return f"Assigned ({len(assigned_documents)})"
        return "No PDFs assigned"

    def _detail_docs_status(
        self,
        context: ResolvedItemContext,
        *,
        assigned_documents: Sequence[Any],
    ) -> str:
        del context
        if assigned_documents:
            return f"Assigned ({len(assigned_documents)})"
        return "No PDFs assigned"

    def _context_document_state(self, context: ResolvedItemContext) -> tuple[bool, str]:
        assigned_refs = self.ui_state_store.get_document_assignments(
            context.item_id,
            dataset_scope=context.dataset_scope,
        )
        assigned_documents = self.document_store.references_from_stored(assigned_refs)
        if assigned_documents:
            return True, self._detail_docs_status(
                context,
                assigned_documents=assigned_documents,
            )

        return False, self._detail_docs_status(
            context,
            assigned_documents=[],
        )

    def _classification_requires_documents(
        self,
        context: ResolvedItemContext,
        *,
        document_mode: DocumentMode,
    ) -> bool:
        """Return whether the selected classification path must load assigned PDFs.

        Inputs:
            context: Resolved Material Master item context.
            document_mode: Requested document evidence mode for this run.

        Expected output:
            True when missing PDFs should block job submission or execution.
        """

        if self._composition_mode_for_context(context) != "material_master":
            return True
        return document_mode == "with_documents"

    def _resolve_item_context(self, item_id: str) -> ResolvedItemContext:
        if item_id.startswith("mm:"):
            _, _, raw_id = item_id.partition(":")
            try:
                source_row_id = int(raw_id)
            except ValueError as exc:
                raise KeyError(f"Invalid Material Master item id: {item_id}") from exc
            if source_row_id not in self.serving_store._source_by_row_id.index:  # noqa: SLF001
                raise KeyError(f"Material Master item {item_id} was not found in the current serving dataset")
            row = self.serving_store._source_by_row_id.loc[source_row_id]  # noqa: SLF001
            return ResolvedItemContext(
                item_id=item_id,
                item_type="mm",
                dataset_scope=self.dataset_signature,
                source_row_id=source_row_id,
                product_code=_optional_text(row.get(PRODUCT_CODE_COLUMN)) or "",
                priority=_optional_text(row.get(PRIORITY_COLUMN)),
                priority_detail=_optional_text(row.get(PRIORITY_DETAIL_COLUMN)),
                business_segment=_optional_text(row.get(BUSINESS_SEGMENT_COLUMN)),
                site=_optional_text(row.get(SITE_COLUMN)),
                pn_revised_standardized=_optional_text(row.get(PN_COLUMN)),
                part_description=_optional_text(row.get(PART_DESCRIPTION_COLUMN)),
                new_part_description=_optional_text(row.get(NEW_PART_DESCRIPTION_COLUMN)),
                total_weight_gram=_optional_float(row.get(TOTAL_WEIGHT_COLUMN)),
                material_content_method=_optional_text(row.get(MATERIAL_CONTENT_METHOD_COLUMN)),
                material_identified=_optional_text(row.get(MATERIAL_IDENTIFIED_COLUMN)),
                date_started=_optional_text(row.get(DATE_STARTED_COLUMN)),
                date_completed=_optional_text(row.get(DATE_COMPLETED_COLUMN)),
            )

        raise KeyError(f"Material Master item {item_id} was not found in the current serving dataset")

    def _selected_source_row_id_for_context(self, context: ResolvedItemContext) -> Optional[int]:
        if context.item_type != "mm":
            return None
        if context.source_row_id is None:
            raise ValueError(f"Material Master item {context.item_id} is missing source_row_id")
        return int(context.source_row_id)

    @staticmethod
    def _resolution_mode_for_context(context: ResolvedItemContext) -> ResolutionMode:
        del context
        return "auto"

    def _build_predict_input_for_context(
        self,
        context: ResolvedItemContext,
        *,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
    ) -> MetalCompositionPredictInput:
        composition_mode = self._composition_mode_for_context(context)
        diagram_payloads = (
            self._load_item_diagram_payloads(context)
            if self._classification_requires_documents(context, document_mode=document_mode)
            else []
        )
        return MetalCompositionPredictInput(
            product_code=context.product_code,
            selected_source_row_id=self._selected_source_row_id_for_context(context),
            resolution_mode=self._resolution_mode_for_context(context),
            composition_mode=composition_mode,
            document_mode=document_mode,
            diagram_payloads=diagram_payloads,
            include_token_usage=include_token_usage,
        )

    def _build_batch_request_for_context(
        self,
        context: ResolvedItemContext,
        *,
        document_mode: DocumentMode = "text_only",
        include_token_usage: bool = False,
    ) -> tuple[MetalCompositionBatchRequestItem, List[DiagramPayload]]:
        composition_mode = self._composition_mode_for_context(context)
        diagram_payloads = (
            self._load_item_diagram_payloads(context)
            if self._classification_requires_documents(context, document_mode=document_mode)
            else []
        )
        return (
            MetalCompositionBatchRequestItem(
                product_code=context.product_code,
                selected_source_row_id=self._selected_source_row_id_for_context(context),
                resolution_mode=self._resolution_mode_for_context(context),
                document_mode=document_mode,
                include_token_usage=include_token_usage,
            ),
            diagram_payloads,
        )

    def _load_item_diagram_payloads(self, context: ResolvedItemContext) -> List[DiagramPayload]:
        assigned_refs = self.ui_state_store.get_document_assignments(
            context.item_id,
            dataset_scope=context.dataset_scope,
        )
        if assigned_refs:
            document_paths = [reference.path for reference in self.document_store.references_from_stored(assigned_refs)]
        else:
            document_paths = []
        return self._load_diagram_payloads(document_paths)

    @staticmethod
    def _load_diagram_payloads(document_paths: Sequence[str]) -> List[DiagramPayload]:
        payloads: List[DiagramPayload] = []
        for raw_path in document_paths:
            file_path = Path(raw_path).expanduser().resolve()
            with file_path.open("rb") as handle:
                payloads.append(
                    DiagramPayload(
                        filename=file_path.name,
                        content_type="application/pdf",
                        data=handle.read(),
                        source_filename=file_path.name,
                    )
                )
        return payloads


@lru_cache(maxsize=1)
def get_metal_composition_service() -> MetalCompositionService:
    """Dependency factory for the metal composition router."""

    return MetalCompositionService()


def warm_metal_composition_service() -> MetalCompositionService:
    """Prewarm the metal composition service during application startup."""

    return get_metal_composition_service()
