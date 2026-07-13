"""Pydantic models for the metal composition API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


ResolutionMode = Literal["auto"]
SourceKind = Literal["mm"]
ClassificationJobType = Literal["single", "batch", "chat"]
ClassificationJobStatus = Literal["queued", "running", "completed", "failed", "partial_failed"]
CompositionMode = Literal["diagram_manual", "material_master"]
DocumentMode = Literal["text_only", "with_documents"]


class MetalCompositionCandidate(BaseModel):
    """Candidate Material Master row returned when product code lookup is ambiguous."""

    source_row_id: int
    source_kind: SourceKind = "mm"
    pn_revised_standardized: Optional[str] = None
    part_description: Optional[str] = None
    new_part_description: Optional[str] = None
    priority_detail: Optional[str] = None
    site: Optional[str] = None
    business_segment: Optional[str] = None
    total_weight_gram: Optional[float] = None
    date_started: Optional[str] = None
    date_completed: Optional[str] = None


class MetalTopLevelComposition(BaseModel):
    steel: float = 0.0
    aluminum: float = 0.0
    copper: float = 0.0
    cast_iron: float = 0.0


class MetalSteelSubtypeComposition(BaseModel):
    electrical_steel: float = 0.0
    cold_rolled_coil_steel: float = 0.0
    hot_rolled_coil_steel: float = 0.0
    stainless_steel_304: float = 0.0
    stainless_steel_316: float = 0.0
    stainless_steel_bar: float = 0.0
    duplex_steel: float = 0.0
    cast_steel: float = 0.0


class CompositionEvidenceProvenance(BaseModel):
    dominant_source: str = "unknown"
    top_level_sources: Dict[str, str] = Field(default_factory=dict)
    steel_subtype_sources: Dict[str, str] = Field(default_factory=dict)
    needs_human_review: bool = False
    notes: List[str] = Field(default_factory=list)


class CompositionSourceDocument(BaseModel):
    filename: str
    page_number: Optional[int] = None


class CompositionPresentationRow(BaseModel):
    type: str
    weight_grams: float = 0.0
    source_documents: List[CompositionSourceDocument] = Field(default_factory=list)
    source_status: str = "none"


class FinalMetalComposition(BaseModel):
    is_metal_item: bool
    total_weight_grams: Optional[float] = None
    estimated_total_metal_grams: float = 0.0
    top_level_grams: MetalTopLevelComposition
    steel_subtype_grams: MetalSteelSubtypeComposition
    provenance: CompositionEvidenceProvenance = Field(default_factory=CompositionEvidenceProvenance)
    confidence: float = 0.0
    reasoning: str
    metal_rows: List[CompositionPresentationRow] = Field(default_factory=list)
    steel_subtype_rows: List[CompositionPresentationRow] = Field(default_factory=list)


class BlockingReason(BaseModel):
    code: str
    message: str
    docs_status: str


class HTSCitation(BaseModel):
    page_number: Optional[int] = None
    page_label: Optional[str] = None
    chapter_number: Optional[int] = None
    heading_code: Optional[str] = None
    source_url: Optional[str] = None
    ruling_number: Optional[str] = None


class HTSCandidate(BaseModel):
    code: str
    description: str
    digits: int
    confidence: float = 0.0
    reasoning: str = ""
    validation_status: Optional[str] = None
    normalized_from: Optional[str] = None
    resolution_basis: Optional[str] = None
    citations: List[HTSCitation] = Field(default_factory=list)
    origins: List[str] = Field(default_factory=list)
    hana_router_stage: Optional[str] = None
    specificity_supported: Optional[int] = None
    matched_path: Optional[str] = None
    matched_terms: List[str] = Field(default_factory=list)
    matched_phrases: List[str] = Field(default_factory=list)
    retrieval_score: float = 0.0
    missing_discriminators: List[str] = Field(default_factory=list)


class HTSClassification(BaseModel):
    status: Literal["completed", "failed", "unavailable", "omitted"]
    best_candidate: Optional[HTSCandidate] = None
    candidates: List[HTSCandidate] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    needs_human_review: bool = True


class Section232Assessment(BaseModel):
    status: Literal["completed", "failed", "unavailable", "omitted"]
    decision: Literal["subject", "not_subject", "needs_review"] = "needs_review"
    confidence: float = 0.0
    basis_summary: str = ""
    needs_human_review: bool = False
    weight_rule_applied: bool = False
    supporting_hts_candidates: List[str] = Field(default_factory=list)
    chapter_99_candidates: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class Section232TopLevelComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steel: float = 0.0
    aluminum: float = 0.0
    copper: float = 0.0


class Section232DirectClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hts_code: str
    supporting_hts_candidates: List[str] = Field(default_factory=list)
    total_weight_grams: Optional[float] = None
    top_level_grams: Section232TopLevelComposition
    metal_share_certainty: Literal["exact", "estimated"] = "exact"

    @model_validator(mode="after")
    def validate_hts_code_present(self) -> "Section232DirectClassificationRequest":
        if not str(self.hts_code or "").strip():
            raise ValueError("hts_code is required")
        return self


class Section232ClassificationResponse(BaseModel):
    section_232_assessment: Section232Assessment
    section_232_reasoner_output: Dict[str, Any] = Field(default_factory=dict)


class MetalCompositionAgentOutputs(BaseModel):
    diagram: Dict[str, Any] = Field(default_factory=dict)
    hts_fact_profile: Dict[str, Any] = Field(default_factory=dict)
    hana_tree_search: Dict[str, Any] = Field(default_factory=dict)
    hts_resolution: Dict[str, Any] = Field(default_factory=dict)
    section_232_reasoner: Dict[str, Any] = Field(default_factory=dict)


class TokenUsageEntry(BaseModel):
    phase: str
    task: str
    model: str
    call_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    usage_available: bool = False


class TokenUsageSummary(BaseModel):
    entries: List[TokenUsageEntry] = Field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    missing_usage_entry_count: int = 0


class MetalCompositionResponse(BaseModel):
    status: Literal["needs_disambiguation", "completed", "not_found", "failed", "blocked"]
    product_code: str
    document_mode: Optional[DocumentMode] = None
    selected_source: Optional[MetalCompositionCandidate] = None
    candidates: List[MetalCompositionCandidate] = Field(default_factory=list)
    final_composition: Optional[FinalMetalComposition] = None
    hts_classification: Optional[HTSClassification] = None
    section_232_assessment: Optional[Section232Assessment] = None
    agent_outputs: Optional[MetalCompositionAgentOutputs] = None
    token_usage: Optional[TokenUsageSummary] = None
    timing: Dict[str, Any] = Field(default_factory=dict)
    blocking_reason: Optional[BlockingReason] = None
    error: Optional[str] = None


class ItemPredictRequest(BaseModel):
    """Background prediction request for one Material Master item.

    Inputs:
        item_id: Material Master item identifier in the form ``mm:<source_row_id>``.
        document_mode: ``text_only`` ignores assigned PDFs; ``with_documents`` uses them as extra HTS evidence.
        include_token_usage: API-only diagnostic flag that records provider token metadata in the saved result.

    Expected output:
        A validated request object used to submit one background classification job.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    document_mode: DocumentMode = "text_only"
    include_token_usage: bool = False


class ItemPredictBatchRequest(BaseModel):
    """Background prediction request for multiple Material Master items.

    Inputs:
        item_ids: Material Master item identifiers in request order.
        document_mode: Batch-wide document evidence mode.
        include_token_usage: API-only diagnostic flag that records provider token metadata in saved results.

    Expected output:
        A validated request object used to submit one background classification job.
    """

    model_config = ConfigDict(extra="forbid")

    item_ids: List[str] = Field(default_factory=list)
    document_mode: DocumentMode = "text_only"
    include_token_usage: bool = False


class MetalCompositionAppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_material_master_metal_composition: bool = True
    updated_at: Optional[str] = None


class MaterialMasterHanaRefreshResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"]
    uploaded_filename: str
    uploaded_size_bytes: int
    source_path: Optional[str] = None
    stored_workbook_path: str
    sheet_name: str
    hana_schema: str
    hana_table: str
    source_row_count: int
    prepared_row_count: int
    refresh_result: Dict[str, Any] = Field(default_factory=dict)
    cleared_classification_count: int = 0
    cancelled_job_count: int = 0
    service_cache_invalidated: bool = True


class MetalCompositionAppSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_material_master_metal_composition: bool = True


class MetalCompositionFacetOption(BaseModel):
    value: str
    count: int = 0


class MetalCompositionItemSummary(BaseModel):
    item_id: str
    item_type: Literal["mm"]
    source_row_id: Optional[int] = None
    product_code: str
    priority: Optional[str] = None
    business_segment: Optional[str] = None
    pn_revised_standardized: Optional[str] = None
    new_part_description: Optional[str] = None
    part_description: Optional[str] = None
    site: Optional[str] = None
    total_weight_gram: Optional[float] = None
    has_documents: bool = False
    docs_status: str = "No documents"
    is_classified: bool = False
    classification_status: str = "not_classified"
    last_classified_at: Optional[str] = None


class MetalCompositionItemListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[MetalCompositionItemSummary] = Field(default_factory=list)
    facets: Dict[str, List[MetalCompositionFacetOption]] = Field(default_factory=dict)


class DocumentReference(BaseModel):
    path: str
    relative_path: str
    file_name: str
    source: Literal["uploaded"]
    size_bytes: Optional[int] = None


class DocumentAssignmentRequest(BaseModel):
    document_paths: List[str] = Field(default_factory=list)


class Section232SourceSummary(BaseModel):
    source_id: str
    filename: str
    size_bytes: int = 0
    page_count: int = 0
    extraction_status: Literal["completed", "failed", "partial"] = "completed"
    hts_mention_count: int = 0
    uploaded_at: str
    warnings: List[str] = Field(default_factory=list)


Section232SourceUpdateMode = Literal["append", "replace"]


class Section232SourceListResponse(BaseModel):
    total: int
    eligible_hts_code_count: int = 0
    items: List[Section232SourceSummary] = Field(default_factory=list)


class Section232SourceUploadResponse(BaseModel):
    update_mode: Section232SourceUpdateMode
    source_count: int = 0
    uploaded_hts_code_count: int = 0
    total_eligible_hts_code_count: int = 0
    items: List[Section232SourceSummary] = Field(default_factory=list)


class Section232DraftBatchSummary(BaseModel):
    batch_id: str
    status: str
    source_count: int = 0
    source_filenames: List[str] = Field(default_factory=list)
    rule_candidate_count: int = 0
    pending_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    warning_count: int = 0
    created_at: str


class Section232MatchEvidence(BaseModel):
    source_id: str
    source_filename: Optional[str] = None
    page_number: int = 0
    matched_text: str = ""
    normalized_hts_code: str = ""
    context_text: str = ""
    text_sources: List[Literal["plain", "layout"]] = Field(default_factory=list)


class Section232SourceDocumentMetadata(BaseModel):
    source_id: str
    filename: Optional[str] = None
    uploaded_at: Optional[str] = None


class Section232DraftRuleItem(BaseModel):
    candidate_id: str
    legal_hts_code: str
    hts_code: str
    description: Optional[str] = None
    catalog_match_type: Literal["exact", "family", "missing"] = "missing"
    catalog_representative_code: Optional[str] = None
    catalog_family_match_count: int = 0
    rule_type: Literal["include", "remove", "rate_schedule"]
    coverage_effect: Literal["include", "remove"]
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    rate_text: Optional[str] = None
    metal_scope: str
    source_document_ids: List[str] = Field(default_factory=list)
    source_filenames: List[str] = Field(default_factory=list)
    source_documents: List[Section232SourceDocumentMetadata] = Field(default_factory=list)
    source_uploaded_at: Optional[str] = None
    source_pages: List[int] = Field(default_factory=list)
    source_excerpt: str = ""
    match_evidence: List[Section232MatchEvidence] = Field(default_factory=list)
    interpreter_confidence: float = 0.0
    catalog_match_found: bool = False
    catalog_warning: Optional[str] = None
    candidate_quality: Literal["normal", "suspect"] = "normal"
    candidate_flags: List[str] = Field(default_factory=list)
    processed_at: Optional[str] = None
    review_decision: Literal["pending", "accepted", "rejected"] = "pending"


class Section232DraftRuleListResponse(BaseModel):
    total: int
    limit: int = 0
    offset: int = 0
    items: List[Section232DraftRuleItem] = Field(default_factory=list)


class Section232DraftRuleReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accepted", "rejected"]


class Section232DraftRuleReviewResponse(BaseModel):
    item: Section232DraftRuleItem


class Section232DraftRuleBulkReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_mode: Literal["explicit", "all"] = "explicit"
    candidate_ids: List[str] = Field(default_factory=list)
    excluded_candidate_ids: List[str] = Field(default_factory=list)
    decision: Literal["accepted", "rejected"]

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "Section232DraftRuleBulkReviewRequest":
        def _normalize(values: List[str], *, field_name: str) -> List[str]:
            normalized_values: List[str] = []
            seen_values: set[str] = set()
            for candidate_id in values:
                normalized_candidate_id = str(candidate_id or "").strip()
                if not normalized_candidate_id:
                    raise ValueError(f"{field_name} must contain non-empty values")
                if normalized_candidate_id in seen_values:
                    continue
                seen_values.add(normalized_candidate_id)
                normalized_values.append(normalized_candidate_id)
            return normalized_values

        self.candidate_ids = _normalize(self.candidate_ids, field_name="candidate_ids")
        self.excluded_candidate_ids = _normalize(
            self.excluded_candidate_ids,
            field_name="excluded_candidate_ids",
        )
        if self.selection_mode == "explicit" and not self.candidate_ids:
            raise ValueError("candidate_ids must contain at least one value when selection_mode='explicit'")
        if self.selection_mode == "all" and self.candidate_ids:
            raise ValueError("candidate_ids must be empty when selection_mode='all'")
        return self


class Section232DraftRuleBulkReviewResponse(BaseModel):
    updated_count: int = 0
    candidate_ids: List[str] = Field(default_factory=list)
    decision: Literal["accepted", "rejected"]


class Section232DraftRuleDeleteResponse(BaseModel):
    batch_id: str
    deleted_hts_code: str
    deleted_count: int = 0


class Section232PublishedRuleDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    published_by: str

    @model_validator(mode="after")
    def validate_published_by_present(self) -> "Section232PublishedRuleDeleteRequest":
        if not str(self.published_by or "").strip():
            raise ValueError("published_by is required")
        return self


class Section232PublishedRuleDeleteResponse(BaseModel):
    deleted_hts_code: str
    removed_rule_count: int = 0
    published_version: str
    ruleset_summary: "Section232RulesetSummaryResponse"


class Section232ReviewHistoryItem(BaseModel):
    version: str
    published_at: Optional[str] = None
    published_by: Optional[str] = None
    candidate_id: str
    legal_hts_code: str
    hts_code: str
    rule_type: Literal["include", "remove", "rate_schedule"]
    coverage_effect: Literal["include", "remove"]
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    metal_scope: str
    source_document_ids: List[str] = Field(default_factory=list)
    source_filenames: List[str] = Field(default_factory=list)
    source_documents: List[Section232SourceDocumentMetadata] = Field(default_factory=list)
    source_uploaded_at: Optional[str] = None
    processed_at: Optional[str] = None


class Section232ReviewRow(Section232DraftRuleItem):
    history: List[Section232ReviewHistoryItem] = Field(default_factory=list)


class Section232ReviewWorkspaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["draft", "published"]
    batch: Optional[Section232DraftBatchSummary] = None
    version: Optional[str] = None
    source_filenames: List[str] = Field(default_factory=list)
    total: int = 0
    limit: int = 0
    offset: int = 0
    rows: List[Section232ReviewRow] = Field(default_factory=list)


class Section232EligibleCodeListResponse(BaseModel):
    total: int = 0
    codes: List[str] = Field(default_factory=list)


class Section232EligibleCodeDetail(BaseModel):
    code: str
    description: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None


class Section232EligibleCodeDetailListResponse(BaseModel):
    total: int = 0
    limit: int = 0
    offset: int = 0
    query: Optional[str] = None
    items: List[Section232EligibleCodeDetail] = Field(default_factory=list)


class Section232RulesetSummaryResponse(BaseModel):
    active_ruleset_version: Optional[str] = None
    eligible_hts_code_count: int = 0
    pending_draft_batch_count: int = 0
    pending_draft_batches: List[Section232DraftBatchSummary] = Field(default_factory=list)
    last_published_at: Optional[str] = None


class Section232DraftBatchProcessResponse(BaseModel):
    batch: Section232DraftBatchSummary
    ruleset_summary: Section232RulesetSummaryResponse
    token_usage: Optional[TokenUsageSummary] = None
    total: int = 0
    limit: int = 0
    offset: int = 0
    items: List[Section232DraftRuleItem] = Field(default_factory=list)


class Section232CancelDraftBatchResponse(BaseModel):
    batch_id: str
    deleted_source_count: int = 0
    deleted_source_filenames: List[str] = Field(default_factory=list)
    deleted_draft_rule_count: int = 0
    ruleset_summary: Section232RulesetSummaryResponse


class Section232PublishDraftBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    published_by: str

    @model_validator(mode="after")
    def validate_published_by_present(self) -> "Section232PublishDraftBatchRequest":
        if not str(self.published_by or "").strip():
            raise ValueError("published_by is required")
        return self


class Section232PublishDraftBatchResponse(BaseModel):
    published_version: str
    accepted_rule_count: int = 0
    ruleset_summary: Section232RulesetSummaryResponse


class ClassificationStatsResponse(BaseModel):
    saved_classification_count: int = 0
    latest_classified_at: Optional[str] = None


class ClassificationResetResponse(BaseModel):
    cleared_classification_count: int = 0
    cancelled_job_count: int = 0


class Section232ResetResponse(BaseModel):
    cleared_source_count: int = 0
    cleared_draft_batch_count: int = 0
    cleared_draft_rule_count: int = 0
    cleared_delete_override_count: int = 0
    cleared_published_ruleset_count: int = 0
    cleared_published_rule_count: int = 0


class HTSCatalogSourceFileSummary(BaseModel):
    filename: str
    source_kind: Literal["chapter", "code_map"]
    chapter_number: Optional[int] = None
    size_bytes: int = 0
    uploaded_at: str


class HTSCatalogSummaryResponse(BaseModel):
    managed_file_count: int = 0
    managed_chapter_file_count: int = 0
    loaded_chapters: List[int] = Field(default_factory=list)
    has_code_map: bool = False
    last_refresh_status: str = "unknown"
    last_refresh_at: Optional[str] = None
    last_refresh_error: Optional[str] = None
    catalog_row_count: int = 0
    code_map_row_count: int = 0


class HTSCatalogSourceListResponse(BaseModel):
    total: int = 0
    summary: HTSCatalogSummaryResponse = Field(default_factory=HTSCatalogSummaryResponse)
    items: List[HTSCatalogSourceFileSummary] = Field(default_factory=list)


class HTSCatalogSourceUploadResponse(BaseModel):
    uploaded_file_count: int = 0
    overwritten_file_count: int = 0
    items: List[HTSCatalogSourceFileSummary] = Field(default_factory=list)
    summary: HTSCatalogSummaryResponse = Field(default_factory=HTSCatalogSummaryResponse)


class HTSCatalogSourceDeleteResponse(BaseModel):
    deleted_filename: str
    summary: HTSCatalogSummaryResponse = Field(default_factory=HTSCatalogSummaryResponse)


class MetalCompositionItemDetail(BaseModel):
    item_id: str
    item_type: Literal["mm"]
    source_row_id: Optional[int] = None
    dataset_signature: Optional[str] = None
    product_code: str
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
    docs_status: str = "No documents"
    assigned_documents: List[DocumentReference] = Field(default_factory=list)
    latest_classification: Optional[MetalCompositionResponse] = None
    last_classified_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ItemClassificationResponse(BaseModel):
    item_id: str
    last_classified_at: Optional[str] = None
    result: MetalCompositionResponse


class MissingDocumentsBatchItem(BaseModel):
    item_id: str
    product_code: str
    priority: Optional[str] = None
    docs_status: str = "No documents"


class MissingDocumentsConfirmationDetail(BaseModel):
    code: Literal["pdf_required_blocked"] = "pdf_required_blocked"
    message: str
    items: List[MissingDocumentsBatchItem] = Field(default_factory=list)


class ItemClassifyBatchRequest(BaseModel):
    item_ids: List[str] = Field(default_factory=list)
    confirm_missing_documents: bool = False


class ItemClassifyBatchResponse(BaseModel):
    total: int
    completed: int
    failed: int
    results: List[ItemClassificationResponse] = Field(default_factory=list)


class ClassificationJobRef(BaseModel):
    job_id: str
    job_type: ClassificationJobType
    status: ClassificationJobStatus


class ClassificationJobSubmissionResponse(BaseModel):
    job_id: str
    job_type: ClassificationJobType
    status: ClassificationJobStatus
    submitted_at: str
    total_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


class ClassificationJobStatusResponse(BaseModel):
    job_id: str
    job_type: ClassificationJobType
    status: ClassificationJobStatus
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    error_message: Optional[str] = None


class ClassificationJobConflictDetail(BaseModel):
    code: Literal["classification_already_running"] = "classification_already_running"
    message: str
    item_ids: List[str] = Field(default_factory=list)
