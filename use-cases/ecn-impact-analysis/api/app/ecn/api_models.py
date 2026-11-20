"""API envelope models for all ECN impact HTTP endpoints.

Each model enforces ``extra="forbid"`` so that the API surface is closed
against unintended fields.  String-width limits are aligned to the HANA
``NVARCHAR`` column sizes defined in the schema layer.

Imports from ``app.ecn.models`` (Task 1) provide the shared enumerations.
Imports from ``app.ecn.analysis_models`` (Task 2) provide analysis-specific
types such as ``AnalysisSections`` and ``AnalysisCompleteness``.

``AnalysisFilterState`` extends ``JobState`` with ``COMPLETED_WITH_GAPS``,
which is an API/result projection: the HANA job table stores only
``JobState.COMPLETED``; COMPLETED_WITH_GAPS is derived from the stored
analysis completeness and is never persisted in the job row.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StringConstraints,
    model_validator,
)

from app.ecn.analysis_models import (
    AnalysisCompleteness,
    AnalysisSections,
    DataReadinessDescription,
    ImpactNextSteps,
    ImpactFactors,
    ImpactSeverity,
)
from app.ecn.models import (
    ArtifactKind,
    BusinessDirection,
    BusinessEvidenceSection,
    EvidenceKind,
    EvidenceTier,
    JobState,
)
from app.ecn.rdf import ProjectionState

# ---------------------------------------------------------------------------
# Analysis filter state (API projection enum)
# ---------------------------------------------------------------------------


class AnalysisFilterState(StrEnum):
    """Extend the mutable job state with the COMPLETED_WITH_GAPS API projection.

    The six values expose the full lifecycle visible to API clients.  The
    HANA job table uses ``JobState`` (which has no COMPLETED_WITH_GAPS value);
    COMPLETED_WITH_GAPS is derived from the stored analysis completeness at
    read time and is therefore an API-only concept.
    """

    READY = "READY"
    QUEUED = "QUEUED"
    BLOCKED = "BLOCKED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_GAPS = "COMPLETED_WITH_GAPS"
    FAILED = "FAILED"


class AnalysisFilterGroup(StrEnum):
    """Grouped inbox filters; these values are never returned as row states."""

    SUMMARIES_AVAILABLE = "SUMMARIES_AVAILABLE"
    IN_PROGRESS = "IN_PROGRESS"


AnalysisStateFilter = AnalysisFilterState | AnalysisFilterGroup


# ---------------------------------------------------------------------------
# List query
# ---------------------------------------------------------------------------


class EcnListQuery(BaseModel):
    """Validate all server-side list filters and the closed sort allowlist.

    Args:
        page: 1-based page number.  Must be >= 1.  Defaults to ``1``.
        page_size: Number of items per page.  Must be in ``[1, 100]``.
            Defaults to ``25``.
        query: Free-text search string.  Stripped, max 500 characters; ``None``
            means no filter.
        ecn_number: Exact or prefix ECN number filter.  Stripped, max 80
            characters; ``None`` means no filter.
        analysis_state: Filter by one lifecycle state or inbox summary group;
            ``None`` means no filter.
        has_warnings: ``True`` for ECNs with at least one warning; ``False``
            for ECNs with no warnings; ``None`` means no filter.
        plant: Case-insensitive substring of a plant code or name (e.g.
            ``"PL10"``).  Only ECNs whose plants were already assessed can
            match; ``None`` means no filter.
        sort: Column to sort by.  Must be one of the documented values.
            Defaults to ``"uploaded_at"``.
        order: Sort direction.  Either ``"asc"`` or ``"desc"``.
            Defaults to ``"desc"``.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    query: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        ]
        | None
    ) = None
    ecn_number: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
        ]
        | None
    ) = None
    analysis_state: AnalysisStateFilter | None = None
    has_warnings: bool | None = None
    plant: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
        ]
        | None
    ) = None
    sort: Literal[
        "uploaded_at",
        "resolution_date",
        "affected_plants",
        "ecn_number",
        "description",
        "affected_item_count",
        "ebom_match_count",
        "mbom_match_count",
        "warning_count",
        "analysis_state",
    ] = "uploaded_at"
    order: Literal["asc", "desc"] = "desc"


# ---------------------------------------------------------------------------
# Artifact view
# ---------------------------------------------------------------------------


class ArtifactView(BaseModel):
    """Expose safe immutable source-artifact metadata.

    Args:
        artifact_id: UUID of the source artifact.
        kind: File-type classification.
        file_name: Original uploaded file name.  Max 500 characters.
        sha256: Lowercase hex digest of the artifact bytes.  Must be exactly
            64 characters.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID
    kind: ArtifactKind
    file_name: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


# ---------------------------------------------------------------------------
# Impact counts
# ---------------------------------------------------------------------------


class ImpactCounts(BaseModel):
    """Expose deterministic occurrence/evidence counts as nonnegative values.

    Args:
        affected_item_count: Total unique technical items changed by this ECN,
            including affected and resulting source objects.
        raw_ecn_match_count: Number of BOM rows matching raw ECN part numbers.
        ebom_match_count: Number of EBOM matches.
        ebom_context_count: Number of EBOM context-window rows.
        mbom_match_count: Number of MBOM matches.
        warning_count: Number of evidence warnings.
    """

    model_config = ConfigDict(extra="forbid")

    affected_item_count: NonNegativeInt
    raw_ecn_match_count: NonNegativeInt
    ebom_match_count: NonNegativeInt
    ebom_context_count: NonNegativeInt
    mbom_match_count: NonNegativeInt
    warning_count: NonNegativeInt


# ---------------------------------------------------------------------------
# Evidence summary (compact per-section deterministic evidence)
# ---------------------------------------------------------------------------


class EvidenceSummary(BaseModel):
    """Expose compact deterministic evidence in its fixed section.

    Args:
        evidence_id: Unique identifier for this evidence item.
        kind: Epistemic category.
        tier: Data-quality tier; ``None`` for tier-less evidence kinds.
        summary: One-sentence human-readable summary.
        quality_qualifiers: List of derivation quality flags.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]
    kind: EvidenceKind
    tier: EvidenceTier | None
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    quality_qualifiers: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ]


class EvidenceSectionSummaries(BaseModel):
    """Group deterministic evidence under the same six UI/analysis sections.

    Field names match ``AnalysisSectionKey`` values so that
    ``getattr(sections, key.value)`` retrieves the correct list.

    Args:
        executive_summary: Evidence summaries for the executive summary section.
        change_intent: Evidence summaries for the change-intent section.
        affected_materials: Evidence summaries for the affected-materials section.
        ebom_mbom_impact: Evidence summaries for the EBOM/MBOM impact section.
        configuration_scope: Evidence summaries for the configuration-scope section.
        uncertainties_and_missing_evidence: Evidence summaries for the
            uncertainties and missing-evidence section.
    """

    model_config = ConfigDict(extra="forbid")

    executive_summary: list[EvidenceSummary]
    change_intent: list[EvidenceSummary]
    affected_materials: list[EvidenceSummary]
    ebom_mbom_impact: list[EvidenceSummary]
    configuration_scope: list[EvidenceSummary]
    uncertainties_and_missing_evidence: list[EvidenceSummary]


# ---------------------------------------------------------------------------
# Analysis job and stored analysis views
# ---------------------------------------------------------------------------


class AnalysisJobView(BaseModel):
    """Expose one mutable technical analysis state without internal payloads.

    Args:
        job_id: UUID of the analysis job.
        state: Current lifecycle state.
        attempt_count: Number of execution attempts made so far (>= 0).
        error_code: Stable machine-readable error code; ``None`` when no error.
            Max 80 characters.
        error_message: Safe human-readable error message; ``None`` when no error.
            Max 2000 characters.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    state: JobState
    attempt_count: NonNegativeInt
    error_code: Annotated[str, StringConstraints(max_length=80)] | None
    error_message: Annotated[str, StringConstraints(max_length=2000)] | None


class StoredAnalysisView(BaseModel):
    """Expose one immutable validated analysis result.

    Args:
        analysis_id: UUID of the stored analysis record.
        completeness: Overall completeness classification derived from sections.
        change_brief: One AI-generated decision headline; ``None`` for legacy analyses.
        sections: All six validated analysis sections.
        impact_factors: Business severity factors plus deterministic readiness.
        overall_impact: Fixed overall business-impact severity.
        created_at: Timestamp when this analysis was stored.
        evidence_hash: 64-character hex digest of the evidence bundle used.
        prompt_version: Version tag of the prompt template that produced this
            analysis.
        output_schema_version: Version tag of the structured output schema.
        model_name: Gen AI Hub model name used for this analysis.
        model_deployment: Gen AI Hub deployment name; ``None`` when not recorded.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    completeness: AnalysisCompleteness
    change_brief: Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"^[^\r\n]+$")] | None = None
    sections: AnalysisSections
    impact_factors: ImpactFactors = Field(default_factory=ImpactFactors)
    overall_impact: ImpactSeverity = ImpactSeverity.UNKNOWN
    data_readiness_descriptions: list[DataReadinessDescription] = Field(default_factory=list)
    additional_next_steps: ImpactNextSteps = Field(default_factory=ImpactNextSteps)
    created_at: datetime
    evidence_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    prompt_version: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    output_schema_version: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    model_deployment: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    )


# ---------------------------------------------------------------------------
# List item and response
# ---------------------------------------------------------------------------


class EcnListItem(BaseModel):
    """Represent one immutable uploaded ECN occurrence in the historical table.

    Args:
        ecn_row_id: UUID of the normalized ECN occurrence.
        ecn_number: Source-system ECN identifier; ``None`` when absent in source.
        display_ecn_number: Display-safe ECN number (never null; falls back to a
            generated label when the source number is absent).
        title: ECN title; ``None`` when absent.
        description_summary: Short description; ``None`` when absent.
        source_ordinal: Zero-based index within the source file.
        resolution_date: Source-system ResolutionDate (the date the ECN was
            released); ``None`` when the source carries no value.
        uploaded_at: When this occurrence was stored.
        source_file_name: Original uploaded file name.
        counts: Deterministic impact and evidence counts.
        analysis_state: Derived lifecycle state including COMPLETED_WITH_GAPS.
        has_partial_analysis: Whether a failed job has compatible validated
            stages newer than its last accepted analysis.
        completeness: Stored analysis completeness; ``None`` when no analysis
            exists yet.
        overall_impact: Stored business-impact severity; ``None`` before analysis.
        affected_plants: Plant labels such as ``"PL10 Plant A"`` from the
            ECN's "Plants affected" table; ``[]`` when no plant is linked and
            ``None`` while the snapshot has not been assessed yet.
        review_outcome: Newest engineering review decision for this assessment
            (``ACCEPTED``, ``CHANGES_REQUESTED`` or a retired outcome); ``None``
            when nobody has reviewed it.
    """

    model_config = ConfigDict(extra="forbid")

    ecn_row_id: UUID
    ecn_number: (
        Annotated[str, StringConstraints(min_length=1, max_length=80)] | None
    )
    display_ecn_number: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    title: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None
    description_summary: (
        Annotated[str, StringConstraints(min_length=1, max_length=2000)] | None
    )
    source_ordinal: NonNegativeInt
    resolution_date: datetime | None = None
    uploaded_at: datetime
    source_file_name: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    counts: ImpactCounts
    analysis_state: AnalysisFilterState
    completeness: AnalysisCompleteness | None
    overall_impact: ImpactSeverity | None = None
    has_partial_analysis: bool = False
    affected_plants: list[str] | None = None
    review_outcome: Annotated[str, StringConstraints(max_length=40)] | None = None


class EcnListResponse(BaseModel):
    """Return one stable server-side page and its totals.

    Args:
        items: The ECN list items for the requested page.
        page: 1-based page number returned.
        page_size: Items per page (1–100).
        total_items: Total number of items matching the query.
        total_pages: Total number of pages at the current page_size.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[EcnListItem]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_items: NonNegativeInt
    total_pages: NonNegativeInt


# ---------------------------------------------------------------------------
# Detail response
# ---------------------------------------------------------------------------


class LinkedEcnReference(BaseModel):
    """An explicit source-description reference and its available navigation target.

    A null row ID means the referenced ECN has not been imported. When there
    are multiple occurrences, prefer the current upload, then the newest upload.
    """

    model_config = ConfigDict(extra="forbid")

    ecn_number: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    ecn_row_id: UUID | None = None


class EcnDetailResponse(BaseModel):
    """Return one exact ECN occurrence using metadata-only initial hydration.

    Args:
        ecn_row_id: UUID of the normalized ECN occurrence.
        ecn_number: Source-system ECN identifier; ``None`` when absent.
        display_ecn_number: Display-safe ECN number.
        title: ECN title; ``None`` when absent.
        description_summary: Short description; ``None`` when absent.
        description_text: Full description text; ``None`` when absent.
        linked_ecns: Unique non-self references parsed from source descriptions,
            with an occurrence UUID when the referenced ECN is available.
        source_ordinal: Zero-based index within the source file.
        source_status: Source-system release or approval state; ``None`` when absent.
        requestor_name: Name carried in the ECN requestor field; ``None`` when absent.
        affected_products: Declared affected product/model labels from the source.
        resolution_date: Source-system ResolutionDate (the date the ECN was
            released); ``None`` when the source carries no value.
        uploaded_at: When this occurrence was stored.
        source_file_name: Original uploaded file name.
        ecn_artifact: Provenance of the ECN JSON artifact.
        bom_artifact: Provenance of the BOM workbook artifact.
        rule_artifact: Provenance of the conditional-rule workbook; ``None``
            when not supplied.
        counts: Deterministic impact and evidence counts.
        job: Current analysis job state.
        analysis: Stored validated analysis; ``None`` when not yet completed.
    """

    model_config = ConfigDict(extra="forbid")

    ecn_row_id: UUID
    ecn_number: (
        Annotated[str, StringConstraints(min_length=1, max_length=80)] | None
    )
    display_ecn_number: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    title: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None
    description_summary: (
        Annotated[str, StringConstraints(min_length=1, max_length=2000)] | None
    )
    description_text: str | None
    linked_ecns: list[LinkedEcnReference] = Field(default_factory=list)
    source_ordinal: NonNegativeInt
    source_status: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    requestor_name: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    affected_products: list[Annotated[str, StringConstraints(min_length=1, max_length=500)]] = Field(default_factory=list)
    resolution_date: datetime | None = None
    uploaded_at: datetime
    source_file_name: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    ecn_artifact: ArtifactView
    bom_artifact: ArtifactView
    rule_artifact: ArtifactView | None
    counts: ImpactCounts
    job: AnalysisJobView
    analysis: StoredAnalysisView | None


class BusinessEvidenceSourceView(BaseModel):
    """Expose an exact source coordinate without artifact content or credentials.

    Args:
        artifact_kind: Source artifact type.
        json_path: Exact JSON path when the source is an ECN document.
        sheet_name: Workbook sheet when the source is tabular.
        source_row_number: One-based workbook row when available.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_kind: ArtifactKind
    json_path: str | None
    sheet_name: str | None
    source_row_number: NonNegativeInt | None


class BusinessEvidenceTechnicalView(BaseModel):
    """Expose internal audit fields only inside an optional technical disclosure.

    Args:
        internal_evidence_ids: Stored evidence identities represented by the row.
        raw_kinds: Original evidence-kind enum values.
        raw_tiers: Original evidence-tier enum values.
        raw_values: Original JSON-compatible evidence payloads.
        quality_qualifiers: Exact stored source-quality codes.
        related_ids: Internal related domain identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    internal_evidence_ids: list[str]
    raw_kinds: list[str]
    raw_tiers: list[str]
    raw_values: list[dict[str, Any]]
    quality_qualifiers: list[str]
    related_ids: list[str]


class BusinessEvidenceItemView(BaseModel):
    """Expose one readable material, BOM, configuration, quality, or gap record.

    Args:
        label: Short display name with no raw enum formatting.
        summary: Business-readable explanation.
        material_number: Material or component number when available.
        description: Source material description when available.
        revision: Source revision when available.
        direction: Added, removed, affected, or matched direction when applicable.
        parent: Immediate BOM parent when available.
        bom_path: Ordered BOM parent path.
        quantity: Source quantity rendered as text.
        effectivity: Readable effectivity interval.
        plant: Plant code or bounded site name when available.
        matched_row_count: Number of underlying stored evidence facts.
        sources: Up to three representative exact source coordinates.
        technical_details: Optional raw audit values for explicit disclosure.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    summary: str
    material_number: str | None
    description: str | None
    revision: str | None
    direction: BusinessDirection | None
    parent: str | None
    bom_path: list[str]
    quantity: str | None
    effectivity: str | None
    plant: str | None
    matched_row_count: NonNegativeInt
    sources: list[BusinessEvidenceSourceView]
    technical_details: BusinessEvidenceTechnicalView | None = None


class BusinessEvidencePageResponse(BaseModel):
    """Return one stable page from a single business evidence section."""

    model_config = ConfigDict(extra="forbid")

    section: BusinessEvidenceSection
    items: list[BusinessEvidenceItemView]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_items: NonNegativeInt
    total_pages: NonNegativeInt


class BusinessEvidenceDetailResponse(BaseModel):
    """Return one exact evidence lookup as a readable row plus source details."""

    model_config = ConfigDict(extra="forbid")

    ecn_row_id: UUID
    item: BusinessEvidenceItemView


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Return one successful deterministic publication and initial job counts.

    The six state counts (queued + blocked + running + completed + completed_with_gaps +
    failed) must equal ``imported_ecn_count``; the model-level validator
    enforces this equation.

    Args:
        upload_id: UUID of the created upload batch.
        request_id: UUID of the idempotency request record.
        created: ``True`` when the upload was newly created; ``False`` when it
            was a duplicate and an existing record was returned.
        imported_ecn_count: Number of ECN occurrences imported from this upload.
        bom_artifact_id: UUID of the BOM artifact (may be reused).
        bom_reused: ``True`` when a matching BOM artifact already existed.
        queued_count: ECNs in QUEUED state after upload.
        blocked_count: ECNs held for provider availability after upload.
        running_count: ECNs in RUNNING state after upload.
        completed_count: ECNs in COMPLETED state (no gaps) after upload.
        completed_with_gaps_count: ECNs with COMPLETED_WITH_GAPS after upload.
        failed_count: ECNs in FAILED state after upload.
        list_url: Relative URL of the ECN list endpoint for this upload.
        progress_url: Relative URL of the progress endpoint for this upload.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: UUID
    request_id: UUID
    created: bool
    imported_ecn_count: NonNegativeInt
    bom_artifact_id: UUID
    bom_reused: bool
    ready_count: NonNegativeInt = 0
    queued_count: NonNegativeInt
    blocked_count: NonNegativeInt = 0
    running_count: NonNegativeInt
    completed_count: NonNegativeInt
    completed_with_gaps_count: NonNegativeInt
    failed_count: NonNegativeInt
    list_url: str
    progress_url: str

    @model_validator(mode="after")
    def validate_state_total(self) -> "UploadResponse":
        """Require each imported occurrence to occupy exactly one reported state."""
        state_total = (
            self.queued_count
            + self.blocked_count
            + self.running_count
            + self.completed_count
            + self.completed_with_gaps_count
            + self.failed_count
            + self.ready_count
        )
        if state_total != self.imported_ecn_count:
            raise ValueError(
                "upload state counts must equal imported_ecn_count"
            )
        return self


# ---------------------------------------------------------------------------
# Progress response
# ---------------------------------------------------------------------------


class AnalysisProgressResponse(BaseModel):
    """Return mutually exclusive aggregate analysis-state counts.

    Two equations are enforced by the model-level validator:

    - ``unfinished_count == queued_count + blocked_count + running_count``
    - ``total_count == unfinished_count + completed_count
      + completed_with_gaps_count + failed_count``

    Args:
        upload_id: UUID of the upload batch being tracked; ``None`` for a
            global progress query.
        queued_count: ECNs in QUEUED state.
        blocked_count: ECNs waiting for provider availability.
        running_count: ECNs in RUNNING state.
        completed_count: ECNs in COMPLETED state (no gaps).
        completed_with_gaps_count: ECNs with COMPLETED_WITH_GAPS status.
        failed_count: ECNs in FAILED state.
        unfinished_count: Sum of queued, blocked, and running counts.
        total_count: Sum of all six state counts.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: UUID | None
    ready_count: NonNegativeInt = 0
    queued_count: NonNegativeInt
    blocked_count: NonNegativeInt = 0
    running_count: NonNegativeInt
    completed_count: NonNegativeInt
    completed_with_gaps_count: NonNegativeInt
    failed_count: NonNegativeInt
    unfinished_count: NonNegativeInt
    total_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_totals(self) -> "AnalysisProgressResponse":
        """Ensure aggregate fields equal their state components."""
        expected_unfinished = (
            self.queued_count + self.blocked_count + self.running_count
        )
        expected_total = (
            expected_unfinished
            + self.completed_count
            + self.completed_with_gaps_count
            + self.failed_count
            + self.ready_count
        )
        if self.unfinished_count != expected_unfinished:
            raise ValueError(
                "unfinished_count does not equal queued_count + blocked_count + running_count"
            )
        if self.total_count != expected_total:
            raise ValueError(
                "total_count does not equal all terminal and unfinished states"
            )
        return self


# ---------------------------------------------------------------------------
# Retry and history
# ---------------------------------------------------------------------------


class RetryResponse(BaseModel):
    """Confirm that one failed job was requeued.

    Args:
        ecn_row_id: UUID of the ECN occurrence whose job was requeued.
        job_id: UUID of the new queued job.
        state: Always ``"QUEUED"``; the literal enforces the post-condition.
    """

    model_config = ConfigDict(extra="forbid")

    ecn_row_id: UUID
    job_id: UUID
    state: Literal["QUEUED"]


class RdfProjectionView(BaseModel):
    """Expose bounded status for one rebuildable derived ECN RDF projection.

    Args:
        ecn_row_id: Canonical selected ECN occurrence identity.
        status: Current durable projection lifecycle state.
        source_sha256: Canonical hash of the authoritative relational JSON.
        projection_version: Deterministic mapper version represented by status.
        verified_node_count: Graph membership count after successful read-back.
        verified_triple_count: Owned graph-triple count after successful read-back.
        error_code: Safe graph failure code, when the last rebuild failed.
        error_message: Safe bounded graph failure diagnostic, when applicable.
        last_attempt_at: Most recent rebuild attempt in UTC.
        current_at: Successful verification timestamp in UTC.
    """

    model_config = ConfigDict(extra="forbid")

    ecn_row_id: UUID
    status: ProjectionState
    source_sha256: str
    projection_version: str
    verified_node_count: NonNegativeInt | None
    verified_triple_count: NonNegativeInt | None
    error_code: str | None
    error_message: str | None
    last_attempt_at: datetime | None
    current_at: datetime | None


class ProjectionBatchResult(BaseModel):
    """Return aggregate results for trusted post-commit projection attempts."""

    model_config = ConfigDict(extra="forbid")

    requested: NonNegativeInt
    current: NonNegativeInt
    failed: NonNegativeInt


class HistoryResetRequest(BaseModel):
    """Require the exact destructive confirmation phrase.

    Args:
        confirmation: Must be exactly ``"RESET HISTORY"`` to prevent
            accidental deletion.
    """

    model_config = ConfigDict(extra="forbid")

    confirmation: Literal["RESET HISTORY"]


class HistoryResetResponse(BaseModel):
    """Report exact dependency-ordered deletion counts.

    All fields are nonnegative integers.  The order reflects the foreign-key
    deletion cascade: attempt records first, source artifacts last.

    Args:
        deleted_attempt_count: Number of job attempt records deleted.
        deleted_analysis_count: Number of stored analysis records deleted.
        deleted_job_count: Number of analysis job records deleted.
        deleted_ecn_count: Number of ECN occurrence records deleted.
        deleted_upload_count: Number of upload batch records deleted.
        deleted_bom_row_count: Number of BOM row records deleted.
        deleted_source_artifact_count: Number of source artifact records
            deleted.
    """

    model_config = ConfigDict(extra="forbid")

    deleted_attempt_count: NonNegativeInt
    deleted_analysis_count: NonNegativeInt
    deleted_job_count: NonNegativeInt
    deleted_ecn_count: NonNegativeInt
    deleted_upload_count: NonNegativeInt
    deleted_bom_row_count: NonNegativeInt
    deleted_source_artifact_count: NonNegativeInt


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Expose only a stable code, safe message, and optional safe details.

    Width limits are aligned to the HANA ``NVARCHAR(80)`` and
    ``NVARCHAR(2000)`` error-log columns so that error payloads may be stored
    verbatim without truncation.

    Args:
        code: Stable machine-readable error code slug.  Max 80 characters.
        message: Safe human-readable error message.  Max 2000 characters.
        details: Optional structured diagnostic details; must not contain
            source content.  ``None`` when no additional context is available.
    """

    model_config = ConfigDict(extra="forbid")

    code: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
    ]
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    details: dict[str, Any] | None = None  # JsonValue recursive alias requires Any in Pydantic models
