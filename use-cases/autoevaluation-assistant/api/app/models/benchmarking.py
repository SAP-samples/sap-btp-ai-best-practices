"""Models for persisted assessment profiles and imported peer benchmarks."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class AssessmentProfile(BaseModel):
    """Persisted company context that defines one assessment's benchmark cohort.

    Inputs:
        Assessment identity, display label, optional source-company identity,
        normalized customer class, and exact level-one NACE label.

    Outputs:
        Validated profile shared by HANA/memory repositories and API routes.
    """

    assessment_id: str
    display_name: str
    source_company_id: str | None = None
    customer_class: str
    nace1: str


class BenchmarkImportInfo(BaseModel):
    """Safe metadata for one versioned benchmark import without workbook bytes.

    Inputs:
        Import identity, source audit metadata, normalized counts, status, and
        creation/activation timestamps.

    Outputs:
        Validated metadata suitable for score context and administration APIs.
    """

    import_id: str
    source_filename: str
    source_sha256: str
    scoring_version: str
    row_count: int = 0
    company_count: int = 0
    questionnaire_count: int = 0
    question_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    status: str
    is_active: bool = False
    created_at: datetime | None = None
    activated_at: datetime | None = None


class BenchmarkScopeScore(BaseModel):
    """One imported questionnaire score at overall, dimension, or topic scope.

    Inputs:
        Scope type, optional canonical dimension/question identifiers, and the
        deterministic calculated score.

    Outputs:
        Validated scope row used by pure peer aggregation.
    """

    scope_type: str
    dimension: str | None = None
    question_id: str | None = None
    calculated_score: float


class BenchmarkPeerSubmission(BaseModel):
    """Imported questionnaire submission plus all calculated score scopes.

    Inputs:
        Source identities, exact cohort attributes, release/date metadata, and
        score rows loaded only from the active import.

    Outputs:
        Validated submission consumed by latest-release cohort selection.
    """

    source_company_id: str
    questionnaire_id: str
    customer_class: str
    nace1: str | None = None
    submission_date: date | None = None
    extraction_date: date | None = None
    release_status: str | None = None
    scores: list[BenchmarkScopeScore] = Field(default_factory=list)


class BenchmarkMetric(BaseModel):
    """Company-versus-peer comparison for one score scope.

    Inputs:
        Current company score, optional peer average/best, sample size,
        positioning status, and deterministic localized commentary.

    Outputs:
        Validated nested metric embedded at overall, dimension, and topic scope.
    """

    company_score: float = 0.0
    peer_average: float | None = None
    best_peer: float | None = None
    sample_size: int = 0
    positioning: str = "unavailable"
    commentary_key: str = "benchmark.unavailable"
    commentary: str = "Peer comparison unavailable."


class BenchmarkContext(BaseModel):
    """Version and cohort audit context attached to a score response.

    Inputs:
        Availability/reason, safe active-import metadata, exact cohort values,
        peer sample size, and deterministic cohort definition.

    Outputs:
        Validated context proving which persisted dataset produced metrics.
    """

    available: bool = False
    reason: str | None = "no_active_dataset"
    import_id: str | None = None
    source_sha256: str | None = None
    source_filename: str | None = None
    scoring_version: str | None = None
    dataset_activated_at: datetime | None = None
    customer_class: str
    nace1: str | None = None
    peer_sample_size: int = 0
    cohort_definition: str = "exact_customer_class_and_nace1"


class BenchmarkCohortOption(BaseModel):
    """One exact customer-class and NACE-1 pair in the active import.

    Inputs:
        Normalized customer class and exact level-one NACE label.

    Outputs:
        Validated identity-free selector option for assessment profiles.
    """

    customer_class: str
    nace1: str


class BenchmarkOptionsResponse(BaseModel):
    """Active benchmark class/NACE selector options for profile editing.

    Inputs:
        Availability/reason, active import identity, independent label lists,
        and exact valid cohort pairs.

    Outputs:
        Identity-free options response for the Assessment UI.
    """

    available: bool
    reason: str | None = None
    import_id: str | None = None
    customer_classes: list[str] = Field(default_factory=list)
    nace1_sectors: list[str] = Field(default_factory=list)
    cohorts: list[BenchmarkCohortOption] = Field(default_factory=list)


class BenchmarkImportHistoryResponse(BaseModel):
    """Safe active and bounded recent benchmark import summaries.

    Inputs:
        Availability/reason, optional active metadata, and recent version list.

    Outputs:
        Administration response excluding workbook BLOBs and peer identities.
    """

    available: bool
    reason: str | None = None
    active: BenchmarkImportInfo | None = None
    recent: list[BenchmarkImportInfo] = Field(default_factory=list)


class PublicBenchmarkWarning(BaseModel):
    """Identity-free aggregate warning returned by benchmark import APIs.

    Inputs:
        Stable warning code, generic public message, aggregate count, and any
        worksheet row numbers recoverable without exposing raw sample values.

    Outputs:
        Validated warning safe for clients outside the trusted parser boundary.
    """

    code: str
    message: str
    count: int = Field(ge=1)
    row_numbers: list[int] = Field(default_factory=list)


class PublicBenchmarkRowError(BaseModel):
    """Identity-free sampled validation error returned by import APIs.

    Inputs:
        Worksheet row number, stable error code, and generic public message.

    Outputs:
        Validated error that preserves actionable location without raw values.
    """

    row_number: int = Field(ge=0)
    code: str
    message: str


class PublicBenchmarkValidationSummary(BaseModel):
    """Public benchmark validation result with peer identifiers removed.

    Inputs:
        Safe import audit fields, aggregate cohort/count information, sanitized
        warning/error entries, and write outcome flags.

    Outputs:
        API response model retaining UI-required validation information while
        excluding internal raw warning samples and source-identity messages.
    """

    import_id: str | None = None
    source_filename: str
    source_sha256: str
    scoring_version: str
    row_count: int = 0
    company_count: int = 0
    questionnaire_count: int = 0
    question_count: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    nace1_counts: dict[str, int] = Field(default_factory=dict)
    nace2_counts: dict[str, int] = Field(default_factory=dict)
    nace3_counts: dict[str, int] = Field(default_factory=dict)
    accepted_count: int = 0
    rejected_count: int = 0
    warnings: list[PublicBenchmarkWarning] = Field(default_factory=list)
    sampled_errors: list[PublicBenchmarkRowError] = Field(default_factory=list)
    success: bool = False
    status: str = "rejected"
    write_completed: bool = False
    no_op: bool = False
