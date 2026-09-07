"""Strict models for deterministic asynchronous assessment PDF reports."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.language import DEFAULT_LANGUAGE
from app.services.customer_class_scope import CustomerClass


AssessmentReportJobStatus = Literal[
    "pending",
    "generating",
    "rendering",
    "completed",
    "failed",
]
"""Allowed lifecycle states for an asynchronous assessment report job."""

AssessmentReportPositioning = Literal[
    "below_peers",
    "in_line_with_peers",
    "above_peers",
    "unavailable",
]
"""Trusted qualitative benchmark classification snapshotted at enqueue."""

AssessmentReportUnavailableReason = Literal[
    "no_active_dataset",
    "insufficient_peer_sample",
]
"""Allowed reasons for withholding peer benchmark disclosure."""


def _expected_report_positioning(
    company_score: float,
    peer_average: float | None,
) -> AssessmentReportPositioning:
    """Classify one frozen company score against its peer average.

    Inputs:
        company_score: Trusted company score on the zero-to-one-hundred scale.
        peer_average: Optional arithmetic peer average for the same scope.

    Outputs:
        AssessmentReportPositioning: Deterministic classification using the
        same inclusive ninety-to-one-hundred-ten-percent bounds as scoring.
    """

    if peer_average is None:
        return "unavailable"
    if peer_average == 0:
        return "in_line_with_peers" if company_score == 0 else "above_peers"
    if company_score < 0.9 * peer_average:
        return "below_peers"
    if company_score <= 1.1 * peer_average:
        return "in_line_with_peers"
    return "above_peers"


class AssessmentReportRequest(BaseModel):
    """Request to snapshot an assessment and enqueue report generation.

    Inputs:
        assessment_id: Assessment whose persisted profile and answers are used.
        language: Requested English or Italian report language.
        customer_class: Deprecated optional browser field accepted but ignored.
        sector: Deprecated optional browser field accepted but ignored.

    Outputs:
        Validated enqueue request consumed by the assessment report API.
    """

    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    language: str = DEFAULT_LANGUAGE
    customer_class: CustomerClass | None = None
    sector: str | None = None


class _FrozenReportModel(BaseModel):
    """Base configuration for immutable, versioned report snapshot nodes.

    Inputs:
        Subclass-declared report fields.

    Outputs:
        Strict Pydantic nodes that reject unknown fields and assignment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AssessmentReportMetric(_FrozenReportModel):
    """Exact frozen company and peer values for one report scope.

    Inputs:
        company_score: Trusted company score on the fixed zero-to-one-hundred scale.
        peer_average: Optional arithmetic peer average for the exact cohort.
        best_peer: Optional independent maximum peer value for the exact cohort.
        sample_size: Number of peer submissions contributing at this scope.
        positioning: Deterministic relative classification copied at enqueue.
        commentary_key: Stable deterministic commentary key.
        commentary: Already-localized deterministic qualitative commentary.

    Outputs:
        Immutable metric used by charts and qualitative rows.
    """

    company_score: float = Field(ge=0, le=100)
    peer_average: float | None = Field(default=None, ge=0, le=100)
    best_peer: float | None = Field(default=None, ge=0, le=100)
    sample_size: int = Field(ge=0)
    positioning: AssessmentReportPositioning
    commentary_key: str
    commentary: str

    @model_validator(mode="after")
    def validate_peer_metric_consistency(self) -> Self:
        """Reject contradictory peer disclosure fields within one metric.

        Inputs:
            self: Fully parsed company, peer, sample, positioning, and commentary data.

        Outputs:
            Self: The unchanged metric when peer values form one coherent state.

        Raises:
            ValueError: If peer values are partial, under-disclosed, inconsistent
            with availability positioning, or use the wrong commentary key.
        """

        has_average = self.peer_average is not None
        has_best = self.best_peer is not None
        if has_average != has_best:
            raise ValueError("Peer average and best peer must be present together.")
        if not has_average:
            if self.sample_size != 0:
                raise ValueError("A metric without peer values must have sample size zero.")
            if self.positioning != "unavailable":
                raise ValueError("A metric without peer values must be unavailable.")
        else:
            if self.sample_size < 3:
                raise ValueError("Disclosed peer values require at least three peers.")
            if self.best_peer < self.peer_average:
                raise ValueError("Best peer cannot be below the peer average.")
            if self.positioning == "unavailable":
                raise ValueError("A metric with peer values must have a peer position.")
        expected_positioning = _expected_report_positioning(
            self.company_score,
            self.peer_average,
        )
        if self.positioning != expected_positioning:
            raise ValueError(
                "Metric positioning must match company and peer-average values."
            )
        if self.commentary_key != f"benchmark.{self.positioning}":
            raise ValueError("Metric commentary key must match its positioning.")
        return self


class AssessmentReportTopic(_FrozenReportModel):
    """One class-applicable localized topic in the immutable report tree.

    Inputs:
        question_id: Canonical assessment question identifier.
        dimension: Canonical parent dimension identifier.
        topic_title: Localized approved report title captured at enqueue.
        answered: Whether the persisted assessment selected an applicable answer.
        metric: Frozen company and peer scope values.

    Outputs:
        Immutable topic node rendered without selected-answer prompt data.
    """

    question_id: str
    dimension: str
    topic_title: str
    answered: bool
    metric: AssessmentReportMetric


class AssessmentReportDimension(_FrozenReportModel):
    """One applicable dimension and its ordered localized topic children.

    Inputs:
        dimension: Canonical dimension identifier.
        display_name: Localized dimension title captured at enqueue.
        metric: Frozen aggregate dimension metric.
        topics: Ordered tuple of class-applicable topics only.

    Outputs:
        Immutable dimension node used by overview and detailed analysis sections.
    """

    dimension: str
    display_name: str
    metric: AssessmentReportMetric
    topics: tuple[AssessmentReportTopic, ...]

    @model_validator(mode="after")
    def validate_topic_membership(self) -> Self:
        """Require unique topic IDs whose parent matches this dimension.

        Inputs:
            self: Fully parsed dimension node and ordered topic children.

        Outputs:
            Self: The unchanged dimension when topic membership is coherent.

        Raises:
            ValueError: If a topic names another parent or repeats an ID.
        """

        topic_ids: set[str] = set()
        for topic in self.topics:
            if topic.dimension != self.dimension:
                raise ValueError("Topic dimension must match its parent dimension.")
            if topic.question_id in topic_ids:
                raise ValueError("Question IDs must be unique within a dimension.")
            topic_ids.add(topic.question_id)
        return self


class AssessmentReportProvenance(_FrozenReportModel):
    """Frozen active-import and exact-cohort provenance for a report.

    Inputs:
        benchmark_available: Whether at least three distinct peers were eligible.
        unavailable_reason: Stable reason when peer disclosure is unavailable.
        import_id: Captured active import identifier, when one exists.
        source_filename: Safe imported workbook filename.
        source_sha256: Captured workbook digest.
        dataset_activated_at: Activation timestamp used as the dataset date.
        scoring_version: Imported deterministic scoring-contract version.
        customer_class: Exact persisted profile class.
        nace1: Exact persisted profile NACE-1 label.
        peer_sample_size: Actual eligible distinct-peer count.
        cohort_definition: Stable exact class-plus-NACE definition marker.

    Outputs:
        Immutable provenance proving which dataset and cohort produced metrics.
    """

    benchmark_available: bool
    unavailable_reason: AssessmentReportUnavailableReason | None
    import_id: str | None = Field(default=None, min_length=1, max_length=64)
    source_filename: str | None = Field(default=None, min_length=1, max_length=512)
    source_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    dataset_activated_at: datetime | None
    scoring_version: str | None = Field(default=None, min_length=1, max_length=64)
    customer_class: str
    nace1: str
    peer_sample_size: int = Field(ge=0)
    cohort_definition: Literal["exact_customer_class_and_nace1"] = (
        "exact_customer_class_and_nace1"
    )

    @model_validator(mode="after")
    def validate_availability_consistency(self) -> Self:
        """Keep disclosure state, reason, import, and cohort count coherent.

        Inputs:
            self: Fully parsed benchmark availability and provenance values.

        Outputs:
            Self: The unchanged provenance when disclosure invariants hold.

        Raises:
            ValueError: If an available cohort lacks an import/minimum sample or
            an unavailable cohort lacks a reason/has a disclosure-sized sample.
        """

        import_metadata = (
            self.import_id,
            self.source_filename,
            self.source_sha256,
            self.dataset_activated_at,
            self.scoring_version,
        )
        has_complete_import = all(value is not None for value in import_metadata)
        has_any_import = any(value is not None for value in import_metadata)

        if self.benchmark_available:
            if self.unavailable_reason is not None:
                raise ValueError("Available benchmarks cannot have an unavailable reason.")
            if not has_complete_import:
                raise ValueError("Available benchmarks require complete import provenance.")
            if self.peer_sample_size < 3:
                raise ValueError("Available benchmarks require at least three peers.")
        elif self.unavailable_reason == "no_active_dataset":
            if self.peer_sample_size != 0:
                raise ValueError("No active dataset must have a zero peer sample.")
            if has_any_import:
                raise ValueError("No active dataset cannot retain import provenance.")
        elif self.unavailable_reason == "insufficient_peer_sample":
            if self.peer_sample_size >= 3:
                raise ValueError("Insufficient samples must contain fewer than three peers.")
            if not has_complete_import:
                raise ValueError(
                    "An insufficient peer sample requires complete import provenance."
                )
        else:
            raise ValueError("Unavailable benchmarks require an explicit reason.")
        return self


class AssessmentReportSource(_FrozenReportModel):
    """Complete report-specific snapshot persisted before asynchronous work.

    Inputs:
        Assessment/profile identity, localized labels, completion counts, frozen
        metrics, ordered dimensions/topics, and captured import provenance.

    Outputs:
        Strict current source consumed without live repository reads.
    """

    schema_version: Literal[4] = 4
    assessment_id: str
    display_name: str
    source_company_id: str | None = None
    customer_class: str
    customer_class_label: str
    nace1: str
    language: Literal["en", "it"] = DEFAULT_LANGUAGE
    generated_at: datetime
    applicable_question_count: int = Field(ge=0)
    answered_question_count: int = Field(ge=0)
    overall: AssessmentReportMetric
    dimensions: tuple[AssessmentReportDimension, ...]
    provenance: AssessmentReportProvenance

    @model_validator(mode="after")
    def validate_snapshot_consistency(self) -> Self:
        """Reject contradictory counts, cohort metadata, hierarchy, and disclosure.

        Inputs:
            self: Fully parsed report tree and captured provenance.

        Outputs:
            Self: The unchanged source when all cross-node invariants hold.

        Raises:
            ValueError: If completion counts, class/NACE provenance, unique scope
            identities, sample bounds, or unavailable peer values conflict.
        """

        if self.customer_class != self.provenance.customer_class:
            raise ValueError("Snapshot and provenance customer class must match.")
        if self.nace1 != self.provenance.nace1:
            raise ValueError("Snapshot and provenance NACE-1 must match.")

        dimension_ids: set[str] = set()
        question_ids: set[str] = set()
        metrics = [self.overall]
        answered_count = 0
        topic_count = 0
        for dimension in self.dimensions:
            if dimension.dimension in dimension_ids:
                raise ValueError("Dimension IDs must be unique in a report snapshot.")
            dimension_ids.add(dimension.dimension)
            metrics.append(dimension.metric)
            for topic in dimension.topics:
                if topic.question_id in question_ids:
                    raise ValueError("Question IDs must be unique in a report snapshot.")
                question_ids.add(topic.question_id)
                topic_count += 1
                answered_count += int(topic.answered)
                metrics.append(topic.metric)

        if self.applicable_question_count != topic_count:
            raise ValueError("Applicable question count must match snapshotted topics.")
        if self.answered_question_count != answered_count:
            raise ValueError("Answered question count must match answered topics.")
        if self.answered_question_count > self.applicable_question_count:
            raise ValueError("Answered question count cannot exceed applicable questions.")

        for metric in metrics:
            if metric.sample_size > self.provenance.peer_sample_size:
                raise ValueError("Metric sample size cannot exceed the cohort peer count.")
            if not self.provenance.benchmark_available and (
                metric.peer_average is not None or metric.best_peer is not None
            ):
                raise ValueError("Unavailable benchmarks cannot contain peer values.")
        return self


class AssessmentReportJobResponse(BaseModel):
    """Response returned immediately after a report job is accepted.

    Inputs:
        job_id: Persisted queue identity.
        status: Initial pending lifecycle state.
        language: Snapshot language.

    Outputs:
        Minimal HTTP 202 response used by Score polling.
    """

    job_id: str
    status: AssessmentReportJobStatus
    language: str


class AssessmentReportJobStatusResponse(BaseModel):
    """Polling response for one asynchronous report job.

    Inputs:
        Current persisted lifecycle, progress, result, error, and expiry fields.

    Outputs:
        API-safe job state used to update Score and start PDF downloads.
    """

    job_id: str
    assessment_id: str
    language: str
    status: AssessmentReportJobStatus
    progress_message: str | None = None
    download_ready: bool = False
    file_name: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class AssessmentReportDownload(BaseModel):
    """Internal binary payload returned for a completed report job.

    Inputs:
        job_id: Completed queue identity.
        file_name: Safe persisted attachment filename.
        content: Stored PDF BLOB bytes.

    Outputs:
        Download data converted into an ``application/pdf`` response.
    """

    job_id: str
    file_name: str
    content: bytes
