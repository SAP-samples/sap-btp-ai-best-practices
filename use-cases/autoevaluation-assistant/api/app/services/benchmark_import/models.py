"""Pure data models shared by benchmark parsing, scoring, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class BenchmarkWarning(BaseModel):
    """One aggregate, non-blocking workbook quality warning.

    Inputs:
        code: Stable machine-readable warning category.
        message: Human-readable explanation of the quality issue.
        count: Number of source defects represented by the aggregate.
        samples: Bounded representative source values or row references.

    Outputs:
        JSON-safe warning embedded in a benchmark validation summary.
    """

    code: str
    message: str
    count: int = Field(ge=1)
    samples: list[str] = Field(default_factory=list)


class BenchmarkRowError(BaseModel):
    """One sampled blocking workbook validation error.

    Inputs:
        row_number: One-based worksheet row number, or zero for file errors.
        code: Stable machine-readable validation category.
        message: Human-readable explanation of the rejected input.

    Outputs:
        JSON-safe sampled error embedded in a validation summary.
    """

    row_number: int = Field(ge=0)
    code: str
    message: str


class BenchmarkValidationSummary(BaseModel):
    """JSON-safe dry-run or write result for one benchmark workbook.

    Inputs:
        Source identity, normalized cohort counts, validation totals, warnings,
        sampled errors, and optional write outcome fields.

    Outputs:
        Serializable summary used by CLIs and future API routes.
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
    warnings: list[BenchmarkWarning] = Field(default_factory=list)
    sampled_errors: list[BenchmarkRowError] = Field(default_factory=list)
    success: bool = False
    status: str = "rejected"
    write_completed: bool = False
    no_op: bool = False


class BenchmarkValidationError(ValueError):
    """Report a workbook validation failure before any HANA write.

    Inputs:
        summary: Structured validation summary containing sampled errors.

    Outputs:
        Exception carrying both a message and JSON-safe validation summary.
    """

    def __init__(self, summary: BenchmarkValidationSummary) -> None:
        """Store the rejected validation summary and initialize ``ValueError``.

        Inputs:
            summary: Failed validation result with one or more sampled errors.

        Outputs:
            None. The exception message uses the first sampled error when present.
        """

        self.summary = summary
        message = (
            summary.sampled_errors[0].message
            if summary.sampled_errors
            else "Benchmark workbook validation failed"
        )
        super().__init__(message)


@dataclass(frozen=True)
class BenchmarkCompany:
    """Normalized company profile owned by one benchmark import version.

    Inputs:
        Import/company identity and workbook profile fields after normalization.

    Outputs:
        Immutable company row ready for HANA batch insertion.
    """

    import_id: str
    source_company_id: str
    company_name: str | None
    customer_class: str
    revenue: Decimal | None
    employees: int | None
    nace1: str | None
    nace2: str | None
    nace3: str | None
    company_size: str | None
    legal_form: str | None
    geographic_presence: str | None
    is_listed: bool | None
    is_public_contracting_client: bool | None
    uses_self_governance_code: bool | None


@dataclass(frozen=True)
class BenchmarkSubmission:
    """Normalized questionnaire submission metadata for one company.

    Inputs:
        Import/questionnaire identity, company link, dates, and source statuses.

    Outputs:
        Immutable submission row ready for HANA batch insertion.
    """

    import_id: str
    questionnaire_id: str
    source_company_id: str
    submission_date: date | None
    extraction_date: date | None
    release_status: str | None
    raw_assessment_category: str | None


@dataclass(frozen=True)
class BenchmarkResponse:
    """Canonical response identity plus source audit fields.

    Inputs:
        Import/submission/question identity, mapped answer, source values, and
        response flags.

    Outputs:
        Immutable response row used by scoring and HANA persistence.
    """

    import_id: str
    questionnaire_id: str
    question_id: str
    canonical_answer_id: str
    source_answer_id: str
    source_answer_text: str
    source_answer_value: str | None
    level: int
    selected: bool
    optional: bool
    managed: bool
    validation_status: str | None


@dataclass(frozen=True)
class BenchmarkScore:
    """Calculated score at overall, dimension, or per-topic/question scope.

    Inputs:
        Deterministic score identity, import/submission scope, calculated score,
        and optional workbook-supplied audit score.

    Outputs:
        Immutable score row ready for reconciliation and HANA persistence.
    """

    score_id: str
    import_id: str
    questionnaire_id: str
    scope_type: str
    dimension: str | None
    topic: str | None
    question_id: str | None
    calculated_score: float
    supplied_score: float | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    """Complete validated benchmark import payload before persistence.

    Inputs:
        Original workbook bytes, normalized entities, scores, and summary.

    Outputs:
        Immutable top-level dataset passed to the HANA repository.
    """

    workbook_bytes: bytes
    companies: list[BenchmarkCompany]
    submissions: list[BenchmarkSubmission]
    responses: list[BenchmarkResponse]
    scores: list[BenchmarkScore]
    summary: BenchmarkValidationSummary


@dataclass(frozen=True)
class BenchmarkWriteResult:
    """Outcome from atomic benchmark version persistence.

    Inputs:
        Import identity, status, and whether the active SHA caused a no-op.

    Outputs:
        Immutable result used to update the public validation summary.
    """

    import_id: str
    status: str
    no_op: bool
