"""Models for persisted assessment responses and score payloads."""

from pydantic import BaseModel, Field

from app.models.benchmarking import BenchmarkContext, BenchmarkMetric
from app.services.customer_class_scope import CustomerClass


class AssessmentResponsesRequest(BaseModel):
    """Request to replace persisted user answers for submitted questions.

    Inputs:
        assessment_id: Assessment instance whose answers are being saved.
        customer_class: Company class used to filter unavailable answers.
        source: Source label, either manual UI selection or AI apply.
        answers: Selected answer item IDs keyed by question ID.

    Outputs:
        Validated request object consumed by assessment response routes.
    """

    assessment_id: str
    customer_class: CustomerClass | None = None
    source: str = "manual"
    answers: dict[str, list[str]] = Field(default_factory=dict)


class AssessmentResponsesResponse(BaseModel):
    """Persisted answer selections returned to the Assessment UI.

    Inputs:
        Field values loaded from future assessment response persistence.

    Outputs:
        Validated response object containing saved answer IDs by question.
    """

    assessment_id: str
    customer_class: CustomerClass | None = None
    answers: dict[str, list[str]] = Field(default_factory=dict)


class ApplyAiMarksRequest(BaseModel):
    """Request to apply AI verified marks to the user response state.

    Inputs:
        assessment_id: Assessment instance whose answers are being updated.
        customer_class: Company class used to filter unavailable answers.
        question_id: Assessment question receiving verified AI marks.
        task_id: Optional AI review task that produced the marks.
        answer_item_ids: AI-selected answer item IDs to apply.

    Outputs:
        Validated request object consumed by future AI-apply routes.
    """

    assessment_id: str
    customer_class: CustomerClass | None = None
    question_id: str
    task_id: str | None = None
    answer_item_ids: list[str] = Field(default_factory=list)


class ScoreBenchmarkRow(BaseModel):
    """One benchmark row loaded from HANA or in-memory test state.

    Inputs:
        customer_class: Customer class the benchmark applies to.
        sector: Optional sector-specific scope, or ``None`` for class fallback.
        dimension: Optional dimension scope.
        question_id: Optional question scope.
        benchmark_score: Benchmark score on the same 0-100 scale as responses.
        sample_size: Number of benchmark observations, when available.

    Outputs:
        Validated benchmark row consumed by the pure scoring service.
    """

    customer_class: str
    sector: str | None = None
    dimension: str | None = None
    question_id: str | None = None
    benchmark_score: float
    sample_size: int = 0


class QuestionScore(BaseModel):
    """Report-ready score details for one assessment question.

    Inputs:
        Field values calculated by the scoring service for a single question.

    Outputs:
        Validated question score including weighting, selected answers,
        applicability, and optional peer benchmark comparisons.
    """

    question_id: str
    dimension: str
    section: str
    topic_title: str = ""
    question_text: str
    score: float
    raw_score: float
    max_allowed_level: int
    weight: float
    benchmark: BenchmarkMetric = Field(default_factory=BenchmarkMetric)
    benchmark_score: float | None = None
    benchmark_delta: float | None = None
    same_sector_benchmark_score: float | None = None
    same_sector_benchmark_delta: float | None = None
    same_sector_sample_size: int = 0
    same_size_benchmark_score: float | None = None
    same_size_benchmark_delta: float | None = None
    same_size_sample_size: int = 0
    selected_answer_item_ids: list[str] = Field(default_factory=list)
    applicable: bool


class DimensionScore(BaseModel):
    """Report-ready aggregate score for one assessment dimension.

    Inputs:
        Field values calculated by the scoring service for one dimension.

    Outputs:
        Validated dimension score with question counts and optional benchmark
        comparisons for future report generation.
    """

    dimension: str
    display_name: str
    score: float
    benchmark: BenchmarkMetric = Field(default_factory=BenchmarkMetric)
    benchmark_score: float | None = None
    benchmark_delta: float | None = None
    same_sector_benchmark_score: float | None = None
    same_sector_benchmark_delta: float | None = None
    same_sector_sample_size: int = 0
    same_size_benchmark_score: float | None = None
    same_size_benchmark_delta: float | None = None
    same_size_sample_size: int = 0
    question_count: int
    answered_question_count: int


class AssessmentScoreResponse(BaseModel):
    """Report-ready score payload for the current assessment state.

    Inputs:
        Field values calculated from persisted selections, framework metadata,
        and optional benchmark rows.

    Outputs:
        Validated score response containing final, dimension, and question
        details suitable for later API routes and PDF report generation.
    """

    assessment_id: str
    customer_class: str
    sector: str | None = None
    final_score: float
    benchmark_context: BenchmarkContext | None = None
    benchmark: BenchmarkMetric = Field(default_factory=BenchmarkMetric)
    benchmark_score: float | None = None
    benchmark_delta: float | None = None
    same_sector_benchmark_score: float | None = None
    same_sector_benchmark_delta: float | None = None
    same_sector_sample_size: int = 0
    same_size_benchmark_score: float | None = None
    same_size_benchmark_delta: float | None = None
    same_size_sample_size: int = 0
    applicable_question_count: int
    answered_question_count: int
    dimensions: list[DimensionScore] = Field(default_factory=list)
    questions: list[QuestionScore] = Field(default_factory=list)
