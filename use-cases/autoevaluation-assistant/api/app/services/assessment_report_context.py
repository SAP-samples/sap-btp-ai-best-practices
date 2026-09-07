"""Build strict deterministic snapshots from persisted assessment data."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.models.benchmarking import BenchmarkContext, BenchmarkMetric
from app.models.language import normalize_language
from app.models.reports import (
    AssessmentReportDimension,
    AssessmentReportMetric,
    AssessmentReportProvenance,
    AssessmentReportSource,
    AssessmentReportTopic,
)
from app.services.assessment_scoring import calculate_assessment_score
from app.services.customer_class_scope import customer_class_options


def _snapshot_metric(metric: BenchmarkMetric) -> AssessmentReportMetric:
    """Copy one trusted score metric into the frozen report model.

    Inputs:
        metric: Deterministic company/peer comparison from assessment scoring.

    Outputs:
        AssessmentReportMetric: Independent immutable report value snapshot.
    """

    return AssessmentReportMetric(
        company_score=metric.company_score,
        peer_average=metric.peer_average,
        best_peer=metric.best_peer,
        sample_size=metric.sample_size,
        positioning=metric.positioning,
        commentary_key=metric.commentary_key,
        commentary=metric.commentary,
    )


def _snapshot_provenance(
    context: BenchmarkContext,
    *,
    customer_class: str,
    nace1: str,
) -> AssessmentReportProvenance:
    """Copy active-import/cohort context into immutable report provenance.

    Inputs:
        context: Deterministic benchmark context returned by scoring.
        customer_class: Persisted profile class used for the exact cohort.
        nace1: Persisted profile NACE-1 label used for the exact cohort.

    Outputs:
        AssessmentReportProvenance: Frozen availability and version metadata.
    """

    return AssessmentReportProvenance(
        benchmark_available=context.available,
        unavailable_reason=context.reason,
        import_id=context.import_id,
        source_filename=context.source_filename,
        source_sha256=context.source_sha256,
        dataset_activated_at=context.dataset_activated_at,
        scoring_version=context.scoring_version,
        customer_class=customer_class,
        nace1=nace1,
        peer_sample_size=context.peer_sample_size,
    )


def _customer_class_label(customer_class: str, language: str) -> str:
    """Return the frozen localized label for one exact persisted class.

    Inputs:
        customer_class: Valid persisted class identifier.
        language: Normalized report language.

    Outputs:
        str: Configured localized class label, falling back to the identifier.
    """

    return next(
        (
            option["label"]
            for option in customer_class_options(language)
            if option["value"] == customer_class
        ),
        customer_class,
    )


def build_assessment_report_source(
    repository: Any,
    assessment_id: str,
    language: str,
    generated_at: datetime | None = None,
) -> AssessmentReportSource:
    """Snapshot one persisted profile, active import, score tree, and provenance.

    Inputs:
        repository: Assessment repository exposing profile, framework, response,
            active-import, and exact-cohort peer reads.
        assessment_id: Assessment instance to snapshot.
        language: Requested English or Italian report language.
        generated_at: Optional deterministic enqueue timestamp for tests.

    Outputs:
        AssessmentReportSource: Frozen report-specific score tree persisted
        before asynchronous rendering and never recalculated by the worker.

    Raises:
        ValueError: If language is unsupported or the assessment has no persisted
        profile defining both class and NACE-1 cohort context.
    """

    normalized_language = normalize_language(language)
    profile_loader = getattr(repository, "get_assessment_profile", None)
    profile = profile_loader(assessment_id) if callable(profile_loader) else None
    if profile is None:
        raise ValueError(
            "A persisted assessment profile is required before generating a report."
        )

    # Capture the active import exactly once. Peer rows are requested by this
    # immutable ID so a concurrent activation cannot mix two dataset versions.
    active_import_loader = getattr(repository, "get_active_benchmark_import", None)
    active_import = active_import_loader() if callable(active_import_loader) else None
    peer_submissions = []
    if active_import is not None:
        peer_submissions = repository.list_benchmark_peer_submissions(
            active_import.import_id,
            profile.customer_class,
            profile.nace1,
        )

    dimensions = repository.list_dimensions(language=normalized_language)
    questions = repository.list_all_questions(language=normalized_language)
    # Read persisted answers once so every snapshotted score uses one revision.
    selected_answers = repository.get_assessment_responses(assessment_id)
    score = calculate_assessment_score(
        assessment_id=assessment_id,
        customer_class=profile.customer_class,
        sector=profile.nace1,
        dimensions=dimensions,
        questions=questions,
        selected_answers=selected_answers,
        benchmarks=[],
        benchmark_import=active_import,
        peer_submissions=peer_submissions,
        nace1=profile.nace1,
        source_company_id=profile.source_company_id,
        language=normalized_language,
    )
    if score.benchmark_context is None:
        raise ValueError("Assessment score did not return benchmark provenance.")

    topics_by_dimension: dict[str, list[AssessmentReportTopic]] = defaultdict(list)
    for question_score in score.questions:
        topics_by_dimension[question_score.dimension].append(
            AssessmentReportTopic(
                question_id=question_score.question_id,
                dimension=question_score.dimension,
                topic_title=question_score.topic_title or question_score.section,
                answered=bool(question_score.selected_answer_item_ids),
                metric=_snapshot_metric(question_score.benchmark),
            )
        )

    report_dimensions = tuple(
        AssessmentReportDimension(
            dimension=dimension_score.dimension,
            display_name=dimension_score.display_name,
            metric=_snapshot_metric(dimension_score.benchmark),
            topics=tuple(topics_by_dimension[dimension_score.dimension]),
        )
        for dimension_score in score.dimensions
        if topics_by_dimension[dimension_score.dimension]
    )
    return AssessmentReportSource(
        assessment_id=assessment_id,
        display_name=profile.display_name,
        source_company_id=profile.source_company_id,
        customer_class=profile.customer_class,
        customer_class_label=_customer_class_label(
            profile.customer_class,
            normalized_language,
        ),
        nace1=profile.nace1,
        language=normalized_language,
        generated_at=generated_at or datetime.now(timezone.utc),
        applicable_question_count=score.applicable_question_count,
        answered_question_count=score.answered_question_count,
        overall=_snapshot_metric(score.benchmark),
        dimensions=report_dimensions,
        provenance=_snapshot_provenance(
            score.benchmark_context,
            customer_class=profile.customer_class,
            nace1=profile.nace1,
        ),
    )
