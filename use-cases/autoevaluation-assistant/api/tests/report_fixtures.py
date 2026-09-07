"""Deterministic snapshot fixtures shared by report tests and visual QA."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.models.reports import (
    AssessmentReportDimension,
    AssessmentReportMetric,
    AssessmentReportProvenance,
    AssessmentReportSource,
    AssessmentReportTopic,
)
from app.services.customer_class_scope import (
    available_question_ids,
    customer_class_options,
)
from app.services.benchmark_aggregation import (
    PeerScopeAggregate,
    build_benchmark_metric,
)
from app.services.framework_importer import ITALIAN_DIMENSION_FILES, load_framework_seed
from app.services.framework_topics import topic_title_for


SNAPSHOT_TIME = datetime(2026, 7, 13, 9, 30, tzinfo=timezone.utc)
"""Stable report enqueue time used by deterministic fixtures."""


def make_metric(
    company_score: float,
    *,
    peer_average: float | None = 48.2345,
    best_peer: float | None = 91.3456,
    sample_size: int = 4,
    language: str = "en",
) -> AssessmentReportMetric:
    """Build one deterministic report metric.

    Inputs:
        company_score: Exact snapshotted company value.
        peer_average: Optional arithmetic cohort average.
        best_peer: Optional independent cohort maximum.
        sample_size: Distinct peer count for the scope.
        language: Commentary language, ``en`` or ``it``.

    Outputs:
        AssessmentReportMetric: Frozen metric suitable for source fixtures.
    """

    metric = build_benchmark_metric(
        company_score,
        PeerScopeAggregate(
            peer_average=peer_average,
            best_peer=best_peer,
            sample_size=sample_size,
        ),
        available=peer_average is not None,
        language=language,
    )
    return AssessmentReportMetric.model_validate(metric.model_dump())


def make_compact_source(language: str = "en") -> AssessmentReportSource:
    """Build a compact valid source for lifecycle and rendering tests.

    Inputs:
        language: Snapshot language, ``en`` or ``it``.

    Outputs:
        AssessmentReportSource: One-dimension, one-topic frozen snapshot.
    """

    dimension_name = "Strategia" if language == "it" else "Strategy"
    topic_title = (
        "Definizione degli Obiettivi"
        if language == "it"
        else "Definition of Objectives"
    )
    metric = make_metric(37.1234, language=language)
    return AssessmentReportSource(
        assessment_id="assessment-1",
        display_name="assessment",
        source_company_id="company-current",
        customer_class="class_3",
        customer_class_label="Media - Classe 3" if language == "it" else "Medium - Class 3",
        nace1="Electricity and gas",
        language=language,
        generated_at=SNAPSHOT_TIME,
        applicable_question_count=1,
        answered_question_count=1,
        overall=metric,
        dimensions=(
            AssessmentReportDimension(
                dimension="Strategy",
                display_name=dimension_name,
                metric=metric,
                topics=(
                    AssessmentReportTopic(
                        question_id="Q.STR.03.01",
                        dimension="Strategy",
                        topic_title=topic_title,
                        answered=True,
                        metric=metric,
                    ),
                ),
            ),
        ),
        provenance=AssessmentReportProvenance(
            benchmark_available=True,
            unavailable_reason=None,
            import_id="import-a",
            source_filename="benchmark-a.xlsx",
            source_sha256="a" * 64,
            dataset_activated_at=SNAPSHOT_TIME,
            scoring_version="assessment-v1",
            customer_class="class_3",
            nace1="Electricity and gas",
            peer_sample_size=4,
        ),
    )


def make_class_source(
    repo_root: Path,
    *,
    customer_class: str,
    language: str = "en",
    include_peers: bool = True,
    long_labels: bool = True,
) -> AssessmentReportSource:
    """Build a representative report fixture for any configured customer class.

    Inputs:
        repo_root: API root containing canonical framework source files.
        customer_class: Configured class whose applicable question scope is frozen.
        language: Snapshot language, ``en`` or ``it``.
        include_peers: Whether peer average/best layers are available.
        long_labels: Whether one topic receives a deliberate wrapping stress label.

    Outputs:
        AssessmentReportSource: Frozen source containing all class-applicable topics.
    """

    seed = load_framework_seed(
        repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    applicable_ids = set(available_question_ids(customer_class))
    class_labels = {
        option["value"]: option["label"]
        for option in customer_class_options(language)
    }
    peer_average = 48.2345 if include_peers else None
    best_peer = 91.3456 if include_peers else None
    sample_size = 4 if include_peers else 0
    topics_by_dimension: dict[str, list[AssessmentReportTopic]] = {}
    for index, question in enumerate(seed.questions, start=1):
        if question.question_id not in applicable_ids:
            continue
        title = topic_title_for(question.question_id, language)
        if long_labels and question.question_id == "Q.FDR.12.01":
            title = (
                "Riservatezza delle informazioni aziendali nei processi interni, esterni e nelle responsabilità di governance delegate"
                if language == "it"
                else "Confidentiality of company information across internal, external, and delegated governance responsibilities"
            )
        metric = make_metric(
            37.1234 if index == 1 else float((index * 13) % 101),
            peer_average=peer_average,
            best_peer=best_peer,
            sample_size=sample_size,
            language=language,
        )
        topics_by_dimension.setdefault(question.dimension, []).append(
            AssessmentReportTopic(
                question_id=question.question_id,
                dimension=question.dimension,
                topic_title=title,
                answered=True,
                metric=metric,
            )
        )

    dimension_labels = {
        dimension: labels[1] if language == "it" else dimension
        for dimension, labels in ITALIAN_DIMENSION_FILES.items()
    }
    dimensions: list[AssessmentReportDimension] = []
    for index, (dimension, topics) in enumerate(topics_by_dimension.items(), start=1):
        dimension_metric = make_metric(
            float(30 + index * 7),
            peer_average=peer_average,
            best_peer=best_peer,
            sample_size=sample_size,
            language=language,
        )
        dimensions.append(
            AssessmentReportDimension(
                dimension=dimension,
                display_name=dimension_labels.get(dimension, dimension),
                metric=dimension_metric,
                topics=tuple(topics),
            )
        )

    overall_metric = make_metric(
        37.1234,
        peer_average=peer_average,
        best_peer=best_peer,
        sample_size=sample_size,
        language=language,
    )
    answered_count = sum(
        topic.answered for dimension in dimensions for topic in dimension.topics
    )
    return AssessmentReportSource(
        assessment_id=f"assessment-{customer_class.replace('_', '-')}",
        display_name=f"Benchmark assessment for {customer_class.replace('_', ' ')}",
        source_company_id="company-current",
        customer_class=customer_class,
        customer_class_label=class_labels[customer_class],
        nace1="Electricity and gas",
        language=language,
        generated_at=SNAPSHOT_TIME,
        applicable_question_count=len(applicable_ids),
        answered_question_count=answered_count,
        overall=overall_metric,
        dimensions=tuple(dimensions),
        provenance=AssessmentReportProvenance(
            benchmark_available=include_peers,
            unavailable_reason=None if include_peers else "insufficient_peer_sample",
            import_id="import-a",
            source_filename="benchmark-a.xlsx",
            source_sha256="a" * 64,
            dataset_activated_at=SNAPSHOT_TIME,
            scoring_version="assessment-v1",
            customer_class=customer_class,
            nace1="Electricity and gas",
            peer_sample_size=sample_size,
        ),
    )


def make_class3_source(
    repo_root: Path,
    *,
    language: str = "en",
    include_peers: bool = True,
    long_labels: bool = True,
) -> AssessmentReportSource:
    """Build the legacy representative class-three report fixture.

    Inputs:
        repo_root: API root containing canonical framework source files.
        language: Snapshot language, ``en`` or ``it``.
        include_peers: Whether peer average/best layers are available.
        long_labels: Whether one topic receives a deliberate wrapping stress label.

    Outputs:
        AssessmentReportSource: Generic class-scoped fixture fixed to class three.
    """

    return make_class_source(
        repo_root,
        customer_class="class_3",
        language=language,
        include_peers=include_peers,
        long_labels=long_labels,
    )
