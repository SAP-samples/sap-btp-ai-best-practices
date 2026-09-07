"""Calculate report-ready company scores and active-import peer comparisons."""

from __future__ import annotations

from collections import defaultdict

from app.models.assessment import AssessmentDimension, AssessmentQuestion
from app.models.benchmarking import (
    BenchmarkContext,
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
)
from app.models.scoring import (
    AssessmentScoreResponse,
    DimensionScore,
    QuestionScore,
    ScoreBenchmarkRow,
)
from app.services.benchmark_aggregation import (
    MINIMUM_PEER_COMPANIES,
    PeerBenchmarkAggregate,
    PeerScopeAggregate,
    aggregate_peer_scope_scores,
    build_benchmark_metric,
    select_latest_released_submissions,
)
from app.services.customer_class_scope import max_allowed_level, normalize_customer_class
from app.services.question_scoring import (
    QuestionScoreCalculation,
    calculate_question_score_details,
)


def _round_score(value: float) -> float:
    """Round one score or delta to the API's stable four-decimal precision.

    Inputs:
        value: Raw score or calculated difference.

    Outputs:
        float: Four-decimal score value.
    """

    return round(float(value), 4)


def _benchmark_delta(score: float, peer_average: float | None) -> float | None:
    """Return company minus peer average for the temporary compatibility alias.

    Inputs:
        score: Current company score.
        peer_average: Optional active-cohort arithmetic average.

    Outputs:
        float | None: Rounded delta, or ``None`` when peers are unavailable.
    """

    return None if peer_average is None else _round_score(score - peer_average)


def _empty_peer_aggregate() -> PeerBenchmarkAggregate:
    """Return an empty peer aggregate used for unavailable benchmark contexts.

    Inputs:
        None.

    Outputs:
        PeerBenchmarkAggregate: Empty overall, dimension, and topic scopes.
    """

    return PeerBenchmarkAggregate(
        overall=PeerScopeAggregate(),
        dimensions={},
        topics={},
    )


def _benchmark_state(
    *,
    benchmark_import: BenchmarkImportInfo | None,
    peer_submissions: list[BenchmarkPeerSubmission],
    customer_class: str,
    nace1: str | None,
    source_company_id: str | None,
) -> tuple[BenchmarkContext, PeerBenchmarkAggregate]:
    """Build active version context and exact-cohort aggregate for one score.

    Inputs:
        benchmark_import: Safe metadata for the active persisted import.
        peer_submissions: Active-import submissions available to the repository.
        customer_class: Normalized persisted assessment class.
        nace1: Exact persisted level-one NACE label.
        source_company_id: Optional assessed company identity to exclude.

    Outputs:
        tuple[BenchmarkContext, PeerBenchmarkAggregate]: Audit context plus
        available aggregates, or an explicit unavailable context and empty data.
    """

    if benchmark_import is None:
        return (
            BenchmarkContext(
                available=False,
                reason="no_active_dataset",
                customer_class=customer_class,
                nace1=nace1,
            ),
            _empty_peer_aggregate(),
        )

    if not nace1:
        return (
            BenchmarkContext(
                available=False,
                reason="missing_profile_context",
                import_id=benchmark_import.import_id,
                source_sha256=benchmark_import.source_sha256,
                source_filename=benchmark_import.source_filename,
                scoring_version=benchmark_import.scoring_version,
                dataset_activated_at=benchmark_import.activated_at,
                customer_class=customer_class,
                nace1=None,
            ),
            _empty_peer_aggregate(),
        )

    selected_peers = select_latest_released_submissions(
        peer_submissions,
        customer_class=customer_class,
        nace1=nace1,
        excluded_source_company_id=source_company_id,
    )
    available = len(selected_peers) >= MINIMUM_PEER_COMPANIES
    context = BenchmarkContext(
        available=available,
        reason=None if available else "insufficient_peer_sample",
        import_id=benchmark_import.import_id,
        source_sha256=benchmark_import.source_sha256,
        source_filename=benchmark_import.source_filename,
        scoring_version=benchmark_import.scoring_version,
        dataset_activated_at=benchmark_import.activated_at,
        customer_class=customer_class,
        nace1=nace1,
        peer_sample_size=len(selected_peers),
    )
    return (
        context,
        aggregate_peer_scope_scores(selected_peers)
        if available
        else _empty_peer_aggregate(),
    )


def calculate_assessment_score(
    assessment_id: str,
    customer_class: str | None,
    sector: str | None,
    dimensions: list[AssessmentDimension],
    questions: list[AssessmentQuestion],
    selected_answers: dict[str, list[str]],
    benchmarks: list[ScoreBenchmarkRow] | None = None,
    benchmark_import: BenchmarkImportInfo | None = None,
    peer_submissions: list[BenchmarkPeerSubmission] | None = None,
    nace1: str | None = None,
    source_company_id: str | None = None,
    language: str = "en",
) -> AssessmentScoreResponse:
    """Calculate company scores and imported exact-cohort peer comparisons.

    Inputs:
        assessment_id: Assessment instance being scored.
        customer_class: Persisted class used for question applicability.
        sector: Deprecated legacy display/query value retained in the response.
        dimensions: Localized framework dimensions from the repository.
        questions: Localized framework questions and answer catalogs from HANA.
        selected_answers: Persisted answer IDs keyed by canonical question ID.
        benchmarks: Deprecated legacy benchmark rows; intentionally ignored so
            callers can never fabricate or fall back to runtime seed values.
        benchmark_import: Safe active-import version metadata, if one exists.
        peer_submissions: Score-bearing submissions loaded only from that active
            import; pure logic applies exact cohort/latest-release rules.
        nace1: Exact persisted profile NACE-1 label.
        source_company_id: Optional profile company ID excluded from peers.
        language: Normalized response language for deterministic commentary.

    Outputs:
        AssessmentScoreResponse: Applicable topic scores, dimension/overall
        scores, active dataset context, nested peer metrics, peer-average aliases,
        and null/zero deprecated same-sector/same-size compatibility fields.
    """

    # Legacy rows are accepted only so older internal callers do not crash. They
    # must never influence a score after versioned benchmark imports are enabled.
    _ = benchmarks
    normalized_class = normalize_customer_class(customer_class)
    applicable_questions = [
        question
        for question in questions
        if max_allowed_level(question.question_id, normalized_class) > 0
    ]
    applicable_count = len(applicable_questions)
    question_weight = 1.0 / applicable_count if applicable_count else 0.0

    calculation_rows: list[tuple[AssessmentQuestion, QuestionScoreCalculation]] = []
    for question in applicable_questions:
        calculation_rows.append(
            (
                question,
                calculate_question_score_details(
                    question,
                    selected_answers.get(question.question_id, []),
                    max_allowed_level(question.question_id, normalized_class),
                ),
            )
        )

    question_scores_by_dimension: dict[str, list[float]] = defaultdict(list)
    answered_by_dimension: dict[str, int] = defaultdict(int)
    for question, calculation in calculation_rows:
        question_scores_by_dimension[question.dimension].append(calculation.score)
        if calculation.selected_answer_ids:
            answered_by_dimension[question.dimension] += 1

    company_dimension_scores = {
        dimension: _round_score(sum(values) / len(values))
        for dimension, values in question_scores_by_dimension.items()
        if values
    }
    final_score = (
        _round_score(
            sum(calculation.score for _question, calculation in calculation_rows)
            / applicable_count
        )
        if applicable_count
        else 0.0
    )

    benchmark_context, peer_aggregate = _benchmark_state(
        benchmark_import=benchmark_import,
        peer_submissions=peer_submissions or [],
        customer_class=normalized_class,
        nace1=nace1,
        source_company_id=source_company_id,
    )
    overall_benchmark = build_benchmark_metric(
        final_score,
        peer_aggregate.overall,
        available=benchmark_context.available,
        language=language,
    )

    dimensions_by_name = {dimension.dimension: dimension for dimension in dimensions}
    dimension_scores: list[DimensionScore] = []
    for dimension_name, dimension in dimensions_by_name.items():
        if dimension_name not in company_dimension_scores:
            continue
        score = company_dimension_scores[dimension_name]
        peer_metric = build_benchmark_metric(
            score,
            peer_aggregate.dimensions.get(dimension_name),
            available=benchmark_context.available,
            language=language,
        )
        dimension_scores.append(
            DimensionScore(
                dimension=dimension_name,
                display_name=dimension.display_name,
                score=score,
                benchmark=peer_metric,
                benchmark_score=peer_metric.peer_average,
                benchmark_delta=_benchmark_delta(score, peer_metric.peer_average),
                # Deprecated pre-import fields are intentionally never fabricated.
                same_sector_benchmark_score=None,
                same_sector_benchmark_delta=None,
                same_sector_sample_size=0,
                same_size_benchmark_score=None,
                same_size_benchmark_delta=None,
                same_size_sample_size=0,
                question_count=len(question_scores_by_dimension[dimension_name]),
                answered_question_count=answered_by_dimension[dimension_name],
            )
        )

    question_scores: list[QuestionScore] = []
    for question, calculation in calculation_rows:
        peer_metric = build_benchmark_metric(
            calculation.score,
            peer_aggregate.topics.get(question.question_id),
            available=benchmark_context.available,
            language=language,
        )
        question_scores.append(
            QuestionScore(
                question_id=question.question_id,
                dimension=question.dimension,
                section=question.section,
                topic_title=question.topic_title or question.section,
                question_text=question.question,
                score=_round_score(calculation.score),
                raw_score=_round_score(calculation.selected_points),
                max_allowed_level=max_allowed_level(
                    question.question_id,
                    normalized_class,
                ),
                weight=_round_score(question_weight),
                benchmark=peer_metric,
                benchmark_score=peer_metric.peer_average,
                benchmark_delta=_benchmark_delta(
                    calculation.score,
                    peer_metric.peer_average,
                ),
                # Deprecated pre-import fields are intentionally never fabricated.
                same_sector_benchmark_score=None,
                same_sector_benchmark_delta=None,
                same_sector_sample_size=0,
                same_size_benchmark_score=None,
                same_size_benchmark_delta=None,
                same_size_sample_size=0,
                selected_answer_item_ids=calculation.selected_answer_ids,
                applicable=True,
            )
        )

    return AssessmentScoreResponse(
        assessment_id=assessment_id,
        customer_class=normalized_class,
        sector=sector,
        final_score=final_score,
        benchmark_context=benchmark_context,
        benchmark=overall_benchmark,
        benchmark_score=overall_benchmark.peer_average,
        benchmark_delta=_benchmark_delta(final_score, overall_benchmark.peer_average),
        # Deprecated pre-import fields remain present only for response compatibility.
        same_sector_benchmark_score=None,
        same_sector_benchmark_delta=None,
        same_sector_sample_size=0,
        same_size_benchmark_score=None,
        same_size_benchmark_delta=None,
        same_size_sample_size=0,
        applicable_question_count=applicable_count,
        answered_question_count=sum(
            1
            for _question, calculation in calculation_rows
            if calculation.selected_answer_ids
        ),
        dimensions=dimension_scores,
        questions=question_scores,
    )
