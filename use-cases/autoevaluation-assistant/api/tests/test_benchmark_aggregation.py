"""Focused tests for imported benchmark scoring and cohort aggregation."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.services.assessment_scoring import calculate_assessment_score
from app.services.benchmark_aggregation import (
    PeerScopeAggregate,
    build_benchmark_metric,
)
from app.services.benchmark_import.scoring import calculate_question_score


def _question(
    question_id: str = "Q.STR.03.01",
    *,
    topic_title: str = "Definition of Objectives",
    level_counts: dict[int, int] | None = None,
) -> AssessmentQuestion:
    """Return one configurable canonical scoring question.

    Inputs:
        question_id: Stable framework question identity.
        topic_title: Canonical localized topic title exposed by scoring.
        level_counts: Number of answer items to create per maturity level.

    Outputs:
        AssessmentQuestion: Question fixture with deterministic answer IDs.
    """

    answer_items: list[AnswerItem] = []
    for level, count in (level_counts or {1: 2, 2: 1, 3: 1}).items():
        for item_index in range(1, count + 1):
            answer_items.append(
                AnswerItem(
                    answer_item_id=f"{question_id}-L{level}-{item_index:03d}",
                    question_id=question_id,
                    level=level,
                    item_index=item_index,
                    text=f"Level {level}, item {item_index}",
                )
            )
    return AssessmentQuestion(
        question_id=question_id,
        dimension="Strategy",
        section="Section must not become the topic",
        topic_title=topic_title,
        question=f"Question {question_id}",
        answer_items=answer_items,
    )


def _benchmark_import() -> object:
    """Return active import metadata accepted by the scoring service.

    Inputs:
        None.

    Outputs:
        BenchmarkImportInfo: Active version metadata without workbook bytes.
    """

    from app.models.benchmarking import BenchmarkImportInfo

    return BenchmarkImportInfo(
        import_id="import-active",
        source_filename="benchmark.xlsx",
        source_sha256="a" * 64,
        scoring_version="assessment-v1",
        row_count=300,
        company_count=6,
        questionnaire_count=8,
        question_count=50,
        accepted_count=300,
        rejected_count=0,
        status="active",
        is_active=True,
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        activated_at=datetime(2026, 7, 2, tzinfo=timezone.utc),
    )


def _peer_submission(
    company_id: str,
    questionnaire_id: str,
    *,
    customer_class: str = "class_1",
    nace1: str = "Energy",
    submitted: date = date(2026, 1, 1),
    release_status: str = "REL",
    overall: float = 50.0,
    dimension: float = 50.0,
    topic: float = 50.0,
) -> object:
    """Return one imported peer submission with all required score scopes.

    Inputs:
        Company, cohort, release, date, and score values for the fixture.

    Outputs:
        BenchmarkPeerSubmission: Immutable peer submission used by aggregation.
    """

    from app.models.benchmarking import BenchmarkPeerSubmission, BenchmarkScopeScore

    return BenchmarkPeerSubmission(
        source_company_id=company_id,
        questionnaire_id=questionnaire_id,
        customer_class=customer_class,
        nace1=nace1,
        submission_date=submitted,
        extraction_date=submitted,
        release_status=release_status,
        scores=[
            BenchmarkScopeScore(scope_type="overall", calculated_score=overall),
            BenchmarkScopeScore(
                scope_type="dimension",
                dimension="Strategy",
                calculated_score=dimension,
            ),
            BenchmarkScopeScore(
                scope_type="topic",
                dimension="Strategy",
                question_id="Q.STR.03.01",
                calculated_score=topic,
            ),
        ],
    )


def test_shared_question_formula_filters_and_deduplicates_in_order() -> None:
    """Verify imported and live scoring share one detailed maturity calculation."""

    from app.services.question_scoring import calculate_question_score_details

    question = _question()
    selected = [
        "Q.STR.03.01-L1-001",
        "Q.STR.03.01-L1-001",
        "Q.STR.03.01-L1-002",
        "Q.STR.03.01-L2-001",
        "Q.STR.03.01-L3-001",
    ]

    detail = calculate_question_score_details(question, selected, maximum_level=2)

    assert detail.selected_answer_ids == [
        "Q.STR.03.01-L1-001",
        "Q.STR.03.01-L1-002",
        "Q.STR.03.01-L2-001",
    ]
    assert detail.selected_points == 2.0
    assert detail.score == 100.0
    assert calculate_question_score(question, selected, maximum_level=2) == detail.score


def test_latest_released_submission_selection_is_exact_and_excludes_company() -> None:
    """Verify cohort selection keeps one latest released exact-cohort submission."""

    from app.services.benchmark_aggregation import select_latest_released_submissions

    submissions = [
        _peer_submission("peer-1", "old", submitted=date(2026, 1, 1)),
        _peer_submission("peer-1", "new", submitted=date(2026, 2, 1)),
        _peer_submission("peer-2", "draft", release_status="DRAFT"),
        _peer_submission("peer-3", "other-class", customer_class="class_2"),
        _peer_submission("peer-4", "other-nace", nace1="Manufacturing"),
        _peer_submission("current", "self"),
        _peer_submission("peer-5", "exact"),
    ]

    selected = select_latest_released_submissions(
        submissions,
        customer_class="class_1",
        nace1="Energy",
        excluded_source_company_id="current",
    )

    assert [(item.source_company_id, item.questionnaire_id) for item in selected] == [
        ("peer-1", "new"),
        ("peer-5", "exact"),
    ]


def test_peer_aggregation_uses_independent_average_best_and_boundary_positions() -> None:
    """Verify averages, maxima, exact thresholds, and zero-average behavior."""

    from app.services.benchmark_aggregation import (
        aggregate_peer_scope_scores,
        position_against_peers,
    )

    peers = [
        _peer_submission("peer-1", "q1", overall=60, dimension=40, topic=0),
        _peer_submission("peer-2", "q2", overall=90, dimension=60, topic=0),
        _peer_submission("peer-3", "q3", overall=120, dimension=80, topic=0),
    ]

    aggregate = aggregate_peer_scope_scores(peers)

    assert aggregate.overall.peer_average == 90.0
    assert aggregate.overall.best_peer == 120.0
    assert aggregate.overall.sample_size == 3
    assert aggregate.dimensions["Strategy"].peer_average == 60.0
    assert aggregate.dimensions["Strategy"].best_peer == 80.0
    assert aggregate.topics["Q.STR.03.01"].peer_average == 0.0
    assert aggregate.topics["Q.STR.03.01"].best_peer == 0.0
    assert position_against_peers(81.0, 90.0) == "in_line_with_peers"
    assert position_against_peers(99.0, 90.0) == "in_line_with_peers"
    assert position_against_peers(80.9999, 90.0) == "below_peers"
    assert position_against_peers(99.0001, 90.0) == "above_peers"
    assert position_against_peers(0.0, 0.0) == "in_line_with_peers"
    assert position_against_peers(0.0001, 0.0) == "above_peers"
    assert position_against_peers(50.0, None) == "unavailable"


@pytest.mark.parametrize(
    ("company_score", "peer_average", "expected"),
    [
        (120.0, 100.0, "Above the peer average by 20%."),
        (80.0, 100.0, "Below the peer average by 20%."),
        (103.0, 100.0, "In line with the peer average (3% above)."),
        (97.0, 100.0, "In line with the peer average (3% below)."),
        (100.0, 100.0, "In line with the peer average (no difference)."),
        (99.6, 100.0, "In line with the peer average (less than 1% below)."),
        (112.5, 100.0, "Above the peer average by 13%."),
        (1.0, 0.0, "Above the peer average; relative percentage is not applicable."),
        (0.0, 0.0, "In line with the peer average; relative percentage is not applicable."),
    ],
)
def test_deterministic_english_commentary_reports_rounded_relative_difference(
    company_score: float,
    peer_average: float,
    expected: str,
) -> None:
    """Verify every deterministic relative-commentary edge case in English."""

    metric = build_benchmark_metric(
        company_score,
        PeerScopeAggregate(
            peer_average=peer_average,
            best_peer=max(company_score, peer_average),
            sample_size=3,
        ),
        available=True,
        language="en",
    )

    assert metric.commentary == expected


def test_deterministic_italian_commentary_and_unavailable_fallback() -> None:
    """Verify Italian relative wording and withheld-peer fallback."""

    available = build_benchmark_metric(
        80.0,
        PeerScopeAggregate(peer_average=100.0, best_peer=100.0, sample_size=3),
        available=True,
        language="it",
    )
    unavailable = build_benchmark_metric(
        80.0,
        None,
        available=False,
        language="it",
    )

    assert available.commentary == "Sotto la media dei peer del 20%."
    assert unavailable.commentary == "Confronto con i peer non disponibile."


def test_score_marks_small_cohort_unavailable_and_never_uses_legacy_rows() -> None:
    """Verify fewer than three peers disables every metric and fake fallbacks."""

    from app.models.scoring import ScoreBenchmarkRow

    question = _question()
    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Legacy sector must not drive cohorts",
        dimensions=[
            AssessmentDimension(
                dimension="Strategy",
                display_name="Strategy",
                question_count=1,
            )
        ],
        questions=[question],
        selected_answers={
            question.question_id: ["Q.STR.03.01-L1-001"]
        },
        benchmarks=[
            ScoreBenchmarkRow(
                customer_class="class_1",
                benchmark_score=88.0,
                sample_size=99,
            )
        ],
        benchmark_import=_benchmark_import(),
        peer_submissions=[
            _peer_submission("peer-1", "q1"),
            _peer_submission("peer-2", "q2"),
        ],
        nace1="Energy",
        source_company_id=None,
        language="en",
    )

    assert payload.benchmark_context.available is False
    assert payload.benchmark_context.reason == "insufficient_peer_sample"
    assert payload.benchmark_context.peer_sample_size == 2
    assert payload.benchmark.positioning == "unavailable"
    assert payload.benchmark_score is None
    assert payload.benchmark_delta is None
    assert payload.same_sector_benchmark_score is None
    assert payload.same_sector_sample_size == 0
    assert payload.same_size_benchmark_score is None
    assert payload.same_size_sample_size == 0
    assert payload.dimensions[0].benchmark.positioning == "unavailable"
    assert payload.questions[0].benchmark.positioning == "unavailable"


def test_score_exposes_active_context_real_aggregates_aliases_and_applicable_topics() -> None:
    """Verify score output uses exact active peers and filters inapplicable topics."""

    applicable = _question()
    inapplicable = _question(
        "Q.STR.01.01",
        topic_title="Strategic Planning",
        level_counts={1: 1},
    )
    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Ignored legacy sector",
        dimensions=[
            AssessmentDimension(
                dimension="Strategy",
                display_name="Strategy",
                question_count=2,
            )
        ],
        questions=[inapplicable, applicable],
        selected_answers={
            applicable.question_id: [
                "Q.STR.03.01-L1-001",
                "Q.STR.03.01-L1-002",
            ]
        },
        benchmarks=[],
        benchmark_import=_benchmark_import(),
        peer_submissions=[
            _peer_submission("current", "self", overall=100, dimension=100, topic=100),
            _peer_submission("peer-1", "q1", overall=40, dimension=30, topic=20),
            _peer_submission("peer-2", "q2", overall=50, dimension=60, topic=40),
            _peer_submission("peer-3", "q3", overall=90, dimension=90, topic=90),
        ],
        nace1="Energy",
        source_company_id="current",
        language="it",
    )

    assert payload.benchmark_context.available is True
    assert payload.benchmark_context.import_id == "import-active"
    assert payload.benchmark_context.source_sha256 == "a" * 64
    assert payload.benchmark_context.source_filename == "benchmark.xlsx"
    assert payload.benchmark_context.scoring_version == "assessment-v1"
    assert payload.benchmark_context.dataset_activated_at == datetime(
        2026, 7, 2, tzinfo=timezone.utc
    )
    assert payload.benchmark_context.customer_class == "class_1"
    assert payload.benchmark_context.nace1 == "Energy"
    assert payload.benchmark_context.peer_sample_size == 3
    assert payload.benchmark_context.cohort_definition == "exact_customer_class_and_nace1"
    assert payload.benchmark.peer_average == 60.0
    assert payload.benchmark.best_peer == 90.0
    assert payload.benchmark.sample_size == 3
    assert payload.benchmark_score == 60.0
    assert payload.benchmark_delta == -10.0
    assert payload.benchmark.commentary == "Sotto la media dei peer del 17%."
    assert payload.dimensions[0].benchmark.peer_average == 60.0
    assert payload.dimensions[0].benchmark.best_peer == 90.0
    assert [question.question_id for question in payload.questions] == [
        "Q.STR.03.01"
    ]
    assert payload.questions[0].topic_title == "Definition of Objectives"
    assert payload.questions[0].benchmark.peer_average == 50.0
    assert payload.questions[0].benchmark.best_peer == 90.0
    assert payload.questions[0].benchmark.commentary == (
        "In linea con la media dei peer (nessuna differenza)."
    )
    assert payload.same_sector_benchmark_score is None
    assert payload.same_size_benchmark_score is None


def test_score_without_active_dataset_returns_explicit_unavailable_context() -> None:
    """Verify no active dataset produces explicit null peer metrics without fallback."""

    question = _question()
    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=[
            AssessmentDimension(
                dimension="Strategy",
                display_name="Strategy",
                question_count=1,
            )
        ],
        questions=[question],
        selected_answers={},
        benchmarks=[],
        benchmark_import=None,
        peer_submissions=[],
        nace1="Energy",
        source_company_id=None,
        language="en",
    )

    assert payload.benchmark_context.available is False
    assert payload.benchmark_context.reason == "no_active_dataset"
    assert payload.benchmark.peer_average is None
    assert payload.benchmark.best_peer is None
    assert payload.benchmark.positioning == "unavailable"
    assert payload.benchmark.commentary == "Peer comparison unavailable."
