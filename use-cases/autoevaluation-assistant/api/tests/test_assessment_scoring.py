"""Tests for assessment scoring math and report-ready score payloads."""

from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.models.scoring import ScoreBenchmarkRow
from app.services.assessment_scoring import calculate_assessment_score

SECTOR_BENCHMARK_CLASS = "__sector__"


def _question(question_id: str, dimension: str, level_counts: dict[int, int]) -> AssessmentQuestion:
    """Create one scoring fixture question with configurable answer counts."""
    answer_items: list[AnswerItem] = []
    for level, count in level_counts.items():
        for index in range(1, count + 1):
            answer_items.append(
                AnswerItem(
                    answer_item_id=f"{question_id}-L{level}-{index:03d}",
                    question_id=question_id,
                    level=level,
                    item_index=index,
                    text=f"{question_id} level {level} item {index}",
                )
            )
    return AssessmentQuestion(
        question_id=question_id,
        dimension=dimension,
        section="Section",
        question=f"{question_id}?",
        answer_items=answer_items,
    )


def test_score_distributes_one_point_across_items_in_each_level() -> None:
    """Verify selected answer items receive equal point shares within their level."""
    dimensions = [AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=1)]
    questions = [_question("Q.STR.03.01", "Strategy", {1: 2, 2: 4})]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={
            "Q.STR.03.01": [
                "Q.STR.03.01-L1-001",
                "Q.STR.03.01-L2-001",
                "Q.STR.03.01-L2-002",
            ]
        },
        benchmarks=[],
    )

    assert payload.applicable_question_count == 1
    assert payload.questions[0].raw_score == 1.0
    assert payload.questions[0].max_allowed_level == 2
    assert payload.questions[0].score == 50.0
    assert payload.final_score == 50.0


def test_score_ignores_answers_above_customer_class_limit() -> None:
    """Verify stored answers above L_iMax do not contribute to scoring."""
    dimensions = [AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=1)]
    questions = [_question("Q.STR.03.01", "Strategy", {1: 1, 2: 1, 3: 1})]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={
            "Q.STR.03.01": [
                "Q.STR.03.01-L2-001",
                "Q.STR.03.01-L3-001",
            ]
        },
        benchmarks=[],
    )

    assert payload.questions[0].selected_answer_item_ids == ["Q.STR.03.01-L2-001"]
    assert payload.questions[0].raw_score == 1.0
    assert payload.questions[0].score == 50.0


def test_score_excludes_unavailable_questions_from_weighting() -> None:
    """Verify max-level-zero questions are excluded from topic output/weighting."""
    dimensions = [
        AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=2),
    ]
    questions = [
        _question("Q.STR.01.01", "Strategy", {1: 1}),
        _question("Q.STR.03.01", "Strategy", {1: 1, 2: 1}),
    ]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={
            "Q.STR.01.01": ["Q.STR.01.01-L1-001"],
            "Q.STR.03.01": ["Q.STR.03.01-L1-001"],
        },
        benchmarks=[],
    )

    assert [item.question_id for item in payload.questions] == ["Q.STR.03.01"]
    available = payload.questions[0]
    assert available.weight == 1
    assert payload.final_score == 50.0


def test_score_ignores_deprecated_legacy_benchmark_rows_without_active_import() -> None:
    """Verify legacy synthetic rows cannot populate active peer comparisons."""
    dimensions = [AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=1)]
    questions = [_question("Q.STR.03.01", "Strategy", {1: 1, 2: 1})]
    benchmarks = [
        ScoreBenchmarkRow(customer_class="class_1", sector=None, benchmark_score=38.0, sample_size=10),
        ScoreBenchmarkRow(customer_class="class_1", sector=None, benchmark_score=40.0, sample_size=80),
        ScoreBenchmarkRow(customer_class="class_1", sector=None, dimension="Strategy", benchmark_score=45.0),
        ScoreBenchmarkRow(customer_class="class_1", sector=None, question_id="Q.STR.03.01", benchmark_score=50.0),
        ScoreBenchmarkRow(
            customer_class=SECTOR_BENCHMARK_CLASS,
            sector="Energy",
            benchmark_score=55.0,
        ),
        ScoreBenchmarkRow(
            customer_class=SECTOR_BENCHMARK_CLASS,
            sector="Energy",
            dimension="Strategy",
            benchmark_score=60.0,
        ),
        ScoreBenchmarkRow(
            customer_class=SECTOR_BENCHMARK_CLASS,
            sector="Energy",
            question_id="Q.STR.03.01",
            benchmark_score=65.0,
        ),
    ]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={"Q.STR.03.01": ["Q.STR.03.01-L2-001"]},
        benchmarks=benchmarks,
    )

    assert payload.benchmark_context.reason == "no_active_dataset"
    assert payload.benchmark_score is None
    assert payload.benchmark_delta is None
    assert payload.same_sector_benchmark_score is None
    assert payload.same_sector_benchmark_delta is None
    assert payload.same_sector_sample_size == 0
    assert payload.same_size_benchmark_score is None
    assert payload.same_size_benchmark_delta is None
    assert payload.same_size_sample_size == 0
    assert payload.dimensions[0].benchmark_score is None
    assert payload.questions[0].benchmark_score is None


def test_final_score_uses_unrounded_applicable_question_average() -> None:
    """Verify rounded serialized weights cannot push a perfect score over 100."""
    dimensions = [AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=60)]
    questions = [_question("Q.STR.03.01", "Strategy", {1: 1, 2: 1}) for _ in range(60)]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={"Q.STR.03.01": ["Q.STR.03.01-L1-001", "Q.STR.03.01-L2-001"]},
        benchmarks=[],
    )

    assert payload.applicable_question_count == 60
    assert payload.questions[0].weight == 0.0167
    assert payload.final_score == 100.0


def test_score_deduplicates_selected_answer_ids_before_scoring() -> None:
    """Verify direct scoring calls do not double-count repeated answer IDs."""
    dimensions = [AssessmentDimension(dimension="Strategy", display_name="Strategy", question_count=1)]
    questions = [_question("Q.STR.03.01", "Strategy", {1: 1, 2: 1})]

    payload = calculate_assessment_score(
        assessment_id="assessment-1",
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
        selected_answers={
            "Q.STR.03.01": [
                "Q.STR.03.01-L1-001",
                "Q.STR.03.01-L1-001",
                "Q.STR.03.01-L2-001",
            ]
        },
        benchmarks=[],
    )

    assert payload.questions[0].selected_answer_item_ids == [
        "Q.STR.03.01-L1-001",
        "Q.STR.03.01-L2-001",
    ]
    assert payload.questions[0].raw_score == 2.0
    assert payload.questions[0].score == 100.0
    assert payload.final_score == 100.0
