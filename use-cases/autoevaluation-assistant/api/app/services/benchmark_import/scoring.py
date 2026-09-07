"""Pure scoring helpers for normalized benchmark questionnaire responses."""

from __future__ import annotations

from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from app.models.assessment import AssessmentQuestion
from app.services.customer_class_scope import max_allowed_level
from app.services.question_scoring import calculate_question_score_details

from .models import BenchmarkResponse, BenchmarkScore


def calculate_question_score(
    question: AssessmentQuestion,
    selected_answer_ids: list[str],
    maximum_level: int,
) -> float:
    """Calculate one question score using the application's maturity formula.

    Inputs:
        question: Canonical framework question containing answer items by level.
        selected_answer_ids: Source selections mapped to canonical answer IDs.
        maximum_level: Highest level applicable to the source company class.

    Outputs:
        float: Score from 0 to 100, rounded to four decimal places. Each level
        contributes one point divided evenly across its answer items, then the
        selected total is divided by the applicable maximum maturity level.
    """

    return calculate_question_score_details(
        question,
        selected_answer_ids,
        maximum_level,
    ).score


def _score_id(
    import_id: str,
    questionnaire_id: str,
    scope_type: str,
    dimension: str | None,
    question_id: str | None,
) -> str:
    """Return a deterministic score-row identifier within one import version.

    Inputs:
        import_id: Benchmark import version identity.
        questionnaire_id: Source questionnaire identity.
        scope_type: ``overall``, ``dimension``, or ``topic``.
        dimension: Optional canonical dimension scope.
        question_id: Optional canonical question scope.

    Outputs:
        str: UUID5 hex digest suitable for the HANA primary key.
    """

    identity = "|".join(
        [import_id, questionnaire_id, scope_type, dimension or "", question_id or ""]
    )
    return uuid5(NAMESPACE_URL, identity).hex


def calculate_benchmark_scores(
    import_id: str,
    questionnaire_id: str,
    customer_class: str,
    questions: dict[str, AssessmentQuestion],
    responses: list[BenchmarkResponse],
    supplied_dimension_scores: dict[str, float | None],
) -> list[BenchmarkScore]:
    """Calculate question, dimension, and overall benchmark scores.

    Inputs:
        import_id: Benchmark import version identity.
        questionnaire_id: Source questionnaire being calculated.
        customer_class: Normalized class used for per-question maximum levels.
        questions: Canonical framework questions keyed by question ID.
        responses: Canonically mapped source responses for the questionnaire.
        supplied_dimension_scores: Optional source audit scores by dimension.

    Outputs:
        list[BenchmarkScore]: Per-question/topic rows followed by dimension and
        overall rows, all rounded to four decimal places.
    """

    selected_by_question: dict[str, list[str]] = defaultdict(list)
    for response in responses:
        if response.selected:
            selected_by_question[response.question_id].append(
                response.canonical_answer_id
            )

    # Live assessment scoring weights every class-applicable framework question,
    # including unanswered questions at zero, and excludes unavailable questions
    # even if stale/source rows happen to exist for them.
    applicable_question_ids = [
        question_id
        for question_id in questions
        if max_allowed_level(question_id, customer_class) > 0
    ]

    question_values: dict[str, float] = {}
    scores: list[BenchmarkScore] = []
    for question_id in applicable_question_ids:
        question = questions[question_id]
        value = calculate_question_score(
            question,
            selected_by_question[question_id],
            max_allowed_level(question_id, customer_class),
        )
        question_values[question_id] = value
        scores.append(
            BenchmarkScore(
                score_id=_score_id(
                    import_id,
                    questionnaire_id,
                    "topic",
                    question.dimension,
                    question_id,
                ),
                import_id=import_id,
                questionnaire_id=questionnaire_id,
                scope_type="topic",
                dimension=question.dimension,
                topic=question.topic_title or question.section,
                question_id=question_id,
                calculated_score=value,
            )
        )

    question_ids_by_dimension: dict[str, list[str]] = defaultdict(list)
    for question_id in applicable_question_ids:
        question_ids_by_dimension[questions[question_id].dimension].append(question_id)
    for dimension, question_ids in question_ids_by_dimension.items():
        value = round(
            sum(question_values[question_id] for question_id in question_ids)
            / len(question_ids),
            4,
        )
        scores.append(
            BenchmarkScore(
                score_id=_score_id(
                    import_id,
                    questionnaire_id,
                    "dimension",
                    dimension,
                    None,
                ),
                import_id=import_id,
                questionnaire_id=questionnaire_id,
                scope_type="dimension",
                dimension=dimension,
                topic=None,
                question_id=None,
                calculated_score=value,
                supplied_score=supplied_dimension_scores.get(dimension),
            )
        )

    overall = (
        round(sum(question_values.values()) / len(question_values), 4)
        if question_values
        else 0.0
    )
    scores.append(
        BenchmarkScore(
            score_id=_score_id(
                import_id,
                questionnaire_id,
                "overall",
                None,
                None,
            ),
            import_id=import_id,
            questionnaire_id=questionnaire_id,
            scope_type="overall",
            dimension=None,
            topic=None,
            question_id=None,
            calculated_score=overall,
        )
    )
    return scores
