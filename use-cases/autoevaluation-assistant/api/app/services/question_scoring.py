"""Shared pure question-level maturity scoring for live and imported answers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.models.assessment import AssessmentQuestion


@dataclass(frozen=True)
class QuestionScoreCalculation:
    """Detailed deterministic result for one question's selected answers.

    Inputs:
        Ordered valid selections, raw maturity points, normalized score, and
        applicability derived by ``calculate_question_score_details``.

    Outputs:
        Immutable result reusable by live scoring and benchmark imports.
    """

    selected_answer_ids: list[str]
    selected_points: float
    score: float
    applicable: bool


def calculate_question_score_details(
    question: AssessmentQuestion,
    selected_answer_ids: list[str],
    maximum_level: int,
) -> QuestionScoreCalculation:
    """Calculate filtered selections, raw points, and normalized question score.

    Inputs:
        question: Canonical question with answer items grouped by maturity level.
        selected_answer_ids: Candidate selected answer IDs in caller order.
        maximum_level: Highest maturity level applicable to the customer class.

    Outputs:
        QuestionScoreCalculation: Deduplicated applicable selections, their raw
        points, and ``100 * selected_points / maximum_level``. Each included
        level contributes one point divided equally among its answer items.
    """

    return calculate_maturity_score(
        [(item.answer_item_id, item.level) for item in question.answer_items],
        selected_answer_ids,
        maximum_level,
    )


def calculate_maturity_score(
    answer_levels: list[tuple[str, int]],
    selected_answer_ids: list[str],
    maximum_level: int,
) -> QuestionScoreCalculation:
    """Calculate maturity points from raw ordered answer IDs and levels.

    Inputs:
        answer_levels: Ordered ``(answer_id, maturity_level)`` pairs defining
            every answer item in the question or source response catalog.
        selected_answer_ids: Candidate selected answer IDs in caller order.
        maximum_level: Highest maturity level applicable to the company class.

    Outputs:
        QuestionScoreCalculation: Deduplicated in-scope selections, raw points,
        normalized percentage score, and applicability. This lower-level pure
        primitive is shared by live scoring, imports, and synthetic generation.
    """

    if maximum_level <= 0:
        return QuestionScoreCalculation(
            selected_answer_ids=[],
            selected_points=0.0,
            score=0.0,
            applicable=False,
        )

    counts_by_level: dict[int, int] = defaultdict(int)
    for _answer_id, level in answer_levels:
        if level <= maximum_level:
            counts_by_level[level] += 1

    points_by_id = {
        answer_id: 1.0 / counts_by_level[level]
        for answer_id, level in answer_levels
        if level <= maximum_level and counts_by_level[level] > 0
    }
    filtered_ids = list(
        dict.fromkeys(
            answer_id
            for answer_id in selected_answer_ids
            if answer_id in points_by_id
        )
    )
    selected_points = sum(points_by_id[answer_id] for answer_id in filtered_ids)
    return QuestionScoreCalculation(
        selected_answer_ids=filtered_ids,
        selected_points=selected_points,
        score=round(100.0 * selected_points / maximum_level, 4),
        applicable=True,
    )
