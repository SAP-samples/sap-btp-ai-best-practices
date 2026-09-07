"""Tests for editable assessment framework customer-class scope rules."""

from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.customer_class_scope import (
    available_question_ids,
    customer_class_options,
    filter_current_answer_ids,
    load_customer_class_scope,
    max_allowed_level,
    normalize_customer_class,
    scope_question_to_customer_class,
)


def _question(question_id: str) -> AssessmentQuestion:
    """Create one question with five level-coded answer items.

    Inputs:
        question_id: Framework question identifier to place on each item.

    Outputs:
        AssessmentQuestion: Question fixture with one item per maturity level.
    """
    return AssessmentQuestion(
        question_id=question_id,
        dimension="Strategy",
        section="Objectives",
        question="Are objectives monitored?",
        explanation="Review objective evidence.",
        answer_items=[
            AnswerItem(
                answer_item_id=f"{question_id}-L{level}-001",
                question_id=question_id,
                level=level,
                item_index=1,
                text=f"Level {level} answer.",
            )
            for level in range(1, 6)
        ],
    )


def test_scope_config_is_loaded_from_api_config_file() -> None:
    """Verify the editable JSON config has the expected deployable shape."""
    scope = load_customer_class_scope()

    assert scope["schema_version"] == 1
    assert scope["default_customer_class"] == "class_5"
    assert set(scope["classes"]) == {"class_1", "class_2", "class_3", "class_4", "class_5"}
    assert len(scope["question_max_levels"]) == 50


def test_customer_class_counts_match_pdf_summaries() -> None:
    """Verify available question counts match the five PDF summary pages."""
    assert {
        customer_class: len(available_question_ids(customer_class))
        for customer_class in ["class_1", "class_2", "class_3", "class_4", "class_5"]
    } == {
        "class_1": 14,
        "class_2": 22,
        "class_3": 38,
        "class_4": 48,
        "class_5": 50,
    }


def test_strategy_examples_match_class_one_pdf() -> None:
    """Verify the user-provided class-one Strategy examples."""
    assert max_allowed_level("Q.STR.01.01", "class_1") == 0
    assert max_allowed_level("Q.STR.02.01", "class_1") == 0
    assert max_allowed_level("Q.STR.03.01", "class_1") == 2


def test_scope_question_filters_answers_above_class_limit() -> None:
    """Verify scoped questions only expose answer items through the max level."""
    question = _question("Q.STR.03.01")

    scoped = scope_question_to_customer_class(question, "class_1")
    filtered_answers = filter_current_answer_ids(
        question,
        ["Q.STR.03.01-L1-001", "Q.STR.03.01-L3-001"],
        max_level=2,
    )

    assert [item.level for item in scoped.answer_items] == [1, 2]
    assert filtered_answers == ["Q.STR.03.01-L1-001"]


def test_normalize_customer_class_defaults_to_class_five() -> None:
    """Verify missing or unknown customer class preserves legacy full scope."""
    assert normalize_customer_class(None) == "class_5"
    assert normalize_customer_class("not-a-class") == "class_5"


def test_customer_class_options_are_localized_from_config() -> None:
    """Verify selector labels come from the editable backend JSON."""
    assert customer_class_options("en")[0] == {"value": "class_1", "label": "Micro - Class 1"}
    assert customer_class_options("it")[1] == {"value": "class_2", "label": "Piccola - Classe 2"}
