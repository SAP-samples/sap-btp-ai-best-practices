"""Load and apply editable assessment framework customer-class scope rules."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models.assessment import AssessmentQuestion

CustomerClass = str
DEFAULT_CUSTOMER_CLASS = "class_5"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "customer_class_scope.json"


def _validate_scope(scope: dict[str, Any]) -> None:
    """Validate the editable customer-class scope config.

    Inputs:
        scope: Parsed JSON object from the backend config file.

    Outputs:
        None. Raises ``ValueError`` when required config fields are invalid.
    """
    if scope.get("schema_version") != 1:
        raise ValueError("customer class scope schema_version must be 1")

    classes = scope.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("customer class scope must define classes")

    default_class = scope.get("default_customer_class")
    if default_class not in classes:
        raise ValueError("default_customer_class must exist in classes")

    question_max_levels = scope.get("question_max_levels")
    if not isinstance(question_max_levels, dict):
        raise ValueError("customer class scope must define question_max_levels")

    class_ids = set(classes)
    for question_id, levels in question_max_levels.items():
        if not isinstance(question_id, str) or not isinstance(levels, dict):
            raise ValueError("each question_max_levels row must be a class mapping")
        if set(levels) != class_ids:
            raise ValueError(f"{question_id} must include every customer class")
        if any(not isinstance(level, int) or level < 0 or level > 5 for level in levels.values()):
            raise ValueError(f"{question_id} levels must be integers from 0 to 5")


@lru_cache(maxsize=1)
def load_customer_class_scope() -> dict[str, Any]:
    """Load the full editable customer-class scope config.

    Inputs:
        None. The config path is resolved relative to this service module.

    Outputs:
        dict[str, Any]: Raw JSON-compatible config for UI and scoring reuse.
    """
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        scope = json.load(config_file)
    _validate_scope(scope)
    return scope


def normalize_customer_class(value: str | None) -> CustomerClass:
    """Return a known customer class, defaulting unknown values to full scope.

    Inputs:
        value: Optional customer class identifier from a caller.

    Outputs:
        CustomerClass: Configured class id or the configured default class.
    """
    scope = load_customer_class_scope()
    if value in scope["classes"]:
        return value
    return scope["default_customer_class"]


def require_customer_class(value: str) -> CustomerClass:
    """Return an exact configured class or reject invalid persisted context.

    Inputs:
        value: Candidate class identifier supplied for an assessment profile.

    Outputs:
        CustomerClass: The exact configured class identifier.

    Raises:
        ValueError: If ``value`` is not one of the configured class IDs.
    """

    if value not in load_customer_class_scope()["classes"]:
        raise ValueError(f"Unsupported customer class: {value}")
    return value


def customer_class_options(language: str) -> list[dict[str, str]]:
    """Return localized customer-class selector options ordered by rank.

    Inputs:
        language: Preferred label language, falling back to English.

    Outputs:
        list[dict[str, str]]: ``value``/``label`` pairs for UI selectors.
    """
    classes = load_customer_class_scope()["classes"]
    return [
        {"value": class_id, "label": details["labels"].get(language, details["labels"]["en"])}
        for class_id, details in sorted(classes.items(), key=lambda item: item[1]["rank"])
    ]


def max_allowed_level(question_id: str, customer_class: str | None) -> int:
    """Return the max answer level available to a class for one question.

    Inputs:
        question_id: Framework question identifier.
        customer_class: Optional customer class identifier.

    Outputs:
        int: Maximum answer level from 0 to 5, or 0 for unknown questions.
    """
    scope = load_customer_class_scope()
    levels = scope["question_max_levels"].get(question_id)
    if levels is None:
        return 0
    return levels[normalize_customer_class(customer_class)]


def available_question_ids(customer_class: str | None) -> list[str]:
    """Return question IDs with at least one available answer level.

    Inputs:
        customer_class: Optional customer class identifier.

    Outputs:
        list[str]: Question IDs whose max level is greater than zero.
    """
    normalized_class = normalize_customer_class(customer_class)
    return [
        question_id
        for question_id, levels in load_customer_class_scope()["question_max_levels"].items()
        if levels[normalized_class] > 0
    ]


def scope_question_to_max_level(
    question: AssessmentQuestion,
    max_level: int,
) -> AssessmentQuestion:
    """Copy a question with answer items limited to ``max_level``.

    Inputs:
        question: Assessment question to scope.
        max_level: Maximum answer maturity level to keep.

    Outputs:
        AssessmentQuestion: Pydantic copy containing only allowed answer items.
    """
    return question.model_copy(
        update={
            "answer_items": [
                item for item in question.answer_items if item.level <= max_level
            ]
        }
    )


def scope_question_to_customer_class(
    question: AssessmentQuestion,
    customer_class: str | None,
) -> AssessmentQuestion:
    """Copy a question with answers scoped for one customer class.

    Inputs:
        question: Assessment question to scope.
        customer_class: Optional customer class identifier.

    Outputs:
        AssessmentQuestion: Pydantic copy with unavailable answer levels removed.
    """
    return scope_question_to_max_level(
        question,
        max_allowed_level(question.question_id, customer_class),
    )


def filter_current_answer_ids(
    question: AssessmentQuestion,
    answer_ids: list[str],
    max_level: int,
) -> list[str]:
    """Filter selected answer IDs by max allowed level while preserving order.

    Inputs:
        question: Assessment question containing answer item metadata.
        answer_ids: Current selected answer item IDs in caller order.
        max_level: Maximum answer maturity level to keep.

    Outputs:
        list[str]: Input answer IDs that still belong to allowed answer items.
    """
    allowed_ids = {
        item.answer_item_id
        for item in question.answer_items
        if item.level <= max_level
    }
    return [answer_id for answer_id in answer_ids if answer_id in allowed_ids]
