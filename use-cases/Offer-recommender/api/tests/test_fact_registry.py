from __future__ import annotations

from app.nbo.models import Confidence
from app.nbo.fact_registry import HOME_OWNERSHIP_OPTIONS, normalize_answer_value
from app.services.recommendations import RecommendationService


def test_home_ownership_options_use_canonical_values() -> None:
    assert [option.value for option in HOME_OWNERSHIP_OPTIONS] == [
        "HOMEOWNER",
        "RENTER",
        None,
    ]


def test_normalize_answer_value_maps_labels_to_canonical_values() -> None:
    assert normalize_answer_value("home_ownership_status", " homeowner ") == "HOMEOWNER"
    assert normalize_answer_value("home_ownership_status", "rEnTeR") == "RENTER"


def test_normalize_answer_value_maps_stringified_primitives_to_canonical_values() -> None:
    assert normalize_answer_value("prepay_advance_offers_this_month", "0") == 0
    assert normalize_answer_value("prepay_advance_offers_this_month", "2") == 2
    assert normalize_answer_value("payments_on_time", "true") is True
    assert normalize_answer_value("payments_on_time", "false") is False


def test_evaluate_account_preserves_explicit_unknown_customer_answers() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "6001",
        user_answers={"home_ownership_status": None},
    )

    fact = result.facts["home_ownership_status"]

    assert fact.value is None
    assert fact.source.value == "customer_answer"
    assert fact.confidence == Confidence.LOW
    assert fact.evidence == ["Customer explicitly answered unknown in API workflow"]
    assert fact.missing_reason == "customer_unknown"


def test_evaluate_account_treats_invalid_customer_answer_as_unknown() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "6001",
        user_answers={"payments_on_time": "maybe"},
    )

    fact = result.facts["payments_on_time"]

    assert fact.value is None
    assert fact.source.value == "customer_answer"
    assert fact.confidence == Confidence.LOW
    assert fact.evidence == ["Customer explicitly answered unknown in API workflow"]
    assert fact.missing_reason == "customer_unknown"


def test_normalize_answer_value_leaves_unknown_fact_ids_unchanged() -> None:
    assert normalize_answer_value("unknown_fact_id", " homeowner ") == " homeowner "


def test_evaluate_account_ignores_unknown_fact_ids() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "6001",
        user_answers={"_persona_hints": "override", "unknown_fact_id": "value"},
    )

    assert result.facts["_persona_hints"].value != "override"
    assert "unknown_fact_id" not in result.facts


def test_evaluate_account_ignores_known_but_non_customer_answerable_fact_ids() -> None:
    service = RecommendationService()

    baseline = service.evaluate_account("6001")
    result = service.evaluate_account(
        "6001",
        user_answers={
            "customer_type": "COMMERCIAL",
            "has_current_snapshot": False,
            "current_status": "DISCONNECTED",
            "prepay_advance_offers_this_month": 2,
        },
    )

    for fact_id in (
        "customer_type",
        "has_current_snapshot",
        "current_status",
        "prepay_advance_offers_this_month",
    ):
        assert result.facts[fact_id].value == baseline.facts[fact_id].value
        assert result.facts[fact_id].source == baseline.facts[fact_id].source
