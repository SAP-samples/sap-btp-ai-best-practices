from __future__ import annotations

from dataclasses import replace

from app.nbo.decision_engine import _build_rate_plan_offer, _plan_questions
from app.nbo.engine import run_decision_batch
from app.nbo.models import Confidence, DecisionStatus, FactSource, OfferDecision
from app.nbo.decision_facts import compute_account_facts
from app.services.recommendations import RecommendationService, _facts_used_for_explanation


def test_recommendation_service_exposes_honest_service_charge_tier_gap() -> None:
    service = RecommendationService()

    result = service.evaluate_account("6001")

    assert result.final_offer is not None
    assert result.final_offer.program_id == "rate_plan_optimization"
    assert result.facts["service_charge_tier"].value is None
    assert (
        result.facts["service_charge_tier"].missing_reason
        == "Dwelling type and service entrance amps are both required to derive the monthly service charge tier."
    )
    assert result.explanation is not None
    assert any(
        missing
        == "Service charge tier: dwelling type and service entrance data are still missing, so the optimizer cannot fully confirm the right fixed-charge tier."
        for missing in result.explanation.missing_core_facts
    )


def test_explanation_filters_pdf_paths_from_offer_evidence() -> None:
    facts = compute_account_facts("104", RecommendationService().ds)
    offer = OfferDecision(
        program_id="income_qualified_discount",
        display_name="Household Assistance Discount",
        status=DecisionStatus.ELIGIBLE,
        confidence=Confidence.HIGH,
        evidence=["customer-assistance-guide.pdf"],
    )

    lines = _facts_used_for_explanation(offer, facts)

    assert all(
        ".pdf" not in line.casefold()
        and "data/programs/" not in line.casefold()
        for line in lines
    )


def test_rate_plan_offer_marks_summer_usage_fallback_when_recent_window_is_incomplete() -> None:
    service = RecommendationService()
    facts = compute_account_facts("6001", service.ds)

    facts["avg_on_peak_kwh_3m"] = replace(
        facts["avg_on_peak_kwh_3m"],
        value=None,
        missing_reason="Forced missing for regression coverage",
    )
    facts["avg_off_peak_kwh_3m"] = replace(
        facts["avg_off_peak_kwh_3m"],
        value=None,
        missing_reason="Forced missing for regression coverage",
    )
    facts["avg_total_usage_3m"] = replace(
        facts["avg_total_usage_3m"],
        value=None,
        missing_reason="Forced missing for regression coverage",
    )

    result = _build_rate_plan_offer(facts)

    assert result is not None
    assert result.metadata["usage_window_rule"] == "summer_usage_fallback"


def test_question_planner_keeps_snapshot_blocker_without_asking_unusable_prepay_count() -> None:
    service = RecommendationService()

    result = service.evaluate_account("103")
    replanned = _plan_questions(result.blocked_offers, service.catalog_index)

    assert replanned[0].expected_fact == "has_current_snapshot"
    assert all(
        question.expected_fact != "prepay_advance_offers_this_month"
        for question in replanned
    )
    assert any(
        question.expected_fact == "customer_wants_followup"
        for question in replanned[1:]
    )


def test_csr_prepay_count_unlocks_prepay_before_iqd_for_disconnected_mpower() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "10106",
        user_answers={
            "prepay_advance_offers_this_month": 1,
            "household_income_qualified": True,
            "customer_of_record_on_site": True,
            "account_name_type": "PERSONAL",
        },
    )

    assert result.final_offer is not None
    assert result.final_offer.program_id == "prepay_advance"
    assert (
        result.facts["prepay_advance_offers_this_month"].source
        == FactSource.CUSTOMER_ANSWER
    )
    assert result.facts["prepay_advance_offers_this_month"].value == 1


def test_csr_prepay_count_two_or_more_allows_iqd_path() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "10106",
        user_answers={
            "prepay_advance_offers_this_month": 2,
            "household_income_qualified": True,
            "customer_of_record_on_site": True,
            "account_name_type": "PERSONAL",
        },
    )

    assert result.final_offer is not None
    assert result.final_offer.program_id == "income_qualified_discount"
    assert result.facts["prepay_advance_offers_this_month"].value == 2
    prepay_offer = next(
        offer for offer in result.blocked_offers if offer.program_id == "prepay_advance"
    )
    assert prepay_offer.status == DecisionStatus.INELIGIBLE


def test_iqd_questions_are_prioritized_after_prepay_is_exhausted() -> None:
    service = RecommendationService()

    result = service.evaluate_account(
        "10106",
        user_answers={
            "prepay_advance_offers_this_month": 2,
            "household_income_qualified": True,
        },
    )

    assert result.final_offer is None
    question_facts = [question.expected_fact for question in result.questions]
    assert question_facts[:2] == [
        "account_name_type",
        "customer_of_record_on_site",
    ]
    assert "income_assistance_auto_qualifier" not in question_facts


def test_snapshot_blocker_does_not_hide_customer_answerable_paths() -> None:
    service = RecommendationService()

    result = service.evaluate_account("103")

    assert result.questions
    assert result.questions[0].expected_fact == "has_current_snapshot"
    assert any(question.source == "customer" for question in result.questions[1:])


def test_question_planner_orders_reduced_catalog_followup_questions_by_program_rank() -> None:
    service = RecommendationService()

    result = service.evaluate_account("100000")
    replanned = _plan_questions(result.blocked_offers, service.catalog_index)
    positions = {
        question.expected_fact: index
        for index, question in enumerate(replanned)
    }

    assert positions["account_name_type"] < positions["customer_wants_followup"]
    assert positions["customer_of_record_on_site"] < positions["customer_wants_followup"]
    assert positions["household_income_qualified"] < positions["customer_wants_followup"]
    assert positions["income_assistance_auto_qualifier"] < positions["customer_wants_followup"]
    assert positions["battery_ownership"] < positions["customer_wants_followup"]
    assert positions["battery_partner_brand_supported"] < positions["customer_wants_followup"]


def test_recommendation_service_explains_missing_core_facts_before_program_specific_followups() -> None:
    """The service should explain missing foundational account data first."""
    service = RecommendationService()

    result = service.evaluate_account("103")

    assert result.explanation is not None
    assert (
        result.explanation.summary
        == "No primary recommendation is ready yet because the account lacks a current billing snapshot."
    )
    assert any(
        missing.startswith("Current billing snapshot:")
        for missing in result.explanation.missing_core_facts
    )
    assert result.explanation.next_step is not None
    assert (
        result.explanation.next_step
        == "Resolve the missing current billing snapshot before asking downstream program-specific questions."
    )


def test_batch_evaluation_accepts_saved_answer_overlays() -> None:
    """Batch evaluation should apply account-specific saved answers."""
    results = run_decision_batch(
        saved_answers_by_account={
            "10106": {
                "prepay_advance_offers_this_month": 2,
                "household_income_qualified": True,
                "customer_of_record_on_site": True,
                "account_name_type": "PERSONAL",
            }
        }
    )

    account_result = next(result for result in results if result.billing_account == "10106")

    assert account_result.final_offer is not None
    assert account_result.final_offer.program_id == "income_qualified_discount"
    assert account_result.facts["prepay_advance_offers_this_month"].value == 2
