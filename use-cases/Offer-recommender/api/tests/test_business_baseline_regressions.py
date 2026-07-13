from __future__ import annotations

from app.services.recommendations import RecommendationService


def test_6001_exposes_rate_plan_metadata_with_tier2_fixed_charge_fallback() -> None:
    service = RecommendationService()

    result = service.evaluate_account("6001")

    assert result.final_offer is not None
    assert result.final_offer.program_id == "rate_plan_optimization"
    assert result.workflow_stage == "primary_offer_with_followup"
    assert result.questions
    assert result.questions[0].expected_fact == "customer_wants_followup"
    assert result.explanation is not None
    assert (
        result.explanation.summary
        == "Plan Savings Review is the current primary recommendation, with estimated monthly savings of $25.54 compared with the current plan."
    )
    assert "Usage basis: latest 3 valid bills on the account." in result.explanation.facts_used
    assert any(
        "Estimated monthly cost comparison: current plan $143.36 versus recommended plan $117.82."
        == fact
        for fact in result.explanation.facts_used
    )
    assert any(
        missing
        == "Service charge tier: dwelling type and service entrance data are still missing, so the optimizer cannot fully confirm the right fixed-charge tier."
        for missing in result.explanation.missing_core_facts
    )
    assert (
        result.explanation.next_step
        == "Keep the primary recommendation visible and ask whether the customer wants to continue with optional follow-up discovery."
    )
    assert result.final_offer.metadata["usage_window_rule"] == "latest_3_bills"
    assert (
        result.final_offer.metadata["fixed_charge_rule_note"]
        == "Uses dwelling type and service entrance amps when available; otherwise falls back to tier2 until profile data is populated."
    )
    assert result.final_offer.metadata["current_estimated_cost"] == 143.36
    assert result.final_offer.metadata["best_alternative_estimated_cost"] == 117.82
    assert result.final_offer.metadata["estimated_monthly_savings"] == 25.54


def test_103_prioritizes_missing_snapshot_before_program_specific_questions() -> None:
    service = RecommendationService()

    result = service.evaluate_account("103")

    assert result.final_offer is None
    assert result.workflow_stage == "needs_core_facts"
    assert result.questions
    assert result.questions[0].expected_fact == "has_current_snapshot"
    assert result.explanation is not None
    assert (
        result.explanation.summary
        == "No primary recommendation is ready yet because the account lacks a current billing snapshot."
    )
    assert any(
        missing.startswith("Current billing snapshot:")
        for missing in result.explanation.missing_core_facts
    )
    assert (
        result.explanation.next_step
        == "Resolve the missing current billing snapshot before asking downstream program-specific questions."
    )
    assert all(
        question.expected_fact != "prepay_advance_offers_this_month"
        for question in result.questions[1:]
    )
    assert any(question.source == "customer" for question in result.questions[1:])
