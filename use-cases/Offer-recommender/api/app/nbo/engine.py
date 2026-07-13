"""Public engine entry points."""

from __future__ import annotations

from app.nbo.models import Confidence, DecisionResult, Offer, OfferDecision, Recommendation


def _legacy_offer_from_decision(offer: OfferDecision | None, rule_id: str = "DECISION") -> Offer | None:
    if offer is None:
        return None
    return Offer(
        program_name=offer.display_name,
        rule_id=rule_id,
        priority=offer.rank or 999,
        confidence=offer.confidence if isinstance(offer.confidence, Confidence) else Confidence.MANUAL_REVIEW,
        reason="; ".join(offer.reason_codes),
        requires_external=offer.missing_facts,
        metadata=offer.metadata,
    )


def _facts_summary(result: DecisionResult) -> dict:
    summary = {}
    for fact_id, fact in result.facts.items():
        if fact_id.startswith("_"):
            continue
        summary[fact_id] = fact.value
    return summary


def run_batch(skip_pdf_extraction: bool = False) -> list[Recommendation]:
    """Compatibility wrapper returning the legacy recommendation shape."""
    decisions = run_decision_batch(skip_pdf_extraction=skip_pdf_extraction)
    recommendations: list[Recommendation] = []
    for decision in decisions:
        primary = _legacy_offer_from_decision(decision.final_offer)
        secondary = [
            _legacy_offer_from_decision(offer)
            for offer in decision.eligible_offers[1:]
        ]
        recommendations.append(
            Recommendation(
                billing_account=decision.billing_account,
                customer_type=decision.customer_type,
                primary_offer=primary,
                secondary_offers=[offer for offer in secondary if offer is not None],
                facts_summary=_facts_summary(decision),
                flags=list(decision.flags),
                evaluation_trail=list(decision.decision_trace),
            )
        )
    return recommendations


def run_decision_batch(
    skip_pdf_extraction: bool = False,
    saved_answers_by_account: dict[str, dict[str, object]] | None = None,
) -> list[DecisionResult]:
    """Canonical batch entrypoint routed through RecommendationService.

    Inputs:
        skip_pdf_extraction: Legacy compatibility flag that is ignored by the
            current deterministic runtime.
        saved_answers_by_account: Optional account-keyed answer overlays to
            apply during batch evaluation. When omitted, persisted saved
            answers are loaded from the application saved-answer service.

    Output:
        Decision results for every known account.
    """
    del skip_pdf_extraction
    from app.services.recommendations import RecommendationService
    from app.services.saved_answers import normalize_billing_account, saved_answer_service

    service = RecommendationService()
    saved_answers = (
        saved_answers_by_account
        if saved_answers_by_account is not None
        else saved_answer_service.get_all_answer_values()
    )
    return [
        service.evaluate_account(
            account,
            user_answers=saved_answers.get(normalize_billing_account(account), {}),
        )
        for account in service.all_accounts()
    ]
