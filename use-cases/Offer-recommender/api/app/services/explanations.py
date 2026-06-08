"""Build deterministic explanations for recommendation decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.nbo.models import (
    DecisionExplanation,
    DecisionQuestion,
    DecisionStatus,
    FactValue,
    OfferDecision,
    WorkflowStage,
)


class DeterministicExplanationPolisher:
    """Preserve deterministic explanation text without calling external models."""

    def polish(self, explanation: DecisionExplanation) -> DecisionExplanation:
        """Return a deterministic explanation with fallback status.

        Args:
            explanation: Audit explanation built from facts, rules, blockers,
                metadata, and source documents.

        Returns:
            The same explanation content with ``polish_status`` set to
            ``"fallback"`` for API compatibility.
        """
        return replace(explanation, polish_status="fallback")

    def polish_many(
        self,
        explanations: list[DecisionExplanation],
    ) -> list[DecisionExplanation]:
        """Return deterministic explanations for a batch without model calls.

        Args:
            explanations: Audit explanations built for one account evaluation.

        Returns:
            A list of explanations whose content is unchanged and whose
            ``polish_status`` is set to ``"fallback"``.
        """
        return [self.polish(explanation) for explanation in explanations]


def _humanize(value: str) -> str:
    """Convert snake_case identifiers into compact display labels."""
    return value.replace("_", " ").strip()


def _format_value(value: Any) -> str:
    """Format a fact value for concise explanation text."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return "Unknown"
    return str(value)


def _format_currency(value: Any) -> str | None:
    """Format numeric currency values used in rate-plan explanations."""
    if value is None:
        return None
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return None


def _source_documents_for_offer(offer: OfferDecision) -> list[str]:
    """Return source documents carried by offer metadata or evidence fields."""
    sources = offer.metadata.get("source_documents", offer.evidence)
    return [str(source) for source in sources if source]


def _fact_lines(fact_ids: list[str], facts: dict[str, FactValue]) -> list[str]:
    """Build readable fact lines for known facts referenced by a decision."""
    lines: list[str] = []
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is None:
            lines.append(f"{_humanize(fact_id)}: unavailable.")
        elif fact.is_known:
            lines.append(f"{_humanize(fact_id)}: {_format_value(fact.value)}.")
        elif fact.missing_reason:
            lines.append(f"{_humanize(fact_id)}: {fact.missing_reason}")
        else:
            lines.append(f"{_humanize(fact_id)}: unavailable.")
    return lines


def _metadata_fact_lines(offer: OfferDecision) -> list[str]:
    """Build explanation facts from offer metadata, especially rate assignments."""
    metadata = offer.metadata
    lines: list[str] = []
    current_plan = metadata.get("current_rate_plan_display_name") or metadata.get(
        "current_rate_plan"
    )
    alternative_plan = metadata.get("best_alternative_rate_plan_display_name") or metadata.get(
        "best_alternative_rate_plan"
    )
    current_cost = _format_currency(metadata.get("current_estimated_cost"))
    alternative_cost = _format_currency(metadata.get("best_alternative_estimated_cost"))
    savings = _format_currency(metadata.get("estimated_monthly_savings"))

    if current_plan:
        lines.append(f"Current rate plan: {current_plan}.")
    if alternative_plan:
        lines.append(f"Recommended rate plan: {alternative_plan}.")
    if current_cost and alternative_cost:
        lines.append(
            f"Estimated monthly cost comparison: current plan {current_cost} versus recommended plan {alternative_cost}."
        )
    if savings:
        lines.append(f"Estimated monthly savings: {savings}.")
    usage_window_rule = metadata.get("usage_window_rule")
    if usage_window_rule:
        lines.append(f"Usage basis: {_humanize(str(usage_window_rule))}.")
    fixed_charge_note = metadata.get("fixed_charge_rule_note")
    if fixed_charge_note:
        lines.append(str(fixed_charge_note))
    return lines


def _summary_for_offer(
    offer: OfferDecision,
    final_offer: OfferDecision | None,
) -> str:
    """Create deterministic one-sentence text for an offer decision."""
    is_primary = final_offer is not None and offer.program_id == final_offer.program_id
    reason_text = ", ".join(_humanize(code) for code in offer.reason_codes) or "known account facts"

    if offer.status == DecisionStatus.ELIGIBLE and offer.program_id == "rate_plan_optimization":
        savings = _format_currency(offer.metadata.get("estimated_monthly_savings"))
        alternative = (
            offer.metadata.get("best_alternative_rate_plan_display_name")
            or offer.metadata.get("best_alternative_rate_plan")
        )
        if savings and alternative:
            prefix = "Selected" if is_primary else "Eligible"
            return (
                f"{prefix} {offer.display_name} because {alternative} is estimated to lower the monthly bill by {savings}."
            )

    if offer.status == DecisionStatus.ELIGIBLE:
        if is_primary:
            return (
                f"Selected {offer.display_name} because it is the highest-ranked eligible offer for this account."
            )
        return (
            f"{offer.display_name} is eligible based on the documented rules and known account facts."
        )

    if any(code == "suppressed_session_decline" for code in offer.reason_codes):
        return (
            f"{offer.display_name} was suppressed because the user declined it in this session."
        )
    if any(code == "suppressed_current_enrollment" for code in offer.reason_codes):
        return (
            f"{offer.display_name} was suppressed because the account already appears enrolled in the mapped program."
        )
    if any(code == "suppressed_recent_decline" for code in offer.reason_codes):
        return (
            f"{offer.display_name} was suppressed because the same program was declined within the configured suppression window."
        )
    if offer.status == DecisionStatus.NEEDS_INFO:
        blockers = ", ".join(_humanize(fact) for fact in offer.missing_facts)
        return (
            f"{offer.display_name} is not ready because required facts are missing: {blockers}."
        )
    if offer.status == DecisionStatus.MANUAL_REVIEW:
        return f"{offer.display_name} needs manual review because {reason_text}."
    return f"{offer.display_name} is not eligible because {reason_text}."


def _details_for_offer(
    offer: OfferDecision,
    final_offer: OfferDecision | None,
) -> list[str]:
    """Create deterministic detail bullets for an offer decision."""
    details: list[str] = []
    if final_offer and offer.program_id == final_offer.program_id:
        details.append("This offer is currently first after eligibility, suppression, and rank ordering.")
    elif offer.status == DecisionStatus.ELIGIBLE:
        details.append("This offer remains available after the primary recommendation.")

    if offer.reason_codes:
        details.append(
            "Reason: "
            + ", ".join(_humanize(code) for code in offer.reason_codes)
            + "."
        )
    if offer.blocking_facts:
        details.append(
            "Blocking information: "
            + ", ".join(_humanize(fact) for fact in offer.blocking_facts)
            + "."
        )
    if offer.missing_facts:
        details.append(
            "Missing information: "
            + ", ".join(_humanize(fact) for fact in offer.missing_facts)
            + "."
        )
    if offer.metadata.get("documented_eligibility_satisfied") is True:
        details.append("Documented eligibility logic is satisfied.")
    if offer.metadata.get("routing_eligible") is False:
        details.append("Routing conditions are not satisfied.")
    if offer.metadata.get("suppressed") is True:
        details.append("The offer is hidden from eligible options for this evaluation.")
    return details


def _build_offer_explanation(
    offer: OfferDecision,
    final_offer: OfferDecision | None,
    facts: dict[str, FactValue],
) -> DecisionExplanation:
    """Build deterministic audit text and evidence for one offer."""
    blocker_ids = list(dict.fromkeys([*offer.blocking_facts, *offer.missing_facts]))
    metadata_facts = _metadata_fact_lines(offer)
    fact_lines = [*metadata_facts, *_fact_lines(blocker_ids, facts)]
    rules = list(dict.fromkeys([*offer.reason_codes, str(offer.metadata.get("rule_basis", "")).strip()]))
    rules = [rule for rule in rules if rule]
    return DecisionExplanation(
        summary=_summary_for_offer(offer, final_offer),
        details=_details_for_offer(offer, final_offer),
        facts_used=fact_lines,
        rules_used=rules,
        blockers=blocker_ids,
        source_documents=_source_documents_for_offer(offer),
    )


def explain_offer(
    offer: OfferDecision,
    final_offer: OfferDecision | None,
    facts: dict[str, FactValue],
    polisher: DeterministicExplanationPolisher,
) -> OfferDecision:
    """Attach a deterministic explanation to one offer."""
    explanation = _build_offer_explanation(offer, final_offer, facts)
    return replace(offer, explanation=polisher.polish(explanation))


def _program_names(program_ids: list[str], offers_by_program: dict[str, OfferDecision]) -> list[str]:
    """Resolve candidate program identifiers to display names when available."""
    names: list[str] = []
    for program_id in program_ids:
        offer = offers_by_program.get(program_id)
        names.append(offer.display_name if offer else program_id)
    return names


def _build_question_explanation(
    question: DecisionQuestion,
    offers_by_program: dict[str, OfferDecision],
    facts: dict[str, FactValue],
) -> DecisionExplanation:
    """Build deterministic audit text and evidence for one planned question."""
    fact = facts.get(question.expected_fact)
    candidate_names = _program_names(question.candidate_programs, offers_by_program)
    candidate_text = ", ".join(candidate_names[:3])
    if candidate_text:
        summary = (
            f"Ask this to resolve {_humanize(question.expected_fact)} for {candidate_text}."
        )
    elif question.source == "system":
        summary = (
            f"Resolve this system step because {_humanize(question.expected_fact)} is required before the recommendation can continue."
        )
    else:
        summary = (
            f"Ask this to resolve {_humanize(question.expected_fact)} before making the next recommendation decision."
        )

    details = [f"Prompt: {question.prompt}", f"Question source: {question.source}."]
    if question.answer_options:
        details.append(
            "Allowed answers: "
            + ", ".join(option.label for option in question.answer_options)
            + "."
        )
    if candidate_names:
        details.append("Candidate programs: " + ", ".join(candidate_names) + ".")
    fact_lines = _fact_lines([question.expected_fact], facts)
    if fact is not None and fact.missing_reason:
        details.append(f"Current blocker: {fact.missing_reason}")

    return DecisionExplanation(
        summary=summary,
        details=details,
        facts_used=fact_lines,
        rules_used=[question.question_id],
        blockers=[question.expected_fact],
        source_documents=[],
    )


def explain_question(
    question: DecisionQuestion,
    offers_by_program: dict[str, OfferDecision],
    facts: dict[str, FactValue],
    polisher: DeterministicExplanationPolisher,
) -> DecisionQuestion:
    """Attach a deterministic explanation to one question."""
    explanation = _build_question_explanation(question, offers_by_program, facts)
    return replace(question, explanation=polisher.polish(explanation))


def apply_decision_explanations(
    final_offer: OfferDecision | None,
    eligible_offers: list[OfferDecision],
    blocked_offers: list[OfferDecision],
    questions: list[DecisionQuestion],
    facts: dict[str, FactValue],
    workflow_stage: WorkflowStage | None = None,
    polisher: DeterministicExplanationPolisher | None = None,
) -> tuple[OfferDecision | None, list[OfferDecision], list[OfferDecision], list[DecisionQuestion]]:
    """Attach explanations to all offer and question decisions for one account.

    Args:
        final_offer: Selected primary offer, if any.
        eligible_offers: Offers that survived eligibility and suppression checks.
        blocked_offers: Offers that were ineligible, suppressed, or need more data.
        questions: Follow-up or system questions planned for the account.
        facts: Typed account facts used by the deterministic engine.
        workflow_stage: Current workflow stage; reserved for future text tuning.
        polisher: Optional custom deterministic status applicator, mainly for tests.

    Returns:
        The same decision collections with nested explanations attached.
    """
    del workflow_stage
    active_polisher = polisher or DeterministicExplanationPolisher()
    eligible_explanations = [
        _build_offer_explanation(offer, final_offer, facts)
        for offer in eligible_offers
    ]
    blocked_explanations = [
        _build_offer_explanation(offer, final_offer, facts)
        for offer in blocked_offers
    ]
    offer_explanations_with_status = active_polisher.polish_many(
        [*eligible_explanations, *blocked_explanations]
    )
    explained_eligible = [
        replace(offer, explanation=offer_explanations_with_status[index])
        for index, offer in enumerate(eligible_offers)
    ]
    blocked_offset = len(eligible_offers)
    explained_blocked = [
        replace(offer, explanation=offer_explanations_with_status[blocked_offset + index])
        for index, offer in enumerate(blocked_offers)
    ]
    explained_by_program = {
        offer.program_id: offer
        for offer in [*explained_eligible, *explained_blocked]
    }
    explained_final = None
    if final_offer is not None:
        explained_final = next(
            (
                offer
                for offer in explained_eligible
                if offer.program_id == final_offer.program_id
            ),
            explain_offer(final_offer, final_offer, facts, active_polisher),
        )
    question_explanations = [
        _build_question_explanation(question, explained_by_program, facts)
        for question in questions
    ]
    question_explanations_with_status = active_polisher.polish_many(question_explanations)
    explained_questions = [
        replace(question, explanation=question_explanations_with_status[index])
        for index, question in enumerate(questions)
    ]
    return explained_final, explained_eligible, explained_blocked, explained_questions
