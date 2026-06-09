from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from app.nbo.fact_registry import (
    get_fact_definition,
    is_customer_answerable_fact,
    normalize_answer_value,
)
from app.nbo.catalog import load_program_catalog, validate_catalogs
from app.nbo.data_loader import DataStore
from app.nbo.decision_engine import (
    _build_rate_plan_offer,
    _default_question_for_missing_snapshot,
    _evaluate_catalog_offer,
    _plan_questions,
    _sort_offer_decisions,
)
from app.nbo.decision_facts import compute_account_facts
from app.nbo.models import (
    Confidence,
    DecisionResult,
    DecisionStatus,
    FactSource,
    FactValue,
    OfferDecision,
    RecommendationExplanation,
    WorkflowStage,
)
from app.services.explanations import apply_decision_explanations


def normalize_account_number(account: str | None) -> str:
    if account is None:
        return ""
    normalized = str(account).strip()
    if normalized.isdigit() and len(normalized) > 1:
        normalized = str(int(normalized))
    return normalized


def find_matching_account(input_account: str, all_accounts: list[str]) -> str | None:
    input_normalized = normalize_account_number(input_account)

    if input_account in all_accounts:
        return input_account

    for account in all_accounts:
        if normalize_account_number(account) == input_normalized:
            return account

    if input_normalized.isdigit():
        for account in all_accounts:
            if normalize_account_number(account) == input_normalized:
                return account

    return None


def _candidate_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]{3,}", text)


def _select_primary_offer_questions(
    planned_questions: list[Any],
    customer_type: str,
    answers: dict[str, Any],
) -> tuple[list[Any], WorkflowStage]:
    if customer_type != "RESIDENTIAL":
        return [], WorkflowStage.PRIMARY_OFFER_FINAL

    wants_followup = answers.get("customer_wants_followup")
    has_followup_answer = "customer_wants_followup" in answers
    customer_questions = [
        question for question in planned_questions if question.source == "customer"
    ]
    if wants_followup is False:
        return [], WorkflowStage.PRIMARY_OFFER_FINAL
    if has_followup_answer and wants_followup is None:
        return [], WorkflowStage.PRIMARY_OFFER_FINAL

    followup_question = next(
        (
            question
            for question in customer_questions
            if question.expected_fact == "customer_wants_followup"
        ),
        None,
    )
    if wants_followup is True:
        remaining_questions = [
            question
            for question in customer_questions
            if question.expected_fact != "customer_wants_followup"
        ]
        if not remaining_questions:
            return [], WorkflowStage.PRIMARY_OFFER_FINAL
        return remaining_questions, WorkflowStage.PRIMARY_OFFER_WITH_FOLLOWUP

    if followup_question is None:
        return [], WorkflowStage.PRIMARY_OFFER_FINAL

    return [followup_question], WorkflowStage.PRIMARY_OFFER_WITH_FOLLOWUP


def _filter_answered_questions(
    planned_questions: list[Any],
    answers: dict[str, Any],
) -> list[Any]:
    answered_fact_ids = set(answers)
    if not answered_fact_ids:
        return planned_questions
    return [
        question
        for question in planned_questions
        if question.expected_fact not in answered_fact_ids
    ]


EXPLANATION_FACT_LABELS = {
    "customer_type": "Customer type",
    "segment_name": "Residential segment",
    "current_status": "Current account status",
    "current_rate_plan": "Current rate plan",
    "avg_on_peak_kwh_3m": "Latest 3-bill average on-peak usage",
    "avg_off_peak_kwh_3m": "Latest 3-bill average off-peak usage",
    "avg_total_usage_3m": "Latest 3-bill average total usage",
    "has_payment_distress_signal": "Payment distress signal",
    "bill_increase_or_high_usage": "Bill increase or high-usage trigger",
    "payments_on_time": "Payments on time",
}

EXPLANATION_CORE_FACT_IDS = (
    "has_current_snapshot",
    "current_rate_plan",
    "service_charge_tier",
    "payments_on_time",
)

CSR_OPERATIONAL_ANSWER_FACT_IDS = frozenset(
    {
        "prepay_advance_offers_this_month",
    }
)


def _format_currency(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, digits: int = 2) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_boolean(value: Any) -> str | None:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return None


def _fact_line(fact_id: str, fact: FactValue) -> str | None:
    if fact_id == "segment_name" and fact.value:
        return f"{EXPLANATION_FACT_LABELS[fact_id]}: {fact.value}."
    if fact_id == "current_status" and fact.value:
        return f"{EXPLANATION_FACT_LABELS[fact_id]}: {fact.value}."
    if fact_id == "customer_type" and fact.value:
        return f"{EXPLANATION_FACT_LABELS[fact_id]}: {fact.value}."
    if fact_id in {
        "avg_on_peak_kwh_3m",
        "avg_off_peak_kwh_3m",
        "avg_total_usage_3m",
    } and fact.value is not None:
        value = _format_number(fact.value)
        if value is None:
            return None
        return f"{EXPLANATION_FACT_LABELS[fact_id]}: {value} kWh."
    if fact_id in {
        "has_payment_distress_signal",
        "bill_increase_or_high_usage",
        "payments_on_time",
    }:
        value = _format_boolean(fact.value)
        if value is None:
            return None
        return f"{EXPLANATION_FACT_LABELS[fact_id]}: {value}."
    return None


def _missing_fact_line(
    fact_id: str,
    facts: dict[str, FactValue],
    blocked_offers: list[OfferDecision],
) -> str | None:
    fact = facts.get(fact_id)
    if fact_id == "has_current_snapshot":
        reason = (
            fact.missing_reason
            if fact is not None and fact.missing_reason
            else "No dated current billing snapshot is available for this account."
        )
        return f"Current billing snapshot: {reason}"

    if fact_id == "current_rate_plan":
        reason = (
            fact.missing_reason
            if fact is not None and fact.missing_reason
            else "Current rate plan is unavailable because the account lacks a current billing snapshot."
        )
        return f"Current rate plan: {reason}"

    if fact_id == "service_charge_tier":
        return (
            "Service charge tier: dwelling type and service entrance data are still "
            "missing, so the optimizer cannot fully confirm the right fixed-charge tier."
        )

    if fact_id == "payments_on_time":
        if fact is not None and fact.source == FactSource.CUSTOMER_ANSWER:
            if fact.missing_reason == "customer_unknown":
                return (
                    "Payments on time: the customer explicitly answered unknown, so "
                    "payment-sensitive follow-up offers remain gated."
                )
            if fact.value is None:
                return (
                    "Payments on time: the customer has not confirmed this yet, so "
                    "payment-sensitive follow-up offers remain gated."
                )

        gated_offer_names = sorted(
            {
                offer.display_name
                for offer in blocked_offers
                if fact_id in offer.missing_facts
            }
        )
        if gated_offer_names:
            return (
                "Payments on time: still unanswered, which keeps follow-up offers such "
                f"as {', '.join(gated_offer_names[:2])} gated."
            )
        return (
            "Payments on time: still unanswered, so payment-sensitive follow-up offers "
            "remain gated."
        )

    return None


def _explanation_fact_ids(
    final_offer: OfferDecision | None,
    facts: dict[str, FactValue],
) -> list[str]:
    if final_offer and final_offer.program_id == "rate_plan_optimization":
        return [
            "current_rate_plan",
            "avg_on_peak_kwh_3m",
            "avg_off_peak_kwh_3m",
            "avg_total_usage_3m",
            "segment_name",
        ]

    ordered = [
        "customer_type",
        "current_status",
        "current_rate_plan",
        "segment_name",
        "has_payment_distress_signal",
        "bill_increase_or_high_usage",
    ]
    return [fact_id for fact_id in ordered if fact_id in facts]


def _facts_used_for_explanation(
    final_offer: OfferDecision | None,
    facts: dict[str, FactValue],
) -> list[str]:
    lines: list[str] = []

    if final_offer and final_offer.program_id == "rate_plan_optimization":
        metadata = final_offer.metadata
        current_cost = _format_currency(metadata.get("current_estimated_cost"))
        best_cost = _format_currency(metadata.get("best_alternative_estimated_cost"))
        usage_window_rule = metadata.get("usage_window_rule")
        if current_cost and best_cost:
            lines.append(
                "Estimated monthly cost comparison: "
                f"current plan {current_cost} versus recommended plan {best_cost}."
            )
        if usage_window_rule == "latest_3_bills":
            lines.append("Usage basis: latest 3 valid bills on the account.")
        elif usage_window_rule:
            lines.append(f"Usage basis: {usage_window_rule}.")

    for fact_id in _explanation_fact_ids(final_offer, facts):
        fact = facts.get(fact_id)
        if fact is None:
            continue
        line = _fact_line(fact_id, fact)
        if line and line not in lines:
            lines.append(line)

    if final_offer and final_offer.evidence:
        evidence_reference = final_offer.evidence[0]
        evidence_line = None
        if (
            isinstance(evidence_reference, str)
            and ".pdf" not in evidence_reference.casefold()
            and "/" not in evidence_reference
            and "\\" not in evidence_reference
        ):
            evidence_line = f"Decision basis: {evidence_reference}."
        if evidence_line and evidence_line not in lines:
            lines.append(evidence_line)

    return lines[:6]


def _missing_core_facts_for_explanation(
    final_offer: OfferDecision | None,
    blocked_offers: list[OfferDecision],
    questions: list[Any],
    facts: dict[str, FactValue],
) -> list[str]:
    blocked_missing_fact_ids = {
        fact_id
        for offer in blocked_offers
        for fact_id in offer.missing_facts
    }
    question_fact_ids = {question.expected_fact for question in questions}

    missing_fact_ids: list[str] = []
    for fact_id in EXPLANATION_CORE_FACT_IDS:
        fact = facts.get(fact_id)
        is_missing = False

        if fact_id == "has_current_snapshot":
            is_missing = fact is not None and fact.value is False
        else:
            is_missing = fact is None or not fact.is_known

        if fact_id == "service_charge_tier" and (
            final_offer is None or final_offer.program_id != "rate_plan_optimization"
        ):
            continue

        if is_missing and (
            fact_id in blocked_missing_fact_ids
            or fact_id in question_fact_ids
            or fact_id in {"has_current_snapshot", "service_charge_tier"}
        ):
            missing_fact_ids.append(fact_id)

    lines = [
        line
        for fact_id in missing_fact_ids
        if (line := _missing_fact_line(fact_id, facts, blocked_offers))
    ]
    return lines[:4]


def _summary_for_explanation(
    final_offer: OfferDecision | None,
    questions: list[Any],
    facts: dict[str, FactValue],
    workflow_stage: WorkflowStage,
) -> str:
    if final_offer:
        if final_offer.program_id == "rate_plan_optimization":
            savings = _format_currency(
                final_offer.metadata.get("estimated_monthly_savings")
            )
            if savings:
                if workflow_stage == WorkflowStage.PRIMARY_OFFER_WITH_FOLLOWUP:
                    return (
                        "Plan Savings Review is the current primary recommendation, "
                        f"with estimated monthly savings of {savings} compared with "
                        "the current plan."
                    )
                return (
                    "Plan Savings Review remains the best current fit, with "
                    f"estimated monthly savings of {savings} compared with the "
                    "current plan."
                )
        if workflow_stage == WorkflowStage.PRIMARY_OFFER_WITH_FOLLOWUP:
            return (
                f"{final_offer.display_name} is the current primary recommendation "
                "based on the known account facts."
            )
        return (
            f"{final_offer.display_name} remains the best current fit based on the "
            "known account facts."
        )

    has_snapshot = facts.get("has_current_snapshot")
    if (
        has_snapshot is not None
        and has_snapshot.value is False
        and (not questions or questions[0].expected_fact == "has_current_snapshot")
    ):
        return (
            "No primary recommendation is ready yet because the account lacks a "
            "current billing snapshot."
        )

    if workflow_stage == WorkflowStage.SYSTEM_BLOCKED:
        return (
            "No primary recommendation is ready yet because the remaining blockers "
            "depend on system or profile data rather than customer answers."
        )

    if questions:
        return (
            "No primary recommendation is ready yet because key qualification facts "
            "are still missing."
        )

    return "No eligible program is ready with the information currently available."


def _next_step_for_explanation(
    final_offer: OfferDecision | None,
    questions: list[Any],
    workflow_stage: WorkflowStage,
) -> str | None:
    next_question = questions[0] if questions else None
    if final_offer and next_question:
        if next_question.expected_fact == "customer_wants_followup":
            return (
                "Keep the primary recommendation visible and ask whether the customer "
                "wants to continue with optional follow-up discovery."
            )
        return (
            f"Keep the primary recommendation visible and ask next: "
            f"{next_question.prompt}"
        )

    if final_offer and workflow_stage == WorkflowStage.PRIMARY_OFFER_FINAL:
        return "No additional questions are required for the current recommendation."

    if workflow_stage == WorkflowStage.SYSTEM_BLOCKED and next_question:
        return f"Resolve this system blocker next: {next_question.prompt}"

    if next_question and next_question.expected_fact == "has_current_snapshot":
        return (
            "Resolve the missing current billing snapshot before asking downstream "
            "program-specific questions."
        )

    if next_question:
        return f"Ask next: {next_question.prompt}"

    return None


def _build_recommendation_explanation(
    final_offer: OfferDecision | None,
    blocked_offers: list[OfferDecision],
    questions: list[Any],
    facts: dict[str, FactValue],
    workflow_stage: WorkflowStage,
) -> RecommendationExplanation:
    return RecommendationExplanation(
        summary=_summary_for_explanation(
            final_offer,
            questions,
            facts,
            workflow_stage,
        ),
        facts_used=_facts_used_for_explanation(final_offer, facts),
        missing_core_facts=_missing_core_facts_for_explanation(
            final_offer,
            blocked_offers,
            questions,
            facts,
        ),
        next_step=_next_step_for_explanation(
            final_offer,
            questions,
            workflow_stage,
        ),
    )


def _can_apply_answer_overlay(fact_id: str, facts: dict[str, FactValue]) -> bool:
    """Return whether a supplied answer may override the computed fact value."""
    if not is_customer_answerable_fact(fact_id):
        return False

    if fact_id not in CSR_OPERATIONAL_ANSWER_FACT_IDS:
        return True

    if fact_id == "prepay_advance_offers_this_month":
        current_status = facts.get("current_status")
        is_mpower_enrolled = facts.get("is_mpower_enrolled")
        if (
            current_status is None
            or str(current_status.value or "").strip().upper() != "DISCONNECTED"
            or is_mpower_enrolled is None
            or is_mpower_enrolled.value is not True
        ):
            return False

    existing = facts.get(fact_id)
    if existing is None:
        return True

    return not (
        existing.is_known
        and existing.source in {FactSource.EXTERNAL, FactSource.WORKBOOK}
    )


class RecommendationService:
    """Canonical deterministic recommendation service shared by all surfaces."""

    def __init__(self) -> None:
        self.ds = DataStore()
        validate_catalogs(self.ds)
        self.catalog_entries = load_program_catalog(self.ds)
        self.catalog_index = {
            entry["program_id"]: entry for entry in self.catalog_entries
        }

    def all_accounts(self) -> list[str]:
        return self.ds.all_accounts()

    def match_account(self, raw_account: str) -> str | None:
        return find_matching_account(raw_account.strip(), self.ds.all_accounts())

    def extract_account_from_text(self, text: str) -> str | None:
        for token in _candidate_tokens(text):
            matched = self.match_account(token)
            if matched:
                return matched
        return None

    def _event_records(self, facts: dict[str, FactValue]) -> list[dict[str, Any]]:
        event_fact = facts.get("_program_event_history")
        if event_fact is None or not event_fact.is_known or not isinstance(event_fact.value, list):
            return []
        return [record for record in event_fact.value if isinstance(record, dict)]

    def _offer_matches_record(self, offer: OfferDecision, record: dict[str, Any]) -> bool:
        program_id = str(record.get("PROGRAM ID", "") or "").strip()
        program_code = str(record.get("PROGRAM CODE", "") or "").strip()
        aliases = set(offer.metadata.get("program_code_aliases", []))
        return offer.program_id == program_id or (program_code and program_code in aliases)

    def _parse_event_date(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _current_enrollment_detected(self, offer: OfferDecision, facts: dict[str, FactValue]) -> bool:
        if offer.program_id == "rate_plan_optimization":
            return False
        current_program_codes = set(facts.get("current_program_codes").value or [])
        aliases = set(offer.metadata.get("program_code_aliases", []))
        return bool(current_program_codes & aliases)

    def _recent_decline_detected(self, offer: OfferDecision, facts: dict[str, FactValue]) -> bool:
        records = self._event_records(facts)
        if not records:
            return False
        suppression_window = timedelta(days=int(facts["_decline_suppression_days"].value))
        cutoff = datetime.now(timezone.utc) - suppression_window
        for record in records:
            event_type = str(record.get("EVENT TYPE", "") or "").strip().casefold()
            if event_type != "declined":
                continue
            if not self._offer_matches_record(offer, record):
                continue
            event_date = self._parse_event_date(record.get("EVENT DATE"))
            if event_date is not None and event_date >= cutoff:
                return True
        return False

    def _suppressed_offer(
        self,
        offer: OfferDecision,
        reason_code: str,
        source_documents: list[str],
    ) -> OfferDecision:
        updated_reason_codes = list(offer.reason_codes)
        if reason_code not in updated_reason_codes:
            updated_reason_codes.append(reason_code)
        metadata = dict(offer.metadata)
        metadata["suppressed"] = True
        return replace(
            offer,
            status=DecisionStatus.INELIGIBLE,
            confidence=Confidence.LOW,
            reason_codes=updated_reason_codes,
            metadata=metadata,
            evidence=list({*offer.evidence, *source_documents}),
        )

    def evaluate_account(
        self,
        account: str,
        user_answers: dict[str, Any] | None = None,
        declined_programs: list[str] | None = None,
    ) -> DecisionResult:
        matched_account = self.match_account(account)
        if matched_account is None:
            raise ValueError(f"Unknown billing account: {account}")

        answers = user_answers or {}
        declined = set(declined_programs or [])
        facts = compute_account_facts(matched_account, self.ds)
        applied_answer_count = 0

        for fact_id, value in answers.items():
            try:
                definition = get_fact_definition(fact_id)
            except KeyError:
                continue
            if not _can_apply_answer_overlay(fact_id, facts):
                continue

            normalized_value = normalize_answer_value(fact_id, value)
            if normalized_value is None:
                confidence = Confidence.LOW
                evidence = ["Customer explicitly answered unknown in API workflow"]
                missing_reason = "customer_unknown"
            else:
                confidence = Confidence.MEDIUM
                evidence = ["Customer provided via API workflow"]
                missing_reason = None
            facts[fact_id] = FactValue(
                fact_id=fact_id,
                value=normalized_value,
                value_type=definition.value_type,
                source=FactSource.CUSTOMER_ANSWER,
                confidence=confidence,
                evidence=evidence,
                missing_reason=missing_reason,
            )
            applied_answer_count += 1

        customer_type = facts["customer_type"].value or "UNKNOWN"
        trace: list[str] = ["Typed facts computed"]
        if applied_answer_count:
            trace.append(f"User answers applied: {applied_answer_count} facts")

        flags: list[str] = []
        if declined:
            flags.append(f"Declined programs suppressed: {', '.join(sorted(declined))}")
        if not self._event_records(facts):
            flags.append("Program event history unavailable; 90-day decline suppression not applied")

        if not facts["has_current_snapshot"].value:
            trace.append("No current snapshot available")

        offers: list[OfferDecision] = []
        if customer_type == "RESIDENTIAL":
            rate_offer = _build_rate_plan_offer(facts, self.ds.rate_plan_display_name)
            if rate_offer:
                offers.append(rate_offer)

        eligible_catalog_entries = [
            entry for entry in self.catalog_entries if entry["customer_type"] == customer_type
        ]
        for entry in eligible_catalog_entries:
            offers.append(_evaluate_catalog_offer(entry, facts))

        for offer in offers:
            if offer is None:
                continue
            if offer.program_id in self.catalog_index:
                offer.metadata["program_code_aliases"] = list(
                    self.catalog_index[offer.program_id].get("program_code_aliases", [])
                )

        hints = facts["_persona_hints"].value if "_persona_hints" in facts else {}
        unsuppressed_eligible_offers = _sort_offer_decisions(
            [offer for offer in offers if offer and offer.status == DecisionStatus.ELIGIBLE],
            self.catalog_index,
            hints,
        )
        blocked_offers = [offer for offer in offers if offer and offer.status != DecisionStatus.ELIGIBLE]
        suppression_reasons: list[str] = []
        current_enrollment_detected = False
        eligible_offers: list[OfferDecision] = []
        for offer in unsuppressed_eligible_offers:
            if offer.program_id in declined:
                blocked_offers.append(
                    self._suppressed_offer(offer, "suppressed_session_decline", [])
                )
                suppression_reasons.append(
                    f"{offer.display_name}: suppressed because the user declined it in this session."
                )
                continue
            if self._current_enrollment_detected(offer, facts):
                blocked_offers.append(
                    self._suppressed_offer(
                        offer,
                        "suppressed_current_enrollment",
                        ["Program contract / current-program-code suppression"],
                    )
                )
                suppression_reasons.append(
                    f"{offer.display_name}: suppressed because the account already has the mapped program code."
                )
                current_enrollment_detected = True
                continue
            if self._recent_decline_detected(offer, facts):
                blocked_offers.append(
                    self._suppressed_offer(
                        offer,
                        "suppressed_recent_decline",
                        ["Program event history 90-day decline suppression"],
                    )
                )
                suppression_reasons.append(
                    f"{offer.display_name}: suppressed because the same program was declined within the last 90 days."
                )
                continue
            eligible_offers.append(offer)

        blocked_offers = _sort_offer_decisions(
            blocked_offers,
            self.catalog_index,
            hints,
        )
        final_offer = eligible_offers[0] if eligible_offers else None
        question_source_offers = [
            offer
            for offer in blocked_offers
            if "suppressed_" not in " ".join(offer.reason_codes)
        ]
        planned_questions = _filter_answered_questions(
            _plan_questions(question_source_offers, self.catalog_index),
            answers,
        )
        customer_questions = [question for question in planned_questions if question.source == "customer"]
        system_questions = [question for question in planned_questions if question.source != "customer"]

        if final_offer:
            trace.append(f"Final offer selected: {final_offer.display_name}")
            questions, workflow_stage = _select_primary_offer_questions(
                planned_questions,
                customer_type,
                answers,
            )
            if questions and questions[0].expected_fact == "customer_wants_followup":
                trace.append("Primary offer kept visible while follow-up discovery remains optional")
            elif questions:
                trace.append("Primary offer kept visible while additional discovery continues")
        else:
            trace.append("No eligible final offer")
            snapshot_question = next(
                (
                    question
                    for question in planned_questions
                    if question.expected_fact == "has_current_snapshot"
                ),
                None,
            )
            if snapshot_question is not None:
                questions = [
                    snapshot_question,
                    *[
                        question
                        for question in planned_questions
                        if question.expected_fact != "has_current_snapshot"
                    ],
                ]
                workflow_stage = WorkflowStage.NEEDS_CORE_FACTS
            elif customer_questions:
                questions = customer_questions + system_questions
                workflow_stage = WorkflowStage.NEEDS_CORE_FACTS
            elif system_questions:
                questions = system_questions
                workflow_stage = WorkflowStage.SYSTEM_BLOCKED
                trace.append("No customer-answerable questions remain; waiting on system or profile data")
            else:
                questions = [_default_question_for_missing_snapshot()]
                workflow_stage = WorkflowStage.NEEDS_CORE_FACTS
            if questions and questions[0].expected_fact == "has_current_snapshot":
                trace.append("Prioritized foundational question: has_current_snapshot")

        if customer_type == "COMMERCIAL":
            taxonomy = facts["business_taxonomy"].value
            trace.append(f"Commercial taxonomy: {taxonomy}")
            if taxonomy == "GENERAL_BUSINESS":
                flags.append("Commercial routing used fallback taxonomy")

        if customer_type == "RESIDENTIAL":
            rate_offer = next(
                (offer for offer in offers if offer and offer.program_id == "rate_plan_optimization"),
                None,
            )
            if rate_offer and rate_offer.status == DecisionStatus.MANUAL_REVIEW:
                flags.append("Residential tariff simulation unsupported for current rate plan")

        if current_enrollment_detected:
            flags.append("Current enrollment suppression applied")

        source_documents = sorted(
            {
                source
                for offer in offers
                for source in offer.metadata.get("source_documents", offer.evidence)
            }
        )
        routing_stage = workflow_stage.value if workflow_stage is not None else None
        final_offer, eligible_offers, blocked_offers, questions = apply_decision_explanations(
            final_offer,
            eligible_offers,
            blocked_offers,
            questions,
            facts,
            workflow_stage,
        )
        explanation = _build_recommendation_explanation(
            final_offer,
            blocked_offers,
            questions,
            facts,
            workflow_stage,
        )

        return DecisionResult(
            billing_account=matched_account,
            customer_type=customer_type,
            final_offer=final_offer,
            eligible_offers=eligible_offers,
            blocked_offers=blocked_offers,
            questions=questions,
            facts=facts,
            decision_trace=trace,
            ai_trace=[],
            flags=flags,
            workflow_stage=workflow_stage,
            explanation=explanation,
            source_documents=source_documents,
            suppression_reasons=suppression_reasons,
            routing_stage=routing_stage,
            current_enrollment_detected=current_enrollment_detected,
        )
