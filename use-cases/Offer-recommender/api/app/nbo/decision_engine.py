"""Executable hybrid decision engine with typed facts and nested catalog logic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.nbo.catalog import (
    load_program_catalog,
    load_program_rule_matrix,
    load_tariff_catalog,
    validate_catalogs,
)
from app.nbo.data_loader import DataStore
from app.nbo.decision_facts import compute_account_facts
from app.nbo.fact_registry import get_fact_definition, is_customer_answerable_fact
from app.nbo.models import (
    AnswerOptionData,
    Confidence,
    DecisionQuestion,
    DecisionResult,
    DecisionStatus,
    OfferDecision,
    RecommendationExplanation,
)

FOUNDATIONAL_FACT_IDS = frozenset(
    {
        "has_current_snapshot",
        "current_rate_plan",
        "prepay_advance_offers_this_month",
    }
)
FOUNDATIONAL_FACT_ORDER = {
    "has_current_snapshot": 0,
    "current_rate_plan": 1,
    "prepay_advance_offers_this_month": 2,
}
RATE_PLAN_USAGE_WINDOW_RULE = "latest_3_bills"
RATE_PLAN_USAGE_WINDOW_RULE_SUMMER_FALLBACK = "summer_usage_fallback"
FIXED_CHARGE_RULE_NOTE = (
    "Uses dwelling type and service entrance amps when available; otherwise "
    "falls back to tier2 until profile data is populated."
)


@dataclass
class LogicOutcome:
    result: bool | None
    missing_facts: set[str] = field(default_factory=set)
    failed_facts: set[str] = field(default_factory=set)
    matched_facts: set[str] = field(default_factory=set)


def _evaluate_predicate(predicate: dict, facts: dict) -> bool | None:
    fact = facts.get(predicate["fact_id"])
    if fact is None or not fact.is_known:
        return None
    value = fact.value
    target = predicate.get("value")
    op = predicate["op"]

    if op == "eq":
        return value == target
    if op == "lt":
        return value < target
    if op == "lte":
        return value <= target
    if op == "gt":
        return value > target
    if op == "gte":
        return value >= target
    if op == "in":
        return value in target
    if op == "contains":
        return target in value
    raise ValueError(f"Unsupported predicate operator: {op}")


def _evaluate_logic(node: dict | list | None, facts: dict) -> LogicOutcome:
    if not node:
        return LogicOutcome(result=True)
    if isinstance(node, list):
        return _evaluate_logic({"all_of": node}, facts)

    if "fact_id" in node:
        result = _evaluate_predicate(node, facts)
        fact_id = node["fact_id"]
        if result is True:
            return LogicOutcome(result=True, matched_facts={fact_id})
        if result is False:
            return LogicOutcome(result=False, failed_facts={fact_id})
        return LogicOutcome(result=None, missing_facts={fact_id})

    if "all_of" in node:
        outcomes = [_evaluate_logic(child, facts) for child in node["all_of"]]
        if any(outcome.result is False for outcome in outcomes):
            return LogicOutcome(
                result=False,
                failed_facts={fact for outcome in outcomes for fact in outcome.failed_facts},
                missing_facts={fact for outcome in outcomes for fact in outcome.missing_facts},
                matched_facts={fact for outcome in outcomes for fact in outcome.matched_facts},
            )
        if any(outcome.result is None for outcome in outcomes):
            return LogicOutcome(
                result=None,
                missing_facts={fact for outcome in outcomes for fact in outcome.missing_facts},
                matched_facts={fact for outcome in outcomes for fact in outcome.matched_facts},
            )
        return LogicOutcome(
            result=True,
            matched_facts={fact for outcome in outcomes for fact in outcome.matched_facts},
        )

    if "any_of" in node:
        outcomes = [_evaluate_logic(child, facts) for child in node["any_of"]]
        true_outcomes = [outcome for outcome in outcomes if outcome.result is True]
        if true_outcomes:
            return LogicOutcome(
                result=True,
                matched_facts={fact for outcome in true_outcomes for fact in outcome.matched_facts},
            )
        if any(outcome.result is None for outcome in outcomes):
            return LogicOutcome(
                result=None,
                missing_facts={fact for outcome in outcomes for fact in outcome.missing_facts},
                failed_facts={fact for outcome in outcomes for fact in outcome.failed_facts},
            )
        return LogicOutcome(
            result=False,
            failed_facts={fact for outcome in outcomes for fact in outcome.failed_facts},
        )

    if "none_of" in node:
        outcomes = [_evaluate_logic(child, facts) for child in node["none_of"]]
        true_outcomes = [outcome for outcome in outcomes if outcome.result is True]
        if true_outcomes:
            return LogicOutcome(
                result=False,
                failed_facts={fact for outcome in true_outcomes for fact in outcome.matched_facts},
            )
        if any(outcome.result is None for outcome in outcomes):
            return LogicOutcome(
                result=None,
                missing_facts={fact for outcome in outcomes for fact in outcome.missing_facts},
            )
        return LogicOutcome(result=True)

    raise ValueError(f"Unsupported logic node: {node}")


def _confidence_for_status(status: DecisionStatus, missing_facts: list[str], manual: bool) -> Confidence:
    if status == DecisionStatus.ELIGIBLE and not missing_facts and not manual:
        return Confidence.HIGH
    if status == DecisionStatus.ELIGIBLE:
        return Confidence.MEDIUM
    if status == DecisionStatus.NEEDS_INFO:
        return Confidence.MEDIUM
    if status == DecisionStatus.MANUAL_REVIEW or manual:
        return Confidence.MANUAL_REVIEW
    return Confidence.LOW


def _persona_boost(entry: dict, hints: dict[str, str]) -> int:
    family = entry.get("offer_family", "")
    if not hints:
        return 0
    if family == "payment_assistance" and hints.get("payment_assistance") == "boost":
        return -1
    if family == "credit_earning" and hints.get("renewable_programs") == "boost":
        return -1
    if family == "followup_rebates" and hints.get("tech_programs") == "boost":
        return -1
    return 0


def _sort_offer_decisions(offers: list[OfferDecision], catalog_index: dict[str, dict], hints: dict[str, str]) -> list[OfferDecision]:
    def key(decision: OfferDecision):
        entry = catalog_index.get(decision.program_id, {})
        return (
            decision.rank if decision.rank is not None else 9999,
            entry.get("priority_group", 9999),
            _persona_boost(entry, hints),
            decision.display_name,
        )

    return sorted(offers, key=key)


def _season_for_month(month: int) -> str:
    if month in {7, 8}:
        return "summer_peak"
    if month in {5, 6, 9, 10}:
        return "summer"
    return "winter"


def _fixed_charge_tier(rate_entry: dict, facts: dict) -> str | None:
    fixed_charges = rate_entry.get("fixed_charges", {})
    if not fixed_charges:
        return None

    service_charge_tier = facts.get("service_charge_tier")
    if (
        service_charge_tier is not None
        and service_charge_tier.is_known
        and service_charge_tier.value in fixed_charges
    ):
        return service_charge_tier.value

    if "tier2" in fixed_charges:
        return "tier2"

    return next(iter(fixed_charges), None)


def _get_fixed_charge(rate_entry: dict, facts: dict) -> float:
    tier = _fixed_charge_tier(rate_entry, facts)
    if tier is None:
        return 0.0
    return float(rate_entry.get("fixed_charges", {}).get(tier, 0.0))


def _resolve_usage_fact(facts: dict, primary_fact_id: str, fallback_fact_id: str):
    primary_fact = facts.get(primary_fact_id)
    if primary_fact is not None and primary_fact.is_known and primary_fact.value is not None:
        return primary_fact, False

    fallback_fact = facts.get(fallback_fact_id)
    used_fallback = fallback_fact is not None and fallback_fact.is_known and fallback_fact.value is not None
    return fallback_fact, used_fallback


def _estimate_rate_plan_cost_details(
    rate_entry: dict,
    facts: dict,
    use_summer_rates: bool = True,
) -> tuple[float | None, str]:
    """Estimate a monthly cost and report whether the estimator needed summer usage facts."""
    snapshot_fact = facts.get("has_current_snapshot")
    current_plan = facts.get("current_rate_plan")
    if snapshot_fact is None or not snapshot_fact.value or current_plan is None:
        return None, RATE_PLAN_USAGE_WINDOW_RULE

    avg_on, used_summer_on = _resolve_usage_fact(facts, "avg_on_peak_kwh_3m", "avg_on_peak_summer")
    avg_off, used_summer_off = _resolve_usage_fact(facts, "avg_off_peak_kwh_3m", "avg_off_peak_summer")
    avg_super, used_summer_super = _resolve_usage_fact(
        facts, "avg_super_off_peak_kwh_3m", "avg_super_off_peak_summer"
    )
    avg_total, used_summer_total = _resolve_usage_fact(facts, "avg_total_usage_3m", "avg_total_usage_summer")
    avg_demand = facts.get("avg_on_peak_daily_kw_3m")
    usage_window_rule = (
        RATE_PLAN_USAGE_WINDOW_RULE_SUMMER_FALLBACK
        if used_summer_on or used_summer_off or used_summer_super or used_summer_total
        else RATE_PLAN_USAGE_WINDOW_RULE
    )

    if use_summer_rates:
        season = "summer"
    else:
        snapshot_date = facts.get("snapshot_read_date")
        month = snapshot_date.value.month if snapshot_date and snapshot_date.is_known else None
        if month is None:
            return None, usage_window_rule
        season = _season_for_month(month)

    rates = rate_entry["energy_rates"].get(season, {})
    demand_rates = rate_entry.get("demand_rates", {}).get(season, {})
    if not rates and not demand_rates:
        rates = rate_entry["energy_rates"].get("summer", {})
        demand_rates = rate_entry.get("demand_rates", {}).get("summer", {})

    kind = rate_entry["rate_kind"]
    fixed_charge = _get_fixed_charge(rate_entry, facts)

    if kind == "flat":
        if avg_total is None or not avg_total.is_known or avg_total.value is None:
            return None, usage_window_rule
        if "all_kwh" not in rates:
            return None, usage_window_rule
        energy_cost = avg_total.value * rates["all_kwh"]
        return round(fixed_charge + energy_cost, 2), usage_window_rule

    if kind == "tou":
        if avg_on is None or avg_off is None or not avg_on.is_known or not avg_off.is_known:
            return None, usage_window_rule
        if avg_on.value is None or avg_off.value is None:
            return None, usage_window_rule
        if "on_peak" not in rates or "off_peak" not in rates:
            return None, usage_window_rule
        energy_cost = avg_on.value * rates["on_peak"] + avg_off.value * rates["off_peak"]
        return round(fixed_charge + energy_cost, 2), usage_window_rule

    if kind == "tou_with_super_off_peak":
        if avg_on is None or avg_off is None or avg_super is None:
            return None, usage_window_rule
        if not avg_on.is_known or not avg_off.is_known or not avg_super.is_known:
            return None, usage_window_rule
        if None in {avg_on.value, avg_off.value, avg_super.value}:
            return None, usage_window_rule
        if {"on_peak", "off_peak", "super_off_peak"} - set(rates):
            return None, usage_window_rule
        estimated_off_peak = max(avg_off.value - avg_super.value, 0.0)
        energy_cost = (
            avg_on.value * rates["on_peak"]
            + estimated_off_peak * rates["off_peak"]
            + avg_super.value * rates["super_off_peak"]
        )
        return round(fixed_charge + energy_cost, 2), usage_window_rule

    if kind == "demand":
        if avg_on is None or avg_off is None or avg_super is None or avg_demand is None:
            return None, usage_window_rule
        if not avg_on.is_known or not avg_off.is_known or not avg_super.is_known or not avg_demand.is_known:
            return None, usage_window_rule
        if None in {avg_on.value, avg_off.value, avg_super.value, avg_demand.value}:
            return None, usage_window_rule
        if {"on_peak", "off_peak", "super_off_peak"} - set(rates):
            return None, usage_window_rule
        demand_rate = demand_rates.get("avg_on_peak_daily_kw")
        if demand_rate is None:
            return None, usage_window_rule
        estimated_off_peak = max(avg_off.value - avg_super.value, 0.0)
        energy_cost = (
            avg_on.value * rates["on_peak"]
            + estimated_off_peak * rates["off_peak"]
            + avg_super.value * rates["super_off_peak"]
        )
        demand_cost = avg_demand.value * demand_rate
        return round(fixed_charge + energy_cost + demand_cost, 2), usage_window_rule

    return None, usage_window_rule


def _estimate_rate_plan_cost(rate_entry: dict, facts: dict, use_summer_rates: bool = True) -> float | None:
    cost, _ = _estimate_rate_plan_cost_details(rate_entry, facts, use_summer_rates=use_summer_rates)
    return cost


def _status_for_missing_facts(missing_facts: set[str], manual: bool) -> DecisionStatus:
    if any(is_customer_answerable_fact(fact_id) for fact_id in missing_facts):
        return DecisionStatus.NEEDS_INFO
    if manual:
        return DecisionStatus.MANUAL_REVIEW
    return DecisionStatus.MANUAL_REVIEW


def _build_rate_plan_offer(
    facts: dict,
    rate_plan_display_name: Callable[[str | None], str | None] | None = None,
) -> OfferDecision | None:
    """Build the rate-plan optimization decision from typed account facts.

    Inputs:
        facts: Account facts produced by ``compute_account_facts``.
        rate_plan_display_name: Optional resolver that maps internal rate-plan
            IDs to HANA-backed business offering names for user-facing metadata.

    Output:
        An ``OfferDecision`` for rate-plan optimization, or ``None`` for
        non-residential accounts.
    """
    def _display_plan(rate_plan: str | None) -> str | None:
        """Return a user-facing plan label while preserving ID fallback behavior."""
        if rate_plan is None:
            return None
        if rate_plan_display_name is None:
            return rate_plan
        return rate_plan_display_name(rate_plan) or rate_plan

    if facts["customer_type"].value != "RESIDENTIAL":
        return None
    if not facts["has_current_snapshot"].value:
        return OfferDecision(
            program_id="rate_plan_optimization",
            display_name="Plan Savings Review",
            status=DecisionStatus.NEEDS_INFO,
            rank=20,
            confidence=Confidence.MANUAL_REVIEW,
            reason_codes=["missing_current_snapshot"],
            blocking_facts=["has_current_snapshot"],
            missing_facts=["has_current_snapshot"],
            evidence=["Deterministic tariff catalog"],
            metadata={"source_documents": ["rate-plan-guide.pdf"]},
        )
    current_plan = facts["current_rate_plan"].value
    current_plan_display_name = _display_plan(current_plan)
    if not facts["current_rate_supported_for_optimization"].value:
        return OfferDecision(
            program_id="rate_plan_optimization",
            display_name="Plan Savings Review",
            status=DecisionStatus.MANUAL_REVIEW,
            rank=20,
            confidence=Confidence.MANUAL_REVIEW,
            reason_codes=["unsupported_current_rate_plan"],
            blocking_facts=["current_rate_plan"],
            missing_facts=[],
            evidence=["Deterministic tariff catalog does not cover current rate plan"],
            metadata={
                "source_documents": ["rate-plan-guide.pdf"],
                "current_rate_plan": current_plan,
                "current_rate_plan_display_name": current_plan_display_name,
            },
        )

    tariffs = {
        entry["rate_plan"]: entry
        for entry in load_tariff_catalog()
        if entry.get("simulation_supported")
    }
    current_entry = tariffs.get(current_plan)
    current_cost, current_usage_window_rule = (
        _estimate_rate_plan_cost_details(current_entry, facts)
        if current_entry
        else (None, RATE_PLAN_USAGE_WINDOW_RULE)
    )
    if current_cost is None:
        return OfferDecision(
            program_id="rate_plan_optimization",
            display_name="Plan Savings Review",
            status=DecisionStatus.NEEDS_INFO,
            rank=20,
            confidence=Confidence.MANUAL_REVIEW,
            reason_codes=["missing_usage_profile"],
            blocking_facts=["avg_on_peak_kwh_3m", "avg_off_peak_kwh_3m", "avg_total_usage_3m"],
            missing_facts=["avg_on_peak_kwh_3m", "avg_off_peak_kwh_3m", "avg_total_usage_3m"],
            evidence=["Deterministic tariff catalog"],
            metadata={"source_documents": ["rate-plan-guide.pdf"]},
        )

    best_alt: tuple[str, float, str] | None = None
    missing_facts: set[str] = set()
    for rate_plan, entry in tariffs.items():
        if rate_plan == current_plan or not entry.get("new_enrollment_allowed", True):
            continue

        outcome = _evaluate_logic(entry.get("eligibility_logic"), facts)
        if outcome.result is False:
            continue
        if outcome.result is None:
            missing_facts.update(outcome.missing_facts)
            continue

        alt_cost, alt_usage_window_rule = _estimate_rate_plan_cost_details(entry, facts)
        if alt_cost is None:
            continue
        if alt_cost < current_cost - 0.01:
            if best_alt is None or alt_cost < best_alt[1]:
                best_alt = (rate_plan, alt_cost, alt_usage_window_rule)

    if best_alt is not None:
        savings = round(current_cost - best_alt[1], 2)
        usage_window_rule = (
            RATE_PLAN_USAGE_WINDOW_RULE_SUMMER_FALLBACK
            if current_usage_window_rule == RATE_PLAN_USAGE_WINDOW_RULE_SUMMER_FALLBACK
            or best_alt[2] == RATE_PLAN_USAGE_WINDOW_RULE_SUMMER_FALLBACK
            else RATE_PLAN_USAGE_WINDOW_RULE
        )
        return OfferDecision(
            program_id="rate_plan_optimization",
            display_name="Plan Savings Review",
            status=DecisionStatus.ELIGIBLE,
            rank=20,
            confidence=Confidence.HIGH,
            reason_codes=["lower_estimated_bill"],
            evidence=["Deterministic tariff simulation"],
            metadata={
                "source_documents": ["rate-plan-guide.pdf"],
                "current_rate_plan": current_plan,
                "current_rate_plan_display_name": current_plan_display_name,
                "current_rate_plan_new_enrollment_allowed": bool(
                    current_entry and current_entry.get("new_enrollment_allowed", True)
                ),
                "usage_window_rule": usage_window_rule,
                "fixed_charge_rule_note": FIXED_CHARGE_RULE_NOTE,
                "current_estimated_cost": current_cost,
                "best_alternative_rate_plan": best_alt[0],
                "best_alternative_rate_plan_display_name": _display_plan(best_alt[0]),
                "best_alternative_estimated_cost": best_alt[1],
                "estimated_monthly_savings": savings,
            },
        )

    if missing_facts:
        status = _status_for_missing_facts(missing_facts, manual=False)
        return OfferDecision(
            program_id="rate_plan_optimization",
            display_name="Plan Savings Review",
            status=status,
            rank=20,
            confidence=_confidence_for_status(status, sorted(missing_facts), False),
            reason_codes=["missing_alternative_rate_eligibility_facts"],
            blocking_facts=sorted(missing_facts),
            missing_facts=sorted(missing_facts),
            evidence=["Deterministic tariff simulation"],
            metadata={"source_documents": ["rate-plan-guide.pdf"]},
        )

    return OfferDecision(
        program_id="rate_plan_optimization",
        display_name="Plan Savings Review",
        status=DecisionStatus.INELIGIBLE,
        rank=20,
        confidence=Confidence.LOW,
        reason_codes=["no_cheaper_supported_rate_plan"],
        evidence=["Deterministic tariff simulation"],
        metadata={"source_documents": ["rate-plan-guide.pdf"]},
    )


def _evaluate_catalog_offer(entry: dict, facts: dict) -> OfferDecision:
    exclusion_outcome = (
        _evaluate_logic(entry.get("exclusion_logic"), facts)
        if entry.get("exclusion_logic")
        else LogicOutcome(result=False)
    )
    eligibility_outcome = (
        _evaluate_logic(entry["eligibility_logic"], facts)
        if entry.get("eligibility_logic")
        else LogicOutcome(result=True)
    )
    routing_outcome = (
        _evaluate_logic(entry.get("routing_logic"), facts)
        if entry.get("routing_logic")
        else LogicOutcome(result=True)
    )

    missing_facts: set[str] = set()
    blocking_facts: set[str] = set()
    metadata = {
        "source_documents": list(entry["evidence_references"]),
        "legacy_program_ids": list(entry.get("legacy_program_ids", [])),
        "rule_basis": entry.get("rule_basis", "doc_backed"),
        "documented_eligibility_satisfied": False,
        "routing_eligible": routing_outcome.result is True,
        **entry.get("offer_metadata", {}),
    }

    if exclusion_outcome.result is True:
        blocking_facts.update(exclusion_outcome.failed_facts or exclusion_outcome.matched_facts)
        status = DecisionStatus.INELIGIBLE
        reason_codes = ["matched_exclusion_logic"]
    elif eligibility_outcome.result is False:
        blocking_facts.update(eligibility_outcome.failed_facts)
        status = DecisionStatus.INELIGIBLE
        reason_codes = ["failed_documented_eligibility"]
    elif eligibility_outcome.result is None:
        missing_facts.update(eligibility_outcome.missing_facts)
        if exclusion_outcome.result is None:
            missing_facts.update(exclusion_outcome.missing_facts)
        status = _status_for_missing_facts(missing_facts, manual=False)
        reason_codes = ["missing_documented_eligibility_facts"]
    else:
        metadata["documented_eligibility_satisfied"] = True
        if exclusion_outcome.result is None:
            missing_facts.update(exclusion_outcome.missing_facts)
            status = _status_for_missing_facts(missing_facts, entry["manual_curation_required"])
            reason_codes = ["missing_exclusion_facts"]
        elif routing_outcome.result is False:
            blocking_facts.update(routing_outcome.failed_facts)
            status = DecisionStatus.INELIGIBLE
            reason_codes = ["routing_conditions_not_met"]
        elif routing_outcome.result is None:
            missing_facts.update(routing_outcome.missing_facts)
            status = _status_for_missing_facts(missing_facts, entry["manual_curation_required"])
            reason_codes = ["missing_routing_facts"]
        elif entry["manual_curation_required"]:
            status = DecisionStatus.MANUAL_REVIEW
            reason_codes = ["manual_curation_required"]
        else:
            status = DecisionStatus.ELIGIBLE
            reason_codes = ["documented_eligibility_satisfied"]

    confidence = _confidence_for_status(status, sorted(missing_facts), entry["manual_curation_required"])
    return OfferDecision(
        program_id=entry["program_id"],
        display_name=entry["display_name"],
        status=status,
        rank=entry["program_rank"],
        confidence=confidence,
        reason_codes=reason_codes,
        blocking_facts=sorted(blocking_facts),
        missing_facts=sorted(missing_facts),
        evidence=entry["evidence_references"],
        metadata=metadata,
    )


def _question_index_for_programs(programs: list[dict]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for entry in programs:
        for template in entry["question_templates"]:
            index[(entry["program_id"], template["expected_fact"])] = template
    return index


def _build_answer_options(definition) -> list[AnswerOptionData]:
    if not definition.answer_options:
        return []
    return [
        AnswerOptionData(
            label=opt.label,
            value=opt.value,
            description=opt.description,
        )
        for opt in definition.answer_options
    ]


def _fact_has_question_path(fact_id: str) -> bool:
    definition = get_fact_definition(fact_id)
    return bool(
        definition.question_prompt
        and (definition.answer_options or definition.question_source == "system")
    )


def _plan_questions(offers: list[OfferDecision], catalog_index: dict[str, dict]) -> list[DecisionQuestion]:
    by_fact: dict[str, DecisionQuestion] = {}
    template_index = _question_index_for_programs(list(catalog_index.values()))

    for offer in offers:
        if offer.status not in {DecisionStatus.NEEDS_INFO, DecisionStatus.MANUAL_REVIEW}:
            continue
        if any(
            not _fact_has_question_path(fact_id)
            for fact_id in offer.missing_facts
        ):
            continue
        for fact_id in offer.missing_facts:
            definition = get_fact_definition(fact_id)
            template = template_index.get((offer.program_id, fact_id))
            prompt = template["prompt"] if template else definition.question_prompt
            if not prompt:
                continue
            question_id = template["question_id"] if template else f"fact_{fact_id}"
            priority = template["priority"] if template else (offer.rank or 100)
            answer_options = _build_answer_options(definition)
            existing = by_fact.get(fact_id)
            if existing is None:
                by_fact[fact_id] = DecisionQuestion(
                    question_id=question_id,
                    prompt=prompt,
                    expected_fact=fact_id,
                    candidate_programs=[offer.program_id],
                    priority=priority,
                    source=definition.question_source,
                    answer_options=answer_options,
                )
            else:
                if offer.program_id not in existing.candidate_programs:
                    existing.candidate_programs.append(offer.program_id)
                existing.priority = min(existing.priority, priority)

    return sorted(
        by_fact.values(),
        key=lambda q: (
            0 if q.expected_fact in FOUNDATIONAL_FACT_IDS else 1,
            FOUNDATIONAL_FACT_ORDER.get(q.expected_fact, 99),
            0 if q.source == "customer" else 1,
            q.priority,
            q.question_id,
        ),
    )


def _default_question_for_missing_snapshot() -> DecisionQuestion:
    definition = get_fact_definition("has_current_snapshot")
    return DecisionQuestion(
        question_id="retrieve_current_snapshot",
        prompt=definition.question_prompt or "Retrieve a current billing snapshot for this account.",
        expected_fact="has_current_snapshot",
        candidate_programs=[],
        priority=1,
        source=definition.question_source,
    )


def run_decision_batch(skip_pdf_extraction: bool = False) -> list[DecisionResult]:
    """Execute the decision engine for all accounts."""
    del skip_pdf_extraction

    ds = DataStore()
    validate_catalogs(ds)
    catalog_entries = load_program_catalog(ds)
    catalog_index = {entry["program_id"]: entry for entry in catalog_entries}
    rule_matrix = load_program_rule_matrix()

    results: list[DecisionResult] = []
    for account in ds.all_accounts():
        facts = compute_account_facts(account, ds)
        customer_type = facts["customer_type"].value or "UNKNOWN"
        trace: list[str] = ["Typed facts computed"]
        if not facts["has_current_snapshot"].value:
            trace.append("No current snapshot available")

        offers: list[OfferDecision] = []
        if customer_type == "RESIDENTIAL":
            rate_offer = _build_rate_plan_offer(facts, ds.rate_plan_display_name)
            if rate_offer:
                offers.append(rate_offer)
        eligible_catalog_entries = [
            entry for entry in catalog_entries if entry["customer_type"] == customer_type
        ]
        for entry in eligible_catalog_entries:
            offers.append(_evaluate_catalog_offer(entry, facts))

        hints = facts["_persona_hints"].value if "_persona_hints" in facts else {}
        eligible_offers = _sort_offer_decisions(
            [offer for offer in offers if offer and offer.status == DecisionStatus.ELIGIBLE],
            catalog_index,
            hints,
        )
        blocked_offers = _sort_offer_decisions(
            [offer for offer in offers if offer and offer.status != DecisionStatus.ELIGIBLE],
            catalog_index,
            hints,
        )
        final_offer = eligible_offers[0] if eligible_offers else None
        questions = _plan_questions(blocked_offers, catalog_index)
        if not questions and not final_offer:
            questions = [_default_question_for_missing_snapshot()]

        from app.services.explanations import apply_decision_explanations

        final_offer, eligible_offers, blocked_offers, questions = apply_decision_explanations(
            final_offer,
            eligible_offers,
            blocked_offers,
            questions,
            facts,
        )
        result_explanation = None
        if final_offer and final_offer.explanation:
            result_explanation = RecommendationExplanation(
                summary=final_offer.explanation.summary,
                facts_used=list(final_offer.explanation.facts_used),
                missing_core_facts=list(final_offer.explanation.blockers),
                next_step=questions[0].explanation.summary
                if questions and questions[0].explanation
                else None,
            )
        elif questions and questions[0].explanation:
            result_explanation = RecommendationExplanation(
                summary=questions[0].explanation.summary,
                facts_used=list(questions[0].explanation.facts_used),
                missing_core_facts=list(questions[0].explanation.blockers),
                next_step=questions[0].prompt,
            )

        results.append(
            DecisionResult(
                billing_account=account,
                customer_type=customer_type,
                final_offer=final_offer,
                eligible_offers=eligible_offers,
                blocked_offers=blocked_offers,
                questions=questions,
                facts=facts,
                decision_trace=trace,
                ai_trace=[],
                flags=[],
                explanation=result_explanation,
                source_documents=sorted(
                    {
                        source
                        for offer in offers
                        for source in offer.metadata.get("source_documents", offer.evidence)
                    }
                ),
                routing_stage="batch",
            )
        )

    return results
