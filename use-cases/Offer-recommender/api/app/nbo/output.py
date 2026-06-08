"""Output writers for decision results and legacy compatibility objects."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.nbo.config import OUTPUT_DIR
from app.nbo.models import Confidence, DecisionResult, DecisionStatus, FactValue, Recommendation

log = logging.getLogger(__name__)


def _today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _serialize_fact(fact: FactValue) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "value": fact.value,
        "value_type": fact.value_type,
        "source": fact.source.value,
        "confidence": fact.confidence.value,
        "evidence": fact.evidence,
        "missing_reason": fact.missing_reason,
    }


def _serialize_dataclass(obj: Any) -> Any:
    if is_dataclass(obj):
        data = asdict(obj)
        for key, value in list(data.items()):
            if hasattr(value, "value"):
                data[key] = value.value
        return data
    return obj


def _write_decision_excel(results: list[DecisionResult], path: Path) -> Path:
    decisions_rows = []
    offer_rows = []
    question_rows = []

    for result in results:
        final_offer = result.final_offer
        decisions_rows.append(
            {
                "BILLING ACCOUNT#": result.billing_account,
                "CUSTOMER_TYPE": result.customer_type,
                "FINAL_OFFER": final_offer.display_name if final_offer else "",
                "FINAL_STATUS": final_offer.status.value if final_offer else "",
                "FINAL_CONFIDENCE": final_offer.confidence.value if final_offer else "",
                "FINAL_REASON_CODES": "; ".join(final_offer.reason_codes) if final_offer else "",
                "WORKFLOW_STAGE": result.workflow_stage.value if result.workflow_stage else "",
                "ROUTING_STAGE": result.routing_stage or "",
                "SOURCE_DOCUMENTS": "; ".join(result.source_documents),
                "SUPPRESSION_REASONS": "; ".join(result.suppression_reasons),
                "CURRENT_ENROLLMENT_DETECTED": result.current_enrollment_detected,
                "QUESTIONS": "; ".join(q.prompt for q in result.questions),
                "FLAGS": "; ".join(result.flags),
                "DECISION_TRACE": "; ".join(result.decision_trace),
                "EXPLANATION_SUMMARY": result.explanation.summary if result.explanation else "",
                "EXPLANATION_DETAILS": "; ".join(result.explanation.facts_used) if result.explanation else "",
                "EXPLANATION_NEXT_STEP": result.explanation.next_step if result.explanation else "",
            }
        )

        for offer in result.eligible_offers + result.blocked_offers:
            offer_rows.append(
                {
                    "BILLING ACCOUNT#": result.billing_account,
                    "PROGRAM_ID": offer.program_id,
                    "DISPLAY_NAME": offer.display_name,
                    "STATUS": offer.status.value,
                    "RANK": offer.rank,
                    "CONFIDENCE": offer.confidence.value,
                    "REASON_CODES": "; ".join(offer.reason_codes),
                    "BLOCKING_FACTS": "; ".join(offer.blocking_facts),
                    "MISSING_FACTS": "; ".join(offer.missing_facts),
                    "EVIDENCE": "; ".join(offer.evidence),
                    "METADATA": json.dumps(offer.metadata, default=str),
                    "EXPLANATION_SUMMARY": offer.explanation.summary if offer.explanation else "",
                    "EXPLANATION_DETAILS": "; ".join(offer.explanation.details) if offer.explanation else "",
                    "EXPLANATION_FACTS_USED": "; ".join(offer.explanation.facts_used) if offer.explanation else "",
                    "EXPLANATION_RULES_USED": "; ".join(offer.explanation.rules_used) if offer.explanation else "",
                    "EXPLANATION_BLOCKERS": "; ".join(offer.explanation.blockers) if offer.explanation else "",
                    "EXPLANATION_POLISH_STATUS": offer.explanation.polish_status if offer.explanation else "",
                }
            )

        for question in result.questions:
            question_rows.append(
                {
                    "BILLING ACCOUNT#": result.billing_account,
                    "QUESTION_ID": question.question_id,
                    "PROMPT": question.prompt,
                    "EXPECTED_FACT": question.expected_fact,
                    "SOURCE": question.source,
                    "PRIORITY": question.priority,
                    "CANDIDATE_PROGRAMS": "; ".join(question.candidate_programs),
                    "EXPLANATION_SUMMARY": question.explanation.summary if question.explanation else "",
                    "EXPLANATION_DETAILS": "; ".join(question.explanation.details) if question.explanation else "",
                    "EXPLANATION_BLOCKERS": "; ".join(question.explanation.blockers) if question.explanation else "",
                    "EXPLANATION_POLISH_STATUS": question.explanation.polish_status if question.explanation else "",
                }
            )

    summary_rows = [
        {"Metric": "Total accounts", "Value": len(results)},
        {"Metric": "Accounts with final offer", "Value": sum(1 for r in results if r.final_offer is not None)},
        {"Metric": "Accounts without final offer", "Value": sum(1 for r in results if r.final_offer is None)},
        {"Metric": "Questions generated", "Value": sum(len(r.questions) for r in results)},
        {
            "Metric": "Distinct commercial final offers",
            "Value": len({
                r.final_offer.display_name
                for r in results
                if r.customer_type == "COMMERCIAL" and r.final_offer
            }),
        },
    ]

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        pd.DataFrame(decisions_rows).to_excel(writer, sheet_name="Decisions", index=False)
        pd.DataFrame(offer_rows).to_excel(writer, sheet_name="Offer Decisions", index=False)
        pd.DataFrame(question_rows).to_excel(writer, sheet_name="Questions", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Data Quality", index=False)

    log.info("Excel written: %s", path)
    return path


def _write_legacy_excel(results: list[Recommendation], path: Path) -> Path:
    rows = []
    for rec in results:
        rows.append(
            {
                "BILLING ACCOUNT#": rec.billing_account,
                "CUSTOMER_TYPE": rec.customer_type,
                "PRIMARY_OFFER": rec.primary_offer.program_name if rec.primary_offer else "",
                "FLAGS": "; ".join(rec.flags),
                "EVALUATION_TRAIL": "; ".join(rec.evaluation_trail),
            }
        )
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Recommendations", index=False)
    log.info("Excel written: %s", path)
    return path


def write_excel(results, path: Path | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = OUTPUT_DIR / f"nbo_recommendations_{_today_stamp()}.xlsx"
    if not results:
        return _write_decision_excel([], path)
    if isinstance(results[0], DecisionResult):
        return _write_decision_excel(results, path)
    return _write_legacy_excel(results, path)


def _serialize_decision_result(result: DecisionResult) -> dict[str, Any]:
    return {
        "billing_account": result.billing_account,
        "customer_type": result.customer_type,
        "final_offer": _serialize_dataclass(result.final_offer) if result.final_offer else None,
        "eligible_offers": [_serialize_dataclass(offer) for offer in result.eligible_offers],
        "blocked_offers": [_serialize_dataclass(offer) for offer in result.blocked_offers],
        "questions": [_serialize_dataclass(question) for question in result.questions],
        "facts": {fact_id: _serialize_fact(fact) for fact_id, fact in result.facts.items()},
        "decision_trace": result.decision_trace,
        "ai_trace": result.ai_trace,
        "flags": result.flags,
        "workflow_stage": result.workflow_stage.value if result.workflow_stage else None,
        "explanation": _serialize_dataclass(result.explanation) if result.explanation else None,
        "source_documents": result.source_documents,
        "suppression_reasons": result.suppression_reasons,
        "routing_stage": result.routing_stage,
        "current_enrollment_detected": result.current_enrollment_detected,
    }


def _serialize_legacy_recommendation(rec: Recommendation) -> dict[str, Any]:
    return {
        "billing_account": rec.billing_account,
        "customer_type": rec.customer_type,
        "primary_offer": _serialize_dataclass(rec.primary_offer) if rec.primary_offer else None,
        "secondary_offers": [_serialize_dataclass(offer) for offer in rec.secondary_offers],
        "facts_summary": rec.facts_summary,
        "flags": rec.flags,
        "evaluation_trail": rec.evaluation_trail,
    }


def write_json(results, path: Path | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = OUTPUT_DIR / f"nbo_recommendations_{_today_stamp()}.json"

    if results and isinstance(results[0], DecisionResult):
        payload = [_serialize_decision_result(result) for result in results]
    else:
        payload = [_serialize_legacy_recommendation(rec) for rec in results]

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("JSON written: %s", path)
    return path
