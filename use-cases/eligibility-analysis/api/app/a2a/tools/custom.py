"""Custom LangGraph tools for eligibility rules and customer logs."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from ...config.eligibility_config import EligibilitySettings
from ...models.eligibility import RULE_DESCRIPTIONS, RuleCode
from ...services.eligibility.customer_log import CustomerLogService

_RULE_LOGIC: Dict[RuleCode, str] = {
    RuleCode.R13: "Due date must be strictly after purchase date (invoice not overdue).",
    RuleCode.R1: "Due date minus purchase date must be at least NDDT days.",
    RuleCode.R16: "Due date minus issuance date must be less than TEIH days.",
    RuleCode.R17: "Purchase date minus issuance date must be at least ISSPUR days.",
    RuleCode.R11: "Invoice currency must be in the eligible currencies list.",
    RuleCode.R2: "Doc number + fiscal year + reference number must be unique within a batch.",
}


def to_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, list):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    return value


def _rule_payload(rule_code: RuleCode) -> Dict[str, Any]:
    return {
        "rule_code": rule_code.value,
        "description": RULE_DESCRIPTIONS.get(rule_code, "Unknown rule"),
        "logic": _RULE_LOGIC.get(rule_code, ""),
    }


def _rejection_details(rejection_rules: Optional[List[str]]) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []
    if not rejection_rules:
        return details
    for code in rejection_rules:
        try:
            rule_enum = RuleCode(code)
            description = RULE_DESCRIPTIONS.get(rule_enum, "Unknown rule")
        except ValueError:
            description = "Unknown rule"
        details.append({"rule_code": code, "description": description})
    return details


def _entries_payload(entries: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for entry in entries:
        entry_payload = to_json(entry)
        entry_payload["rejection_details"] = _rejection_details(entry.rejection_rules)
        results.append(entry_payload)
    return results


def _suggestions_for_rule(rule_code: RuleCode) -> List[str]:
    if rule_code == RuleCode.R16:
        return [
            "Reduce the tenor by moving the due date earlier or using an earlier issuance date.",
            "Increase the TEIH threshold (maximum tenor) in eligibility settings if business rules allow.",
        ]
    if rule_code == RuleCode.R1:
        return [
            "Increase the due date so it is at least NDDT days after the purchase date.",
            "Decrease the NDDT threshold in eligibility settings if policy allows.",
            "Adjust the purchase date if the transaction timing permits.",
        ]
    if rule_code == RuleCode.R17:
        return [
            "Issue the invoice earlier so it is at least ISSPUR days before purchase.",
            "Decrease the ISSPUR threshold in eligibility settings if policy allows.",
        ]
    if rule_code == RuleCode.R13:
        return [
            "Ensure the due date is strictly after the purchase date.",
            "Adjust the purchase date if timing permits and policy allows.",
        ]
    if rule_code == RuleCode.R11:
        return [
            "Use an eligible currency for the invoice.",
            "Add the currency to the eligible list if policy allows.",
        ]
    if rule_code == RuleCode.R2:
        return [
            "Remove duplicate document keys (doc number + fiscal year + reference number) within the batch.",
            "Consolidate duplicate rows before submitting the offer file.",
        ]
    return []


def _rule_diagnostics_from_entry(
    rule_code: RuleCode,
    entry: Dict[str, Any],
    settings: Dict[str, Any],
    service: CustomerLogService,
    doc_number: Optional[str],
    fiscal_year: Optional[str],
    reference_number: Optional[str],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"rule_code": rule_code.value}
    if rule_code == RuleCode.R16:
        issuance_date = entry.get("issuance_date")
        due_date = entry.get("due_date")
        if issuance_date and due_date:
            tenor_days = (due_date - issuance_date).days
            diagnostics.update(
                {
                    "issuance_date": issuance_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "tenor_days": tenor_days,
                    "teih_max_days": settings.get("teih"),
                }
            )
        else:
            diagnostics["missing_fields"] = ["issuance_date", "due_date"]
        return diagnostics
    if rule_code == RuleCode.R1:
        purchase_date = entry.get("purchase_date")
        due_date = entry.get("due_date")
        if purchase_date and due_date:
            days_to_due = (due_date - purchase_date).days
            diagnostics.update(
                {
                    "purchase_date": purchase_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "days_to_due": days_to_due,
                    "nddt_min_days": settings.get("nddt"),
                }
            )
        else:
            diagnostics["missing_fields"] = ["purchase_date", "due_date"]
        return diagnostics
    if rule_code == RuleCode.R13:
        purchase_date = entry.get("purchase_date")
        due_date = entry.get("due_date")
        if purchase_date and due_date:
            diagnostics.update(
                {
                    "purchase_date": purchase_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "is_overdue": due_date <= purchase_date,
                }
            )
        else:
            diagnostics["missing_fields"] = ["purchase_date", "due_date"]
        return diagnostics
    if rule_code == RuleCode.R17:
        purchase_date = entry.get("purchase_date")
        issuance_date = entry.get("issuance_date")
        if purchase_date and issuance_date:
            days_since_issuance = (purchase_date - issuance_date).days
            diagnostics.update(
                {
                    "purchase_date": purchase_date.isoformat(),
                    "issuance_date": issuance_date.isoformat(),
                    "days_since_issuance": days_since_issuance,
                    "isspur_min_days": settings.get("isspur"),
                }
            )
        else:
            diagnostics["missing_fields"] = ["purchase_date", "issuance_date"]
        return diagnostics
    if rule_code == RuleCode.R11:
        invoice_currency = entry.get("original_currency")
        diagnostics.update(
            {
                "invoice_currency": invoice_currency,
                "eligible_currencies": settings.get("eligible_currencies"),
            }
        )
        if not invoice_currency:
            diagnostics["missing_fields"] = ["original_currency"]
        return diagnostics
    if rule_code == RuleCode.R2:
        duplicates_in_history = None
        if doc_number and fiscal_year and reference_number:
            duplicates_in_history = service.check_duplicate_in_history(
                doc_number=doc_number,
                fiscal_year=fiscal_year,
                reference_number=reference_number,
            )
        diagnostics.update(
            {
                "doc_number": doc_number,
                "fiscal_year": fiscal_year,
                "reference_number": reference_number,
                "note": "Duplicate checks are per batch; history may be incomplete.",
                "duplicate_found_in_history": duplicates_in_history,
            }
        )
        return diagnostics
    return diagnostics


@tool("list_eligibility_rules")
def list_eligibility_rules() -> Dict[str, Any]:
    """List all eligibility rules and their descriptions."""
    settings = EligibilitySettings().to_dict()
    rules = [_rule_payload(rule) for rule in RuleCode]
    return {"rules": rules, "settings_defaults": settings}


@tool("get_rule_details")
def get_rule_details(rule_code: str) -> Dict[str, Any]:
    """Get detailed information about a single eligibility rule by code (e.g., R1)."""
    try:
        rule_enum = RuleCode(rule_code)
    except ValueError as exc:
        raise ValueError(f"Unknown rule code: {rule_code}") from exc
    settings = EligibilitySettings().to_dict()
    payload = _rule_payload(rule_enum)
    payload["settings_defaults"] = settings
    return payload


@tool("get_eligibility_settings")
def get_eligibility_settings() -> Dict[str, Any]:
    """Get the current default eligibility settings (environment overrides applied)."""
    return EligibilitySettings().to_dict()


@tool("get_invoice_history")
def get_invoice_history(
    invoice_ref: str,
    seller_id: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Fetch recent log entries for an invoice, optionally filtered by seller_id."""
    if not invoice_ref:
        raise ValueError("invoice_ref is required")
    limit = max(1, min(int(limit), 50))
    service = CustomerLogService()
    entries = service.get_invoice_history(invoice_ref=invoice_ref, seller_id=seller_id, limit=limit)
    results = _entries_payload(entries)
    return {"count": len(results), "entries": results}


@tool("explain_invoice_noneligibility")
def explain_invoice_noneligibility(
    invoice_ref: str,
    seller_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Explain non-eligibility details for a specific invoice and suggest fixes."""
    if not invoice_ref:
        raise ValueError("invoice_ref is required")
    service = CustomerLogService()
    entries = service.get_invoice_history(
        invoice_ref=invoice_ref, seller_id=seller_id, limit=1
    )
    if not entries:
        return {
            "invoice_ref": invoice_ref,
            "seller_id": seller_id,
            "message": "No matching invoice found in customer logs.",
        }

    entry = entries[0]
    entry_data = {
        "issuance_date": entry.issuance_date,
        "due_date": entry.due_date,
        "purchase_date": entry.purchase_date,
        "original_currency": entry.original_currency,
    }
    settings = EligibilitySettings().to_dict()
    failed_rules = entry.rejection_rules or []
    rule_details = []
    for code in failed_rules:
        try:
            rule_enum = RuleCode(code)
        except ValueError:
            continue
        rule_details.append(
            {
                "rule_code": rule_enum.value,
                "description": RULE_DESCRIPTIONS.get(rule_enum, "Unknown rule"),
                "diagnostics": _rule_diagnostics_from_entry(
                    rule_enum,
                    entry_data,
                    settings,
                    service,
                    entry.doc_number,
                    entry.fiscal_year,
                    entry.reference_number,
                ),
                "suggestions": _suggestions_for_rule(rule_enum),
            }
        )

    return {
        "invoice_ref": entry.invoice_ref,
        "seller_id": entry.seller_id,
        "processed_date": to_json(entry.processed_date),
        "failed_rules": rule_details,
        "settings_defaults": settings,
        "note": (
            "Rule diagnostics use stored dates and current default settings. "
            "If settings were overridden during analysis, results may differ."
        ),
    }


@tool("get_seller_summary")
def get_seller_summary(seller_id: str) -> Dict[str, Any]:
    """Get eligibility and non-eligibility summary statistics for a seller."""
    if not seller_id:
        raise ValueError("seller_id is required")
    service = CustomerLogService()
    summary = service.get_seller_summary(seller_id)
    return to_json(summary)


@tool("search_invoices")
def search_invoices(
    seller_id: Optional[str] = None,
    debtor_id: Optional[str] = None,
    invoice_ref: Optional[str] = None,
    rule_codes: Optional[List[str]] = None,
    match_all_rules: bool = False,
    is_eligible: Optional[bool] = None,
    processed_from: Optional[str] = None,
    processed_to: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Search invoice logs with flexible filters (seller, debtor, rules, eligibility, dates)."""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    normalized_rules = [code for code in (rule_codes or []) if code]
    service = CustomerLogService()
    entries, total = service.search_invoices(
        seller_id=seller_id,
        debtor_id=debtor_id,
        invoice_ref=invoice_ref,
        rule_codes=normalized_rules or None,
        match_all_rules=bool(match_all_rules),
        is_eligible=is_eligible,
        processed_from=processed_from,
        processed_to=processed_to,
        limit=limit,
        offset=offset,
    )
    results = _entries_payload(entries)
    return {"count": len(results), "total": total, "entries": results}


@tool("get_invoices_with_rule")
def get_invoices_with_rule(
    rule_code: str,
    seller_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    only_noneligible: bool = True,
) -> Dict[str, Any]:
    """List invoices that triggered a specific rejection rule (e.g., R2 for duplicates)."""
    if not rule_code:
        raise ValueError("rule_code is required")
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    service = CustomerLogService()
    entries, total = service.search_invoices(
        seller_id=seller_id,
        rule_codes=[rule_code],
        match_all_rules=False,
        is_eligible=False if only_noneligible else None,
        limit=limit,
        offset=offset,
    )
    results = _entries_payload(entries)
    return {
        "rule_code": rule_code,
        "seller_id": seller_id,
        "count": len(results),
        "total": total,
        "entries": results,
    }


@tool("get_duplicate_invoice_groups")
def get_duplicate_invoice_groups(
    seller_id: Optional[str] = None,
    min_count: int = 2,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Return invoice keys that appear multiple times in the log history."""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    min_count = max(2, int(min_count))
    service = CustomerLogService()
    groups = service.get_duplicate_invoice_groups(
        seller_id=seller_id,
        min_count=min_count,
        limit=limit,
        offset=offset,
    )
    return {"count": len(groups), "groups": to_json(groups)}


@tool("get_top_noneligible_debtors")
def get_top_noneligible_debtors(limit: int = 10, seller_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the debtors with the highest non-eligibility counts."""
    limit = max(1, min(int(limit), 50))
    service = CustomerLogService()
    rows = service.get_top_noneligible_debtors(limit=limit, seller_id=seller_id)
    return {"count": len(rows), "debtors": to_json(rows)}


@tool("get_top_noneligible_sellers")
def get_top_noneligible_sellers(limit: int = 10) -> Dict[str, Any]:
    """Return the sellers with the highest non-eligibility counts."""
    limit = max(1, min(int(limit), 50))
    service = CustomerLogService()
    rows = service.get_top_noneligible_sellers(limit=limit)
    return {"count": len(rows), "sellers": to_json(rows)}


@tool("get_major_noneligibility_cause")
def get_major_noneligibility_cause(seller_id: str) -> Dict[str, Any]:
    """Return the top non-eligibility rule (by count/percentage) for a seller."""
    if not seller_id:
        raise ValueError("seller_id is required")
    service = CustomerLogService()
    summary = service.get_seller_summary(seller_id)
    if not summary.rejection_by_rule:
        return {
            "seller_id": seller_id,
            "message": "No non-eligibility data available for this seller.",
        }
    top = summary.rejection_by_rule[0]
    return {
        "seller_id": seller_id,
        "rule_code": top.rule_code,
        "rule_description": top.rule_description,
        "count": top.count,
        "percentage": top.percentage,
        "total_non_eligible": summary.total_rejected,
    }


CUSTOM_TOOLS = [
    list_eligibility_rules,
    get_rule_details,
    get_eligibility_settings,
    get_invoice_history,
    explain_invoice_noneligibility,
    get_seller_summary,
    get_major_noneligibility_cause,
    search_invoices,
    get_invoices_with_rule,
    get_duplicate_invoice_groups,
    get_top_noneligible_debtors,
    get_top_noneligible_sellers,
]
