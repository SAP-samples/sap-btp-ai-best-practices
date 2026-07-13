"""Shared label formatting helpers for report rendering and prompts."""

from __future__ import annotations

from typing import Dict


def humanize_identifier(name: str) -> str:
    """Convert machine identifiers into report-friendly title case labels."""
    text = str(name or "").strip()
    if not text:
        return ""
    return " ".join(text.replace("_", " ").split()).title()


# Brief descriptions for known rule names, shown in the Rule Funnel table
# and provided to the LLM so it can explain exclusions meaningfully.
RULE_DESCRIPTIONS: Dict[str, str] = {
    "cohort_target_offer_file_date": "Keeps only invoices whose offer file date matches the target cohort date",
    "exclude_status_not_eligible": "Removes invoices with a status indicating they are not eligible for funding",
    "exclude_negative_purchase_price": "Removes invoices where the purchase price is negative",
    "deduplicate_invoice_or_document": "Removes duplicate invoices based on invoice reference or document number",
    "exclude_missing_company_code": "Removes invoices with a missing or blank company code",
    "exclude_missing_customer": "Removes invoices with a missing or blank customer identifier",
    "exclude_zero_amount": "Removes invoices where the purchase price is zero or negative",
    "exclude_past_due": "Removes invoices whose due date has already passed",
}
