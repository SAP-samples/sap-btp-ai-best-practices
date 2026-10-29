"""
Canonical payment-advice schema for the extractor (UC-01).

Single source of truth for the ``payment_advice_canonical`` shape (from the
workshop spec). Used to:

1. create/configure the canonical SAP Document AI schema (field definitions);
2. normalize numbers/dates during recombination (``CANONICAL_SCHEMA``);
3. validate a produced canonical payload (``validate_canonical``).

The raw LLM mapping (raw extraction -> this shape) is added on top of this module.

Field-type note: ``payment_currency`` is the ISO currency *code* (e.g. USD, CAD),
so it is modelled as a string, not a numeric/amount type — otherwise number
normalization would corrupt it. Amounts are numbers; dates are dates.
"""

from __future__ import annotations

from typing import Any

from dox_client import FieldDefinition

CANONICAL_SCHEMA_NAME = "payment_advice_canonical"
# Must be one of SAP Document AI's supported documentTypes (from /capabilities):
# invoice, paymentAdvice, purchaseOrder, businessCard, shippingDocument, custom.
CANONICAL_DOCUMENT_TYPE = "paymentAdvice"

# (name, formatting_type, description). formatting_type drives both the SAP field
# definition and the aggregation normalization ("string" | "number" | "date").
HEADER_FIELD_META: tuple[tuple[str, str, str], ...] = (
    ("payer_name", "string", "Organization sending the payment (e.g. a retailer or distributor)."),
    ("payee_name", "string", "Legal entity receiving the payment (the payee); used for company-code context."),
    ("payment_reference", "string", "Payment, remittance, check or document number linking the advice to the bank item."),
    ("payment_date", "date", "Payment, transfer or remittance date."),
    ("payment_currency", "string", "Payment currency ISO code, e.g. USD or CAD."),
    ("payment_amount", "number", "Net payment/remittance total (not the gross invoice total)."),
)

LINE_FIELD_META: tuple[tuple[str, str, str], ...] = (
    ("customer_account_reference", "string", "Raw payer-side site, group, DC, store or customer key."),
    ("invoice_reference", "string", "Payee invoice number or primary AR document reference."),
    ("customer_document_reference", "string", "Alternative customer document, external reference, PO or transaction reference."),
    ("invoice_date", "date", "Invoice or document date."),
    ("line_type", "string", "Explicit type such as invoice, credit memo, deduction or payment; null if not printed."),
    ("gross_amount", "number", "Signed amount before explicit discounts or deductions."),
    ("discount_amount", "number", "Cash, terms or settlement discount."),
    ("deduction_amount", "number", "Other deduction or allowance shown on the same row."),
    ("net_amount", "number", "Signed amount paid, payable or allocated after stated differences."),
    ("deduction_reason", "string", "Raw deduction code and description, e.g. shortage or price difference."),
)

# Minimal fields a canonical payload must carry to be useful downstream.
REQUIRED_HEADER_FIELDS = ("payee_name", "payment_reference", "payment_amount")

HEADER_FIELD_NAMES = tuple(name for name, _, _ in HEADER_FIELD_META)
LINE_FIELD_NAMES = tuple(name for name, _, _ in LINE_FIELD_META)

# Derived output fields: decided by the customer's rules (rule_engine), never
# extracted, so they are not part of the Document AI canonical schema. They are
# part of the canonical result and editable during review like extracted fields.
DERIVED_HEADER_FIELD_META: tuple[tuple[str, str, str], ...] = (
    ("payer_account", "string", "Payee's customer account of the payer, from the customer's rules."),
    ("company_code", "string", "Payee company code receiving the payment, from the customer's rules."),
)
DERIVED_LINE_FIELD_META: tuple[tuple[str, str, str], ...] = (
    ("customer_account", "string", "Payee's customer account the line belongs to, from the customer's rules "
                                   "(e.g. a region or store code mapped to an account)."),
)
DERIVED_HEADER_FIELD_NAMES = tuple(name for name, _, _ in DERIVED_HEADER_FIELD_META)
DERIVED_LINE_FIELD_NAMES = tuple(name for name, _, _ in DERIVED_LINE_FIELD_META)


def _field_definition(name: str, formatting_type: str, description: str) -> FieldDefinition:
    """Build a SAP FieldDefinition for one canonical field.

    Mirrors the proven DocumentAI-Agent approach: set only name, a unique label,
    description, and formattingType. The ``formatting`` object is left at SAP's
    default (an explicit separators/format object triggered HTTP 500 on the field
    endpoint), and labels are unique across header + line items as SAP requires.
    """
    return FieldDefinition(
        name=name,
        description=description,
        label=name.replace("_", " ").title(),
        formattingType=formatting_type,
    )


def build_field_definitions() -> tuple[list[FieldDefinition], list[FieldDefinition]]:
    """Return (header_fields, line_item_fields) as SAP FieldDefinition objects."""
    header = [_field_definition(*meta) for meta in HEADER_FIELD_META]
    line = [_field_definition(*meta) for meta in LINE_FIELD_META]
    return header, line


# Lightweight schema dict for aggregation normalization (name + formattingType only).
CANONICAL_SCHEMA: dict[str, Any] = {
    "headerFields": [{"name": n, "formattingType": t} for n, t, _ in HEADER_FIELD_META],
    "lineItemFields": [{"name": n, "formattingType": t} for n, t, _ in LINE_FIELD_META],
}


def validate_canonical(header: dict[str, Any], line_items: list[dict[str, Any]]) -> list[str]:
    """
    Validate a canonical payload; return a list of human-readable issues.

    Checks required header fields are present and non-empty, and that line items
    only use canonical field names. An empty list means the payload is valid.
    """
    issues: list[str] = []
    for required in REQUIRED_HEADER_FIELDS:
        value = header.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(f"missing required header field: {required}")

    allowed = set(LINE_FIELD_NAMES)
    for index, line in enumerate(line_items):
        unknown = set(line) - allowed
        if unknown:
            issues.append(f"line {index}: unknown fields {sorted(unknown)}")
    return issues
