"""Build and create the S/4 Payment Advice (`API_PAYMENT_ADVICE_SRV`) for a checked advice.

`build_payload` is the single place that maps our advice onto S/4 fields, so
feedback from the S/4 functional review is a one-function change:

Header (`A_PaymentAdvice`): company code from the matched receivables, the
payer from the customer's rules (else the lead matched customer), account type `D` (customer), advice type (default `04`; S/4 then
assigns the advice number), currency, payment date, paid amount and the
customer's payment reference as header text.

Items (`to_PaymentAdviceItem`, deep insert):
- matched line: `DocumentReferenceID` (XBLNR) plus `AccountingDocument` and
  `FiscalYear` (the document number takes priority in S/4 matching), gross,
  cash discount, deduction and net amounts, and the reason code when S/4 has to
  book a residual. A customer other than the header account is sent as the
  alternative account (lead payer), with its company code (`AlternativeCompanyCode`;
  S/4 otherwise looks the customer up in company code "").
- deduction line: no XBLNR (it is not a payee document); the customer's
  deduction reference goes to `AssignmentReference` (18) and `DocumentItemText`
  (50) with the reason code in `PaymentDifferenceReason`; its rule-derived
  customer account becomes the alternative account when it differs from the header.

Amounts keep the advice signs: invoices positive, deductions and credits negative.
Reason codes go through the demo-only `reason_aliases` (`S4_REASON_CODE_ALIASES`).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from .client import S4Client, odata_entity, odata_str
from .validation import dec

SERVICE = "API_PAYMENT_ADVICE_SRV"
ACCOUNT_TYPE_CUSTOMER = "D"
KEY_FIELDS = ("CompanyCode", "PaymentAdviceAccountType", "PaymentAdviceAccount", "PaymentAdvice")


def odata_date(value: Any) -> str | None:
    """Convert `YYYY-MM-DD` to the OData V2 `/Date(<ms>)/` literal at UTC midnight (None when empty/invalid)."""
    try:
        day = date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
    return f"/Date({int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)})/"


def _item(line: dict[str, Any], derived: dict[str, Any], reason_aliases: dict[str, str]) -> dict[str, Any]:
    """Map one checked advice line to a payment advice item (reason code through ``reason_aliases``)."""
    reason = reason_aliases.get(line["reason_code"], line["reason_code"]) if line["reason_code"] else line["reason_code"]
    alternative = {"PaymentAdviceAltvAccountType": ACCOUNT_TYPE_CUSTOMER, "AlternativeCompanyCode": derived["company_code"]}
    item: dict[str, Any] = {"Currency": derived["currency"],
                            "GrossAmountInPaymentCurrency": line["gross"], "NetPaymentAmountInPaytCurrency": line["net"]}
    if line["kind"] == "matched":
        s4 = line["s4_item"]
        item.update(DocumentReferenceID=s4["document_reference"] or line["reference"][:16],
                    AccountingDocument=s4["accounting_document"], FiscalYear=s4["fiscal_year"],
                    CashDiscountAmountInPaytCrcy=line["discount"])
        if line["deduction"] != "0.00":
            item["DeductionAmountInPaytCurrency"] = line["deduction"]
        if line["needs_reason"] and reason:
            item["PaymentDifferenceReason"] = reason
        if s4["customer"] != derived["customer"]:
            item.update(PaymentAdviceAltvAccount=s4["customer"], **alternative)
    else:
        item.update(AssignmentReference=line["reference"][:18],
                    DocumentItemText=f"{line['reference']} {line['nature'].replace('_', ' ')}"[:50],
                    PaymentDifferenceReason=reason)
        # The deduction's customer comes only from the customer's rules (e.g. region code).
        if line.get("account") and line["account"] != derived["customer"]:
            item.update(PaymentAdviceAltvAccount=line["account"], **alternative)
    return item


def build_payload(result: dict[str, Any], check: dict[str, Any], advice_type: str = "04",
                  reason_aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the deep-insert JSON for `POST A_PaymentAdvice`.

    Args:
        result: Advice result (`header` + `line_items`).
        check: `validation.check_advice` output; must be `ready`.
        advice_type: S/4 payment advice type (number range decides the number).
        reason_aliases: Demo-only {documented reason code: S/4 reason code}; None in production.

    Returns:
        Payload dict ready to POST.

    Raises:
        ValueError: When the check still has blocking issues.
    """
    if not check["ready"]:
        raise ValueError("The advice has blocking S/4 issues; resolve them before building the payload.")
    header, derived = result.get("header") or {}, check["derived"]
    payload: dict[str, Any] = {
        "CompanyCode": derived["company_code"],
        "PaymentAdviceAccountType": ACCOUNT_TYPE_CUSTOMER,
        "PaymentAdviceAccount": derived["customer"],
        "PaymentAdviceType": advice_type,
        "PaymentCurrency": derived["currency"],
        "PaidAmountInPaytCurrency": str(dec(header["payment_amount"])),
        "PaymentAdviceHeaderText": str(header.get("payment_reference") or "")[:25],
    }
    if odata_date(header.get("payment_date")):
        payload["PaymentDate"] = odata_date(header.get("payment_date"))
    if header.get("payer_name"):
        payload["BusinessPartnerName"] = str(header["payer_name"])[:55]
    # Official SAP samples send the navigation property as a plain array.
    payload["to_PaymentAdviceItem"] = [_item(line, derived, reason_aliases or {}) for line in check["lines"]]
    return payload


def key_path(key: dict[str, str]) -> str:
    """Return the OData entity path `/A_PaymentAdvice(...)` for a payment advice key."""
    return "/A_PaymentAdvice(" + ",".join(f"{field}={odata_str(key[field])}" for field in KEY_FIELDS) + ")"


def create(client: S4Client, payload: dict[str, Any]) -> dict[str, str]:
    """POST the payment advice and return its key (S/4 assigns `PaymentAdvice`)."""
    created = odata_entity(client.post_json(SERVICE, "/A_PaymentAdvice", payload))
    return {field: created.get(field) or payload.get(field, "") for field in KEY_FIELDS}


def read_back(client: S4Client, key: dict[str, str]) -> dict[str, Any]:
    """Read the stored payment advice with its items, proving what S/4 persisted."""
    entity = odata_entity(client.get_json(SERVICE, key_path(key), {"$expand": "to_PaymentAdviceItem"}))
    items = (entity.get("to_PaymentAdviceItem") or {}).get("results") or []
    return {**{k: v for k, v in entity.items() if not k.startswith("__") and k != "to_PaymentAdviceItem"},
            "items": [{k: v for k, v in item.items() if not k.startswith("__")} for item in items]}
