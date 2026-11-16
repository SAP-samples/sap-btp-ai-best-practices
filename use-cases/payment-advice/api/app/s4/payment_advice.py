"""Build and create the S/4 Payment Advice (`API_PAYMENT_ADVICE_SRV`) for a checked advice.

`build_payload` is the single place that maps our advice onto S/4 fields, so
feedback from the S/4 functional review is a one-function change:

Header (`A_PaymentAdvice`): the company code that receives the payment
(`derivation.derive`), the payer from the customer's rules (else the lead
matched customer), account type `D` (customer), advice type (default `04`; S/4
then assigns the advice number), currency, payment date, paid amount and the
customer's payment reference as header text.

Items (`to_PaymentAdviceItem`, deep insert):
- matched line: `AccountingDocument` and `FiscalYear` of the S/4 item the line
  resolved to (the document number takes priority in S/4 matching), plus the
  `DocumentReferenceID` (XBLNR) S/4 itself holds for that document. When S/4 has
  no reference on the document, none is sent: the advice reference is never
  passed off as an S/4 reference. `AccountingDocumentItem` is added when the
  document has several customer items or the item is a residual item, so S/4
  applies exactly that item. Gross, cash discount, deduction and net amounts
  follow, and the reason code when S/4 has to book a residual. A customer other
  than the header account is sent as the alternative account (lead payer), with
  its company code (`AlternativeCompanyCode`; S/4 otherwise looks the customer
  up in company code ""). An item in another company code than the header
  always carries the alternative account fields, with the company code of the
  item: that is how the advice names an item to be cleared across company codes.
- matched line that pays every open item of one document (instalments): one
  item per S/4 item, each with its open amount as gross and net.
- residual line (reviewer's choice, e.g. a reversed invoice the customer paid):
  same item as a deduction, with its own sign and gross = net (the cash
  received); S/4 books an open item.
- deduction line: no XBLNR (it is not a payee document); the customer's
  deduction reference goes to `AssignmentReference` (18) and `DocumentItemText`
  (50) with the reason code in `PaymentDifferenceReason`; its rule-derived
  customer account becomes the alternative account when it differs from the header.

Amounts keep the advice signs: invoices positive, deductions and credits negative,
and carry exactly the decimals of the currency (`amounts.fmt`).
Reason codes go through the demo-only `reason_aliases` (`S4_REASON_CODE_ALIASES`).

`compare_read_back` proves what S/4 stored: amounts are compared as numbers,
account and document numbers without leading zeros, every other field as text,
with the stored items ordered by their S/4 item number.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from .amounts import dec, fmt
from .client import S4Client, odata_entity, odata_str
from .validation import parse_iso_date

SERVICE = "API_PAYMENT_ADVICE_SRV"
ACCOUNT_TYPE_CUSTOMER = "D"
KEY_FIELDS = ("CompanyCode", "PaymentAdviceAccountType", "PaymentAdviceAccount", "PaymentAdvice")


def odata_date(value: Any) -> str | None:
    """Convert `YYYY-MM-DD` to the OData V2 `/Date(<ms>)/` literal at UTC midnight (None when empty/invalid)."""
    day = parse_iso_date(value)
    if day is None:
        return None
    return f"/Date({int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)})/"


def _items(line: dict[str, Any], derived: dict[str, Any], reason_aliases: dict[str, str]) -> list[dict[str, Any]]:
    """Map one checked advice line to its payment advice item(s) (reason code through ``reason_aliases``).

    Returns:
        One item; several only for a matched line that pays every open item of one S/4 document.
    """
    reason = reason_aliases.get(line["reason_code"], line["reason_code"]) if line["reason_code"] else line["reason_code"]
    currency = derived["currency"]
    alternative = {"PaymentAdviceAltvAccountType": ACCOUNT_TYPE_CUSTOMER, "AlternativeCompanyCode": derived["company_code"]}
    base: dict[str, Any] = {"Currency": currency,
                            "GrossAmountInPaymentCurrency": line["gross"], "NetPaymentAmountInPaytCurrency": line["net"]}
    if line["kind"] != "matched":
        item = dict(base)
        if line["kind"] == "residual":  # no invoice to take a discount from: the open item is the cash received
            item["GrossAmountInPaymentCurrency"] = line["net"]
        item.update(AssignmentReference=line["reference"][:18],
                    DocumentItemText=f"{line['reference']} "
                                     f"{'residual' if line['kind'] == 'residual' else line['nature'].replace('_', ' ')}"[:50],
                    PaymentDifferenceReason=reason)
        # The deduction's customer comes only from the customer's rules (e.g. region code).
        if line.get("account") and line["account"] != derived["customer"]:
            item.update(PaymentAdviceAltvAccount=line["account"], **alternative)
        return [item]
    targets = line.get("s4_items") or [line["s4_item"]]
    items = []
    for s4 in targets:
        item = dict(base)
        if len(targets) > 1:  # the line pays the whole document: every open item in full
            amount = fmt(Decimal(s4["open_amount"] or s4["amount"]), currency)
            item.update(GrossAmountInPaymentCurrency=amount, NetPaymentAmountInPaytCurrency=amount)
        if s4["document_reference"]:
            item["DocumentReferenceID"] = s4["document_reference"]
        item.update(AccountingDocument=s4["accounting_document"], FiscalYear=s4["fiscal_year"])
        if line.get("item_targeted"):
            item["AccountingDocumentItem"] = s4["accounting_document_item"]
        item["CashDiscountAmountInPaytCrcy"] = line["discount"]
        if Decimal(line["deduction"]):
            item["DeductionAmountInPaytCurrency"] = line["deduction"]
        if line["needs_reason"] and reason:
            item["PaymentDifferenceReason"] = reason
        if s4["company_code"] != derived["company_code"]:  # cross-company-code item: named with its own company code
            item.update(PaymentAdviceAltvAccount=s4["customer"], **{**alternative, "AlternativeCompanyCode": s4["company_code"]})
        elif s4["customer"] != derived["customer"]:
            item.update(PaymentAdviceAltvAccount=s4["customer"], **alternative)
        items.append(item)
    return items


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
        "PaidAmountInPaytCurrency": fmt(dec(header["payment_amount"], derived["currency"]), derived["currency"]),
        "PaymentAdviceHeaderText": str(header.get("payment_reference") or "")[:25],
    }
    if odata_date(header.get("payment_date")):
        payload["PaymentDate"] = odata_date(header.get("payment_date"))
    if header.get("payer_name"):
        payload["BusinessPartnerName"] = str(header["payer_name"])[:55]
    # Official SAP samples send the navigation property as a plain array.
    payload["to_PaymentAdviceItem"] = [item for line in check["lines"] for item in _items(line, derived, reason_aliases or {})]
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


HEADER_CHECK_FIELDS = ("CompanyCode", "PaymentAdviceAccount", "PaymentCurrency", "PaidAmountInPaytCurrency",
                       "PaymentAdviceHeaderText", "PaymentDate")
# Numbers S/4 may return with or without leading zeros.
NUMBER_FIELDS = frozenset({"PaymentAdviceAccount", "PaymentAdviceAltvAccount", "AccountingDocument", "AccountingDocumentItem"})


def _same(field: str, sent: Any, stored: Any) -> bool:
    """Compare a sent and a stored value by the kind of field: amount, account/document number, or text."""
    if "Amount" in field:
        try:
            return Decimal(str(sent or 0)) == Decimal(str(stored or 0))
        except InvalidOperation:
            return False
    if field in NUMBER_FIELDS:
        return str(sent or "").lstrip("0") == str(stored or "").lstrip("0")
    return str(sent if sent is not None else "") == str(stored if stored is not None else "")


def compare_read_back(payload: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    """Check that S/4 stored what was sent (header fields and every item field that was sent).

    Args:
        payload: The posted deep-insert payload.
        stored: `read_back` output (header fields plus `items`).

    Returns:
        `{"items_sent", "items_stored", "differences": [text, ...]}`; no differences means S/4
        holds exactly what was posted.
    """
    sent_items = payload.get("to_PaymentAdviceItem") or []
    # S/4 numbers the items in the order they were sent; sort by that number rather than trusting the response order.
    stored_items = sorted(stored.get("items") or [], key=lambda item: str(item.get("PaymentAdviceItem") or ""))
    differences = [f"Header {field}: sent {payload.get(field)!r}, S/4 has {stored.get(field)!r}"
                   for field in HEADER_CHECK_FIELDS if field in payload and not _same(field, payload.get(field), stored.get(field))]
    if len(sent_items) != len(stored_items):
        differences.append(f"{len(sent_items)} items sent, S/4 holds {len(stored_items)}")
    for number, (sent, kept) in enumerate(zip(sent_items, stored_items), 1):
        differences += [f"Item {number} {field}: sent {value!r}, S/4 has {kept.get(field)!r}"
                        for field, value in sent.items() if not _same(field, value, kept.get(field))]
    return {"items_sent": len(sent_items), "items_stored": len(stored_items), "differences": differences}
