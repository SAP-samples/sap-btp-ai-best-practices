"""Validate an advice against its S/4 receivables and derive the payment advice header.

Pure functions: no S/4 or HANA access. Input is the advice `result`
(`header` + `line_items` as produced by UC-01/UC-02) plus the output of
`lookup.find_receivables`; output is a JSON-serialisable check dict:

    {"derived": {company_code, customer, customers, currency},
     "lines":   [{index, row_id, reference, kind, status, matched_by, s4_item, ...}],
     "issues":  [{severity: "blocking"|"warning", code, line, message}],
     "ready":   True when there is no blocking issue}

Line kinds:
- `matched`: the reference is an S/4 receivable (invoice or credit memo).
- `deduction`: not in S/4 and a customer deduction (chargeback, allowance,
  credit memo, or a negative/unknown line); needs a 3-character reason code.
- `unresolved`: not in S/4 but should be (an invoice, or a positive line).

S/4 books any difference on a matched line (advice gross vs S/4 open amount,
or net vs gross - discount - deduction) as a residual item, which needs a
reason code; without one the line is blocking.

Accounts and company code combine both sources:
- S/4 is authoritative for invoices: the receivable's customer must equal the
  line's rule-derived ``customer_account`` (else ``customer_mismatch``).
- Deduction lines exist only on the remittance, so their rule-derived account
  is the only source; it must exist in the company code (``unknown_customer``).
- The header account is the rule-derived payer (``payer_account``) when given,
  else the customer with the largest matched amount.
- The company code comes from the matched receivables; the rule-derived one
  (mapped through ``aliases``, e.g. CA01 -> Z291 in a demo system) must agree.
  When no invoice matched, rule company code + payer are used (warning).
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from collections.abc import Callable

from .lookup import ReceivableItem, normalize_reference, pad_account

CENT = Decimal("0.01")
TOLERANCE = Decimal("0.01")
XBLNR_LENGTH = 16
ASSIGNMENT_LENGTH = 18
DEDUCTION_NATURES = {"chargeback", "allowance_reversal", "credit_memo"}


def dec(value: Any) -> Decimal:
    """Convert an extracted number (float, str or None) to a cent-rounded Decimal."""
    return Decimal(str(value if value not in (None, "") else 0)).quantize(CENT)


def _issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, line: int | None = None) -> None:
    """Append one validation finding."""
    issues.append({"severity": severity, "code": code, "line": line, "message": message})


def _classify(index: int, row: dict[str, Any], items: list[ReceivableItem], issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one advice line against its S/4 matches and record line-level issues."""
    reference = normalize_reference(row.get("invoice_reference"))
    gross, discount, net = dec(row.get("gross_amount")), dec(row.get("discount_amount")), dec(row.get("net_amount"))
    deduction = dec(row.get("deduction_amount"))
    reason = str(row.get("reason_code") or "").strip()
    nature = row.get("document_nature") or "unknown"
    line = {"index": index, "row_id": row.get("row_id"), "reference": reference, "nature": nature,
            "gross": str(gross), "discount": str(discount), "deduction": str(deduction), "net": str(net),
            "reason_code": reason or None, "matched_by": None, "s4_item": None, "needs_reason": False,
            "account": pad_account(row.get("customer_account"))}
    label = f"Line {index + 1} ({reference or 'no reference'})"
    if len(items) > 1:
        line.update(kind="unresolved", status="ambiguous")
        docs = ", ".join(f"{i.company_code}/{i.accounting_document}/{i.fiscal_year}" for i in items)
        _issue(issues, "blocking", "ambiguous_reference", f"{label} matches {len(items)} S/4 items ({docs}).", index)
        return line
    if items:
        item = items[0]
        line.update(kind="matched", status="cleared" if item.is_cleared else "open",
                    matched_by=item.matched_by, s4_item=item.to_dict())
        if item.is_cleared:
            _issue(issues, "blocking", "invoice_cleared",
                   f"{label} is already cleared in S/4 (clearing document {item.clearing_document}).", index)
        if line["account"] and item.customer and line["account"] != item.customer:
            _issue(issues, "blocking", "customer_mismatch",
                   f"{label}: the customer rules give account {line['account']} but the S/4 invoice belongs to "
                   f"customer {item.customer}.", index)
        differences = []
        if abs(gross - item.amount) > TOLERANCE:
            differences.append(f"advice gross {gross} vs S/4 open amount {item.amount}")
        if abs(gross - discount - deduction - net) > TOLERANCE:
            differences.append(f"net {net} differs from gross - discount - deduction")
        if differences:
            line["needs_reason"] = True
            if reason:
                _issue(issues, "warning", "amount_difference",
                       f"{label}: {'; '.join(differences)}; S/4 books the residual with reason {reason}.", index)
            else:
                _issue(issues, "blocking", "difference_without_reason",
                       f"{label}: {'; '.join(differences)} and the line has no reason code.", index)
        return line
    if nature == "invoice" or (nature not in DEDUCTION_NATURES and net > 0):
        line.update(kind="unresolved", status="not_found")
        hint = f" (longer than {XBLNR_LENGTH} characters, so it cannot be an S/4 reference)" if len(reference) > XBLNR_LENGTH else ""
        _issue(issues, "blocking", "invoice_not_found", f"{label} was not found as an S/4 customer item{hint}.", index)
        return line
    line.update(kind="deduction", status="not_found")
    if len(reason) != 3:
        _issue(issues, "blocking", "missing_reason_code", f"{label} is a deduction without a 3-character reason code.", index)
    if len(reference) > ASSIGNMENT_LENGTH:
        _issue(issues, "warning", "reference_truncated",
               f"{label} reference exceeds {ASSIGNMENT_LENGTH} characters; the full value is kept in the item text.", index)
    return line


def _derive(lines: list[dict[str, Any]], header: dict[str, Any], issues: list[dict[str, Any]],
            aliases: dict[str, str], customer_lookup: Callable[[list[str], str], set[str]] | None) -> dict[str, Any]:
    """Derive company code, header account, all customers and currency (see module docstring)."""
    matched = [line["s4_item"] for line in lines if line["kind"] == "matched"]
    rule_code = str(header.get("company_code") or "").strip().upper() or None
    rule_code_s4 = aliases.get(rule_code, rule_code) if rule_code else None
    payer = pad_account(header.get("payer_account"))
    advice_currency = str(header.get("payment_currency") or "").strip().upper() or None
    derived: dict[str, Any] = {"company_code": None, "customer": None, "customers": [], "currency": None,
                               "payer_account": payer, "rule_company_code": rule_code}
    totals: dict[str, Decimal] = defaultdict(Decimal)
    if matched:
        company_codes = sorted({item["company_code"] for item in matched})
        currencies = sorted({item["currency"] for item in matched})
        for item in matched:
            totals[item["customer"]] += Decimal(item["amount"])
        derived.update(company_code=company_codes[0] if len(company_codes) == 1 else None,
                       currency=currencies[0] if len(currencies) == 1 else None)
        if len(company_codes) > 1:
            _issue(issues, "blocking", "multiple_company_codes", f"Matched items span company codes {', '.join(company_codes)}.")
        if len(currencies) > 1:
            _issue(issues, "blocking", "multiple_currencies", f"Matched items use currencies {', '.join(currencies)}.")
        if advice_currency and derived["currency"] and advice_currency != derived["currency"]:
            _issue(issues, "blocking", "currency_mismatch",
                   f"Advice currency {advice_currency} differs from S/4 currency {derived['currency']}.")
        if rule_code_s4 and derived["company_code"] and rule_code_s4 != derived["company_code"]:
            via = f" (alias of {rule_code})" if rule_code_s4 != rule_code else ""
            _issue(issues, "blocking", "company_code_mismatch",
                   f"The customer rules give company code {rule_code_s4}{via} but the S/4 invoices are in "
                   f"{derived['company_code']}.")
    elif rule_code_s4 and payer:
        derived.update(company_code=rule_code_s4, currency=advice_currency)
        _issue(issues, "warning", "derived_from_rules",
               f"No line matched an S/4 invoice; company code {rule_code_s4} and payer {payer} come from the customer's rules.")
        if not advice_currency:
            _issue(issues, "blocking", "missing_currency", "No S/4 invoice matched and the advice has no currency.")
    else:
        _issue(issues, "blocking", "no_invoice_resolved",
               "No line resolved to an S/4 customer item and the customer's rules give no company code and payer, "
               "so the advice cannot be assigned.")
        return derived
    lead = sorted(totals, key=lambda c: (-totals[c], c))
    derived["customer"] = payer or (lead[0] if lead else None)
    line_accounts = [line["account"] for line in lines if line["account"]]
    derived["customers"] = list(dict.fromkeys([c for c in [derived["customer"], *lead, *line_accounts] if c]))
    if customer_lookup and derived["company_code"]:
        needed = sorted({a for a in [payer, *line_accounts] if a})
        existing = customer_lookup(needed, derived["company_code"]) if needed else set()
        for account in needed:
            if account not in existing:
                _issue(issues, "blocking", "unknown_customer",
                       f"Customer {account} from the customer's rules does not exist in company code {derived['company_code']}.")
    if len(derived["customers"]) > 1:
        _issue(issues, "warning", "multiple_customers",
               f"Items belong to customers {', '.join(derived['customers'])}; {derived['customer']} is the advice "
               "account and the others are sent as alternative accounts (lead payer).")
    return derived


def check_advice(result: dict[str, Any], found: dict[str, list[ReceivableItem]],
                 verification: dict[str, Any] | None = None, *,
                 customer_lookup: Callable[[list[str], str], set[str]] | None = None,
                 aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate one advice against its S/4 lookup results.

    Args:
        result: Advice result with `header` and `line_items`.
        found: `lookup.find_receivables` output keyed by normalized reference.
        verification: Optional UC-01 `verify_canonical` dict (duplicate groups).
        customer_lookup: Optional `(accounts, company_code) -> existing accounts`
            (e.g. `lookup.existing_customers`); None skips the existence check.
        aliases: Rule company code -> S/4 company code (demo systems only).

    Returns:
        Check dict described in the module docstring.
    """
    header, rows = result.get("header") or {}, result.get("line_items") or []
    issues: list[dict[str, Any]] = []
    lines = [_classify(i, row, found.get(normalize_reference(row.get("invoice_reference")), []), issues)
             for i, row in enumerate(rows)]
    derived = _derive(lines, header, issues, aliases or {}, customer_lookup)
    documents: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for line in lines:
        if line["kind"] == "matched":
            item = line["s4_item"]
            documents[(item["company_code"], item["accounting_document"], item["fiscal_year"])].append(line["index"])
    for (_, document, year), indexes in documents.items():
        if len(indexes) > 1:
            _issue(issues, "blocking", "duplicate_application",
                   f"Lines {', '.join(str(i + 1) for i in indexes)} all apply S/4 document {document}/{year}.")
    for group in (verification or {}).get("suspected_duplicate_groups") or []:
        _issue(issues, "warning", "suspected_duplicates",
               f"Lines {', '.join(str(i + 1) for i in group)} look like duplicates; confirm before posting.")
    payment = header.get("payment_amount")
    total = sum((dec(line["net"]) for line in lines), Decimal("0"))
    if payment in (None, ""):
        _issue(issues, "blocking", "missing_payment_amount", "The advice has no payment amount.")
    elif abs(total - dec(payment)) > TOLERANCE:
        _issue(issues, "blocking", "totals_mismatch", f"Line nets sum to {total} but the payment amount is {dec(payment)}.")
    return {"derived": derived, "lines": lines, "issues": issues,
            "ready": not any(issue["severity"] == "blocking" for issue in issues)}
