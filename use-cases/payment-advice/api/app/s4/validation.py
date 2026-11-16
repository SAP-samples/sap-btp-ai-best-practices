"""Validate an advice against its S/4 receivables and derive the payment advice header.

Pure functions: no S/4 or HANA access. Input is the advice `result`
(`header` + `line_items` as produced by extraction and interpretation) plus the
candidates `lookup.find_receivables` found; output is a JSON-serialisable check:

    {"derived": {company_code, customer, customers, currency, ...},
     "lines":   [{index, row_id, reference, kind, status, matched_by, s4_item, ...}],
     "issues":  [{severity: "blocking"|"warning", code, line, message}],
     "ready":   True when there is no blocking issue,
     "summary": {lines, matched, matched_by, resolution, company_codes, deductions, residuals, unresolved,
                 blocking, warnings}}

Each line is first resolved to one S/4 document (`resolution.resolve`), then
classified:

- `matched`: the line applies an S/4 receivable (invoice or credit memo). Its
  gross is compared with what is still OPEN in S/4, and the net with
  gross - discount - deduction. S/4 books a difference as a residual item,
  which needs a reason code. A deduction amount on the line is such a
  difference as well, so it never reaches S/4 without a reason code.
- `deduction`: nothing in S/4, and the line is a customer deduction
  (chargeback, allowance, credit memo, or a negative/unknown line). Needs a
  3-character reason code. A deduction that quotes an invoice of the opposite
  sign is a deduction too; the invoice is shown as `related_document`.
- `residual`: the reviewer chose to post the line without an invoice
  (`s4_treatment == "residual"`); S/4 books an open item on the customer.
- `unresolved`: the line should apply an S/4 document but none could be
  chosen: not found, several candidates (`candidates` lists them for the
  reviewer's pick), wrong sign, or a special G/L item.

Header values are derived in `derivation.derive`; the advice has one company
code (the one that receives the payment) while its lines may apply items of
other company codes. After that the advice-level checks run: lines outside the
advice company code, customers and reason codes per company code
(`company_codes`), a document applied twice, totals, payment reference and date.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any

from .amounts import dec, fmt, parse_amount, sign_of, tolerance  # noqa: F401  (dec is re-exported for callers)
from .company_codes import (REASON_LENGTH, CustomerLookup, ReasonCodes, check_cross_company, check_customers,
                            check_reasons)
from .derivation import derive, issue, line_ranges, lines_by_company_code, matched_items
from .lookup import ReceivableItem, pad_account
from .references import XBLNR_LENGTH, line_reference, pick_of
from .resolution import LineContext, Resolution, is_open, item_sign, open_balance, resolve

ASSIGNMENT_LENGTH = 18
HEADER_TEXT_LENGTH = 25
# `PaymentAdviceItem` has 5 digits in S/4.
MAX_ITEMS = 99999
DEDUCTION_NATURES = {"chargeback", "allowance_reversal", "credit_memo"}
AMOUNT_FIELDS = ("gross_amount", "discount_amount", "deduction_amount", "net_amount")


def parse_iso_date(value: Any) -> date | None:
    """Return the date of a `YYYY-MM-DD...` text, or None when it is empty or not a date."""
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _candidate(item: ReceivableItem) -> dict[str, Any]:
    """Describe one S/4 item for the reviewer's picker (`pick` is what the UI saves as `s4_document`)."""
    state = "reversed" if item.is_reversed else "cleared" if item.is_cleared else "open"
    return {**item.to_dict(), "open_amount": str(open_balance(item)), "state": state,
            "pick": pick_of(item.company_code, item.accounting_document, item.fiscal_year, item.accounting_document_item)}


def _document(item: ReceivableItem) -> str:
    """Short name of an S/4 document for messages: `COMPANYCODE/DOCUMENT/YEAR`."""
    return f"{item.company_code}/{item.accounting_document}/{item.fiscal_year}"


def _line(index: int, row: dict[str, Any], currency: str | None,
          issues: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Decimal]]:
    """Start the check line of one advice row and parse its amounts (a bad amount blocks and counts as zero)."""
    reference, source = line_reference(row)
    amounts: dict[str, Decimal] = {}
    invalid: list[str] = []
    for name in AMOUNT_FIELDS:
        value = parse_amount(row.get(name), currency)
        if value is None:
            invalid.append(f"{name.replace('_', ' ')} {row.get(name)!r}")
        amounts[name] = value if value is not None else parse_amount(None, currency)
    reason = str(row.get("reason_code") or "").strip()
    line = {"index": index, "row_id": row.get("row_id"), "reference": reference,
            "nature": row.get("document_nature") or "unknown",
            "gross": fmt(amounts["gross_amount"], currency), "discount": fmt(amounts["discount_amount"], currency),
            "deduction": fmt(amounts["deduction_amount"], currency), "net": fmt(amounts["net_amount"], currency),
            "reason_code": reason or None, "matched_by": None, "s4_item": None, "needs_reason": False,
            "account": pad_account(row.get("customer_account"))}
    if source != "invoice_reference":
        line["reference_source"] = source
    if invalid:
        issue(issues, "blocking", "invalid_amount",
              f"Line {index + 1} ({reference or 'no reference'}): {', '.join(invalid)} is not a valid amount"
              + (f" in {currency}" if currency else "") + ".", index)
    return line, amounts


def _context(row: dict[str, Any], amounts: dict[str, Decimal], rule_code: str | None, currency: str | None) -> LineContext:
    """Build what the line and the customer's rules tell the resolution."""
    limit = tolerance(currency)
    has_gross = row.get("gross_amount") not in (None, "")
    gross, net = amounts["gross_amount"], amounts["net_amount"]
    plain = (not amounts["discount_amount"] and not amounts["deduction_amount"]
             and (not has_gross or abs(gross - net) <= limit))
    residual = row.get("s4_treatment") == "residual"  # a residual line applies no document, so a pick has no meaning
    return LineContext(gross=gross if has_gross else None, sign=sign_of(gross) or sign_of(net), company_code=rule_code,
                       account=pad_account(row.get("customer_account")),
                       pick=None if residual else (row.get("s4_document") or None), plain=plain, tolerance=limit)


def _reason_issue(line: dict[str, Any], label: str, what: str, issues: list[dict[str, Any]]) -> None:
    """Block a deduction or residual line whose reason code does not have exactly 3 characters."""
    if len(line["reason_code"] or "") != REASON_LENGTH:
        issue(issues, "blocking", "missing_reason_code", f"{label} is {what} without a 3-character reason code.", line["index"])


def _truncation_issue(line: dict[str, Any], label: str, issues: list[dict[str, Any]]) -> None:
    """Warn when a reference without S/4 document is longer than the assignment field."""
    if len(line["reference"]) > ASSIGNMENT_LENGTH:
        issue(issues, "warning", "reference_truncated",
              f"{label} reference exceeds {ASSIGNMENT_LENGTH} characters; the full value is kept in the item text.", line["index"])


def _deduction(line: dict[str, Any], label: str, issues: list[dict[str, Any]], related: ReceivableItem | None = None) -> dict[str, Any]:
    """Classify a line as a new customer deduction (optionally quoting an S/4 document of the opposite sign)."""
    line.update(kind="deduction", status="not_found")
    if related is not None:
        line["related_document"] = _document(related)
    elif line["nature"] == "credit_memo":
        issue(issues, "warning", "credit_not_found",
              f"{label} is a credit memo that S/4 does not know; it is sent as a new deduction. If it is a credit memo "
              "of the payee, correct its reference.", line["index"])
    _reason_issue(line, label, "a deduction", issues)
    _truncation_issue(line, label, issues)
    return line


def _unresolved(line: dict[str, Any], label: str, resolution: Resolution, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a line whose S/4 document could not be chosen and say what the reviewer can do."""
    options = resolution.candidates
    line.update(kind="unresolved", status="not_found" if resolution.status == "suggested" else "ambiguous",
                candidates=[_candidate(item) for item in options],
                resolution={"method": None, "steps": resolution.steps, "warning": None})
    listed = ", ".join(dict.fromkeys(_document(item) for item in options))
    if resolution.status == "stale_pick":
        issue(issues, "blocking", "selection_stale",
              f"{label}: the chosen S/4 document is no longer among the candidates ({listed}); choose again.", line["index"])
    elif resolution.status == "multi_item":
        issue(issues, "blocking", "multi_item_document",
              f"{label}: S/4 document {listed} has {len(options)} open items and the advice amount fits neither one "
              "item nor their total; choose the item.", line["index"])
    elif resolution.status == "suggested":
        issue(issues, "blocking", "invoice_not_found",
              f"{label} was not found as an S/4 customer item. S/4 has '{options[0].variant}' ({listed}) but its amount "
              "differs; choose it or correct the reference.", line["index"])
    else:
        issue(issues, "blocking", "ambiguous_reference",
              f"{label} matches {len(options)} S/4 items ({listed}) that the rules cannot tell apart; choose the document.",
              line["index"])
    return line


def _not_found(line: dict[str, Any], label: str, net: Decimal, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a line with no S/4 candidate: an invoice is unresolved, anything else is a deduction."""
    if line["nature"] == "invoice" or (line["nature"] not in DEDUCTION_NATURES and net > 0):
        line.update(kind="unresolved", status="not_found")
        reference = line["reference"]
        hint = f" (longer than {XBLNR_LENGTH} characters, so it cannot be an S/4 reference)" if len(reference) > XBLNR_LENGTH else ""
        issue(issues, "blocking", "invoice_not_found", f"{label} was not found as an S/4 customer item{hint}.", line["index"])
        return line
    return _deduction(line, label, issues)


def _matched(line: dict[str, Any], label: str, amounts: dict[str, Decimal], context: LineContext,
             resolution: Resolution, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a line that resolved to S/4 item(s) and run the checks of a matched line."""
    index, items = line["index"], resolution.items
    item = items[0]
    line.update(matched_by=item.matched_by, s4_item=item.to_dict(),
                resolution={"method": resolution.method, "steps": resolution.steps, "warning": resolution.warning})
    if context.sign and item_sign(item) and item_sign(item) != context.sign:
        line.update(kind="unresolved", status="sign_mismatch")
        issue(issues, "blocking", "sign_mismatch",
              f"{label} is a {'payment' if context.sign > 0 else 'credit'} of {line['gross']} but S/4 document "
              f"{_document(item)} is a {'credit' if item_sign(item) < 0 else 'debit'} item of {item.amount}. Correct the "
              "reference, set the nature to a deduction, or exclude the line.", index)
        return line
    if all(i.special_gl for i in items):
        line.update(kind="unresolved", status="special_gl")
        issue(issues, "blocking", "special_gl_item",
              f"{label} is special G/L item {_document(item)} (indicator {item.special_gl}), which a payment advice "
              "cannot apply like an invoice.", index)
        return line
    status = "reversed" if item.is_reversed else "cleared" if item.is_cleared else "open"
    line.update(kind="matched", status=status, item_targeted=resolution.item_targeted,
                open_amount=str(sum((open_balance(i) for i in items), Decimal("0"))))
    if len(items) > 1:
        line["s4_items"] = [i.to_dict() for i in items]
    if item.is_reversed:
        issue(issues, "blocking", "invoice_reversed",
              f"{label} is reversed in S/4 (reversal document {item.reverse_document}); it cannot be paid.", index)
    elif item.is_cleared:
        issue(issues, "blocking", "invoice_cleared",
              f"{label} is already cleared in S/4 (clearing document {item.clearing_document}).", index)
    if item.residual_of:
        issue(issues, "warning", "residual_item_applied",
              f"{label}: invoice {item.residual_of} is cleared; the line is applied to its open residual item "
              f"{item.accounting_document}/{item.fiscal_year} of {open_balance(item)}.", index)
    if resolution.warning:
        issue(issues, "warning", "reference_normalized" if item.variant else "reference_disambiguated",
              f"{label} was matched to S/4 document {_document(item)} because {resolution.warning}; confirm it.", index)
    if line["account"] and item.customer and line["account"] not in (item.customer, item.branch_account):
        issue(issues, "blocking", "customer_mismatch",
              f"{label}: the customer rules give account {line['account']} but the S/4 invoice belongs to "
              f"customer {item.customer}.", index)
    _differences(line, label, amounts, context, items, status, issues)
    return line


def _differences(line: dict[str, Any], label: str, amounts: dict[str, Decimal], context: LineContext,
                 items: list[ReceivableItem], status: str, issues: list[dict[str, Any]]) -> None:
    """Compare the advice amounts with S/4; a difference becomes a residual in S/4 and needs a reason code."""
    gross, net = amounts["gross_amount"], amounts["net_amount"]
    original = sum((i.amount for i in items), Decimal("0"))
    # An open item is compared with what is still open; a cleared or reversed one with what it was.
    expected = sum((open_balance(i) for i in items), Decimal("0")) if status == "open" else original
    differences = []
    if abs(gross - expected) > context.tolerance:
        partial = sum((i.partial_payments for i in items), Decimal("0"))
        detail = f" (document amount {original}, partial payments {partial})" if status == "open" and partial else ""
        differences.append(f"advice gross {gross} vs S/4 open amount {expected}{detail}")
    if abs(gross - amounts["discount_amount"] - amounts["deduction_amount"] - net) > context.tolerance:
        differences.append(f"net {net} differs from gross - discount - deduction")
    if amounts["deduction_amount"]:
        # A deduction taken on the invoice line is a difference too, even when the line adds up.
        differences.append(f"the line deducts {amounts['deduction_amount']} from the invoice")
    if not differences:
        return
    line["needs_reason"] = True
    reason, text = line["reason_code"] or "", "; ".join(differences)
    if len(reason) == REASON_LENGTH:
        issue(issues, "warning", "amount_difference", f"{label}: {text}; S/4 books the residual with reason {reason}.", line["index"])
    elif reason:
        issue(issues, "blocking", "missing_reason_code",
              f"{label}: {text}, and reason code '{reason}' does not have 3 characters.", line["index"])
    else:
        issue(issues, "blocking", "difference_without_reason", f"{label}: {text} and the line has no reason code.", line["index"])


def _classify(line: dict[str, Any], row: dict[str, Any], amounts: dict[str, Decimal], context: LineContext,
              resolution: Resolution, candidates: list[ReceivableItem], issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one advice line against its S/4 resolution and record line-level issues."""
    label = f"Line {line['index'] + 1} ({line['reference'] or 'no reference'})"
    if row.get("s4_treatment") == "residual":
        line.update(kind="residual", status="residual")
        shown = resolution.items or resolution.candidates or candidates
        if shown:  # the reversed / cleared document stays visible next to the residual
            line.update(matched_by=shown[0].matched_by, s4_item=shown[0].to_dict())
        _reason_issue(line, label, "posted as a residual", issues)
        still_open = [i for i in candidates if is_open(i)]
        if still_open:
            issue(issues, "warning", "residual_with_open_invoice",
                  f"{label} is open in S/4 as {still_open[0].accounting_document}/{still_open[0].fiscal_year}; "
                  "posting it as a residual leaves that invoice open.", line["index"])
        _truncation_issue(line, label, issues)
        return line
    # A customer deduction that quotes one of our documents of the opposite sign deducts from it; it does not pay it.
    opposite = bool(context.sign and candidates) and all(item_sign(i) and item_sign(i) != context.sign for i in candidates)
    if opposite and line["nature"] in DEDUCTION_NATURES:
        return _deduction(line, label, issues, related=candidates[0])
    if resolution.status == "none":
        return _not_found(line, label, amounts["net_amount"], issues)
    if resolution.status != "resolved":
        return _unresolved(line, label, resolution, issues)
    return _matched(line, label, amounts, context, resolution, issues)


def _currency_hint(header: dict[str, Any], found: dict[str, list[ReceivableItem]]) -> str | None:
    """Return the currency amounts are read in: the advice currency, else the one currency of all candidates."""
    advice = str(header.get("payment_currency") or "").strip().upper()
    if advice:
        return advice
    currencies = {item.currency for items in found.values() for item in items if item.currency}
    return next(iter(currencies)) if len(currencies) == 1 else None


def _consensus(resolutions: list[Resolution]) -> str | None:
    """Return the company code all lines with exactly one, open, reference-matched candidate agree on."""
    codes = {r.items[0].company_code for r in resolutions
             if r.status == "resolved" and r.method == "unique" and is_open(r.items[0])
             and r.items[0].matched_by == "DocumentReferenceID" and not r.items[0].variant}
    return next(iter(codes)) if len(codes) == 1 else None


def _check_header(header: dict[str, Any], lines: list[dict[str, Any]], currency: str | None,
                  issues: list[dict[str, Any]]) -> None:
    """Advice-level checks: one application per S/4 item, reference, date, item count and totals."""
    applied: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for line in lines:
        if line["kind"] == "matched":
            for item in matched_items(line):
                applied[(item["company_code"], item["accounting_document"], item["fiscal_year"],
                         item["accounting_document_item"])].append(line["index"])
    for (company_code, document, year, _), indexes in applied.items():
        if len(indexes) > 1:
            issue(issues, "blocking", "duplicate_application",
                  f"Lines {line_ranges(indexes)} all apply S/4 document {company_code}/{document}/{year}.")
    reference = str(header.get("payment_reference") or "").strip()
    if not reference:
        issue(issues, "blocking", "missing_payment_reference",
              "The advice has no payment reference. It becomes the header text of the S/4 advice and is what protects "
              "against posting the same remittance twice.")
    elif len(reference) > HEADER_TEXT_LENGTH:
        issue(issues, "warning", "payment_reference_truncated",
              f"The payment reference has {len(reference)} characters; S/4 keeps the first {HEADER_TEXT_LENGTH}.")
    if parse_iso_date(header.get("payment_date")) is None:
        issue(issues, "warning", "missing_payment_date",
              "The advice has no valid payment date (YYYY-MM-DD); the S/4 advice is created without one.")
    count = sum(len(matched_items(line)) if line["kind"] == "matched" else 1 for line in lines)
    if count > MAX_ITEMS:
        issue(issues, "blocking", "too_many_items", f"The advice has {count} items; an S/4 payment advice holds at most {MAX_ITEMS}.")
    payment = header.get("payment_amount")
    total = sum((Decimal(line["net"]) for line in lines), Decimal("0"))
    if payment in (None, ""):
        issue(issues, "blocking", "missing_payment_amount", "The advice has no payment amount.")
    elif (paid := parse_amount(payment, currency)) is None:
        issue(issues, "blocking", "invalid_amount", f"The payment amount {payment!r} is not a valid amount"
              + (f" in {currency}" if currency else "") + ".")
    elif abs(total - paid) > tolerance(currency):
        issue(issues, "blocking", "totals_mismatch", f"Line nets sum to {total} but the payment amount is {paid}.")


def _summary(lines: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Count how the lines were matched: the quality of the match at a glance."""
    matched = [line for line in lines if line["kind"] == "matched"]
    by_field: dict[str, int] = defaultdict(int)
    by_method = {"unique": 0, "auto": 0, "reviewer": 0}
    for line in matched:
        by_field[line["matched_by"]] += 1
        by_method[line["resolution"]["method"]] += 1
    count = lambda kind: sum(1 for line in lines if line["kind"] == kind)  # noqa: E731
    by_code = {code: len(indexes) for code, indexes in sorted(lines_by_company_code(lines).items())}
    return {"lines": len(lines), "matched": len(matched), "matched_by": dict(by_field), "resolution": by_method,
            "company_codes": by_code,
            "deductions": count("deduction"), "residuals": count("residual"), "unresolved": count("unresolved"),
            "blocking": sum(1 for i in issues if i["severity"] == "blocking"),
            "warnings": sum(1 for i in issues if i["severity"] == "warning")}


def check_advice(result: dict[str, Any], found: dict[str, list[ReceivableItem]],
                 verification: dict[str, Any] | None = None, *,
                 customer_lookup: CustomerLookup | None = None, aliases: dict[str, str] | None = None,
                 reason_codes: ReasonCodes | None = None, reason_aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate one advice against its S/4 lookup results.

    Args:
        result: Advice result with `header` and `line_items`.
        found: Candidates keyed by normalized reference (`lookup.find_receivables`,
            ideally after `resolution.apply_linked`).
        verification: Optional extraction verification dict (duplicate groups).
        customer_lookup: Optional `(accounts, company_code) -> existing accounts`,
            as a set or as `{account: {posting_blocked, deletion_flag}}`, called
            once per company code the advice touches; None skips the customer checks.
        aliases: Rule company code -> S/4 company code (demo systems only).
        reason_codes: Optional `company code -> valid reason codes` (None when S/4
            cannot tell), called once per company code; None skips the reason-code check.
        reason_aliases: Documented reason code -> S/4 reason code (demo systems only).

    Returns:
        Check dict described in the module docstring.
    """
    header, rows = result.get("header") or {}, result.get("line_items") or []
    issues: list[dict[str, Any]] = []
    aliases = aliases or {}
    rule_code = str(header.get("company_code") or "").strip().upper() or None
    rule_code = aliases.get(rule_code, rule_code) if rule_code else None
    currency = _currency_hint(header, found)
    # Rows the reviewer excluded are not posted; the others keep their table position in messages.
    drafts = []
    for index, row in enumerate(rows):
        if not row.get("excluded"):
            line, amounts = _line(index, row, currency, issues)
            drafts.append((line, row, amounts, _context(row, amounts, rule_code, currency), found.get(line["reference"], [])))
    resolutions = [resolve(candidates, context) for _, _, _, context, candidates in drafts]
    consensus = None if rule_code else _consensus(resolutions)
    if consensus:
        # Lines with one candidate are the evidence; every other line is resolved again inside their company code.
        resolutions = [first if first.method == "unique" else resolve(candidates, context, consensus)
                       for first, (_, _, _, context, candidates) in zip(resolutions, drafts)]
    lines = [_classify(line, row, amounts, context, resolution, candidates, issues)
             for (line, row, amounts, context, candidates), resolution in zip(drafts, resolutions)]
    derived = derive(lines, header, issues, aliases)
    check_cross_company(lines, derived, issues)
    check_customers(lines, derived, issues, customer_lookup)
    for group in (verification or {}).get("suspected_duplicate_groups") or []:
        issue(issues, "warning", "suspected_duplicates",
              f"Lines {', '.join(str(i + 1) for i in group)} look like duplicates; confirm before posting.")
    _check_header(header, lines, currency, issues)
    check_reasons(lines, derived["company_code"], reason_codes, reason_aliases or {}, issues)
    return {"derived": derived, "lines": lines, "issues": issues,
            "ready": not any(entry["severity"] == "blocking" for entry in issues), "summary": _summary(lines, issues)}
