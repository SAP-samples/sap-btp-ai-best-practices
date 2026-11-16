"""Checks that depend on the company code of each item of an advice.

Pure functions: no S/4 or HANA access (S/4 is reached only through the lookup
callables the caller passes in). One advice has one company code on its header,
the one that receives the payment (`derivation.derive`), but its matched lines
may apply S/4 items of other company codes. Three checks therefore run per
company code instead of once per advice:

- `check_cross_company`: a line applied to an item outside the advice company
  code must have its customer confirmed by something other than the automatic
  choice among several documents; otherwise the reviewer confirms the document.
- `check_customers`: every account is checked in the company code it is used
  in. Rule accounts must exist there (`unknown_customer`); a posting block
  blocks (`customer_blocked`); a deletion flag warns
  (`customer_marked_for_deletion`).
- `check_reasons`: a reason code must exist in the advice company code, and a
  code on a matched line outside it in the company code of that item as well.
  S/4 keeps reason codes per company code, and the residual item is posted
  where the invoice is; requiring both is the strict reading.

Example:
    derived = derive(lines, header, issues, aliases)
    check_cross_company(lines, derived, issues)
    check_customers(lines, derived, issues, customer_lookup)
    check_reasons(lines, derived["company_code"], reason_codes, reason_aliases, issues)
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .derivation import issue, matched_items
from .references import pick_of

REASON_LENGTH = 3
# `(accounts, company code) -> existing accounts`, as a set or as {account: {posting_blocked, deletion_flag}}.
CustomerLookup = Callable[[list[str], str], Any]
# `company code -> reason codes valid there`, or None when S/4 cannot tell.
ReasonCodes = Callable[[str], "set[str] | None"]
# Ways a line reached its document that need no further confirmation of the customer.
SELF_CONFIRMING = frozenset({"unique", "reviewer"})


def _label(line: dict[str, Any]) -> str:
    """Name a line the way every message of the check does."""
    return f"Line {line['index'] + 1} ({line['reference'] or 'no reference'})"


def _outside(line: dict[str, Any], home: str) -> list[dict[str, Any]]:
    """Return the S/4 items of a matched line that lie outside the advice company code."""
    if line["kind"] != "matched":
        return []
    return [item for item in matched_items(line) if item["company_code"] != home]


def check_cross_company(lines: list[dict[str, Any]], derived: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """Block a cross-company-code line whose customer nothing confirms, and offer its document for confirmation.

    A line outside the advice company code is accepted when one of these holds:
    the reference had exactly one candidate in S/4, the reviewer chose the
    document, the customer's rules name the account of the line (a wrong
    account is reported as `customer_mismatch`), or the customer is the payer
    or the customer of a matched item in the advice company code. What is left
    is a document picked by the rules among several, for a customer the advice
    does not otherwise know. That line gets `candidates`, so the reviewer can
    confirm it with "Use this document".

    Args:
        lines: Classified advice lines; a blocked line receives `candidates`.
        derived: `derivation.derive` output.
        issues: Receives the findings.
    """
    home = derived["company_code"]
    if not home:
        return
    known = {derived["payer_account"]} | {item["customer"] for line in lines if line["kind"] == "matched"
                                          for item in matched_items(line) if item["company_code"] == home}
    for line in lines:
        outside = _outside(line, home)
        if not outside or line["account"] or line["resolution"]["method"] in SELF_CONFIRMING:
            continue
        item = outside[0]
        if item["customer"] in known:
            continue
        line["candidates"] = [{**entry, "open_amount": entry["open_amount"] if entry["open_amount"] is not None else entry["amount"],
                               "state": line["status"],
                               "pick": pick_of(entry["company_code"], entry["accounting_document"], entry["fiscal_year"],
                                               entry["accounting_document_item"])} for entry in outside]
        issue(issues, "blocking", "cross_company_unconfirmed",
              f"{_label(line)} was matched to S/4 document {item['company_code']}/{item['accounting_document']}/"
              f"{item['fiscal_year']} of customer {item['customer']}, outside company code {home}. Neither the "
              "customer's rules nor another line confirm that customer; confirm the document or correct the reference.",
              line["index"])


def check_customers(lines: list[dict[str, Any]], derived: dict[str, Any], issues: list[dict[str, Any]],
                    customer_lookup: CustomerLookup | None) -> None:
    """Check existence, posting block and deletion flag of every account in the company code it is used in.

    Args:
        lines: Classified advice lines.
        derived: `derivation.derive` output.
        issues: Receives the findings.
        customer_lookup: Customer master lookup, called once per company code; None skips the check.
    """
    home = derived["company_code"]
    if not (customer_lookup and home):
        return
    by_rule: dict[str, set[str]] = defaultdict(set)  # accounts the customer's rules name, per company code
    in_use: dict[str, set[str]] = defaultdict(set)   # customers of the matched S/4 items, per company code
    if derived["payer_account"]:
        by_rule[home].add(derived["payer_account"])
    if derived["customer"]:
        in_use[home].add(derived["customer"])
    for line in lines:
        items = matched_items(line) if line["kind"] == "matched" else []
        for item in items:
            in_use[item["company_code"]].add(item["customer"])
        # A deduction or residual line has no S/4 document: its account is used in the advice company code.
        for code in {item["company_code"] for item in items} or {home}:
            if line["account"]:
                by_rule[code].add(line["account"])
    reported: set[tuple[str, str]] = set()
    for code in sorted(set(by_rule) | set(in_use)):
        needed = sorted(account for account in by_rule[code] | in_use[code] if account)
        if not needed:
            continue
        answer = customer_lookup(needed, code)
        flags = answer if isinstance(answer, dict) else {account: {} for account in answer}
        for account in sorted(by_rule[code]):
            if account not in flags:
                issue(issues, "blocking", "unknown_customer",
                      f"Customer {account} from the customer's rules does not exist in company code {code}.")
        for account, status in sorted(flags.items()):
            if status.get("posting_blocked") and ("blocked", account) not in reported:
                reported.add(("blocked", account))
                issue(issues, "blocking", "customer_blocked", f"Customer {account} is blocked for posting in S/4.")
            if status.get("deletion_flag") and ("deletion", account) not in reported:
                reported.add(("deletion", account))
                issue(issues, "warning", "customer_marked_for_deletion",
                      f"Customer {account} is marked for deletion in S/4; confirm that it may still receive a payment advice.")


def sent_reason(line: dict[str, Any]) -> str | None:
    """Return the reason code the payload will carry for this line, if any."""
    sends = line["kind"] in ("deduction", "residual") or (line["kind"] == "matched" and line["needs_reason"])
    code = line["reason_code"] or ""
    return code if sends and len(code) == REASON_LENGTH else None


def check_reasons(lines: list[dict[str, Any]], company_code: str | None, reason_codes: ReasonCodes | None,
                  reason_aliases: dict[str, str], issues: list[dict[str, Any]]) -> None:
    """Check every reason code that will be sent against the codes S/4 knows (see module docstring).

    Args:
        lines: Classified advice lines.
        company_code: Company code of the advice; None skips the check.
        reason_codes: `company code -> valid reason codes` (None when S/4 cannot tell); None skips the check.
        reason_aliases: Documented reason code -> S/4 reason code (demo systems only).
        issues: Receives the findings.
    """
    sending = [(line, code) for line in lines if (code := sent_reason(line))]
    if not (reason_codes and company_code and sending):
        return
    valid: dict[str, set[str] | None] = {}

    def codes_of(code: str) -> set[str] | None:
        """Read the reason codes of one company code once; warn when S/4 does not return them."""
        if code not in valid:
            valid[code] = reason_codes(code)
            if valid[code] is None:
                issue(issues, "warning", "reason_codes_not_verified",
                      f"S/4 did not return the reason codes of company code {code}; the codes on this advice were not "
                      "checked there.")
        return valid[code]

    for line, code in sending:
        effective = reason_aliases.get(code, code)
        for where in [company_code, *sorted({item["company_code"] for item in _outside(line, company_code)})]:
            known = codes_of(where)
            if known is not None and effective not in known:
                via = f" (sent as {effective})" if effective != code else ""
                issue(issues, "blocking", "unknown_reason_code",
                      f"{_label(line)}: reason code {code}{via} does not exist in company code {where}.", line["index"])
