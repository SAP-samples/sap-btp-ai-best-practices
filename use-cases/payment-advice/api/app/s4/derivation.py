"""Derive the header of the S/4 payment advice: company code, advice account, customers, currency.

Pure function, no S/4 or HANA access. The remittance itself states none of
these values, so they are combined from two sources:

- S/4 says where each invoice is: every matched receivable carries its company
  code and currency.
- The customer's rules give the payer (`payer_account`), the company code that
  receives the payment (mapped through ``aliases``, e.g. a documented code to
  the code of a demo system) and the account of deduction lines, which exist
  only on the remittance.

Company code of the advice (the header; the company code that receives the
payment):

- All matched items in one company code: that company code. A different rule
  company code blocks (`company_code_mismatch`).
- Matched items in several company codes, one of them the rule company code:
  the rule company code. The items elsewhere are cross-company-code items; the
  advice carries a warning that lists them (`cross_company_items`).
- Several company codes and no rule company code: blocks
  (`multiple_company_codes`). Which company code receives the money is a fact
  about the bank account; the application does not guess it.
- Several company codes, none of them the rule company code: blocks
  (`company_code_mismatch`), because nothing in S/4 confirms the rule.

The currency is the one currency of the matched items; several block.

Advice account: the rule payer when there is one, else the customer with the
largest matched amount in the advice company code. Two customers with exactly
the same largest amount block (`payer_ambiguous`): the application does not
pick one by account number.

The checks that depend on the company code of each item (customer master,
reason codes, unconfirmed cross-company-code lines) are in `company_codes`.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from .lookup import pad_account


def issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, line: int | None = None) -> None:
    """Append one validation finding (`severity` is `blocking` or `warning`)."""
    issues.append({"severity": severity, "code": code, "line": line, "message": message})


def matched_items(line: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the S/4 items a matched line applies (several when it pays all items of a document)."""
    return line.get("s4_items") or [line["s4_item"]]


def line_ranges(indexes: list[int], limit: int = 8) -> str:
    """Name advice lines compactly for a message.

    Args:
        indexes: Zero-based line indexes, any order, duplicates allowed.
        limit: Largest number of ranges written out.

    Returns:
        One-based ranges such as `1-3, 7, 9-12`; beyond ``limit`` ranges the
        rest is summarised as `... (N lines in total)`.
    """
    numbers = sorted({index + 1 for index in indexes})
    ranges: list[list[int]] = []
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    text = ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges[:limit])
    return text if len(ranges) <= limit else f"{text}, ... ({len(numbers)} lines in total)"


def lines_by_company_code(lines: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Return `{company code: [line indexes]}` of the matched lines, by the company code of their S/4 items."""
    by_code: dict[str, list[int]] = defaultdict(list)
    for line in lines:
        if line["kind"] == "matched":
            for code in sorted({item["company_code"] for item in matched_items(line)}):
                by_code[code].append(line["index"])
    return dict(by_code)


def _spread(by_code: dict[str, list[int]], codes: list[str]) -> str:
    """Describe which lines are in which company code: `1000 (lines 1-3), 2000 (lines 4, 6)`."""
    return ", ".join(f"{code} (lines {line_ranges(by_code[code])})" for code in codes)


def _advice_company_code(by_code: dict[str, list[int]], rule_code: str | None, rule_code_s4: str | None,
                         issues: list[dict[str, Any]]) -> str | None:
    """Choose the company code of the advice from the matched items and the rules (see module docstring).

    Args:
        by_code: `lines_by_company_code` output; not empty.
        rule_code: Company code as the rules give it.
        rule_code_s4: The same after alias mapping; None when the rules give none.
        issues: Receives the findings.

    Returns:
        The company code, or None when it cannot be decided.
    """
    codes = sorted(by_code)
    via = f" (alias of {rule_code})" if rule_code_s4 and rule_code_s4 != rule_code else ""
    if len(codes) == 1:
        if rule_code_s4 and rule_code_s4 != codes[0]:
            issue(issues, "blocking", "company_code_mismatch",
                  f"The customer rules give company code {rule_code_s4}{via} but the S/4 invoices are in {codes[0]}.")
        return codes[0]
    if not rule_code_s4:
        issue(issues, "blocking", "multiple_company_codes",
              f"Matched items span company codes {_spread(by_code, codes)} and the customer's rules name none. "
              "Set the company code that receives the payment on the advice header.")
        return None
    if rule_code_s4 not in codes:
        issue(issues, "blocking", "company_code_mismatch",
              f"The customer rules give company code {rule_code_s4}{via} but no matched S/4 item is in it; the items "
              f"are in {_spread(by_code, codes)}.")
        return None
    others = [code for code in codes if code != rule_code_s4]
    issue(issues, "warning", "cross_company_items",
          f"The advice is created in company code {rule_code_s4}{via} and also pays items in "
          f"{_spread(by_code, others)}. S/4 has to clear them across company codes; confirm that this is intended.")
    return rule_code_s4


def derive(lines: list[dict[str, Any]], header: dict[str, Any], issues: list[dict[str, Any]],
           aliases: dict[str, str]) -> dict[str, Any]:
    """Derive company code, header account, all customers and currency (see module docstring).

    Args:
        lines: Classified advice lines (`validation._classify` output).
        header: Advice header.
        issues: Receives the findings.
        aliases: Rule company code -> S/4 company code (demo systems only).

    Returns:
        `{company_code, customer, customers, currency, payer_account, rule_company_code}`.
    """
    matched = [item for line in lines if line["kind"] == "matched" for item in matched_items(line)]
    rule_code = str(header.get("company_code") or "").strip().upper() or None
    rule_code_s4 = aliases.get(rule_code, rule_code) if rule_code else None
    payer = pad_account(header.get("payer_account"))
    advice_currency = str(header.get("payment_currency") or "").strip().upper() or None
    derived: dict[str, Any] = {"company_code": None, "customer": None, "customers": [], "currency": None,
                               "payer_account": payer, "rule_company_code": rule_code}
    everywhere: dict[str, Decimal] = defaultdict(Decimal)  # matched amount per customer, all company codes
    at_home: dict[str, Decimal] = defaultdict(Decimal)     # the same, items in the advice company code only
    if matched:
        company_code = _advice_company_code(lines_by_company_code(lines), rule_code, rule_code_s4, issues)
        currencies = sorted({item["currency"] for item in matched})
        for item in matched:
            everywhere[item["customer"]] += Decimal(item["amount"])
            if item["company_code"] == company_code:
                at_home[item["customer"]] += Decimal(item["amount"])
        derived.update(company_code=company_code, currency=currencies[0] if len(currencies) == 1 else None)
        if len(currencies) > 1:
            issue(issues, "blocking", "multiple_currencies", f"Matched items use currencies {', '.join(currencies)}.")
        if advice_currency and derived["currency"] and advice_currency != derived["currency"]:
            issue(issues, "blocking", "currency_mismatch",
                  f"Advice currency {advice_currency} differs from S/4 currency {derived['currency']}.")
    elif rule_code_s4 and payer:
        # A residual line's S/4 document (e.g. the reversed invoice) still tells the currency.
        known = sorted({line["s4_item"]["currency"] for line in lines if line["kind"] == "residual" and line["s4_item"]})
        currency = advice_currency or (known[0] if len(known) == 1 else None)
        derived.update(company_code=rule_code_s4, currency=currency)
        issue(issues, "warning", "derived_from_rules",
              f"No line matched an S/4 invoice; company code {rule_code_s4} and payer {payer} come from the customer's rules.")
        if not currency:
            issue(issues, "blocking", "missing_currency", "No S/4 invoice matched and the advice has no currency.")
    else:
        issue(issues, "blocking", "no_invoice_resolved",
              "No line resolved to an S/4 customer item and the customer's rules give no company code and payer, "
              "so the advice cannot be assigned.")
        return derived
    # The header account must exist in the advice company code, so the lead payer is taken from the items there.
    totals = at_home or everywhere
    lead = sorted(totals, key=lambda c: (-totals[c], c))
    if not payer and len(lead) > 1 and totals[lead[0]] == totals[lead[1]]:
        issue(issues, "blocking", "payer_ambiguous",
              f"Customers {lead[0]} and {lead[1]} have the same matched amount and the customer's rules name no payer; "
              "set the payer account on the advice header.")
    derived["customer"] = payer or (lead[0] if lead else None)
    others = sorted(everywhere, key=lambda c: (-everywhere[c], c))
    line_accounts = [line["account"] for line in lines if line["account"]]
    derived["customers"] = list(dict.fromkeys([c for c in [derived["customer"], *lead, *others, *line_accounts] if c]))
    if len(derived["customers"]) > 1:
        issue(issues, "warning", "multiple_customers",
              f"Items belong to customers {', '.join(derived['customers'])}; {derived['customer']} is the advice "
              "account and the others are sent as alternative accounts (lead payer).")
    return derived
