"""Collect the S/4 candidates of an advice: the S/4 reads of the check, in order.

This module holds the input/output side of matching; the decisions are pure and
live in `resolution` and `validation`. `collect_candidates` runs three reads:

1. References. Every line reference is looked up in the customer items, field
   by field (`lookup.find_receivables`). A reference goes on to the next field
   while it has no payable item in the advice's scope, where the scope is the
   company code and the customer accounts the customer's rules give.
2. Variants. A reference of an invoice-like line that S/4 does not know at all
   is tried again in other spellings (`references.reference_variants`). A hit
   is kept as a candidate marked with the spelling that matched; validation
   applies it only when the amount confirms it.
3. Linked items. For every candidate document, the items that refer to it
   (partial payments, residual items) are read and turned into open balances
   (`resolution.apply_linked`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from .amounts import parse_amount
from .client import S4Client
from .lookup import ReceivableItem, find_linked_items, find_receivables, pad_account
from .references import line_reference, reference_variants
from .resolution import apply_linked, payable_in_scope
from .validation import DEDUCTION_NATURES


def rule_company_code(header: dict[str, Any], aliases: dict[str, str]) -> str | None:
    """Return the company code the customer's rules give, mapped to the S/4 code (None when there is none)."""
    code = str(header.get("company_code") or "").strip().upper() or None
    return aliases.get(code, code) if code else None


def _invoice_like(row: dict[str, Any]) -> bool:
    """Return True for a line that should apply an S/4 document: an invoice, or a positive line that is no deduction."""
    nature = row.get("document_nature") or "unknown"
    net = parse_amount(row.get("net_amount"))
    return nature == "invoice" or (nature not in DEDUCTION_NATURES and net is not None and net > 0)


def _add_variants(client: S4Client, found: dict[str, list[ReceivableItem]], references: set[str],
                  trace: list[dict[str, Any]] | None) -> None:
    """Look unknown references up in other spellings and add the hits as candidates of the original reference."""
    wanted: dict[str, list[str]] = defaultdict(list)
    for reference in sorted(references):
        if not found.get(reference):
            for _, variant in reference_variants(reference):
                wanted[variant].append(reference)
    if not wanted:
        return
    for variant, items in find_receivables(client, list(wanted), trace=trace).items():
        for reference in wanted.get(variant, []):
            known = {item.key for item in found[reference]}
            found[reference] += [replace(item, reference=reference, variant=variant) for item in items if item.key not in known]


def collect_candidates(client: S4Client, result: dict[str, Any], aliases: dict[str, str],
                       trace: list[dict[str, Any]] | None = None) -> dict[str, list[ReceivableItem]]:
    """Read every S/4 candidate for the lines of one advice (see module docstring).

    Args:
        client: S/4 client.
        result: Advice result (`header` + `line_items`); excluded rows are skipped.
        aliases: Rule company code -> S/4 company code (demo systems only).
        trace: Optional list receiving one entry per S/4 query.

    Returns:
        Candidates keyed by normalized reference, each with its open amount;
        the input `validation.check_advice` expects.
    """
    header = result.get("header") or {}
    company_code = rule_company_code(header, aliases)
    accounts: dict[str, set[str]] = {}
    invoice_like: set[str] = set()
    for row in result.get("line_items") or []:
        reference, _ = line_reference(row)
        if row.get("excluded") or not reference:
            continue
        accounts.setdefault(reference, set())
        if account := pad_account(row.get("customer_account")):
            accounts[reference].add(account)
        if _invoice_like(row):
            invoice_like.add(reference)
    found = find_receivables(
        client, list(accounts), trace=trace,
        needs_more=lambda reference, items: not payable_in_scope(items, company_code, accounts.get(reference, set())))
    _add_variants(client, found, invoice_like, trace)
    linked = find_linked_items(client, [item for items in found.values() for item in items], trace)
    return apply_linked(found, linked)
