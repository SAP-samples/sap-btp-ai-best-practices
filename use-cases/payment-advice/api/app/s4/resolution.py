"""Decide which S/4 document an advice line means when S/4 returns several candidates.

Pure functions: no S/4 or HANA access. Input is what `lookup` found for one
reference plus what the advice line and the customer's rules say; output is a
`Resolution` that names the chosen item(s), or says why no choice was made.

A reference is rarely unique in a real system. The same document number exists
in several company codes and fiscal years, an invoice and its credit memo share
one reference, a cleared invoice lives on as a residual item. The rules below
narrow the candidates in a fixed order. A rule is applied only when it leaves
at least one candidate, and every rule that removed candidates is recorded in
`Resolution.steps`, so the reviewer sees why a document was chosen.

1. Reviewer's pick (`s4_document` on the row) wins; it must still be a candidate.
2. Scope - who and where: the rule customer account (also as branch of a head
   office), then the company code (the one the rules give, else the one the
   other lines of the advice agree on), then the debit/credit side that fits
   the sign of the line.
3. State - inside that scope: live before reversed, open before cleared,
   normal items before special G/L items.
4. Field priority: among what is left, the reference field (XBLNR) before the
   document number (BELNR) before the billing document (VBELN).
5. Amount: the candidate whose open amount, then whose original amount, equals
   the advice gross amount.

Scope comes before state on purpose: our own cleared invoice must be reported
as "already cleared", not silently replaced by another customer's open document
that happens to carry the same reference.

The company code is a preference, not an identity: one advice may pay
documents of several company codes. It therefore decides only against
documents the customer could not mean. When the same customer has a payable
document of the right side in another company code, the company code step is
skipped and the later rules decide. If the document chosen that way lies
outside the preferred company code, the amount must confirm it, and the choice
carries a warning; otherwise the reviewer chooses.

Three situations never resolve without a second piece of evidence, the amount:
a document found only under a lower-priority field while a higher one had a
hit, a reference found only after normalizing its spelling, and a choice
between candidates decided by nothing but the amount. They resolve with a
warning when the amount agrees and stay open for the reviewer when it does not.
A choice also carries a warning when the reference is shared with another
customer and no rule names the customer, or with another company code and
neither a rule nor the other lines name the company code.
Nothing is ever chosen by recency, fiscal year or list order.

`apply_linked` runs before resolution: it turns the partial payments and
residual items read by `lookup.find_linked_items` into open balances.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from .amounts import sign_of
from .lookup import MATCH_FIELDS, ReceivableItem
from .references import parse_pick

FIELD_RANK = {name: rank for rank, name in enumerate(MATCH_FIELDS)}
FIELD_LABEL = {"DocumentReferenceID": "reference", "AccountingDocument": "document number",
               "BillingDocument": "billing document"}
# FollowOnDocumentType values of items that refer to an invoice.
PARTIAL_PAYMENT = "Z"
RESIDUAL_TYPES = frozenset({"V", "P"})  # residual item, payment difference
# Candidates stored on a line for the reviewer's picker; a longer list is not usable on screen.
MAX_CANDIDATES = 20

Group = list[ReceivableItem]


@dataclass(frozen=True)
class LineContext:
    """What one advice line and the customer's rules say about the document it means.

    Args:
        gross: Advice gross amount; None when the line gives none.
        sign: 1 for a positive line, -1 for a negative one, 0 when unknown.
        company_code: Company code from the customer's rules (after alias), if any.
        account: Customer account from the customer's rules (10 digits), if any.
        pick: Reviewer's choice `COMPANYCODE/DOCUMENT/YEAR[/ITEM]`, if any.
        plain: True when the line has no discount and no deduction (net equals gross).
        tolerance: Largest difference at which two amounts count as equal.
    """

    gross: Decimal | None = None
    sign: int = 0
    company_code: str | None = None
    account: str | None = None
    pick: str | None = None
    plain: bool = True
    tolerance: Decimal = Decimal("0.01")


@dataclass
class Resolution:
    """Outcome of resolving one advice line.

    Args:
        status: `none` (nothing found), `resolved` (items chosen), `ambiguous`
            (several candidates left), `suggested` (found only under another
            spelling, amount does not confirm it), `stale_pick` (the reviewer's
            pick is no longer a candidate) or `multi_item` (document with several
            open items and no item fits the amount).
        items: Chosen S/4 items; several only when one line pays all items of a document.
        candidates: Items the reviewer can choose from when nothing was chosen.
        steps: `[{rule, kept, dropped}]` for every rule that removed candidates.
        method: `unique` (one candidate from the start), `auto` (narrowed by rules) or `reviewer`.
        warning: Text when the choice rests on weaker evidence and must be confirmed.
        item_targeted: True when the document has several customer items, so the item number is sent.
    """

    status: str
    items: list[ReceivableItem] = field(default_factory=list)
    candidates: list[ReceivableItem] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    method: str | None = None
    warning: str | None = None
    item_targeted: bool = False


def open_balance(item: ReceivableItem) -> Decimal:
    """Return what is still open on an item: its open amount, else nothing when cleared, else its amount."""
    if item.open_amount is not None:
        return item.open_amount
    return Decimal("0") if item.is_cleared or item.is_reversed else item.amount


def is_open(item: ReceivableItem) -> bool:
    """Return True when the item can still be paid (not cleared, not reversed)."""
    return not item.is_cleared and not item.is_reversed


def item_sign(item: ReceivableItem) -> int:
    """Return 1 for a debit item (invoice), -1 for a credit item, 0 when S/4 gave neither amount nor side."""
    return sign_of(item.amount) or {"S": 1, "H": -1}.get(item.debit_credit, 0)


def _refers_to(link: ReceivableItem, item: ReceivableItem) -> bool:
    """Return True when ``link`` is a different document that refers to ``item`` through its invoice reference."""
    return (link.company_code == item.company_code and link.invoice_reference == item.accounting_document
            and link.invoice_reference_year in ("", item.fiscal_year)
            and link.invoice_item_reference in ("", item.accounting_document_item)
            and (link.accounting_document, link.fiscal_year) != (item.accounting_document, item.fiscal_year))


def apply_linked(found: dict[str, list[ReceivableItem]], linked: list[ReceivableItem]) -> dict[str, list[ReceivableItem]]:
    """Put open balances on the found items and replace cleared invoices by their open residual items.

    - An open invoice with open partial payments (follow-on type `Z`) is open for
      its amount minus those payments.
    - A cleared invoice with open residual items (type `V` or `P`, same side as
      the invoice) is replaced by them: that is what the customer still owes.
    - Any other linked item (for example a credit memo with invoice reference) is
      left alone. It is its own open item and is paid by its own advice line.
    - A residual item can also be found directly, because it inherits the billing
      document number of its invoice. It is then marked as the residual of that
      invoice too, and when it was reached both ways it is kept once.

    Args:
        found: `lookup.find_receivables` output.
        linked: `lookup.find_linked_items` output for the same items.

    Returns:
        New mapping with the same references; every item carries `open_amount`.
    """
    enriched: dict[str, list[ReceivableItem]] = {}
    for reference, items in found.items():
        result: list[ReceivableItem] = []
        for item in items:
            links = [link for link in linked if _refers_to(link, item)]
            residuals = [link for link in links if link.follow_on_type in RESIDUAL_TYPES and is_open(link)
                         and item_sign(link) == item_sign(item)]
            if item.is_cleared and not item.is_reversed and residuals:
                origin = f"{item.accounting_document}/{item.fiscal_year}"
                result += [replace(link, reference=item.reference, matched_by=item.matched_by, variant=item.variant,
                                   residual_of=origin, open_amount=link.amount) for link in residuals]
                continue
            partial = sum((link.amount for link in links if link.follow_on_type == PARTIAL_PAYMENT and is_open(link)),
                          Decimal("0"))
            balance = item.amount + partial if is_open(item) else Decimal("0")
            # An item that is itself a residual item says so, whichever way it was found.
            origin = (f"{item.invoice_reference}/{item.invoice_reference_year}"
                      if item.follow_on_type in RESIDUAL_TYPES and item.invoice_reference else item.residual_of)
            result.append(replace(item, open_amount=balance, partial_payments=partial if is_open(item) else Decimal("0"),
                                  residual_of=origin))
        # One S/4 item is one candidate. The first form found is kept: the lookup returns higher-priority fields
        # first, so that is the one reached through the field the customer most likely quoted.
        unique: dict[tuple[str, ...], ReceivableItem] = {}
        for candidate in result:
            unique.setdefault(candidate.key, candidate)
        enriched[reference] = list(unique.values())
    return enriched


def _groups(items: list[ReceivableItem]) -> list[Group]:
    """Group items by document (company code, fiscal year, document number), keeping their order."""
    documents: dict[tuple[str, str, str], Group] = {}
    for item in items:
        documents.setdefault(item.key[:3], []).append(item)
    return list(documents.values())


def _rank(group: Group) -> int:
    """Return the priority of the field a document was found by (0 is the reference field)."""
    return min(FIELD_RANK.get(item.matched_by, len(FIELD_RANK)) for item in group)


def _narrow(groups: list[Group], steps: list[dict[str, Any]], rule: str, keep) -> list[Group]:
    """Keep the documents that satisfy ``keep`` unless none does; record the rule when it removed some."""
    kept = [group for group in groups if keep(group)]
    if not kept or len(kept) == len(groups):
        return groups
    steps.append({"rule": rule, "kept": len(kept), "dropped": len(groups) - len(kept)})
    return kept


def _amount_matches(group: Group, context: LineContext, balance) -> bool:
    """Return True when the document total or one of its items equals the advice gross (by ``balance``)."""
    if context.gross is None:
        return False
    amounts = [balance(item) for item in group]
    return any(abs(amount - context.gross) <= context.tolerance for amount in [sum(amounts, Decimal("0")), *amounts])


def _confirmed_by_amount(group: Group, context: LineContext) -> bool:
    """Return True when the open or the original amount of the document confirms the advice gross."""
    payable = [item for item in group if is_open(item)] or group
    return _amount_matches(payable, context, open_balance) or _amount_matches(group, context, lambda item: item.amount)


def _capped(groups: list[Group]) -> list[ReceivableItem]:
    """Flatten documents into the candidate list shown to the reviewer (at most MAX_CANDIDATES items)."""
    return [item for group in groups for item in group][:MAX_CANDIDATES]


def _names_item(group: Group) -> bool:
    """Return True when the payload must name the item of the document, not only the document.

    That is the case for a document with several customer items, and for a residual or other follow-on
    item: it sits on a clearing document that holds further customer items the lookup did not return.
    """
    return len(group) > 1 or any(item.residual_of or item.invoice_reference for item in group)


def _pick(groups: list[Group], context: LineContext, steps: list[dict[str, Any]]) -> Resolution:
    """Apply the reviewer's pick: it must name a document (and item) that is still a candidate."""
    pick = parse_pick(context.pick)
    chosen = [group for group in groups if pick and group[0].key[:3] ==
              (pick["company_code"], pick["fiscal_year"], pick["accounting_document"])]
    items = [item for item in (chosen[0] if chosen else []) if not pick["item"] or item.accounting_document_item == pick["item"]]
    if not items:
        return Resolution("stale_pick", candidates=_capped(groups), steps=steps)
    steps.append({"rule": "reviewer_pick", "kept": 1, "dropped": len(groups) - 1})
    if pick["item"]:
        return Resolution("resolved", items=items, steps=steps, method="reviewer", item_targeted=_names_item(chosen[0]))
    return _select_items(chosen[0], context, steps, "reviewer", None)


def _select_items(group: Group, context: LineContext, steps: list[dict[str, Any]], method: str,
                  warning: str | None) -> Resolution:
    """Choose the item(s) of the resolved document that the line pays.

    A document with one customer item is that item. With several (for example
    instalments): the single open one; else the single open item whose open
    amount equals the gross; else all open items when the line pays their total
    without discount or deduction; else the reviewer chooses the item.
    """
    targeted = _names_item(group)
    payable = [item for item in group if is_open(item)]
    if len(payable) <= 1:
        return Resolution("resolved", items=payable or group[:1], steps=steps, method=method, warning=warning,
                          item_targeted=targeted)
    if context.gross is not None:
        exact = [item for item in payable if abs(open_balance(item) - context.gross) <= context.tolerance]
        if len(exact) == 1:
            steps.append({"rule": "item_amount", "kept": 1, "dropped": len(payable) - 1})
            return Resolution("resolved", items=exact, steps=steps, method=method, warning=warning, item_targeted=True)
        total = sum((open_balance(item) for item in payable), Decimal("0"))
        if context.plain and abs(total - context.gross) <= context.tolerance:
            steps.append({"rule": "document_total", "kept": len(payable), "dropped": 0})
            return Resolution("resolved", items=payable, steps=steps, method=method, warning=warning, item_targeted=True)
    return Resolution("multi_item", candidates=payable[:MAX_CANDIDATES], steps=steps)


def _payable_elsewhere(groups: list[Group], preferred: str, context: LineContext) -> bool:
    """Return True when a customer of the preferred company code has a payable document in another one.

    Such a document is one the line may really mean, so the company code must
    not remove it. Payable means: an open, normal item on the side that fits
    the sign of the line.

    Args:
        groups: Candidate documents, already narrowed to the rule customer.
        preferred: Company code of the rules, else the one the other lines agree on.
        context: What the line says (for the sign).
    """
    customers = {group[0].customer for group in groups if group[0].company_code == preferred}
    return any(group[0].company_code != preferred and group[0].customer in customers
               and any(is_open(item) and not item.special_gl for item in group)
               and (not context.sign or item_sign(group[0]) in (0, context.sign))
               for group in groups)


def resolve(candidates: list[ReceivableItem], context: LineContext, consensus: str | None = None) -> Resolution:
    """Resolve one advice line against its S/4 candidates (see the module docstring for the rules).

    Args:
        candidates: Items found for the line's reference, after `apply_linked`.
        context: What the line and the customer's rules say.
        consensus: Company code the uniquely matched lines of the advice agree
            on; used only when the rules give no company code.

    Returns:
        The Resolution; `status == "resolved"` is the only one with chosen items.
    """
    if not candidates:
        return Resolution("none")
    everything = _groups(candidates)
    steps: list[dict[str, Any]] = []
    if context.pick:
        return _pick(everything, context, steps)

    # Scope: who and where. The customer comes first: it identifies the document, the company code only prefers one.
    groups = everything
    if context.account:
        groups = _narrow(groups, steps, "customer", lambda g: context.account in (g[0].customer, g[0].branch_account))
    preferred = context.company_code or consensus
    rival = bool(preferred) and _payable_elsewhere(groups, preferred, context)
    if preferred and not rival:
        groups = _narrow(groups, steps, "company_code" if context.company_code else "advice_company_code",
                         lambda g: g[0].company_code == preferred)
    if context.sign:
        groups = _narrow(groups, steps, "sign", lambda g: item_sign(g[0]) in (0, context.sign))
    scope = groups

    # State: what can still be paid, inside that scope.
    groups = _narrow(groups, steps, "not_reversed", lambda g: not all(item.is_reversed for item in g))
    groups = _narrow(groups, steps, "open", lambda g: any(is_open(item) for item in g))
    groups = _narrow(groups, steps, "normal_item", lambda g: not all(item.special_gl for item in g))

    # Field priority, then the amount.
    best = min(_rank(group) for group in groups)
    groups = _narrow(groups, steps, "field_priority", lambda g: _rank(g) == best)
    by_amount = False
    if len(groups) > 1 and context.gross is not None:
        narrowed = _narrow(groups, steps, "open_amount", lambda g: _amount_matches(
            [item for item in g if is_open(item)] or g, context, open_balance))
        if len(narrowed) == len(groups):
            narrowed = _narrow(groups, steps, "original_amount", lambda g: _amount_matches(g, context, lambda i: i.amount))
        by_amount, groups = len(narrowed) == 1, narrowed
    if len(groups) > 1:
        return Resolution("ambiguous", candidates=_capped(groups), steps=steps)

    chosen = groups[0]
    confirmed = _confirmed_by_amount(chosen, context)
    warning = "it is the only candidate whose amount equals the advice amount" if by_amount else None
    # Found under a lower field although a higher field had a hit: only the amount can justify the switch.
    if any(_rank(group) < _rank(chosen) for group in everything):
        if not confirmed:
            return Resolution("ambiguous", candidates=_capped(scope if len(scope) > 1 else everything), steps=steps)
        warning = (f"it was found by {FIELD_LABEL.get(chosen[0].matched_by, chosen[0].matched_by)} because the document "
                   "with this reference cannot be paid, and its amount equals the advice amount")
    # The same customer has this number in the preferred company code too: outside it, only the amount can decide.
    if rival and chosen[0].company_code != preferred:
        if not confirmed:
            return Resolution("ambiguous", candidates=_capped(scope), steps=steps)
        warning = (f"the customer has a document with this number in company code {preferred} as well, and this one "
                   "is open with the advice amount")
    # Found only under another spelling of the reference: the amount must confirm it.
    if chosen[0].variant:
        if not confirmed:
            return Resolution("suggested", candidates=_capped([chosen]), steps=steps)
        warning = f"the reference was read as '{chosen[0].variant}' and the amount equals the advice amount"
    # The reference is shared with a document that nothing excludes: one in another company code while no rule
    # (and no other line) names the company code, or one of another customer while no rule names the customer.
    def excluded(group: Group) -> bool:
        """Return True when a rule explains why ``group`` is not the document the line means."""
        other_code = group[0].company_code != chosen[0].company_code
        other_customer = group[0].customer != chosen[0].customer
        return ((other_code and bool(context.company_code or consensus)) or (other_customer and bool(context.account))
                or not (other_code or other_customer))

    if warning is None and any(not excluded(group) for group in everything if group is not chosen):
        warning = "other customers or company codes use the same reference and no customer rule confirms this one"
    method = "unique" if len(everything) == 1 else "auto"
    return _select_items(chosen, context, steps, method, warning)


def payable_in_scope(items: list[ReceivableItem], company_code: str | None, accounts: set[str]) -> bool:
    """Return True when the items already contain one that can be paid in the advice's scope.

    This is the lookup's test for "this reference needs no further field": an
    open, live, normal item in the rule company code (when there is one) and for
    one of the rule customer accounts of the lines quoting it (when there are any).

    Args:
        items: Items found so far for one reference.
        company_code: Company code from the customer's rules (after alias), if any.
        accounts: Rule customer accounts of the advice lines that quote the reference.
    """
    return any(is_open(item) and not item.special_gl
               and (not company_code or item.company_code == company_code)
               and (not accounts or item.customer in accounts or item.branch_account in accounts)
               for item in items)
