"""Find the S/4 customer items behind the invoice references quoted on a payment advice.

This reproduces the first thing SAP Cash Application does with an advice line:
take the reference the customer quoted and find the receivable in S/4. The
receivable tells us the company code and customer account, which the advice
itself does not contain.

Source: `API_OPLACCTGDOCITEMCUBE_SRV/A_OperationalAcctgDocItemCube`, restricted
to customer items (`FinancialAccountType eq 'D'`). References are tried against
three fields in order:

1. `DocumentReferenceID` - FI reference (XBLNR), what customers normally quote.
2. `AccountingDocument` - FI document number (BELNR), when they quote that.
3. `BillingDocument` - SD billing document (VBELN).

A reference moves to the next field while the caller still needs more for it.
By default that means "no hit yet"; the S/4 check passes a stricter test ("no
payable item in the advice's company code and customer yet"), so a cleared or
foreign hit in the reference field does not hide the open document that carries
the same number in another field.

`find_receivables` returns EVERY item found, never a choice. Choosing between
several candidates is the job of `app.s4.resolution`. Three facts are settled
here because no later step can change them:

- Reversal documents are skipped: S/4 copies the invoice's XBLNR onto the
  reversal, which is never a payable item.
- When a reversed invoice was posted again under the same reference, only the
  live document is returned.
- Every query is read to its end (`odata.read_all`); an incomplete answer
  raises instead of reporting "not found".
- Numbers are brought to one form. The OData service returns document numbers
  without leading zeros ("100000014", billing document "90000000", item "1")
  and writes "no value" as "0". Every item carries the stored width instead
  (10 digits for documents, 3 for items, "" for none), so a number compares
  equal wherever it comes from: the advice, the reviewer's pick or S/4.

`find_linked_items` reads the items that point at a found document through
`InvoiceReference` (partial payments, residual items); `resolution.apply_linked`
turns them into open balances.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from .client import S4Client, odata_rows, odata_str
from .odata import chunk_values, or_filter, read_all
from .references import is_document_number, normalize_reference, pad_document_number

SERVICE = "API_OPLACCTGDOCITEMCUBE_SRV"
ENTITY = "/A_OperationalAcctgDocItemCube"
MATCH_FIELDS = ("DocumentReferenceID", "AccountingDocument", "BillingDocument")
MAX_LENGTH = {"DocumentReferenceID": 16, "AccountingDocument": 10, "BillingDocument": 10}
# `Ledger` is selected so that the cube returns one row per ledger instead of summing the ledgers
# of one item; the rows are then collapsed by item key.
SELECT = ("CompanyCode,FiscalYear,AccountingDocument,AccountingDocumentItem,Ledger,Customer,BranchAccount,"
          "DocumentReferenceID,BillingDocument,AccountingDocumentType,DebitCreditCode,SpecialGLCode,"
          "AmountInTransactionCurrency,TransactionCurrency,IsCleared,ClearingAccountingDocument,"
          "PostingDate,NetDueDate,IsReversal,IsReversed,ReverseDocument,"
          "InvoiceReference,InvoiceReferenceFiscalYear,InvoiceItemReference,FollowOnDocumentType")
# The entity key: a stable order is what makes paging safe.
ORDER_BY = "CompanyCode,FiscalYear,AccountingDocument,AccountingDocumentItem"
# Kept for callers that size their own filters (20 customer numbers per request).
CHUNK = 20


@dataclass(frozen=True)
class ReceivableItem:
    """One customer line item in S/4 that an advice reference points to.

    Args:
        reference: Advice reference that matched (normalized).
        matched_by: S/4 field the reference matched (see MATCH_FIELDS).
        company_code: S/4 company code.
        fiscal_year: Fiscal year of the accounting document.
        accounting_document: FI document number.
        accounting_document_item: Line item within the document.
        customer: Customer account (10 digits).
        document_reference: XBLNR stored in S/4.
        amount: Amount in transaction currency (positive for invoices, negative for credits).
        currency: Transaction currency.
        is_cleared: True when the item is already cleared (paid).
        clearing_document: Clearing document number when cleared.
        is_reversed: True when the document was reversed in S/4 (no longer payable).
        reverse_document: Reversal document number when reversed.
        debit_credit: `S` (debit) or `H` (credit) as posted.
        special_gl: Special G/L indicator; non-empty for down payments and similar items.
        document_type: Accounting document type.
        branch_account: Branch customer when the item is posted on its head office.
        billing_document: SD billing document on the item.
        invoice_reference: Document this item refers to (partial payment, residual item).
        invoice_reference_year: Fiscal year of that document.
        invoice_item_reference: Item of that document.
        follow_on_type: `Z` partial payment, `V` residual item, `P` payment difference, `G` credit memo.
        open_amount: Amount still open after partial payments; None until `resolution.apply_linked` ran.
        partial_payments: Sum of the open partial payments netted into `open_amount` (zero or negative).
        residual_of: `DOCUMENT/YEAR` of the cleared invoice this residual item stands in for.
        variant: Spelling of the reference that matched when it was not the exact one.
    """

    reference: str
    matched_by: str
    company_code: str
    fiscal_year: str
    accounting_document: str
    accounting_document_item: str
    customer: str
    document_reference: str
    amount: Decimal
    currency: str
    is_cleared: bool
    clearing_document: str
    is_reversed: bool = False
    reverse_document: str = ""
    debit_credit: str = ""
    special_gl: str = ""
    document_type: str = ""
    branch_account: str = ""
    billing_document: str = ""
    invoice_reference: str = ""
    invoice_reference_year: str = ""
    invoice_item_reference: str = ""
    follow_on_type: str = ""
    open_amount: Decimal | None = None
    partial_payments: Decimal = Decimal("0")
    residual_of: str = ""
    variant: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        """The S/4 item key `(company code, fiscal year, document, item)`."""
        return (self.company_code, self.fiscal_year, self.accounting_document, self.accounting_document_item)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (amounts as strings, to keep cents exact)."""
        data = asdict(self)
        data.update(amount=str(self.amount), partial_payments=str(self.partial_payments),
                    open_amount=None if self.open_amount is None else str(self.open_amount))
        return data


def _text(row: dict[str, Any], name: str) -> str:
    """Return one row value as stripped text ("" when absent)."""
    return str(row.get(name) or "").strip()


def _number(value: Any, width: int) -> str:
    """Return an S/4 number in its stored width.

    Args:
        value: Value as the OData service returns it (may lack leading zeros, "0" means none).
        width: Stored width of the field (10 for document numbers, 4 for years, 3 for items).

    Returns:
        Zero-padded digits; "" for an empty or all-zero value; any non-numeric text unchanged.
    """
    text = str(value or "").strip()
    if not (text.isascii() and text.isdigit()):
        return text
    return text.zfill(width) if text.strip("0") else ""


def _item(reference: str, field: str, row: dict[str, Any]) -> ReceivableItem:
    """Convert one cube row into a ReceivableItem (customer padded to 10 digits like the rule accounts)."""
    clearing = _number(row.get("ClearingAccountingDocument"), 10)
    return ReceivableItem(
        reference=reference, matched_by=field, company_code=row["CompanyCode"], fiscal_year=row["FiscalYear"],
        accounting_document=_number(row["AccountingDocument"], 10),
        accounting_document_item=_number(row.get("AccountingDocumentItem"), 3),
        # S/4 may return "10053628" (some systems do) where the rules give "0010053628": compare one form.
        customer=pad_account(row.get("Customer")) or "", document_reference=row.get("DocumentReferenceID", ""),
        amount=Decimal(str(row.get("AmountInTransactionCurrency") or "0")), currency=row.get("TransactionCurrency", ""),
        is_cleared=bool(row.get("IsCleared")) or bool(clearing), clearing_document=clearing,
        is_reversed=bool(row.get("IsReversed")), reverse_document=_number(row.get("ReverseDocument"), 10),
        debit_credit=_text(row, "DebitCreditCode"), special_gl=_text(row, "SpecialGLCode"),
        document_type=_text(row, "AccountingDocumentType"), branch_account=pad_account(row.get("BranchAccount")) or "",
        billing_document=_number(row.get("BillingDocument"), 10),
        invoice_reference=_number(row.get("InvoiceReference"), 10),
        invoice_reference_year=_number(row.get("InvoiceReferenceFiscalYear"), 4),
        invoice_item_reference=_number(row.get("InvoiceItemReference"), 3), follow_on_type=_text(row, "FollowOnDocumentType"),
    )


def _read(client: S4Client, field: str, values: list[str], company_code: str | None,
          trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Read every customer item whose ``field`` equals one of ``values`` (complete read, traced)."""
    clauses = ["FinancialAccountType eq 'D'", or_filter(field, values)]
    if company_code:
        clauses.append(f"CompanyCode eq {odata_str(company_code)}")
    params = {"$filter": " and ".join(clauses), "$select": SELECT}
    rows = read_all(client, SERVICE, ENTITY, params, ORDER_BY)
    if trace is not None:
        trace.append({"service": SERVICE, "entity": ENTITY, "field": field, "filter": params["$filter"], "rows": len(rows)})
    return rows


def find_receivables(client: S4Client, references: list[str], company_code: str | None = None,
                     trace: list[dict[str, Any]] | None = None,
                     needs_more: Callable[[str, list[ReceivableItem]], bool] | None = None,
                     ) -> dict[str, list[ReceivableItem]]:
    """Look up every reference in S/4 customer items.

    Args:
        client: S/4 client.
        references: Advice references (duplicates and blanks are ignored).
        company_code: Optional restriction to one company code.
        trace: Optional list that receives one `{service, entity, field, filter, rows}`
            entry per S/4 query (shown in the UI as the process walkthrough).
        needs_more: `(reference, items found so far) -> bool`; True sends the
            reference to the next field. Default: only while nothing was found.

    Returns:
        Mapping of normalized reference to all its S/4 items. Unmatched references
        map to an empty list; more than one item means the reference needs resolution.
    """
    wanted = list(dict.fromkeys(r for r in map(normalize_reference, references) if r))
    found: dict[str, list[ReceivableItem]] = {ref: [] for ref in wanted}
    # Per reference: the cube can repeat an item per ledger, but two references may share one item.
    seen: dict[str, set[tuple[str, ...]]] = {ref: set() for ref in wanted}
    more = needs_more or (lambda reference, items: not items)
    for field in MATCH_FIELDS:
        # Document numbers are numeric and stored zero-padded to 10 digits; XBLNR is free text.
        numeric = field != "DocumentReferenceID"
        # Two references can share one search value ("123" and "0000000123"): keep them all.
        pending: dict[str, list[str]] = {}
        for ref in wanted:
            if len(ref) > MAX_LENGTH[field] or (numeric and not is_document_number(ref)) or not more(ref, found[ref]):
                continue
            pending.setdefault(pad_document_number(ref) if numeric else ref, []).append(ref)
        for chunk in chunk_values(list(pending), field):
            for row in _read(client, field, chunk, company_code, trace):
                if row.get("IsReversal"):
                    continue  # a reversal document is never payable
                key = (row["CompanyCode"], row["FiscalYear"], row["AccountingDocument"], row.get("AccountingDocumentItem", ""))
                # S/4 returns "100000014" for the document searched as "0100000014": compare in the stored width.
                value = _number(row.get(field), 10) if numeric else normalize_reference(row.get(field))
                for reference in pending.get(value, []):
                    if key not in seen[reference]:
                        seen[reference].add(key)
                        found[reference].append(_item(reference, field, row))
    for reference, items in found.items():
        # An invoice reversed and posted again keeps its reference: the live document wins.
        if any(i.is_reversed for i in items) and not all(i.is_reversed for i in items):
            found[reference] = [i for i in items if not i.is_reversed]
    return found


def find_linked_items(client: S4Client, items: list[ReceivableItem],
                      trace: list[dict[str, Any]] | None = None) -> list[ReceivableItem]:
    """Read the customer items that refer to the given documents through `InvoiceReference`.

    These are the partial payments and residual items of an invoice. They are
    needed to know what is really still open: an invoice with a partial payment
    is open for less than its amount, and a cleared invoice can live on as an
    open residual item.

    Args:
        client: S/4 client.
        items: Items found by `find_receivables`.
        trace: Optional list receiving one entry per S/4 query.

    Returns:
        Linked items (reversal documents left out), each once.
    """
    documents = sorted({item.accounting_document for item in items})
    linked: dict[tuple[str, ...], ReceivableItem] = {}
    for chunk in chunk_values(documents, "InvoiceReference"):
        for row in _read(client, "InvoiceReference", chunk, None, trace):
            if not row.get("IsReversal"):
                item = _item("", "InvoiceReference", row)
                linked.setdefault(item.key, item)
    return list(linked.values())


def pad_account(value: Any) -> str | None:
    """Normalize a customer account to S/4's 10-character form ("10053628" -> "0010053628")."""
    text = str(value or "").strip()
    return (text.zfill(10) if text.isdigit() else text) or None


def existing_customers(client: S4Client, accounts: list[str], company_code: str,
                       trace: list[dict[str, Any]] | None = None) -> set[str]:
    """Return which of ``accounts`` exist as customers in ``company_code`` (padded numbers).

    Reads ``API_BUSINESS_PARTNER/A_CustomerCompany`` (customer company-code data);
    a customer without data in the company code cannot receive a payment advice there.

    Args:
        client: S/4 client.
        accounts: Customer numbers (any padding).
        company_code: Company code the advice will be created in.
        trace: Optional list receiving one entry per S/4 call.

    Returns:
        Set of padded customer numbers found.
    """
    wanted = sorted({padded for padded in map(pad_account, accounts) if padded})
    found: set[str] = set()
    for start in range(0, len(wanted), CHUNK):
        chunk = wanted[start:start + CHUNK]
        params = {"$filter": f"CompanyCode eq {odata_str(company_code)} and ("
                             + " or ".join(f"Customer eq {odata_str(c)}" for c in chunk) + ")",
                  "$select": "Customer,CompanyCode", "$top": "500"}
        rows = odata_rows(client.get_json("API_BUSINESS_PARTNER", "/A_CustomerCompany", params))
        if trace is not None:
            trace.append({"service": "API_BUSINESS_PARTNER", "entity": "/A_CustomerCompany", "field": "Customer",
                          "filter": params["$filter"], "rows": len(rows)})
        found.update(pad_account(row.get("Customer")) for row in rows)
    return found
