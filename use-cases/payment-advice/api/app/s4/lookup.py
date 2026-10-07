"""Find the S/4 customer items behind the invoice references quoted on a payment advice.

This reproduces the first thing SAP Cash Application does with an advice line:
take the reference the customer quoted and find the receivable in S/4. The
receivable tells us the company code and customer account, which the advice
itself does not contain.

Source: `API_OPLACCTGDOCITEMCUBE_SRV/A_OperationalAcctgDocItemCube`, restricted
to customer items (`FinancialAccountType eq 'D'`). References are tried against
three fields in order, and only references still unmatched move to the next
field:

1. `DocumentReferenceID` - FI reference (XBLNR), what customers normally quote.
2. `AccountingDocument` - FI document number (BELNR), when they quote that.
3. `BillingDocument` - SD billing document (VBELN).

`matched_by` on every hit records which field matched, which is the evidence
for the open question "accounting document or document reference ID".
Cleared items are returned too (`is_cleared`) so validation can report "already
paid" instead of "not found".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from .client import S4Client, odata_rows, odata_str

SERVICE = "API_OPLACCTGDOCITEMCUBE_SRV"
ENTITY = "/A_OperationalAcctgDocItemCube"
MATCH_FIELDS = ("DocumentReferenceID", "AccountingDocument", "BillingDocument")
MAX_LENGTH = {"DocumentReferenceID": 16, "AccountingDocument": 10, "BillingDocument": 10}
SELECT = ("CompanyCode,FiscalYear,AccountingDocument,AccountingDocumentItem,Customer,DocumentReferenceID,"
          "BillingDocument,AmountInTransactionCurrency,TransactionCurrency,IsCleared,ClearingAccountingDocument,"
          "PostingDate,NetDueDate")
# OR-filters grow the URL; 20 references keep it far below typical 8 KB gateway limits.
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
        amount: Amount in transaction currency (positive for invoices).
        currency: Transaction currency.
        is_cleared: True when the item is already cleared (paid).
        clearing_document: Clearing document number when cleared.
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (amount as string, to keep cents exact)."""
        return {**asdict(self), "amount": str(self.amount)}


def normalize_reference(value: Any) -> str:
    """Normalize an advice reference the way S/4 stores XBLNR: trimmed and upper case."""
    return str(value or "").strip().upper()


def _item(reference: str, field: str, row: dict[str, Any]) -> ReceivableItem:
    """Convert one cube row into a ReceivableItem (customer padded to 10 digits like the rule accounts)."""
    clearing = str(row.get("ClearingAccountingDocument") or "").strip()
    return ReceivableItem(
        reference=reference, matched_by=field, company_code=row["CompanyCode"], fiscal_year=row["FiscalYear"],
        accounting_document=row["AccountingDocument"], accounting_document_item=row.get("AccountingDocumentItem", ""),
        # S/4 may return "10053628" (some systems do) where the rules give "0010053628": compare one form.
        customer=pad_account(row.get("Customer")) or "", document_reference=row.get("DocumentReferenceID", ""),
        amount=Decimal(str(row.get("AmountInTransactionCurrency") or "0")), currency=row.get("TransactionCurrency", ""),
        is_cleared=bool(row.get("IsCleared")) or bool(clearing), clearing_document=clearing,
    )


def find_receivables(client: S4Client, references: list[str], company_code: str | None = None,
                     trace: list[dict[str, Any]] | None = None) -> dict[str, list[ReceivableItem]]:
    """Look up every reference in S/4 customer items.

    Args:
        client: S/4 client.
        references: Advice references (duplicates and blanks are ignored).
        company_code: Optional restriction to one company code.
        trace: Optional list that receives one `{service, entity, field, filter, rows}`
            entry per S/4 call (shown in the UI as the process walkthrough).

    Returns:
        Mapping of normalized reference to its S/4 items. Unmatched references map
        to an empty list; more than one item means the reference is ambiguous.
    """
    wanted = list(dict.fromkeys(r for r in map(normalize_reference, references) if r))
    found: dict[str, list[ReceivableItem]] = {ref: [] for ref in wanted}
    seen: set[tuple[str, ...]] = set()
    for field in MATCH_FIELDS:
        # Document numbers are numeric and stored zero-padded to 10 digits; XBLNR is free text.
        numeric = field != "DocumentReferenceID"
        pending = {(ref.zfill(10) if numeric else ref): ref for ref in wanted
                   if not found[ref] and len(ref) <= MAX_LENGTH[field] and (ref.isdigit() or not numeric)}
        values = list(pending)
        for start in range(0, len(values), CHUNK):
            chunk = values[start:start + CHUNK]
            clauses = ["FinancialAccountType eq 'D'", "(" + " or ".join(f"{field} eq {odata_str(v)}" for v in chunk) + ")"]
            if company_code:
                clauses.append(f"CompanyCode eq {odata_str(company_code)}")
            params = {"$filter": " and ".join(clauses), "$select": SELECT, "$top": "500"}
            rows = odata_rows(client.get_json(SERVICE, ENTITY, params))
            if trace is not None:
                trace.append({"service": SERVICE, "entity": ENTITY, "field": field, "filter": params["$filter"], "rows": len(rows)})
            for row in rows:
                key = (row["CompanyCode"], row["FiscalYear"], row["AccountingDocument"], row.get("AccountingDocumentItem", ""))
                reference = pending.get(normalize_reference(row.get(field)))
                if key in seen or reference is None:
                    continue  # the cube can repeat a document per ledger
                seen.add(key)
                found[reference].append(_item(reference, field, row))
    return found


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
