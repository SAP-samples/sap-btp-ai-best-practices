"""Exact source selection and explicit conversion into the integrated optimizer schema."""

from decimal import Decimal, InvalidOperation
import pandas as pd
from ...models.workspace import WorkspaceValidationError


def choose_candidates(rows, scope):
    """Return eligible rows in source order; reject invalid or ambiguous selection IDs."""
    if scope.mode == "all_eligible":
        if scope.row_ids:
            raise WorkspaceValidationError("all_eligible requires empty row_ids")
        return [row for row in rows if row["eligible"]]
    wanted = set(scope.row_ids)
    if not wanted or len(wanted) != len(scope.row_ids):
        raise WorkspaceValidationError("Select a nonempty list of unique eligible source rows")
    eligible_ids = {row["row_id"] for row in rows if row["eligible"]}
    if invalid := wanted - eligible_ids:
        raise WorkspaceValidationError("Unknown or ineligible source rows", row_ids=sorted(invalid))
    return [row for row in rows if row["row_id"] in wanted]


def canonical_candidates(analysis, rows, settings):
    """Map current rows into solver fields using saved facility/FX settings only."""
    output = []
    mapping = settings.get("seller_to_facility", {})
    rates = settings.get("currency_rates", {})
    for row in rows:
        invoice = row["invoice"]
        facility = mapping.get(invoice["seller_id"])
        if not facility:
            raise WorkspaceValidationError("Map this seller to a facility", "seller_to_facility", [row["row_id"]])
        if not invoice.get("debtor_id"):
            raise WorkspaceValidationError("Customer ID is required", "debtor_id", [row["row_id"]])
        currency = invoice["original_currency"]
        raw_amount = invoice.get("total_net_value_original")
        if raw_amount is None:
            raw_amount = invoice.get("amount_original")
        try:
            amount = Decimal(str(raw_amount))
            rate = Decimal("1") if currency == "EUR" else Decimal(str(rates[currency]["eur_per_unit"]))
            if not amount.is_finite() or not rate.is_finite() or rate <= 0:
                raise ValueError("Nonfinite amount or nonpositive exchange rate")
        except (InvalidOperation, KeyError, ValueError):
            raise WorkspaceValidationError("Provide a valid amount and explicit currency rate", "currency_rates", [row["row_id"]]) from None
        normalized = (amount * rate).quantize(Decimal("0.01"))
        output.append({"row_id": row["row_id"], "Company Code": str(facility),
                       "Customer": invoice["debtor_id"], "Customer Name": invoice["debtor_name"],
                       "Program": invoice["programa"], "PROGRAMA": invoice["programa"], "Seller": invoice["seller_id"],
                       "Invoice Reference": invoice["invoice_ref"], "optimizer_row_id": row["row_id"],
                       "Fiscal Year": invoice.get("fiscal_year"), "Funding Currency": "EUR",
                       "Document Number": invoice.get("doc_number") or invoice["invoice_ref"],
                       "Reference": invoice["invoice_ref"], "Original Currency": currency,
                       "Original Amount": str(amount), "Normalized Amount": str(normalized),
                       "Currency": "EUR", "Purchase Price": float(normalized),
                       "Amount": float(normalized), "Issuance Date": pd.Timestamp(invoice["issuance_date"]),
                       "Issuance date": pd.Timestamp(invoice["issuance_date"]),
                       "Due Date": pd.Timestamp(invoice["due_date"]),
                       "Offer File Date (UTC)": pd.Timestamp(invoice.get("offer_date") or analysis["analysis_date"]),
                       "Summary File Date": pd.Timestamp(analysis["analysis_date"])})
    return pd.DataFrame(output)
