"""S/4 master data the check needs besides the receivables: customer status and reason codes.

Both reads are GET-only and complete (`odata.read_all`).

- `customer_status` answers, for the accounts an advice uses, whether each one
  exists in the company code (`API_BUSINESS_PARTNER/A_CustomerCompany`), whether
  it is blocked for posting (`A_Customer.PostingIsBlocked`) and whether it is
  marked for deletion (`DeletionIndicator` on either level).
- `reason_codes` returns the payment difference reason codes S/4 knows for a
  company code. The source is `FAR_MANAGE_PAYMENT_ADVICES_SRV/I_PaytDifferenceReasonVH`,
  the value help of the Manage Payment Advices app. It is not a released API,
  so it may be inactive or not authorized in a given system: the function then
  returns None and the check reports that the codes were not verified, instead
  of pretending they were.
"""
from __future__ import annotations

from typing import Any

from .client import S4Client, S4HTTPError, odata_str
from .lookup import pad_account
from .odata import chunk_values, or_filter, read_all

BUSINESS_PARTNER = "API_BUSINESS_PARTNER"
REASON_SERVICE = "FAR_MANAGE_PAYMENT_ADVICES_SRV"
REASON_ENTITY = "/I_PaytDifferenceReasonVH"
# HTTP statuses that mean "this system does not offer the reason-code list to us".
UNAVAILABLE = frozenset({400, 401, 403, 404, 501})


def _trace(trace: list[dict[str, Any]] | None, service: str, entity: str, field: str, params: dict[str, Any], rows: int) -> None:
    """Record one S/4 query in the check trace."""
    if trace is not None:
        trace.append({"service": service, "entity": entity, "field": field, "filter": params["$filter"], "rows": rows})


def customer_status(client: S4Client, accounts: list[str], company_code: str,
                    trace: list[dict[str, Any]] | None = None) -> dict[str, dict[str, bool]]:
    """Return existence and blocks of customers in one company code.

    Args:
        client: S/4 client.
        accounts: Customer numbers (any padding).
        company_code: Company code the advice will be created in.
        trace: Optional list receiving one entry per S/4 query.

    Returns:
        `{padded account: {"posting_blocked": bool, "deletion_flag": bool}}` for the
        accounts that exist in the company code; a missing key means "does not exist".
    """
    wanted = sorted({padded for padded in map(pad_account, accounts) if padded})
    status: dict[str, dict[str, bool]] = {}
    for chunk in chunk_values(wanted, "Customer"):
        params = {"$filter": f"CompanyCode eq {odata_str(company_code)} and {or_filter('Customer', chunk)}",
                  "$select": "Customer,CompanyCode,DeletionIndicator"}
        rows = read_all(client, BUSINESS_PARTNER, "/A_CustomerCompany", params, "Customer,CompanyCode")
        _trace(trace, BUSINESS_PARTNER, "/A_CustomerCompany", "Customer", params, len(rows))
        for row in rows:
            status[pad_account(row.get("Customer"))] = {"posting_blocked": False,
                                                        "deletion_flag": bool(row.get("DeletionIndicator"))}
    for chunk in chunk_values(sorted(status), "Customer"):
        params = {"$filter": or_filter("Customer", chunk), "$select": "Customer,PostingIsBlocked,DeletionIndicator"}
        rows = read_all(client, BUSINESS_PARTNER, "/A_Customer", params, "Customer")
        _trace(trace, BUSINESS_PARTNER, "/A_Customer", "Customer", params, len(rows))
        for row in rows:
            entry = status.get(pad_account(row.get("Customer")))
            if entry is not None:
                entry["posting_blocked"] = bool(row.get("PostingIsBlocked"))
                entry["deletion_flag"] = entry["deletion_flag"] or bool(row.get("DeletionIndicator"))
    return status


def reason_codes(client: S4Client, company_code: str, trace: list[dict[str, Any]] | None = None) -> set[str] | None:
    """Return the payment difference reason codes of a company code, or None when S/4 does not offer the list.

    Args:
        client: S/4 client.
        company_code: Company code the advice will be created in.
        trace: Optional list receiving one entry per S/4 query.

    Returns:
        Set of reason codes; None when the value-help service is inactive or not authorized.

    Raises:
        S4HTTPError: For failures that are not "service unavailable" (for example no connection).
    """
    params = {"$filter": f"CompanyCode eq {odata_str(company_code)}", "$select": "CompanyCode,PaymentDifferenceReason"}
    try:
        rows = read_all(client, REASON_SERVICE, REASON_ENTITY, params, "CompanyCode,PaymentDifferenceReason")
    except S4HTTPError as exc:
        if exc.status_code not in UNAVAILABLE:
            raise
        _trace(trace, REASON_SERVICE, REASON_ENTITY, "CompanyCode", params, 0)
        return None
    _trace(trace, REASON_SERVICE, REASON_ENTITY, "CompanyCode", params, len(rows))
    return {str(row.get("PaymentDifferenceReason") or "").strip() for row in rows}
