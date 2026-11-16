"""Complete reads and URL-sized filters for S/4 OData V2 collections.

Two things every S/4 lookup of this application needs, kept in one place:

- `read_all` reads a collection to the end. It pages with `$top` / `$skip` on a
  stable `$orderby` and asks S/4 for the total (`$inlinecount=allpages`). When
  S/4 reports more rows than were received, it raises `S4IncompleteRead`
  instead of returning a partial list: a truncated answer would make an
  existing invoice look "not found".
- `chunk_values` and `or_filter` split a long list of values into several
  `field eq 'a' or field eq 'b'` filters that each stay below a URL-length
  budget, because OData V2 GET requests carry the filter in the URL.

Example:
    for chunk in chunk_values(references, "DocumentReferenceID"):
        rows = read_all(client, SERVICE, ENTITY,
                        {"$filter": or_filter("DocumentReferenceID", chunk), "$select": SELECT},
                        order_by="CompanyCode,FiscalYear,AccountingDocument,AccountingDocumentItem")
"""
from __future__ import annotations

from typing import Any

from .client import S4Client, S4HTTPError, odata_rows, odata_str

# Rows requested per page. S/4 gateways cap pages well above this value.
PAGE_SIZE = 1000
# Characters of OR terms per request. URL encoding grows the text by roughly a third, and the
# `$select` list is added on top, which keeps the request line far below the usual 8 KB limit.
FILTER_BUDGET = 3000
# Upper bound of OR terms per request, whatever their length.
MAX_TERMS = 80


class S4IncompleteRead(S4HTTPError):
    """S/4 reported more rows than the application received for one query.

    Args:
        target: `<service><entity>` that was read.
        received: Number of rows received.
        expected: Number of rows S/4 reported in `__count`.
    """

    def __init__(self, target: str, received: int, expected: int) -> None:
        super().__init__("GET", target, 0, f"S/4 returned {received} of {expected} rows; the read is incomplete")
        self.received, self.expected = received, expected


def or_filter(field: str, values: list[str]) -> str:
    """Return `(field eq 'a' or field eq 'b' ...)` for the given values (quoted as OData literals)."""
    return "(" + " or ".join(f"{field} eq {odata_str(value)}" for value in values) + ")"


def chunk_values(values: list[str], field: str, budget: int = FILTER_BUDGET, max_terms: int = MAX_TERMS) -> list[list[str]]:
    """Split ``values`` into chunks whose OR filter on ``field`` stays below ``budget`` characters.

    Args:
        values: Values to search for (order is kept).
        field: OData property the values are compared with.
        budget: Maximum characters of OR terms per chunk.
        max_terms: Maximum number of values per chunk.

    Returns:
        List of chunks; every chunk holds at least one value.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for value in values:
        cost = len(f"{field} eq {odata_str(value)} or ")
        if current and (used + cost > budget or len(current) >= max_terms):
            chunks.append(current)
            current, used = [], 0
        current.append(value)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def read_all(client: S4Client, service: str, entity: str, params: dict[str, Any], order_by: str,
             page_size: int = PAGE_SIZE) -> list[dict[str, Any]]:
    """Read every row of an OData V2 collection that matches ``params``.

    Args:
        client: S/4 client.
        service: OData service name, e.g. `API_OPLACCTGDOCITEMCUBE_SRV`.
        entity: Entity set path, e.g. `/A_OperationalAcctgDocItemCube`.
        params: Query options (`$filter`, `$select`); paging options are added here.
        order_by: Comma-separated key properties; a stable order is what makes `$skip` safe.
        page_size: Rows requested per call.

    Returns:
        All matching rows.

    Raises:
        S4IncompleteRead: S/4 reported more rows (`__count`) than were received.
        S4HTTPError: Any failed call.
    """
    rows: list[dict[str, Any]] = []
    expected: int | None = None
    while True:
        page = {**params, "$top": str(page_size), "$orderby": order_by, "$inlinecount": "allpages"}
        if rows:
            page["$skip"] = str(len(rows))
        payload = client.get_json(service, entity, page)
        received = odata_rows(payload)
        rows += received
        count = (payload.get("d") or {}).get("__count")
        if count not in (None, ""):
            expected = int(count)
        # A page shorter than requested is the last one; services without `__count` rely on this alone.
        if len(received) < page_size:
            break
    if expected is not None and len(rows) < expected:
        raise S4IncompleteRead(f"{service}{entity}", len(rows), expected)
    return rows
