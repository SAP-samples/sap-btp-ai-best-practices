from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ...security import get_api_key
from ...models.s4 import Invoice
from ..services.s4 import load_seed, load_records


router = APIRouter(
    prefix="/s4",
    tags=["s4-mock"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/invoices", response_model=List[Invoice])
async def list_invoices(
    vendor_id: Optional[str] = Query(default=None, alias="vendorId"),
    po_id: Optional[str] = Query(default=None, alias="poId"),
    status: Optional[str] = None,
):
    data = load_seed()
    invs = [Invoice(**inv) for inv in data.get("invoices", [])]
    if vendor_id:
        invs = [i for i in invs if i.vendorId == vendor_id]
    if po_id:
        invs = [i for i in invs if i.poId == po_id]
    if status:
        invs = [i for i in invs if i.status.lower() == status.lower()]
    return invs


@router.get("/invoices/{invoice_id}", response_model=Invoice)
async def get_invoice(invoice_id: str):
    data = load_seed()
    for inv in data.get("invoices", []):
        if inv.get("id") == invoice_id:
            return Invoice(**inv)
    raise HTTPException(status_code=404, detail="Invoice not found")


@router.get("/records", response_model=List[Dict[str, Any]])
async def list_raw_records(
    q: Optional[str] = Query(
        default=None,
        description="Generic search across Supplier Name/ID, Journal Entry, and Purchasing Doc List",
    ),
    supplier_id: Optional[str] = Query(default=None, alias="supplierId"),
    supplier_name: Optional[str] = Query(default=None, alias="supplierName"),
    journal_entry: Optional[str] = Query(default=None, alias="journalEntry"),
    purchasing_doc: Optional[str] = Query(default=None, alias="purchasingDocList"),
    invoice_number: Optional[str] = Query(default=None, alias="invoiceNumber"),
):
    rows = load_records()
    data = rows
    if q:
        q_lower = q.lower()
        data = [
            r
            for r in data
            if q_lower in (str(r.get("Supplier Name") or "").lower())
            or q_lower in (str(r.get("Supplier") or "").lower())
            or q_lower in (str(r.get("Journal Entry") or "").lower())
            or q_lower in (str(r.get("Purchasing Doc List") or "").lower())
            or q_lower in (str(r.get("Reference") or "").lower())
        ]
    if supplier_id:
        data = [r for r in data if str(r.get("Supplier") or "") == supplier_id]
    if supplier_name:
        data = [
            r
            for r in data
            if str(r.get("Supplier Name") or "").lower() == supplier_name.lower()
        ]
    if journal_entry:
        data = [r for r in data if str(r.get("Journal Entry") or "") == journal_entry]
    if purchasing_doc:
        data = [
            r for r in data if str(r.get("Purchasing Doc List") or "") == purchasing_doc
        ]
    if invoice_number:
        inv = str(invoice_number)
        data = [
            r
            for r in data
            if inv == str(r.get("Reference") or "").strip()
            or inv == str(r.get("Journal Entry") or "").strip()
            or inv == str(r.get("Assignment") or "").strip()
        ]
    # Enrich with normalized convenience fields expected by the agent
    enriched: List[Dict[str, Any]] = []
    for r in data:
        rr: Dict[str, Any] = dict(r)
        rr["invoiceNumber"] = (
            str(r.get("Reference") or r.get("Journal Entry") or "").strip() or None
        )
        rr["vendorId"] = str(r.get("Supplier") or "").strip() or None
        rr["vendorName"] = str(r.get("Supplier Name") or "").strip() or None
        rr["poId"] = str(r.get("Purchasing Doc List") or "").strip() or None
        enriched.append(rr)
    return enriched
