from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ...security import get_api_key
from ...models.ariba import AribaInvoice
from ..services.ariba import load_invoices


router = APIRouter(
    prefix="/ariba",
    tags=["ariba-mock"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/invoices", response_model=List[AribaInvoice])
async def list_invoices(
    q: Optional[str] = Query(
        default=None, description="Generic search across supplier and invoice number"
    ),
    supplier: Optional[str] = Query(default=None, description="Filter by supplier"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    requester: Optional[str] = Query(default=None, description="Filter by requester"),
    matched_order_id: Optional[str] = Query(
        default=None, alias="matchedOrderId", description="Filter by matched order ID"
    ),
):
    data = [AribaInvoice(**inv) for inv in load_invoices()]
    if q:
        q_lower = q.lower()
        data = [
            d
            for d in data
            if q_lower in (d.supplier or "").lower()
            or q_lower in (d.invoiceNumber or "").lower()
        ]
    if supplier:
        data = [d for d in data if (d.supplier or "").lower() == supplier.lower()]
    if status:
        data = [d for d in data if (d.status or "").lower() == status.lower()]
    if requester:
        data = [d for d in data if (d.requester or "").lower() == requester.lower()]
    if matched_order_id:
        data = [d for d in data if (d.matchedOrderId or "") == matched_order_id]
    return data


@router.get("/invoices/{invoice_number}", response_model=AribaInvoice)
async def get_invoice(invoice_number: str):
    invoices = load_invoices()
    for inv in invoices:
        if (inv.get("invoiceNumber") or "").strip() == invoice_number:
            return AribaInvoice(**inv)
    # Fallback by ID if invoiceNumber doesn't match
    for inv in invoices:
        if (inv.get("id") or "").strip() == invoice_number:
            return AribaInvoice(**inv)
    raise HTTPException(status_code=404, detail="Invoice not found")
