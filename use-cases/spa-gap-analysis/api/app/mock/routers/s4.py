from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...security import get_api_key
from ...models.s4 import BusinessPartner, PurchaseOrder, Invoice
from ..services.s4 import load_seed


router = APIRouter(
    prefix="/s4",
    tags=["s4-mock"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/business-partners", response_model=List[BusinessPartner])
async def list_business_partners(
    q: Optional[str] = Query(default=None, description="Search by name substring")
):
    data = load_seed()
    partners = [BusinessPartner(**bp) for bp in data.get("businessPartners", [])]
    if q:
        q_lower = q.lower()
        partners = [p for p in partners if q_lower in p.name.lower()]
    return partners


@router.get("/business-partners/{bp_id}", response_model=BusinessPartner)
async def get_business_partner(bp_id: str):
    data = load_seed()
    for bp in data.get("businessPartners", []):
        if bp.get("id") == bp_id:
            return BusinessPartner(**bp)
    raise HTTPException(status_code=404, detail="Business Partner not found")


@router.get("/purchase-orders", response_model=List[PurchaseOrder])
async def list_purchase_orders(
    vendor_id: Optional[str] = Query(default=None, alias="vendorId"),
    status: Optional[str] = None,
):
    data = load_seed()
    pos = [PurchaseOrder(**po) for po in data.get("purchaseOrders", [])]
    if vendor_id:
        pos = [p for p in pos if p.vendorId == vendor_id]
    if status:
        pos = [p for p in pos if p.status.lower() == status.lower()]
    return pos


@router.get("/purchase-orders/{po_id}", response_model=PurchaseOrder)
async def get_purchase_order(po_id: str):
    data = load_seed()
    for po in data.get("purchaseOrders", []):
        if po.get("id") == po_id:
            return PurchaseOrder(**po)
    raise HTTPException(status_code=404, detail="Purchase Order not found")


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
