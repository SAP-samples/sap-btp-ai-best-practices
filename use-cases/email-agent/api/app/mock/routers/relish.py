from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from ...security import get_api_key
from ...models.relish import RelishRecord
from ..services.relish import load_records


router = APIRouter(
    prefix="/relish",
    tags=["relish-mock"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/records", response_model=List[RelishRecord])
async def list_records(
    q: Optional[str] = Query(
        default=None,
        description="Generic search across supplierName, invoiceNumber, relishId, and uuid",
    ),
    supplier_id: Optional[str] = Query(default=None, alias="supplierId"),
    supplier_name: Optional[str] = Query(default=None, alias="supplierName"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    inv_status: Optional[str] = Query(default=None, alias="invStatus"),
    company_code: Optional[str] = Query(default=None, alias="companyCode"),
    invoice_number: Optional[str] = Query(default=None, alias="invoiceNumber"),
    payment_method: Optional[str] = Query(default=None, alias="paymentMethod"),
):
    data = [RelishRecord(**rec) for rec in load_records()]
    if q:
        q_lower = q.lower()
        data = [
            d
            for d in data
            if q_lower in (d.supplierName or "").lower()
            or q_lower in (d.invoiceNumber or "").lower()
            or q_lower in (d.relishId or "").lower()
            or q_lower in (d.uuid or "").lower()
        ]
    if supplier_id:
        data = [d for d in data if (d.supplierId or "") == supplier_id]
    if supplier_name:
        data = [
            d for d in data if (d.supplierName or "").lower() == supplier_name.lower()
        ]
    if status:
        data = [d for d in data if (d.status or "").lower() == status.lower()]
    if inv_status:
        data = [d for d in data if (d.invStatus or "").lower() == inv_status.lower()]
    if company_code:
        data = [d for d in data if (d.companyCode or "") == company_code]
    if invoice_number:
        data = [d for d in data if (d.invoiceNumber or "") == invoice_number]
    if payment_method:
        data = [
            d for d in data if (d.paymentMethod or "").lower() == payment_method.lower()
        ]
    return data


@router.get("/records/{identifier}", response_model=RelishRecord)
async def get_record(identifier: str):
    """
    Retrieve a Relish record by invoiceNumber, relishId, or uuid (first match wins).
    """
    records = load_records()
    for rec in records:
        if (rec.get("invoiceNumber") or "").strip() == identifier:
            return RelishRecord(**rec)
    for rec in records:
        if (rec.get("RelishId") or "").strip() == identifier or (
            rec.get("relishId") or ""
        ).strip() == identifier:
            return RelishRecord(**rec)
    for rec in records:
        if (rec.get("UUID") or "").strip() == identifier or (
            rec.get("uuid") or ""
        ).strip() == identifier:
            return RelishRecord(**rec)
    raise HTTPException(status_code=404, detail="Record not found")
