from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ...security import get_api_key
from ...models.vendor_list import VendorListEntry
from ..services.vendor_list import (
    load_vendor_list,
    find_by_domain,
)


router = APIRouter(
    prefix="/vendor-list",
    tags=["vendor-list-mock"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/", response_model=List[VendorListEntry])
async def list_vendor_entries(
    q: Optional[str] = Query(
        default=None,
        description="Search by supplierName, aribaId, taxId, domain, or email",
    ),
    supplier_name: Optional[str] = Query(default=None, alias="supplierName"),
    domain: Optional[str] = Query(default=None, description="Filter by email domain"),
):
    data = [VendorListEntry(**v) for v in load_vendor_list()]
    if q:
        q_lower = q.lower()
        data = [
            d
            for d in data
            if q_lower in (d.supplierName or "").lower()
            or q_lower in (d.aribaId or "").lower()
            or q_lower in (d.taxId or "").lower()
            or q_lower in (d.commsEmail or "").lower()
            or q_lower in (d.remitEmail or "").lower()
            or q_lower in (d.commsDomain or "").lower()
            or q_lower in (d.remitDomain or "").lower()
        ]
    if supplier_name:
        data = [
            d for d in data if (d.supplierName or "").lower() == supplier_name.lower()
        ]
    if domain:
        entries = [VendorListEntry(**v) for v in find_by_domain(domain)]
        return entries
    return data
