"""
purchase_orders_routes.py
-------------------------
FastAPI routes for Purchase Order search.

GET /api/purchase-orders?supplier=<bp>&top=20
    Returns open POs for a given vendor/supplier BP number.

GET /api/purchase-orders/search?q=<vendor_name>&top=10
    Resolves a vendor name to BP first, then returns their POs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from InvoiceProcess.purchase_orders.purchase_orders_service import search_purchase_orders

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Purchase Orders"])


class PurchaseOrder(BaseModel):
    purchase_order: str
    supplier: str
    company_code: str
    purchasing_organization: str
    purchasing_group: str
    document_date: str | None
    currency: str
    status: str
    supplier_name: str


class PurchaseOrdersResponse(BaseModel):
    success: bool
    supplier: str
    count: int
    purchase_orders: list[PurchaseOrder]


@router.get(
    "/purchase-orders",
    response_model=PurchaseOrdersResponse,
    summary="Search Purchase Orders by vendor/supplier BP number",
    description=(
        "Returns open Purchase Orders for a given supplier (Business Partner number). "
        "Tries OData V4 (API_PURCHASEORDER_2) first for S/4HANA 2023+, "
        "falls back to V2 (API_PURCHASEORDER_PROCESS_SRV) automatically.\n\n"
        "**Credentials:** X-SAP-* request headers or .env fallback."
    ),
)
async def list_purchase_orders(
    request: Request,
    supplier: str = Query(..., min_length=1, description="Supplier/vendor BP number"),
    top: int = Query(default=20, ge=1, le=200, description="Max POs to return"),
) -> PurchaseOrdersResponse:
    logger.info("GET /api/purchase-orders | supplier=%s | top=%d", supplier, top)
    try:
        pos = await search_purchase_orders(supplier=supplier, top=top, request=request)
    except RuntimeError as exc:
        logger.error("PO search failed | supplier=%s | error=%s", supplier, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in PO search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(exc).__name__}",
        ) from exc

    return PurchaseOrdersResponse(
        success=True,
        supplier=supplier,
        count=len(pos),
        purchase_orders=[PurchaseOrder(**po) for po in pos],
    )
