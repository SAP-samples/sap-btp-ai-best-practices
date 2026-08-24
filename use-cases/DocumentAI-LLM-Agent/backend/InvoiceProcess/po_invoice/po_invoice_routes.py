"""
po_invoice_routes.py
--------------------
FastAPI router for PO-based Supplier Invoice posting (MIRO equivalent).

POST /api/v1/fi/post-po-invoice
    Accepts extracted invoice data + PO number, auto-matches the vendor BP,
    and posts the document to S/4HANA FI via A_SupplierInvoice with
    to_SuplrInvcItemPurOrdRef deep-create node.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from InvoiceProcess.po_invoice.po_invoice_models import PostPOInvoiceRequest, PostPOInvoiceResponse
from InvoiceProcess.po_invoice.po_invoice_service import post_po_invoice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fi", tags=["FI PO Invoice"])


@router.post(
    "/post-po-invoice",
    response_model=PostPOInvoiceResponse,
    summary="Post a PO-based Supplier Invoice to S/4HANA FI (MIRO equivalent)",
    description=(
        "Posts a vendor invoice referencing a Purchase Order to S/4HANA FI "
        "via the API_SUPPLIERINVOICE_PROCESS_SRV OData API.\n\n"
        "**Replicates MIRO (MM-IV)** using a deep-create POST to A_SupplierInvoice "
        "with the PO line item reference in the nested `to_SuplrInvcItemPurOrdRef` node.\n\n"
        "**Flow:**\n"
        "1. Detect PO number in the extracted invoice data.\n"
        "2. Resolve Business Partner: uses `business_partner` override if provided, "
        "otherwise auto-matches `supplier_name` against S/4HANA Business Partners.\n"
        "3. Fetch CSRF token.\n"
        "4. POST deep-create: invoice header + PO reference line item.\n"
        "5. Return FI document number, company code, and fiscal year.\n\n"
        "**Credentials:** X-SAP-* request headers (set by frontend sessionStorage) "
        "or .env fallback."
    ),
)
async def post_po_invoice_endpoint(
    body: PostPOInvoiceRequest,
    request: Request,
) -> PostPOInvoiceResponse:
    logger.info(
        "POST /api/v1/fi/post-po-invoice | supplier=%r | po=%s | invoice_no=%s | amount=%s %s",
        body.supplier_name,
        body.purchase_order,
        body.invoice_number,
        body.total_amount,
        body.currency,
    )

    try:
        result = post_po_invoice(body, request=request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("PO invoice post failed | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error posting PO FI invoice")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(exc).__name__}: {exc}",
        ) from exc

    return PostPOInvoiceResponse(**result)
