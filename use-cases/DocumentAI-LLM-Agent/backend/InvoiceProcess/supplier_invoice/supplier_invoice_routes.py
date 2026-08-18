"""
supplier_invoice_routes.py
--------------------------
FastAPI router for FI Supplier Invoice posting.

POST /api/v1/fi/post-invoice
    Accepts extracted invoice data, auto-matches the vendor BP,
    and posts the document to S/4HANA FI via A_SupplierInvoice.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from InvoiceProcess.supplier_invoice.supplier_invoice_models import (
    PostInvoiceRequest,
    PostInvoiceResponse,
)
from InvoiceProcess.supplier_invoice.supplier_invoice_service import post_supplier_invoice
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fi", tags=["FI Supplier Invoice"])


@router.get(
    "/config",
    summary="Get FI posting configuration defaults",
    description="Returns the default GL account and other FI posting defaults from server settings.",
)
async def get_fi_config() -> dict:
    return {
        "gl_account": settings.FI_EXPENSE_GL_ACCOUNT,
    }


@router.post(
    "/post-invoice",
    response_model=PostInvoiceResponse,
    summary="Post a Supplier Invoice to S/4HANA FI",
    description=(
        "Accepts invoice data extracted from a document and posts it to S/4HANA FI "
        "via the API_SUPPLIERINVOICE_PROCESS_SRV OData API (A_SupplierInvoice).\n\n"
        "**Flow:**\n"
        "1. Resolve Business Partner: uses `business_partner` override if provided, "
        "otherwise auto-matches the `supplier_name` against S/4HANA Business Partners.\n"
        "2. Fetches a CSRF token from S/4HANA.\n"
        "3. POSTs the invoice header + GL account line item in a single deep-insert.\n"
        "4. Returns the created FI document number, company code, and fiscal year.\n\n"
        "**Credentials:** read from X-SAP-* request headers (set by the frontend "
        "sessionStorage after login), falling back to .env settings."
    ),
)
async def post_invoice(
    body: PostInvoiceRequest,
    request: Request,
) -> PostInvoiceResponse:
    logger.info(
        "POST /api/v1/fi/post-invoice | supplier=%r | invoice_no=%s | amount=%s %s",
        body.supplier_name,
        body.invoice_number,
        body.total_amount,
        body.currency,
    )

    try:
        result = post_supplier_invoice(body, request=request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("FI invoice post failed | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error posting FI invoice")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(exc).__name__}: {exc}",
        ) from exc

    return PostInvoiceResponse(**result)
