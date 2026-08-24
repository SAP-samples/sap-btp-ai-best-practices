"""
so_routes.py
------------
FastAPI router for the Sales Order Process module.

Endpoints:
  POST /api/v1/so/extract        — Upload customer PO PDF, extract with SAP Document AI
  POST /api/v1/so/validate       — Validate extracted PO against S/4HANA master data
  POST /api/v1/so/create         — Create Sales Order in S/4HANA
  GET  /api/v1/so/schema-fields  — Hint fields for SAP_purchaseOrder_schema

Imports use flat paths (no "app." prefix) matching the docai backend layout.
"""

from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from modules.auth.get_token import AuthenticationError
from modules.invoice.process_invoice import InvoiceProcessingError

from SalesOrderProcess.so_extractor import SalesOrderExtractor
from SalesOrderProcess.so_models import (
    CreateSORequest,
    CreateSOResponse,
    ExtractedPurchaseOrder,
    SOValidationResult,
)
from SalesOrderProcess.so_validator import validate_purchase_order
from SalesOrderProcess.so_creator import create_sales_order

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/so", tags=["Sales Order Process"])

_SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_upload(upload: UploadFile) -> Path:
    """Save an uploaded file to a temp path and return the path."""
    suffix = Path(upload.filename or "").suffix.lower() or ".pdf"
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: '{suffix}'. "
                f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            ),
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# POST /api/v1/so/extract
# ---------------------------------------------------------------------------


@router.post(
    "/extract",
    response_model=ExtractedPurchaseOrder,
    summary="Extract Purchase Order data from a PDF using SAP Document AI",
)
async def extract_purchase_order(
    file: UploadFile = File(..., description="Customer PO PDF (or JPEG/PNG/TIFF)"),
    client_id: str = Query("default", description="SAP Document AI client ID"),
) -> ExtractedPurchaseOrder:
    """
    Upload a customer Purchase Order PDF and extract structured data using
    SAP Document AI with schema "SAP_purchaseOrder_schema".

    Returns an ExtractedPurchaseOrder with header fields and line items.
    """
    tmp_path = _save_upload(file)
    try:
        extractor = SalesOrderExtractor()
        result = extractor.extract(tmp_path, client_id=client_id)
        logger.info(
            "PO extracted | customer=%r | po=%r | items=%d",
            result.customer_name,
            result.purchase_order_number,
            len(result.line_items),
        )
        return result

    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except InvoiceProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during PO extraction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /api/v1/so/validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=SOValidationResult,
    summary="Validate extracted PO against S/4HANA master data",
)
async def validate_po(
    po: ExtractedPurchaseOrder,
    request: Request,
) -> SOValidationResult:
    """
    Validate an extracted Purchase Order:
      - Match customer name to a SAP Business Partner (score >= 0.6)
      - Match each line item to a SAP material code (score >= 0.5)

    Returns SOValidationResult including readiness flag and any issues.
    """
    try:
        result = validate_purchase_order(po, request=request)
        logger.info(
            "PO validation | ready=%s | issues=%d",
            result.ready_to_create,
            len(result.issues),
        )
        return result
    except Exception as exc:
        logger.exception("Unexpected error during PO validation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /api/v1/so/create
# ---------------------------------------------------------------------------


@router.post(
    "/create",
    response_model=CreateSOResponse,
    summary="Create a Sales Order in S/4HANA",
)
async def create_so(
    data: CreateSORequest,
    request: Request,
) -> CreateSOResponse:
    """
    Create a Sales Order in S/4HANA via API_SALES_ORDER_SRV/A_SalesOrder.

    Flow:
      1. Fetch CSRF token
      2. POST first item via deep-insert
      3. POST each additional item to A_SalesOrderItem
      4. POST special_instructions as header text TX01 (non-fatal)

    Returns CreateSOResponse with the Sales Order number on success.
    """
    try:
        result = create_sales_order(data, request=request)
        if result.success:
            logger.info(
                "Sales Order created | so=%s | items=%d",
                result.sales_order,
                result.items_created,
            )
        else:
            logger.warning("Sales Order creation failed | error=%s", result.error)
        return result
    except Exception as exc:
        logger.exception("Unexpected error creating Sales Order")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /api/v1/so/schema-fields
# ---------------------------------------------------------------------------


@router.get(
    "/schema-fields",
    summary="List field names extracted by SAP_purchaseOrder_schema",
)
def schema_fields() -> dict:
    """
    Return the standard field names that SAP_purchaseOrder_schema typically
    extracts. Useful as UI hints when building a Purchase Order form.
    """
    return {
        "schema": "SAP_purchaseOrder_schema",
        "document_type": "purchaseOrder",
        "header_fields": [
            "customerName",
            "buyerName",
            "shipToName",
            "purchaseOrderNumber",
            "documentNumber",
            "orderDate",
            "documentDate",
            "deliveryDate",
            "requestedDeliveryDate",
            "currencyCode",
            "currency",
            "grossAmount",
            "totalAmount",
            "specialInstructions",
            "notes",
            "shipToAddress",
            "billToAddress",
            "paymentTerms",
            "incoterms",
        ],
        "line_item_fields": [
            "materialNumber",
            "productCode",
            "itemCode",
            "description",
            "productDescription",
            "quantity",
            "qty",
            "unitOfMeasure",
            "unit",
            "unitPrice",
            "price",
            "totalPrice",
            "amount",
            "currencyCode",
            "lineNumber",
        ],
    }
