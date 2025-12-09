from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from ..security import get_api_key
from ..models.common import ErrorResponse
from ..services.kg_service import KGService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)], tags=["kg"])

kg_service = KGService()


@router.get("/v1/kg/static/list")
async def list_suppliers() -> Dict[str, Any]:
    """List statically packaged suppliers.

    Returns a list of suppliers with id, name, and filename metadata.
    """
    return kg_service.list_static_suppliers()


@router.get("/v1/kg/static/get/{supplier_id}")
async def get_supplier_kg(supplier_id: str) -> Dict[str, Any] | ErrorResponse:
    """Return KG JSON for a given supplier id packaged with the API.

    Args:
        supplier_id: Supplier identifier (e.g., "supplier1").
    """
    try:
        return kg_service.load_static_supplier_kg(supplier_id)
    except FileNotFoundError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load KG for {supplier_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load supplier KG")


@router.post("/v1/kg/create")
async def create_kg(
    supplier_name: Optional[str] = Form(default=None),
    supplier_files: Optional[List[UploadFile]] = File(default=None),
    supplier1_name: Optional[str] = Form(default=None),
    supplier1_files: Optional[List[UploadFile]] = File(default=None),
    supplier2_name: Optional[str] = Form(default=None),
    supplier2_files: Optional[List[UploadFile]] = File(default=None),
) -> Dict[str, Any]:
    """Create per-file KGs for one or more suppliers using image-mode.

    Accepts multipart/form-data with either a single supplier (`supplier_name`,
    `supplier_files`) or the legacy dual-supplier fields (`supplier1_*`,
    `supplier2_*`). Returns identifiers and directories for each processed supplier.
    """
    try:
        suppliers_payload: List[Dict[str, Any]] = []

        if supplier_name:
            suppliers_payload.append({"name": supplier_name, "files": supplier_files or []})
        else:
            if supplier1_name:
                suppliers_payload.append({"name": supplier1_name, "files": supplier1_files or []})
            if supplier2_name:
                suppliers_payload.append({"name": supplier2_name, "files": supplier2_files or []})

        if not suppliers_payload:
            raise HTTPException(status_code=422, detail="At least one supplier name must be provided.")

        return kg_service.create_from_uploads(suppliers_payload)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid KG creation request: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"KG creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UnifyRequest(BaseModel):
    # Either a single generic supplier_id, or legacy supplier1_id/supplier2_id
    supplier_id: Optional[str] = None
    supplier1_id: Optional[str] = None
    supplier2_id: Optional[str] = None
    # Names are optional and currently unused by unification; kept for compatibility
    supplier1_name: Optional[str] = None
    supplier2_name: Optional[str] = None


@router.post("/v1/kg/unify")
async def unify_kg(req: UnifyRequest) -> Dict[str, Any]:
    """Unify per-file KGs for one or two supplier runs.

    Accepts either:
    - single-supplier: { "supplier_id": "..." }
    - dual-supplier (backward compatible): { "supplier1_id": "...", "supplier2_id": "..." }
    - or any combination of the above legacy fields
    Returns a mapping keyed by the provided identifiers, e.g.,
    { "supplier": {...} } or { "supplier1": {...}, "supplier2": {...} }.
    """
    try:
        to_unify: List[tuple[str, str]] = []
        if req.supplier_id:
            to_unify.append(("supplier", req.supplier_id))
        if req.supplier1_id:
            to_unify.append(("supplier1", req.supplier1_id))
        if req.supplier2_id:
            to_unify.append(("supplier2", req.supplier2_id))

        if not to_unify:
            raise HTTPException(status_code=422, detail="At least one supplier id must be provided.")

        results: Dict[str, Any] = {}
        for key, supplier_id in to_unify:
            results[key] = kg_service.unify_supplier(supplier_id)
        return results
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"KG unification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
