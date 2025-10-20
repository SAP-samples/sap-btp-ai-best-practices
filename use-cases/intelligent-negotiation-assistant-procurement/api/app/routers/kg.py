from __future__ import annotations

import logging
from typing import Any, Dict, List

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
    supplier1_name: str = Form(...),
    supplier2_name: str = Form(...),
    supplier1_files: List[UploadFile] = File(default=[]),
    supplier2_files: List[UploadFile] = File(default=[]),
) -> Dict[str, Any]:
    """Create per-file KGs for two suppliers using image-mode and save outputs.

    Accepts multipart/form-data with two supplier names and two lists of PDFs.
    Returns supplier identifiers and directories to be used by the unify call.
    """
    try:
        return kg_service.create_from_uploads(
            supplier1_name=supplier1_name,
            supplier1_files=supplier1_files,
            supplier2_name=supplier2_name,
            supplier2_files=supplier2_files,
        )
    except Exception as e:
        logger.error(f"KG creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UnifyRequest(BaseModel):
    supplier1_id: str
    supplier2_id: str


@router.post("/v1/kg/unify")
async def unify_kg(req: UnifyRequest) -> Dict[str, Any]:
    """Unify per-file KGs for two supplier runs and return unified outputs."""
    try:
        return kg_service.unify_two_suppliers(req.supplier1_id, req.supplier2_id)
    except FileNotFoundError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"KG unification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
