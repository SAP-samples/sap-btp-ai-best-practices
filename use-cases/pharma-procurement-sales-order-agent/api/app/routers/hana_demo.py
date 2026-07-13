import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..utils.hana import HANAConnection
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

class HANAResponse(BaseModel):
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

@router.post("/test", response_model=HANAResponse)
async def test_hana_connection() -> HANAResponse:
    """Test connection to SAP HANA."""
    try:
        hana = HANAConnection()
        result = hana.test_connection()
        return HANAResponse(success=result["success"], message=result["message"])
    except Exception as e:
        logger.error(f"HANA demo error: {e}")
        return HANAResponse(success=False, message=f"Failed to connect: {str(e)}")

