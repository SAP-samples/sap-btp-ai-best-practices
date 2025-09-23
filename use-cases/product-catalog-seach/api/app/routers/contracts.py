import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Get the contracts directory path
CONTRACTS_DIR = Path(__file__).parent.parent.parent / "storage" / "contracts"


@router.get("/contracts/{filename}")
async def get_contract(filename: str):
    """
    Serve contract PDF files from the storage/contracts directory.

    Args:
        filename: The name of the contract file to retrieve

    Returns:
        FileResponse: The requested PDF file

    Raises:
        HTTPException: If the file is not found or access is denied
    """
    try:
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)

        # Construct the full file path
        file_path = CONTRACTS_DIR / safe_filename

        # Check if file exists and is within the contracts directory
        if not file_path.exists():
            logger.warning(f"Contract file not found: {safe_filename}")
            raise HTTPException(status_code=404, detail="Contract file not found")

        # Ensure the file is actually within the contracts directory (security check)
        if not str(file_path.resolve()).startswith(str(CONTRACTS_DIR.resolve())):
            logger.warning(
                f"Attempted access outside contracts directory: {safe_filename}"
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if it's a PDF file
        if not safe_filename.lower().endswith(".pdf"):
            logger.warning(f"Non-PDF file requested: {safe_filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        logger.info(f"Serving contract file: {safe_filename}")

        # Return the file
        return FileResponse(
            path=str(file_path), media_type="application/pdf", filename=safe_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving contract file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
