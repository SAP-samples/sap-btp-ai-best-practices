import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from gen_ai_hub.orchestration.service import OrchestrationService

from ..security import get_api_key
from ..utils.document_intelligence import DocumentTranscriber

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# Shared Orchestration Service instance (can be reused across the app)
ORCHESTRATION_SERVICE = OrchestrationService()

# Reusable transcriber instance configured with defaults
# TRANSCRIBER = DocumentTranscriber(orchestration_service=ORCHESTRATION_SERVICE)

# Reusable transcriber instance with reduced payload size
TRANSCRIBER = DocumentTranscriber(
    orchestration_service=ORCHESTRATION_SERVICE,
    dpi=200,  # lower DPI to reduce bytes
    image_format="jpg",  # JPEG instead of PNG
    jpeg_quality=75,  # adjust if you want stronger compression
    grayscale=False,  # keep color by default
    auto_grayscale_megapixels=4.0,  # optionally grayscale very large pages
)


@router.post("/pdf-to-md")
async def pdf_to_markdown(file: UploadFile = File(...)):
    """Accept a PDF, render pages to images, and extract Markdown."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")

        result = await TRANSCRIBER.transcribe_pdf_to_markdown(content)
        return JSONResponse(result)
    except HTTPException:
        raise
    except ValueError as ve:
        # Expected input issues (e.g., invalid PDF, no pages)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("PDF to Markdown failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf-to-md/stream")
async def pdf_to_markdown_stream(file: UploadFile = File(...)):
    """Stream progress (NDJSON) while rendering and processing pages in parallel."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    content_bytes = await file.read()

    async def event_stream():
        if not content_bytes:
            yield json.dumps({"type": "error", "message": "Empty file."}) + "\n"
            return
        async for event in TRANSCRIBER.stream_transcription_events(content_bytes):
            # Convert dict events to NDJSON
            try:
                yield json.dumps(event) + "\n"
            except Exception as e:
                # Ensure stream continues even if a single event fails to serialize
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
