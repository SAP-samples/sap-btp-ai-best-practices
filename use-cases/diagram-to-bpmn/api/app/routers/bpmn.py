import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from ..models.bpmn import BPMNGenerationResponse
from ..services.bpmn_generator import (
    BPMNGenerator,
    DEFAULT_PROVIDER,
    get_bpmn_generator,
)
from ..security import get_api_key

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES: List[str] = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/svg+xml",
]

router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post(
    "/generate",
    response_model=BPMNGenerationResponse,
    summary="Generate BPMN XML from a process diagram image",
)
async def generate_bpmn_from_image(
    file: UploadFile = File(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: Optional[str] = Form(None),
    generator: BPMNGenerator = Depends(get_bpmn_generator),
) -> BPMNGenerationResponse:
    """Produce BPMN XML by analyzing an uploaded diagram image."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning("Rejected file '%s' with unsupported mime type: %s", file.filename, file.content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        provider_key, resolved_model, chat_response = await run_in_threadpool(
            generator.generate,
            image_bytes,
            provider,
            model,
            file.filename,
            file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not chat_response.success:
        return BPMNGenerationResponse(
            bpmn_xml="",
            provider=provider_key,
            model=resolved_model,
            success=False,
            usage=chat_response.usage,
            error=chat_response.error,
        )

    return BPMNGenerationResponse(
        bpmn_xml=chat_response.text,
        provider=provider_key,
        model=resolved_model,
        success=True,
        usage=chat_response.usage,
    )
