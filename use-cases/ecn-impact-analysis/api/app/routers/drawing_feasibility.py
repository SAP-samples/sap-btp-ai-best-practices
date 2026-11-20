"""Protected transient endpoint for PS4 pipe-drawing feasibility reports."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.dependencies import ServiceContainer, get_services
from app.drawing_feasibility.drawing_input import read_drawing_upload
from app.drawing_feasibility.errors import (
    DrawingServiceUnavailableError,
    DrawingValidationError,
)
from app.drawing_feasibility.models import DrawingFeasibilityReport
from app.observability import extract_client_host_from_request, extract_user_id_from_request

router = APIRouter(tags=["drawing-feasibility"])


@router.post(
    "/drawing-feasibility/assessments",
    response_model=DrawingFeasibilityReport,
)
async def assess_drawing(
    request: Request,
    drawing_file: Annotated[UploadFile, File(description="One PDF, PNG, or JPEG drawing")],
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> DrawingFeasibilityReport:
    """Assess one request-local drawing without ECN persistence or agent coupling."""
    service = services.drawing_feasibility_service
    if service is None:
        raise DrawingServiceUnavailableError()
    form = await request.form()
    if set(form.keys()) != {"drawing_file"} or len(form.getlist("drawing_file")) != 1:
        raise DrawingValidationError(
            "DRAWING_FIELDS_INVALID", "Drawing assessment accepts one drawing file"
        )
    drawing = await read_drawing_upload(
        drawing_file, service.settings.max_upload_bytes
    )
    return await service.assess(
        drawing,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
    )
