"""Read-only paginated assessment routes independent of AI completion."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.dependencies import ServiceContainer, get_services
from app.ecn.evidence.acceptance_plan import build_acceptance_plan
from app.ecn.services.blocking_io import run_blocking

router = APIRouter()


@router.get("/ecns/{ecn_row_id}/snapshot")
async def snapshot(
    ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Return complete deterministic counts, source context and grouped questions."""
    return await run_blocking(services.snapshot_service.overview, str(ecn_row_id))


@router.get("/ecns/{ecn_row_id}/acceptance-plan")
async def acceptance_plan(
    ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Return what accepting this ECN changes per part, and whether it is blocked (no SAP write)."""
    snapshot = await run_blocking(services.snapshot_service.overview, str(ecn_row_id))
    return build_acceptance_plan(snapshot)


@router.get("/ecns/{ecn_row_id}/analysis-stages")
async def analysis_stages(
    ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Return source-compatible validated partial results without waiting for synthesis."""
    return await run_blocking(services.analysis_repository.stage_results_for_ecn, str(ecn_row_id))


@router.get("/ecns/{ecn_row_id}/objects")
async def objects(
    ecn_row_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    query: Annotated[str, Query(max_length=500)] = "",
    technical: bool = True,
    origin: Literal["ecn", "bom"] = "ecn",
    ancestor: Annotated[str | None, Query(max_length=200)] = None,
):
    """List ECN references, optionally narrowed to an ancestor observed in their paths."""
    return await run_blocking(
        services.snapshot_service.objects,
        str(ecn_row_id),
        page=page,
        page_size=page_size,
        query=query,
        technical=technical,
        ancestor=ancestor,
        origin=origin,
    )


@router.get("/ecns/{ecn_row_id}/object")
async def object_details(
    ecn_row_id: UUID,
    number: Annotated[str, Query(min_length=1, max_length=200)],
    services: Annotated[ServiceContainer, Depends(get_services)],
    technical: bool = True,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
):
    """Show exact source coordinates and all observed depths through bounded pages."""
    return await run_blocking(
        services.snapshot_service.object_details,
        str(ecn_row_id),
        number,
        technical=technical,
        page=page,
        page_size=page_size,
    )


@router.get("/ecns/{ecn_row_id}/changes")
async def changes(ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)],
                  page: Annotated[int, Query(ge=1)] = 1,
                  page_size: Annotated[int, Query(ge=1, le=100)] = 25):
    """Return lifecycle actions and evidence-labelled replacement pairs."""
    return await run_blocking(services.snapshot_service.changes, str(ecn_row_id), page=page, page_size=page_size)


@router.get("/ecns/{ecn_row_id}/drawing", response_class=Response,
            responses={200: {"content": {"application/pdf": {}}}})
async def engineering_drawing(
    ecn_row_id: UUID,
    number: Annotated[str, Query(min_length=1, max_length=200)],
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Serve the exact packaged sample PDF through the existing authenticated API."""
    fixture, content = await run_blocking(
        services.snapshot_service.drawing, str(ecn_row_id), number,
    )
    return Response(content, media_type="application/pdf", headers={
        "Content-Disposition": f'inline; filename="{fixture.file_name}"',
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
    })
