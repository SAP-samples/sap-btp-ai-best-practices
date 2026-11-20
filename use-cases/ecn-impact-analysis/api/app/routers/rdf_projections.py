"""Read and retry one selected ECN's rebuildable RDF projection status."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_models import RdfProjectionView
from app.ecn.errors import PersistenceNotFoundError
from app.ecn.rdf import ProjectionRecord
from app.ecn.services.blocking_io import run_blocking


router = APIRouter()


def _view(record: ProjectionRecord) -> RdfProjectionView:
    """Map one durable projection record to its closed HTTP representation."""
    return RdfProjectionView(
        ecn_row_id=record.ecn_row_id,
        status=record.status,
        source_sha256=record.source_sha256,
        projection_version=record.projection_version,
        verified_node_count=record.verified_node_count,
        verified_triple_count=record.verified_triple_count,
        error_code=record.error_code,
        error_message=record.error_message,
        last_attempt_at=record.last_attempt_at,
        current_at=record.current_at,
    )


@router.get("/ecns/{ecn_row_id}/rdf-projection", response_model=RdfProjectionView)
async def get_rdf_projection(
    ecn_row_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> RdfProjectionView:
    """Return derived projection diagnostics without widening ECN detail reads."""
    record = await run_blocking(
        services.rdf_projection_service.get_projection, str(ecn_row_id)
    )
    if record is None:
        raise PersistenceNotFoundError(
            "RDF_PROJECTION_NOT_FOUND",
            "The selected ECN projection was not found.",
            status_code=404,
        )
    return _view(record)


@router.post(
    "/ecns/{ecn_row_id}/rdf-projection/retry",
    response_model=RdfProjectionView,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_rdf_projection(
    ecn_row_id: UUID,
    background_tasks: BackgroundTasks,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> RdfProjectionView:
    """Mark one route-selected projection pending, then rebuild it after response.

    The endpoint accepts no graph, query, version, count, source, or file input.
    Its background task uses only the trusted route ECN identity and persisted
    relational source data.
    """
    canonical_id = str(ecn_row_id)
    record = await run_blocking(
        services.rdf_projection_service.request_rebuild, canonical_id
    )
    background_tasks.add_task(
        run_blocking,
        services.rdf_projection_service.project,
        canonical_id,
    )
    return _view(record)
