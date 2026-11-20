"""Historical ECN list, detail, and evidence routes.

Reads are side-effect free except one idempotent cache fill: the list saves the
plant labels of rows on the requested page that were never assessed before.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_models import (
    AnalysisStateFilter,
    BusinessEvidenceDetailResponse,
    BusinessEvidencePageResponse,
    EcnDetailResponse,
    EcnListQuery,
    EcnListResponse,
)
from app.ecn.evidence.business import BusinessEvidenceSection, paginate_business_evidence
from app.ecn.errors import PersistenceNotFoundError
from app.ecn.services.blocking_io import run_blocking
from app.persistence.codec import (
    ecn_detail_response_from_record,
    evidence_detail_response_from_item,
    business_evidence_page_response,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _fill_affected_plants(services: ServiceContainer, response: EcnListResponse) -> EcnListResponse:
    """Compute plant labels for page rows whose snapshot was never built.

    Each fill builds that ECN's snapshot once and saves the labels, so later
    list reads (including plant filter and sort) see them. A failing snapshot
    leaves the row without plants instead of failing the whole page.

    Args:
        services: Application service container (``snapshot_service`` may be
            ``None`` when snapshots are unavailable).
        response: The page returned by the repository; updated in place.

    Returns:
        The same response with ``affected_plants`` filled where possible.
    """
    snapshots = getattr(services, "snapshot_service", None)
    if snapshots is None:
        return response
    # ponytail: first view of a page of never-opened ECNs pays one snapshot
    # build per row (bounded by page_size). Move this to a post-import
    # background job if that first load becomes too slow.
    for item in response.items:
        if item.affected_plants is not None:
            continue
        try:
            item.affected_plants = snapshots.affected_plants(str(item.ecn_row_id))
        except Exception:  # noqa: BLE001 - one broken snapshot must not hide the list.
            logger.warning("Plant labels unavailable for ECN %s", item.ecn_row_id, exc_info=True)
    return response


@router.get("/ecns", response_model=EcnListResponse)
async def list_ecns(
    services: Annotated[ServiceContainer, Depends(get_services)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    query: Annotated[str | None, Query(min_length=1, max_length=500)] = None,
    ecn_number: Annotated[str | None, Query(min_length=1, max_length=80)] = None,
    analysis_state: Annotated[AnalysisStateFilter | None, Query()] = None,
    has_warnings: Annotated[bool | None, Query()] = None,
    plant: Annotated[str | None, Query(min_length=1, max_length=80)] = None,
    sort: Annotated[
        str,
        Query(
            pattern=(
                "^(uploaded_at|resolution_date|affected_plants|ecn_number|description|affected_item_count|"
                "ebom_match_count|mbom_match_count|warning_count|analysis_state)$"
            )
        ),
    ] = "uploaded_at",
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
) -> EcnListResponse:
    """Return one stable server-filtered page without triggering generation.

    Args:
        page: One-based page number.
        page_size: Bounded number of rows per page.
        query: Optional free-text search.
        ecn_number: Optional exact ECN business identifier.
        analysis_state: Optional derived analysis state or inbox summary group.
        has_warnings: Optional warning-presence filter.
        plant: Optional plant code/name substring (assessed ECNs only).
        sort: Allowlisted sort key.
        order: Ascending or descending sort order.
        services: Application service container.

    Returns:
        Historical immutable ECN occurrence page.
    """
    validated = EcnListQuery(
        page=page,
        page_size=page_size,
        query=query,
        ecn_number=ecn_number,
        analysis_state=analysis_state,
        has_warnings=has_warnings,
        plant=plant,
        sort=sort,  # type: ignore[arg-type] - FastAPI regex closes the allowlist.
        order=order,  # type: ignore[arg-type] - FastAPI regex closes the allowlist.
    )
    response = await run_blocking(services.ecn_repository.list_ecns, validated)
    return await run_blocking(_fill_affected_plants, services, response)


@router.get("/ecns/{ecn_row_id}", response_model=EcnDetailResponse)
async def get_ecn_detail(
    ecn_row_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> EcnDetailResponse:
    """Return exact occurrence provenance, evidence, job, and stored analysis.

    Args:
        ecn_row_id: Internal immutable ECN occurrence UUID.
        services: Application service container.

    Returns:
        Complete read-only detail projection.
    """
    canonical_id = str(ecn_row_id)
    record = await run_blocking(services.ecn_repository.get_ecn_detail, canonical_id)
    if record is None:
        raise PersistenceNotFoundError(
            "ECN_NOT_FOUND",
            "The requested ECN occurrence was not found",
            status_code=404,
        )
    response = ecn_detail_response_from_record(record)
    response.linked_ecns = await run_blocking(services.ecn_repository.get_linked_ecns, record)
    return response


@router.get(
    "/ecns/{ecn_row_id}/evidence/{evidence_id}",
    response_model=BusinessEvidenceDetailResponse,
)
async def get_evidence(
    ecn_row_id: UUID,
    evidence_id: Annotated[
        str, Path(pattern=r"^E-[0-9a-f]{16}$", min_length=18, max_length=18)
    ],
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> BusinessEvidenceDetailResponse:
    """Return evidence only after exact same-occurrence membership verification.

    Args:
        ecn_row_id: Internal immutable ECN occurrence UUID.
        evidence_id: Deterministic `E-` plus sixteen-hex-character identity.
        services: Application service container.

    Returns:
        Full deterministic evidence item with provenance.
    """
    canonical_id = str(ecn_row_id)
    item = await run_blocking(
        services.ecn_repository.get_evidence, canonical_id, evidence_id
    )
    if item is None:
        raise PersistenceNotFoundError(
            "EVIDENCE_NOT_FOUND",
            "The requested evidence was not found for this ECN occurrence",
            status_code=404,
        )
    return evidence_detail_response_from_item(canonical_id, item)


@router.get(
    "/ecns/{ecn_row_id}/evidence",
    response_model=BusinessEvidencePageResponse,
)
async def list_business_evidence(
    ecn_row_id: UUID,
    services: Annotated[ServiceContainer, Depends(get_services)],
    section: Annotated[BusinessEvidenceSection, Query()],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    include_technical: Annotated[bool, Query()] = False,
) -> BusinessEvidencePageResponse:
    """Return one lazy-loaded page of grouped, business-readable evidence.

    Args:
        ecn_row_id: Internal immutable ECN occurrence UUID.
        services: Application service container.
        section: One business change set to retrieve.
        page: One-based page number.
        page_size: Bounded row count from 1 through 100.
        include_technical: Include raw IDs/enums only for explicit audit use.

    Returns:
        Stable page with aggregate and source-provenance fields.
    """
    canonical_id = str(ecn_row_id)
    bundle = await run_blocking(
        services.ecn_repository.get_evidence_bundle, canonical_id
    )
    if bundle is None:
        raise PersistenceNotFoundError(
            "ECN_NOT_FOUND",
            "The requested ECN occurrence was not found",
            status_code=404,
        )
    projected = paginate_business_evidence(
        bundle,
        section,
        page=page,
        page_size=page_size,
    )
    return business_evidence_page_response(
        projected,
        include_technical=include_technical,
    )
