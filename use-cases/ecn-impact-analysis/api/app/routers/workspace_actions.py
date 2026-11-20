"""Explicit model-cost actions and separately attributed human snapshot reviews."""

from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.dependencies import ServiceContainer, get_services
from app.ecn.api_models import AnalysisStateFilter, EcnListQuery
from app.ecn.errors import (
    PersistenceNotFoundError,
    PersistenceStateError,
    SourceValidationError,
)
from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.ingestion import UploadBytesRequest, UploadFileBytes
from app.ecn.services.reviews import actor_id, record_review

router = APIRouter()


class ExplicitSummarySelection(BaseModel):
    """Select exact ECN occurrence identities across one or more list pages."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["explicit"]
    ecn_ids: list[UUID] = Field(min_length=1)


class FilteredSummarySelection(BaseModel):
    """Select all ECNs matching list filters, minus explicit exclusions."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["filtered"]
    query: str = Field(default="", max_length=500)
    ecn_number: str = Field(default="", max_length=80)
    analysis_state: AnalysisStateFilter | None = None
    has_warnings: bool | None = None
    plant: str = Field(default="", max_length=80)
    excluded_ecn_ids: list[UUID] = Field(default_factory=list)


SummarySelection = Annotated[
    ExplicitSummarySelection | FilteredSummarySelection,
    Field(discriminator="mode"),
]


class GenerateSelection(BaseModel):
    """Retry-safe identity and exactly one explicit or filtered selection."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    selection: SummarySelection


class ReviewSubmission(BaseModel):
    """Human engineering decision tied to the displayed immutable evidence.

    Inputs:
        request_id: client-generated UUID that makes retries idempotent.
        snapshot_version / evidence_hash: identity of the snapshot the reviewer saw.
        reviewer: self-reported reviewer name.
        outcome: ``ACCEPTED`` (the ECN and its material master changes are agreed;
            the SAP write is simulated, and a blocked ECN is refused) or
            ``CHANGES_REQUESTED`` (rejected; comments say what must change).
        comments: required for ``CHANGES_REQUESTED``, optional for ``ACCEPTED``.

    Older rows may still carry the retired outcomes REVIEWED,
    CLARIFICATION_REQUESTED and CORRECTION_RECORDED; they stay readable but can
    no longer be submitted.
    """

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    snapshot_version: str = Field(min_length=1, max_length=40)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    outcome: Literal["ACCEPTED", "CHANGES_REQUESTED"]
    comments: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def require_change_description(self):
        """Reject a change request that does not say which changes are needed."""
        if self.outcome == "CHANGES_REQUESTED" and not self.comments.strip():
            raise ValueError("Describe the changes needed before requesting changes.")
        return self


class ReassessmentRequest(BaseModel):
    """Create a new assessment version from exact source JSON and current active datasets."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    assessment_date: date | None = None


def _reassess(services, ecn_row_id, request_id, assessment_date=None):
    """Create one reassessment and return its public result plus trusted RDF IDs."""
    record = services.ecn_repository.get_ecn_detail(ecn_row_id)
    if record is None:
        raise PersistenceNotFoundError(
            "ECN_NOT_FOUND", "The requested ECN no longer exists.", status_code=404
        )
    artifact = services.artifact_repository.get_artifact(
        record.ecn_artifact.artifact_id
    )
    if artifact is None:
        raise PersistenceNotFoundError(
            "EXPORT_NOT_FOUND",
            "The original ECN source is unavailable.",
            status_code=404,
        )
    prior = services.ecn_repository.get_evidence_bundle(ecn_row_id)
    result = services.ingestion_service.ingest_upload(
        UploadBytesRequest(
            request_id,
            UploadFileBytes(
                artifact.file_name, artifact.media_type, artifact.content_bytes
            ),
            None,
            None,
            registered_artifact=artifact,
            source_positions=(record.ecn.source_ordinal + 1,),
            assessment_revision=request_id,
            assessment_date=assessment_date,
            data_group=prior.context.data_group if prior else "customer",
        )
    )
    from uuid import uuid5

    assessment_id = str(
        uuid5(UUID(artifact.artifact_id), f"{record.ecn.source_ordinal}:{request_id}")
    )
    if (
        not result.created
        and services.ecn_repository.get_ecn_detail(assessment_id) is None
    ):
        raise PersistenceStateError(
            "REASSESSMENT_REQUEST_CONFLICT",
            "Use a new request ID for a different assessment.",
            status_code=409,
        )
    return (
        {
            "ecn_row_id": assessment_id,
            "upload_id": result.upload.upload_id,
            "previous_assessment_id": ecn_row_id,
            "model_calls": 0,
        },
        result.projection_ecn_ids,
    )


@router.post("/ecns/{ecn_row_id}/reassess", status_code=201)
async def reassess(
    ecn_row_id: UUID,
    body: ReassessmentRequest,
    background_tasks: BackgroundTasks,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Create a new reviewable snapshot using current datasets; generation remains explicit."""
    response, projection_ecn_ids = await run_blocking(
        _reassess, services, str(ecn_row_id), body.request_id, body.assessment_date
    )
    if projection_ecn_ids:
        background_tasks.add_task(
            run_blocking,
            services.rdf_projection_service.project_many,
            projection_ecn_ids,
        )
    return response


def _resolve_summary_ids(services, selection: SummarySelection) -> list[str]:
    """Resolve explicit or filter-wide membership at submission time.

    Args:
        services: Initialized service container owning the ECN repository.
        selection: Validated discriminated selection request.

    Returns:
        Deduplicated ECN row IDs to persist in the generation batch.

    Raises:
        SourceValidationError: When filters and exclusions resolve to no ECNs.
    """
    if isinstance(selection, ExplicitSummarySelection):
        resolved = list(dict.fromkeys(str(value) for value in selection.ecn_ids))
    else:
        query = EcnListQuery(
            query=selection.query.strip() or None,
            ecn_number=selection.ecn_number.strip() or None,
            analysis_state=selection.analysis_state,
            has_warnings=selection.has_warnings,
            plant=selection.plant.strip() or None,
        )
        excluded = {str(value) for value in selection.excluded_ecn_ids}
        resolved = [
            value
            for value in services.ecn_repository.list_matching_ecn_ids(query)
            if value not in excluded
        ]
    if not resolved:
        raise SourceValidationError(
            "SUMMARY_SELECTION_EMPTY",
            "Select at least one ECN.",
            status_code=400,
        )
    return resolved


@router.post("/summary-batches", status_code=202)
async def generate_summaries(
    selection: GenerateSelection,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Resolve one selection mode, persist its members, and schedule after commit."""
    ecn_ids = await run_blocking(_resolve_summary_ids, services, selection.selection)
    await run_blocking(services.rdf_projection_service.require_all_current, ecn_ids)
    result, jobs = await run_blocking(
        services.workspace_repository.create_batch,
        str(selection.request_id),
        ecn_ids,
    )
    services.runtime.schedule_jobs(jobs)
    return result


@router.get("/summary-batches/{batch_id}")
async def batch_status(
    batch_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Poll durable selection progress without starting provider calls."""
    result = await run_blocking(services.workspace_repository.batch, str(batch_id))
    if result is None:
        raise PersistenceNotFoundError(
            "BATCH_NOT_FOUND", "Generation batch was not found.", status_code=404
        )
    return result


@router.post("/summary-batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Return unstarted selected jobs to Ready; running calls may finish."""
    return await run_blocking(services.workspace_repository.cancel_batch, str(batch_id))


class EcnDeleteRequest(BaseModel):
    """Exact destructive phrase plus the same selection used for summary generation."""

    model_config = ConfigDict(extra="forbid")
    confirmation: str = Field(max_length=20)
    selection: SummarySelection


@router.post("/ecns/delete")
async def delete_ecns(
    body: EcnDeleteRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Delete selected ECN line items, all their versions and derived data.

    Reference datasets (BOM, material master, purchase info, stock, rules)
    are never touched.  Returns committed row counts and ``graph_cleanup``
    (``"complete"`` or ``"failed"``) for the knowledge-graph triples.
    """
    ecn_ids = await run_blocking(_resolve_summary_ids, services, body.selection)
    counts, graph_ok = await run_blocking(
        services.history_service.delete_ecns, body.confirmation, ecn_ids
    )
    if getattr(services, "snapshot_service", None) is not None:
        services.snapshot_service.clear()
    return {
        "ecns": counts.ecns,
        "analyses": counts.analyses,
        "attempts": counts.attempts,
        "jobs": counts.jobs,
        "reviews": counts.reviews,
        "uploads": counts.uploads,
        "source_artifacts": counts.source_artifacts,
        "feasibility_assessments": counts.feasibility_assessments,
        "graph_cleanup": "complete" if graph_ok else "failed",
    }


@router.get("/ecns/{ecn_row_id}/reviews")
async def list_reviews(
    ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Return saved human reviews separately from source facts and generated prose."""
    await run_blocking(services.snapshot_service.overview, str(ecn_row_id))
    return await run_blocking(services.workspace_repository.reviews, str(ecn_row_id))


@router.post("/ecns/{ecn_row_id}/reviews", status_code=201)
async def save_review(
    ecn_row_id: UUID,
    review: ReviewSubmission,
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
):
    """Persist exact snapshot attribution and verify the displayed evidence identity."""
    return await run_blocking(
        record_review,
        services.snapshot_service,
        services.workspace_repository,
        str(ecn_row_id),
        request_id=str(review.request_id),
        reviewer=review.reviewer,
        outcome=review.outcome,
        comments=review.comments,
        actor=actor_id(request.headers.get("X-API-Key", "")),
        evidence_hash=review.evidence_hash,
        snapshot_version=review.snapshot_version,
    )


@router.get("/ecns/{ecn_row_id}/versions")
async def assessment_versions(
    ecn_row_id: UUID, services: Annotated[ServiceContainer, Depends(get_services)]
):
    """Read links to previous immutable assessments of the exact original ECN source position."""
    await run_blocking(services.snapshot_service.overview, str(ecn_row_id))
    return await run_blocking(services.workspace_repository.versions, str(ecn_row_id))
