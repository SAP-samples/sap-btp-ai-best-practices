from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from email.utils import parseaddr
from math import ceil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.engine import Connection

from ..document_requests.agent import DocumentRequestAgent
from ..document_requests.db import create_hana_engine, hana_connection
from ..document_requests.extraction_service import ExtractionService
from ..document_requests.gmail_service import (
    GmailDestinationApiClient,
    GmailFetchService,
    build_mime_reply_with_attachments,
    build_gmail_service,
)
from ..document_requests.models import FetchRunStatus, FetchSummary, GmailEmail
from ..document_requests.processing import (
    SCENARIO_RUN_KEY,
    AgentFactory,
    EmailPipeline,
    RepositoryScope,
    repository_scope,
    run_gmail_fetch,
    run_scenario_seed,
    try_start_run,
)
from ..document_requests.repositories import DocumentRequestRepository
from ..document_requests.response_composer import DocumentResponseComposer
from ..document_requests.scenario_emails import load_scenario_emails as load_scenario_emails_from_dir
from ..document_requests.settings import DocumentRequestSettings
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

DEFAULT_PAGE_SIZE = 50


class ManualEmailRequest(BaseModel):
    """Manual email payload submitted from the local test UI.

    Attributes:
        from_header: Sender header or email address.
        subject: Email subject.
        body: Plain-text email body.
    """

    from_header: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def accept_from_alias(cls, value: object) -> object:
        """Map JSON field "from" to the Python-safe from_header field.

        Args:
            value: Raw request payload.

        Returns:
            Payload with from_header populated when "from" was supplied.
        """
        if isinstance(value, dict) and "from" in value and "from_header" not in value:
            normalized = dict(value)
            normalized["from_header"] = normalized.pop("from")
            return normalized
        return value


class DraftDecisionRequest(BaseModel):
    """Reviewer decision payload for approving or rejecting a draft.

    Attributes:
        body: Current plain-text draft body from the editable review field.
    """

    body: str | None = None


def paginated_response(
    items: list[dict[str, object]],
    page: int,
    page_size: int,
    total: int,
    summary: dict[str, int],
) -> dict[str, object]:
    """Build a standard paginated response for draft list endpoints.

    Args:
        items: Current page rows.
        page: One-based page number.
        page_size: Number of rows requested per page.
        total: Total rows matching the endpoint status filter.
        summary: Status counts across all matching rows.

    Returns:
        JSON-compatible response with items and pagination metadata.
    """
    total_pages = max(1, ceil(total / page_size))
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "summary": summary,
    }


def edited_body_from_payload(payload: DraftDecisionRequest | None, draft: dict[str, object]) -> str:
    """Resolve the final draft body from the request payload or stored fallback.

    Args:
        payload: Optional review decision request body.
        draft: Stored draft row used when older clients submit no body.

    Returns:
        Plain-text draft body to persist and send.
    """
    if payload is not None and payload.body is not None:
        return payload.body
    return str(draft.get("body") or "")


def source_email_from_draft(draft: dict[str, object]) -> GmailEmail:
    """Rebuild source email context needed to refresh a Gmail reply draft.

    Args:
        draft: Draft row joined with the original email metadata.

    Returns:
        GmailEmail model containing threading headers and original body context.
    """
    return GmailEmail(
        gmail_message_id=str(draft.get("gmail_message_id") or ""),
        thread_id=str(draft.get("source_thread_id") or "") or None,
        rfc_message_id=str(draft.get("source_rfc_message_id") or "") or None,
        sender_name=str(draft.get("sender_name") or "") or None,
        sender_email=str(draft.get("sender_email") or "") or None,
        subject=str(draft.get("original_subject") or "") or None,
        email_date=draft.get("email_date") if isinstance(draft.get("email_date"), datetime) else None,
        body=str(draft.get("original_body") or ""),
    )


def get_settings() -> DocumentRequestSettings:
    """Return runtime settings from environment variables.

    Returns:
        DocumentRequestSettings populated from process environment.
    """
    return DocumentRequestSettings.from_env()


def get_connection(settings: DocumentRequestSettings = Depends(get_settings)) -> Iterator[Connection]:
    """Yield a transactional HANA connection.

    Args:
        settings: Runtime settings with HANA connection details.

    Yields:
        Transactional SQLAlchemy connection.
    """
    engine = create_hana_engine(settings)
    with hana_connection(engine) as connection:
        yield connection


def get_repository(connection: Connection = Depends(get_connection)) -> DocumentRequestRepository:
    """Build the HANA repository dependency.

    Args:
        connection: Transactional HANA connection.

    Returns:
        DocumentRequestRepository for document-request persistence.
    """
    return DocumentRequestRepository(connection)


def get_gmail_client(settings: DocumentRequestSettings = Depends(get_settings)) -> GmailDestinationApiClient | None:
    """Build a Gmail API client.

    Args:
        settings: Runtime settings with Gmail destination values.

    Returns:
        GmailDestinationApiClient using BTP Destination authentication,
        or None when running locally without VCAP_SERVICES.
    """
    try:
        return build_gmail_service(settings)
    except Exception:
        return None


def get_gmail_fetch_service(
    settings: DocumentRequestSettings = Depends(get_settings),
    gmail_client: GmailDestinationApiClient | None = Depends(get_gmail_client),
) -> GmailFetchService:
    """Build Gmail fetch service dependency.

    Args:
        settings: Runtime settings with mailbox id.
        gmail_client: Gmail API client.

    Returns:
        GmailFetchService configured for the mailbox.
    """
    if gmail_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gmail destination is not configured. Manual and scenario email processing are still available.",
        )
    return GmailFetchService(gmail_client, mailbox_id=settings.gmail_mailbox_id)


def get_extraction_service(settings: DocumentRequestSettings = Depends(get_settings)) -> ExtractionService:
    """Build extraction service dependency.

    Args:
        settings: Runtime settings with extractor model selection.

    Returns:
        ExtractionService for Gmail email classification.
    """
    return ExtractionService(settings)


def get_response_composer(settings: DocumentRequestSettings = Depends(get_settings)) -> DocumentResponseComposer:
    """Build grounded response composer dependency.

    Args:
        settings: Runtime settings with agent model selection.

    Returns:
        DocumentResponseComposer for HANA-grounded draft bodies.
    """
    return DocumentResponseComposer(settings)


def get_repository_scope(settings: DocumentRequestSettings = Depends(get_settings)) -> RepositoryScope:
    """Return a factory of short transactional repositories.

    Background runs cannot use the request-scoped connection (it closes when
    the response is sent), and each email gets its own transactions.

    Args:
        settings: Runtime settings with HANA connection details.

    Returns:
        Zero-argument callable returning a repository context manager.
    """
    engine = create_hana_engine(settings)
    return lambda: repository_scope(engine)


def get_agent_factory(
    settings: DocumentRequestSettings = Depends(get_settings),
    gmail_client: GmailDestinationApiClient | None = Depends(get_gmail_client),
    composer: DocumentResponseComposer = Depends(get_response_composer),
) -> AgentFactory:
    """Return a factory building a DocumentRequestAgent bound to a repository.

    Args:
        settings: Runtime settings with mailbox and sender configuration.
        gmail_client: Gmail client for draft creation (None without Gmail).
        composer: Response composer for grounded draft bodies.

    Returns:
        Callable mapping a repository to a ready agent.
    """

    def build(repo: DocumentRequestRepository) -> DocumentRequestAgent:
        return DocumentRequestAgent(
            repository=repo,
            gmail_client=gmail_client,
            mailbox_id=settings.gmail_mailbox_id,
            from_email=settings.from_email,
            composer=composer,
        )

    return build


def get_pipeline(
    settings: DocumentRequestSettings = Depends(get_settings),
    scope: RepositoryScope = Depends(get_repository_scope),
    agent_factory: AgentFactory = Depends(get_agent_factory),
    extractor: ExtractionService = Depends(get_extraction_service),
) -> EmailPipeline:
    """Build the email processing pipeline dependency.

    Args:
        settings: Runtime settings.
        scope: Transactional repository factory.
        agent_factory: Agent factory.
        extractor: Email extraction service.

    Returns:
        EmailPipeline used by fetch, scenario and manual processing.
    """
    return EmailPipeline(repository_scope=scope, agent_factory=agent_factory, extractor=extractor, settings=settings)


def get_mailbox_id(settings: DocumentRequestSettings = Depends(get_settings)) -> str:
    """Expose mailbox id as an overrideable dependency.

    Args:
        settings: Runtime settings with mailbox id.

    Returns:
        Gmail mailbox id.
    """
    return settings.gmail_mailbox_id


# Returned when a second approve/reject arrives for a draft already decided
# (double click, two reviewers). Prevents sending the same email twice.
DRAFT_ALREADY_DECIDED = "This draft was already approved or rejected."
RUN_ALREADY_ACTIVE = "A run for this source is already in progress. Poll /emails/fetch-status for its progress."


def run_key_for_source(source: str, settings: DocumentRequestSettings) -> str:
    """Map a UI run source to its HANA run key.

    Args:
        source: "gmail" or "scenarios".
        settings: Runtime settings with the Gmail mailbox id.

    Returns:
        Run key stored in document_request_fetch_state.
    """
    return SCENARIO_RUN_KEY if source == "scenarios" else settings.gmail_mailbox_id


@router.post("/emails/fetch-new", response_model=FetchRunStatus, status_code=status.HTTP_202_ACCEPTED)
def fetch_new_document_request_emails(
    background_tasks: BackgroundTasks,
    settings: DocumentRequestSettings = Depends(get_settings),
    scope: RepositoryScope = Depends(get_repository_scope),
    service: GmailFetchService = Depends(get_gmail_fetch_service),
    pipeline: EmailPipeline = Depends(get_pipeline),
) -> FetchRunStatus:
    """Start a background inbox fetch and return immediately.

    Args:
        background_tasks: FastAPI background task queue (runs after the response).
        settings: Runtime settings.
        scope: Transactional repository factory.
        service: Gmail fetch service.
        pipeline: Email processing pipeline.

    Returns:
        FetchRunStatus with status "running"; poll /emails/fetch-status?source=gmail.

    Raises:
        HTTPException: 409 when a Gmail run is already active.
    """
    started_at = datetime.now(timezone.utc)
    run_key = settings.gmail_mailbox_id
    if not try_start_run(scope, run_key, started_at):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=RUN_ALREADY_ACTIVE)
    background_tasks.add_task(run_gmail_fetch, pipeline, service, run_key, started_at, "/api/document-requests/emails/fetch-new")
    return FetchRunStatus(source="gmail", status="running", started_at=started_at)


@router.post("/emails/seed-scenarios", response_model=FetchRunStatus, status_code=status.HTTP_202_ACCEPTED)
def seed_scenario_emails(
    background_tasks: BackgroundTasks,
    scope: RepositoryScope = Depends(get_repository_scope),
    pipeline: EmailPipeline = Depends(get_pipeline),
) -> FetchRunStatus:
    """Start a background run that resets and reprocesses the scenario emails.

    Args:
        background_tasks: FastAPI background task queue.
        scope: Transactional repository factory.
        pipeline: Email processing pipeline.

    Returns:
        FetchRunStatus with status "running"; poll /emails/fetch-status?source=scenarios.

    Raises:
        HTTPException: 409 when a scenario run is already active.
    """
    scenario_emails = load_scenario_emails()
    started_at = datetime.now(timezone.utc)
    if not try_start_run(scope, SCENARIO_RUN_KEY, started_at):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=RUN_ALREADY_ACTIVE)
    background_tasks.add_task(
        run_scenario_seed, pipeline, scenario_emails, started_at, "/api/document-requests/emails/seed-scenarios"
    )
    return FetchRunStatus(source="scenarios", status="running", started_at=started_at)


@router.get("/emails/fetch-status", response_model=FetchRunStatus)
def get_fetch_status(
    source: str = Query("gmail", pattern="^(gmail|scenarios)$"),
    settings: DocumentRequestSettings = Depends(get_settings),
    repo: DocumentRequestRepository = Depends(get_repository),
) -> FetchRunStatus:
    """Return the state and progress counters of the latest run for a source.

    Args:
        source: "gmail" (inbox fetch) or "scenarios".
        settings: Runtime settings.
        repo: HANA repository.

    Returns:
        FetchRunStatus; status "idle" when the source never ran.
    """
    row = repo.get_fetch_state(run_key_for_source(source, settings))
    if not row:
        return FetchRunStatus(source=source)
    summary_raw = row.get("summary_json")
    return FetchRunStatus(
        source=source,
        status=str(row.get("last_status") or "idle"),
        started_at=row.get("last_started_at"),
        finished_at=row.get("last_finished_at"),
        summary=FetchSummary.model_validate(json.loads(summary_raw)) if summary_raw else FetchSummary(),
        error_message=row.get("error_message"),
    )


@router.post("/emails/manual", response_model=FetchSummary)
def process_manual_email(payload: ManualEmailRequest, pipeline: EmailPipeline = Depends(get_pipeline)) -> FetchSummary:
    """Process one manually entered email synchronously (single email, user waits).

    Args:
        payload: Manual email fields from the UI.
        pipeline: Email processing pipeline.

    Returns:
        FetchSummary for the manual processing run.
    """
    summary = FetchSummary(total=1)
    pipeline.process_batch(
        [manual_payload_to_email(payload)], summary, route="/api/document-requests/emails/manual", actor_type="human"
    )
    return summary


@router.get("/drafts")
def list_active_document_request_drafts(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=DEFAULT_PAGE_SIZE, le=DEFAULT_PAGE_SIZE),
    repo: DocumentRequestRepository = Depends(get_repository),
) -> dict[str, object]:
    """List active draft emails waiting for review.

    Args:
        page: One-based page number.
        page_size: Fixed 50-row page size.
        repo: HANA repository.

    Returns:
        Active draft row collection with pagination metadata.
    """
    return paginated_response(
        items=repo.list_active_drafts(page=page, page_size=page_size),
        page=page,
        page_size=page_size,
        total=repo.count_active_drafts(),
        summary=repo.summarize_active_drafts(),
    )


@router.get("/history")
def list_document_request_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=DEFAULT_PAGE_SIZE, le=DEFAULT_PAGE_SIZE),
    repo: DocumentRequestRepository = Depends(get_repository),
) -> dict[str, object]:
    """List approved and rejected document-request drafts.

    Args:
        page: One-based page number.
        page_size: Fixed 50-row page size.
        repo: HANA repository.

    Returns:
        Historical draft row collection with pagination metadata.
    """
    return paginated_response(
        items=repo.list_history(page=page, page_size=page_size),
        page=page,
        page_size=page_size,
        total=repo.count_history(),
        summary=repo.summarize_history(),
    )


@router.delete("/history")
def clear_document_request_history(repo: DocumentRequestRepository = Depends(get_repository)) -> dict[str, int]:
    """Delete historical document-request drafts and linked email traces.

    Args:
        repo: HANA repository.

    Returns:
        Counts of deleted drafts, extractions, and source emails.
    """
    return repo.clear_history()


@router.post("/drafts/{draft_id}/approve")
def approve_document_request_draft(
    draft_id: str,
    payload: DraftDecisionRequest | None = None,
    repo: DocumentRequestRepository = Depends(get_repository),
    gmail_client: GmailDestinationApiClient = Depends(get_gmail_client),
    mailbox_id: str = Depends(get_mailbox_id),
    settings: DocumentRequestSettings = Depends(get_settings),
) -> dict[str, object]:
    """Persist the reviewed draft body, send the Gmail draft, and approve the workflow.

    Args:
        draft_id: Review draft id to approve.
        payload: Optional reviewed body from the editable draft field.
        repo: HANA repository.
        gmail_client: Gmail API client.
        mailbox_id: Gmail mailbox id.
        settings: Runtime settings with optional From header.

    Returns:
        Updated draft row.
    """
    if not repo.lock_active_draft(draft_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DRAFT_ALREADY_DECIDED)
    draft = repo.get_draft(draft_id)
    gmail_draft_id = draft.get("gmail_draft_id")
    if not gmail_draft_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Draft has no Gmail draft id to send.")
    if gmail_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gmail destination is not configured.")
    reviewed_body = edited_body_from_payload(payload, draft)
    draft = repo.update_draft_body(draft_id, reviewed_body)
    raw_message = build_mime_reply_with_attachments(
        source_email=source_email_from_draft(draft),
        to_email=str(draft.get("to_email") or ""),
        from_email=settings.from_email,
        subject=str(draft.get("subject") or ""),
        body=reviewed_body,
        attachments=repo.get_selected_draft_attachments(draft),
    )
    gmail_client.update_draft(
        user_id=mailbox_id,
        draft_id=str(gmail_draft_id),
        raw_message=raw_message,
        thread_id=str(draft.get("source_thread_id") or "") or None,
    )
    sent = gmail_client.send_draft(user_id=mailbox_id, draft_id=str(gmail_draft_id))
    return repo.mark_draft_sent(draft_id, sent_message_id=sent.get("id"))


@router.post("/drafts/{draft_id}/reject")
def reject_document_request_draft(
    draft_id: str,
    payload: DraftDecisionRequest | None = None,
    repo: DocumentRequestRepository = Depends(get_repository),
    gmail_client: GmailDestinationApiClient | None = Depends(get_gmail_client),
    mailbox_id: str = Depends(get_mailbox_id),
) -> dict[str, object]:
    """Persist the reviewed draft body, delete the Gmail draft, and reject the workflow.

    Args:
        draft_id: Review draft id to reject.
        payload: Optional reviewed body from the editable draft field.
        repo: HANA repository.
        gmail_client: Gmail API client.
        mailbox_id: Gmail mailbox id.

    Returns:
        Updated draft row.
    """
    if not repo.lock_active_draft(draft_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=DRAFT_ALREADY_DECIDED)
    draft = repo.get_draft(draft_id)
    reviewed_body = edited_body_from_payload(payload, draft)
    repo.update_draft_body(draft_id, reviewed_body)
    gmail_draft_id = draft.get("gmail_draft_id")
    if gmail_draft_id and gmail_client:
        gmail_client.delete_draft(user_id=mailbox_id, draft_id=str(gmail_draft_id))
    return repo.mark_draft_rejected(draft_id)


@router.get("/documents/{document_id}")
def get_document_preview(
    document_id: str,
    repo: DocumentRequestRepository = Depends(get_repository),
) -> Response:
    """Serve a shipment document's binary content for browser preview.

    Args:
        document_id: Shipment document primary key.
        repo: HANA repository.

    Returns:
        Binary file content with appropriate Content-Type.
    """
    doc = repo.get_document_content(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return Response(
        content=bytes(doc["file_content"]),
        media_type=doc.get("mime_type") or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{doc["file_name"]}"'},
    )


def manual_payload_to_email(payload: ManualEmailRequest) -> GmailEmail:
    """Convert manual UI input into a normalized email row.

    Args:
        payload: Manual email request.

    Returns:
        GmailEmail-compatible model for processing.
    """
    sender_name, sender_email = parseaddr(payload.from_header)
    message_uuid = uuid.uuid4().hex
    return GmailEmail(
        gmail_message_id=f"manual-{message_uuid}",
        thread_id=f"manual-thread-{message_uuid}",
        rfc_message_id=f"<manual-{message_uuid}@local>",
        sender_name=sender_name or None,
        sender_email=sender_email or payload.from_header,
        subject=payload.subject,
        email_date=datetime.now(timezone.utc),
        body=payload.body,
    )


def load_scenario_emails() -> list[GmailEmail]:
    """Load processable scenario source emails from the repository data folder.

    Returns:
        GmailEmail rows sourced from data/anonymized.
    """
    source_dir = Path(os.getenv("SCENARIO_SOURCE_DIR") or Path(__file__).resolve().parents[3] / "data" / "anonymized")
    return load_scenario_emails_from_dir(source_dir)
