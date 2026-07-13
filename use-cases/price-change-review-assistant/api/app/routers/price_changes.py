from __future__ import annotations

from collections.abc import Iterator
import base64
from datetime import datetime, timezone
from urllib.parse import quote
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.encoders import jsonable_encoder
from sqlalchemy.engine import Connection

from ..price_changes.attachments import (
    AttachmentTooLargeError,
    UnsupportedAttachmentTypeError,
    build_email_attachment,
)
from ..price_changes.batch_resolution import (
    BatchPriceChangeProcessingError,
    BatchPriceChangeProcessor,
    PersistedDraftResult,
)
from ..price_changes.completion_model import CompletionModelClient
from ..price_changes.db import create_hana_engine, hana_connection
from ..price_changes.extraction_service import ExtractionService
from ..price_changes.gmail_service import (
    GmailFetchService,
    build_gmail_query,
    build_gmail_query_start,
    build_gmail_service,
    email_from_message,
)
from ..price_changes.llm_usage_logging import LlmUsageContext, build_llm_usage_context_from_request
from ..price_changes.models import (
    DraftPatchRequest,
    EmailAttachment,
    FetchSummary,
    GmailEmail,
    ManualEmailRequest,
    ProcessingRunCreateRequest,
)
from ..price_changes.progress import HanaProgressReporter
from ..price_changes.repositories import PriceChangeRepository, draft_validation_errors
from ..price_changes.s4_lookup import S4LookupRepository
from ..price_changes.s4_price_write import S4PriceApprovalService, is_successful_write_result
from ..price_changes.settings import PriceChangeSettings
from ..price_changes.token_usage import TokenUsageTracker
from ..price_changes.tools import PriceChangeTools
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])


def get_settings() -> PriceChangeSettings:
    return PriceChangeSettings.from_env()


def get_connection(settings: PriceChangeSettings = Depends(get_settings)) -> Iterator[Connection]:
    engine = create_hana_engine(settings)
    with hana_connection(engine) as connection:
        yield connection


def get_gmail_fetch_service(settings: PriceChangeSettings = Depends(get_settings)) -> GmailFetchService:
    return GmailFetchService(build_gmail_service(settings), mailbox_id=settings.gmail_mailbox_id)


def get_extraction_service(settings: PriceChangeSettings = Depends(get_settings)) -> ExtractionService:
    return ExtractionService(settings)


def get_s4_lookup_repository() -> S4LookupRepository:
    """Create the S/4 lookup repository used by agent tools.

    Returns:
        Repository backed by S/4 OData APIs for supplier, material, and current-price reads.
    """
    return S4LookupRepository.from_env()


def get_s4_price_approval_service(
    s4_lookup_repository: S4LookupRepository = Depends(get_s4_lookup_repository),
) -> S4PriceApprovalService:
    """Create the user-triggered S/4 price approval service.

    Args:
        s4_lookup_repository: S/4 lookup repository that owns the authenticated client.

    Returns:
        S/4 price approval service for PPR0 writes.
    """
    return S4PriceApprovalService(s4_lookup_repository)


def build_s4_approval_failure_detail(
    draft: dict[str, object],
    s4_write_result: dict[str, object],
) -> dict[str, object]:
    """Build the structured HTTP 424 response body for S/4 approval failures.

    Args:
        draft: Draft after recording the S/4 write attempt.
        s4_write_result: S/4 write result.

    Returns:
        JSON-serializable failure detail.
    """
    message = str(
        s4_write_result.get("message")
        or "S/4 price update could not be applied. Review the S/4 write result for details."
    )
    if not message.startswith("S/4 price update could not be applied"):
        message = f"S/4 price update could not be applied: {message}"
    return {
        "message": message,
        "s4_status": s4_write_result.get("status"),
        "s4_write_result": s4_write_result,
        "draft": jsonable_encoder(draft),
    }


def update_summary_from_draft_results(summary: FetchSummary, results: list[PersistedDraftResult]) -> None:
    """Apply persisted draft results to a mutable API processing summary.

    Args:
        summary: Fetch summary to update in place.
        results: Persisted draft results returned by successful or partial batch processing.

    Returns:
        None.
    """
    for result in results:
        summary.drafts_created += 1
        summary.created_draft_ids.append(result.draft_id)
        if result.proposal.status == "ready_for_review":
            summary.ready_for_review += 1
        elif result.proposal.status == "needs_human_review":
            summary.needs_human_review += 1


def process_price_change_email(
    email: GmailEmail,
    summary: FetchSummary,
    settings: PriceChangeSettings,
    repo: PriceChangeRepository,
    extractor: ExtractionService,
    tools: PriceChangeTools,
    attachments: list[EmailAttachment] | None = None,
    token_usage_tracker: TokenUsageTracker | None = None,
    llm_usage_context: LlmUsageContext | None = None,
    progress_reporter: object | None = None,
) -> None:
    """Persist and process one normalized email through extraction and batch completion.

    Args:
        email: Normalized GmailEmail record, including synthetic manual emails.
        summary: Mutable processing summary for the current API call.
        settings: Runtime settings containing extraction and completion model configuration.
        repo: HANA repository used for persistence.
        extractor: Extraction service for quick classification and raw item extraction.
        tools: Tool facade using S/4 for deterministic lookups and HANA draft persistence.
        attachments: Supported email attachments already normalized for persistence.
        token_usage_tracker: Optional tracker for local debugging token reports.
        llm_usage_context: Optional Cloud Logging request context.
        progress_reporter: Optional sanitized progress reporter for UI polling.

    Returns:
        None. The supplied summary is updated in place with processing outcomes.
    """
    attachments = attachments or []
    summary.fetched += 1
    repo.insert_gmail_email(email)
    if progress_reporter is not None:
        progress_reporter.event("email_received", "Email saved for processing")
    stored_attachments = repo.insert_email_attachments(email.gmail_message_id, attachments)
    summary.attachments_downloaded += len(stored_attachments)
    if progress_reporter is not None and stored_attachments:
        progress_reporter.event(
            "parsing_attachment_preview",
            "Parsing attachment preview",
            metadata={"attachments": len(stored_attachments)},
        )
    if progress_reporter is not None:
        progress_reporter.event("classifying_email", "Classifying email")
    extraction_result = extractor.extract(
        email,
        attachments=stored_attachments,
        token_usage_tracker=token_usage_tracker,
        llm_usage_context=llm_usage_context,
    )
    extraction = extraction_result.extraction
    extraction_id = repo.insert_extraction(
        gmail_message_id=email.gmail_message_id,
        model_name=extraction_result.model_name or settings.extractor_model,
        raw_model_output=extraction_result.raw_model_output,
        normalized_json=extraction.model_dump(),
        status="ok" if not extraction_result.validation_errors else "failed",
        validation_errors=extraction_result.validation_errors,
    )
    if extraction_result.validation_errors:
        summary.extraction_failed += 1
        repo.update_email_status(
            email.gmail_message_id,
            "extraction_failed",
            "; ".join(extraction_result.validation_errors),
        )
        return
    if not extraction.is_price_request:
        summary.not_price_request += 1
        repo.update_email_status(email.gmail_message_id, "not_price_request")
        return
    if progress_reporter is not None:
        progress_reporter.event(
            "extracting_requested_price_lines",
            "Extracting requested price lines",
            metadata={"items": len(extraction.items or [])},
        )
    completion_client = CompletionModelClient(
        model_name=settings.agent_model,
        reasoning_effort=settings.agent_reasoning_effort,
    )
    processor = BatchPriceChangeProcessor(
        tools=tools,
        completion_client=completion_client,
        completion_batch_size=settings.completion_batch_size,
    )
    try:
        results = processor.process_email(
            email=email,
            extraction=extraction,
            extraction_id=extraction_id,
            token_usage_tracker=token_usage_tracker,
            progress_reporter=progress_reporter,
            llm_usage_context=llm_usage_context,
        )
    except BatchPriceChangeProcessingError as exc:
        update_summary_from_draft_results(summary, exc.partial_results)
        summary.agent_failed += 1
        repo.update_email_status(email.gmail_message_id, "agent_failed", str(exc))
        raise
    except Exception as exc:
        summary.agent_failed += 1
        repo.update_email_status(email.gmail_message_id, "agent_failed", str(exc))
        raise
    update_summary_from_draft_results(summary, results)
    repo.update_email_status(email.gmail_message_id, "processed")


def manual_request_to_email(request: ManualEmailRequest, received_at: datetime) -> GmailEmail:
    """Convert a manual test request into the GmailEmail shape used by processing.

    Args:
        request: Manual email fields submitted by the UI.
        received_at: UTC timestamp assigned to the synthetic email.

    Returns:
        GmailEmail with a unique manual message id.
    """
    return GmailEmail(
        gmail_message_id=f"manual-{uuid.uuid4().hex}",
        sender_email=request.sender_email,
        subject=request.subject,
        email_date=received_at,
        body=request.body,
        fetched_at=received_at,
        processing_status="fetched",
    )


@router.post("/emails/fetch-new", response_model=FetchSummary, response_model_exclude_none=True)
async def fetch_new_emails(
    request: Request,
    processing_run_id: str | None = Query(None),
    include_token_usage: bool = Query(False),
    settings: PriceChangeSettings = Depends(get_settings),
    connection: Connection = Depends(get_connection),
    service: GmailFetchService = Depends(get_gmail_fetch_service),
    extractor: ExtractionService = Depends(get_extraction_service),
    s4_lookup_repository: S4LookupRepository = Depends(get_s4_lookup_repository),
) -> FetchSummary:
    summary = FetchSummary()
    token_usage_tracker = TokenUsageTracker() if include_token_usage else None
    llm_usage_context = build_llm_usage_context_from_request(request)
    started_at = datetime.now(timezone.utc)
    repo = PriceChangeRepository(connection)
    progress_reporter = None
    if processing_run_id:
        repo.create_processing_run(processing_run_id, "gmail_fetch")
        progress_reporter = HanaProgressReporter(repo, processing_run_id)
        progress_reporter.event("fetching_email_refs", "Fetching new emails")
    repo.record_fetch_started(settings.gmail_mailbox_id, started_at)
    try:
        query_start = build_gmail_query_start(
            now=started_at,
            last_successful_fetch_at=repo.get_last_successful_fetch_at(settings.gmail_mailbox_id),
            fetch_max_days=settings.fetch_max_days,
        )
        refs = service.list_inbox_message_refs(build_gmail_query(query_start))
        messages = service.get_full_messages(refs)
        tools = PriceChangeTools(repo, s4_lookup_repository)
        for message in messages:
            gmail_message_id = str(message["id"])
            if repo.email_exists(gmail_message_id):
                summary.skipped_existing += 1
                continue
            attachment_result = service.download_supported_attachments(message)
            summary.attachments_skipped += attachment_result.skipped
            summary.attachments_failed += attachment_result.failed
            email = email_from_message(message)
            process_price_change_email(
                email=email,
                summary=summary,
                settings=settings,
                repo=repo,
                extractor=extractor,
                tools=tools,
                attachments=attachment_result.attachments,
                token_usage_tracker=token_usage_tracker,
                llm_usage_context=llm_usage_context,
                progress_reporter=progress_reporter,
            )
        if token_usage_tracker is not None:
            summary.token_usage = token_usage_tracker.report()
        repo.record_fetch_finished(
            settings.gmail_mailbox_id,
            started_at,
            datetime.now(timezone.utc),
            "success",
            summary.model_dump(exclude={"token_usage"}),
        )
        if processing_run_id:
            repo.complete_processing_run(processing_run_id)
        return summary
    except Exception as exc:
        repo.record_fetch_finished(
            settings.gmail_mailbox_id,
            started_at,
            datetime.now(timezone.utc),
            "failed",
            summary.model_dump(exclude={"token_usage"}),
            str(exc),
        )
        if processing_run_id:
            if progress_reporter is not None:
                progress_reporter.event("analysis_failed", "Analysis failed", level="error")
            repo.fail_processing_run(processing_run_id, str(exc))
        raise


@router.post("/emails/manual", response_model=FetchSummary, response_model_exclude_none=True)
async def submit_manual_email(
    fastapi_request: Request,
    request: ManualEmailRequest,
    include_token_usage: bool = Query(False),
    settings: PriceChangeSettings = Depends(get_settings),
    connection: Connection = Depends(get_connection),
    extractor: ExtractionService = Depends(get_extraction_service),
    s4_lookup_repository: S4LookupRepository = Depends(get_s4_lookup_repository),
) -> FetchSummary:
    """Process one manually-entered email without calling Gmail.

    Args:
        request: Manual sender, subject, and body fields submitted by the UI.
        include_token_usage: Whether to include local debugging token usage details.
        settings: Runtime settings containing extractor and agent model names.
        connection: HANA connection for persistence.
        extractor: Extraction service dependency.
        s4_lookup_repository: S/4 lookup repository for supplier, material, and current-price reads.

    Returns:
        FetchSummary-compatible processing result for the manual email.
    """
    summary = FetchSummary()
    token_usage_tracker = TokenUsageTracker() if include_token_usage else None
    llm_usage_context = build_llm_usage_context_from_request(fastapi_request)
    repo = PriceChangeRepository(connection)
    progress_reporter = None
    if request.processing_run_id:
        repo.create_processing_run(request.processing_run_id, "manual")
        progress_reporter = HanaProgressReporter(repo, request.processing_run_id)
    tools = PriceChangeTools(repo, s4_lookup_repository)
    received_at = datetime.now(timezone.utc)
    email = manual_request_to_email(request, received_at)
    try:
        process_price_change_email(
            email=email,
            summary=summary,
            settings=settings,
            repo=repo,
            extractor=extractor,
            tools=tools,
            attachments=[],
            token_usage_tracker=token_usage_tracker,
            llm_usage_context=llm_usage_context,
            progress_reporter=progress_reporter,
        )
        if token_usage_tracker is not None:
            summary.token_usage = token_usage_tracker.report()
        if request.processing_run_id:
            repo.complete_processing_run(request.processing_run_id)
        return summary
    except Exception as exc:
        if request.processing_run_id:
            if progress_reporter is not None:
                progress_reporter.event("analysis_failed", "Analysis failed", level="error")
            repo.fail_processing_run(request.processing_run_id, str(exc))
        raise


@router.post("/emails/manual-with-attachments", response_model=FetchSummary, response_model_exclude_none=True)
async def submit_manual_email_with_attachments(
    request: Request,
    sender_email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    attachments: list[UploadFile] | None = File(None),
    processing_run_id: str | None = Form(None),
    include_token_usage: bool = Query(False),
    settings: PriceChangeSettings = Depends(get_settings),
    connection: Connection = Depends(get_connection),
    extractor: ExtractionService = Depends(get_extraction_service),
    s4_lookup_repository: S4LookupRepository = Depends(get_s4_lookup_repository),
) -> FetchSummary:
    """Process one manually-entered email and browser-uploaded attachments.

    Args:
        sender_email: Synthetic sender email address.
        subject: Synthetic email subject.
        body: Plain-text email body.
        attachments: Optional PDF, CSV, or XLSX uploads.
        include_token_usage: Whether to include local debugging token usage details.
        settings: Runtime settings containing extractor and agent model names.
        connection: HANA connection for persistence.
        extractor: Extraction service dependency.
        s4_lookup_repository: S/4 lookup repository for supplier, material, and current-price reads.

    Returns:
        FetchSummary-compatible processing result for the manual email.
    """
    request = ManualEmailRequest(sender_email=sender_email, subject=subject, body=body)
    summary = FetchSummary()
    normalized_attachments: list[EmailAttachment] = []
    for upload in attachments or []:
        content = await upload.read()
        try:
            normalized_attachments.append(
                build_email_attachment(
                    filename=upload.filename or "attachment",
                    mime_type=upload.content_type,
                    content=content,
                    source="manual",
                )
            )
        except UnsupportedAttachmentTypeError:
            summary.attachments_skipped += 1
        except AttachmentTooLargeError as exc:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc

    token_usage_tracker = TokenUsageTracker() if include_token_usage else None
    llm_usage_context = build_llm_usage_context_from_request(request)
    repo = PriceChangeRepository(connection)
    progress_reporter = None
    if processing_run_id:
        repo.create_processing_run(processing_run_id, "manual")
        progress_reporter = HanaProgressReporter(repo, processing_run_id)
    tools = PriceChangeTools(repo, s4_lookup_repository)
    received_at = datetime.now(timezone.utc)
    email = manual_request_to_email(request, received_at)
    try:
        process_price_change_email(
            email=email,
            summary=summary,
            settings=settings,
            repo=repo,
            extractor=extractor,
            tools=tools,
            attachments=normalized_attachments,
            token_usage_tracker=token_usage_tracker,
            llm_usage_context=llm_usage_context,
            progress_reporter=progress_reporter,
        )
        if token_usage_tracker is not None:
            summary.token_usage = token_usage_tracker.report()
        if processing_run_id:
            repo.complete_processing_run(processing_run_id)
        return summary
    except Exception as exc:
        if processing_run_id:
            if progress_reporter is not None:
                progress_reporter.event("analysis_failed", "Analysis failed", level="error")
            repo.fail_processing_run(processing_run_id, str(exc))
        raise


@router.post("/processing-runs")
async def create_processing_run(
    request: ProcessingRunCreateRequest,
    connection: Connection = Depends(get_connection),
) -> dict[str, object]:
    """Create a processing progress run before a long request starts.

    Args:
        request: Browser-generated run id and source type.
        connection: HANA connection for persistence.

    Returns:
        Snapshot containing the newly-created run and initial event.
    """
    repo = PriceChangeRepository(connection)
    repo.create_processing_run(request.processing_run_id, request.source_type)
    repo.append_processing_event(request.processing_run_id, "preparing_analysis", "Preparing analysis")
    return repo.get_processing_run_snapshot(request.processing_run_id)


@router.get("/processing-runs/{processing_run_id}")
async def get_processing_run(
    processing_run_id: str,
    connection: Connection = Depends(get_connection),
) -> dict[str, object]:
    """Return the latest persisted processing progress snapshot.

    Args:
        processing_run_id: Browser-generated processing run id.
        connection: HANA connection for persistence.

    Returns:
        Snapshot containing the run row and ordered sanitized events.
    """
    try:
        return PriceChangeRepository(connection).get_processing_run_snapshot(processing_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing run not found.") from exc


@router.get("/email-attachments/{attachment_id}/download")
async def download_email_attachment(
    attachment_id: str,
    connection: Connection = Depends(get_connection),
) -> Response:
    """Download one persisted email attachment for audit review.

    Args:
        attachment_id: Persisted attachment id.
        connection: HANA connection for persistence.

    Returns:
        HTTP response containing the original attachment bytes.
    """
    try:
        attachment = PriceChangeRepository(connection).get_email_attachment(attachment_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found.") from exc
    filename = str(attachment.get("filename") or "attachment")
    content = base64.b64decode(str(attachment.get("content_base64") or "").encode("ascii"))
    return Response(
        content=content,
        media_type=str(attachment.get("mime_type") or "application/octet-stream"),
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{filename.replace(chr(34), '')}\"; "
                f"filename*=UTF-8''{quote(filename)}"
            )
        },
    )


@router.get("/price-change-drafts")
async def list_price_change_drafts(connection: Connection = Depends(get_connection)) -> dict[str, object]:
    return {"items": PriceChangeRepository(connection).list_price_change_drafts()}


@router.get("/price-change-history")
async def list_price_change_history(
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
    connection: Connection = Depends(get_connection),
) -> dict[str, object]:
    """Return one paginated block of approved/rejected price-change history.

    Args:
        limit: Page size requested by the UI, capped at 50.
        offset: Number of history rows to skip from the newest-first result set.
        connection: HANA connection for persistence.

    Returns:
        History items plus global approved/rejected/total counts and page metadata.
    """
    capped_limit = min(limit, 50)
    repo = PriceChangeRepository(connection)
    summary = repo.price_change_history_summary()
    return {
        "items": repo.list_price_change_history(limit=capped_limit, offset=offset),
        **summary,
        "limit": capped_limit,
        "offset": offset,
    }


@router.delete("/price-change-history")
async def reset_price_change_history(connection: Connection = Depends(get_connection)) -> dict[str, int]:
    """Delete approved/rejected history and private audit records.

    Args:
        connection: HANA connection for persistence.

    Returns:
        Number of terminal history draft rows deleted.
    """
    return PriceChangeRepository(connection).reset_price_change_history()


@router.patch("/price-change-drafts/{draft_id}")
async def patch_price_change_draft(
    draft_id: str,
    request: DraftPatchRequest,
    connection: Connection = Depends(get_connection),
) -> dict[str, object]:
    values = request.model_dump(exclude_unset=True)
    if "material_number" in values:
        values["material_code"] = values.pop("material_number")
    return PriceChangeRepository(connection).patch_price_change_draft(draft_id, values)


@router.post("/price-change-drafts/{draft_id}/approve")
async def approve_price_change_draft(
    draft_id: str,
    connection: Connection = Depends(get_connection),
    s4_price_approval_service: S4PriceApprovalService = Depends(get_s4_price_approval_service),
) -> dict[str, object]:
    repo = PriceChangeRepository(connection)
    try:
        draft = repo.get_price_change_draft(draft_id)
        validation_errors = draft_validation_errors(draft)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        s4_write_result = s4_price_approval_service.approve_draft(draft)
        draft_after_attempt = repo.record_s4_write_attempt(draft_id, s4_write_result)
        if not is_successful_write_result(s4_write_result):
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail=build_s4_approval_failure_detail(draft_after_attempt, s4_write_result),
            )
        return repo.approve_price_change_draft(draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/price-change-drafts/{draft_id}/reject")
async def reject_price_change_draft(
    draft_id: str,
    connection: Connection = Depends(get_connection),
) -> dict[str, object]:
    return PriceChangeRepository(connection).reject_price_change_draft(draft_id)
