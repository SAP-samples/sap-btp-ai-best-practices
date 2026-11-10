"""Email processing pipeline and background run lifecycle.

One email goes through: store source email -> LLM extraction -> agent
(HANA lookup, LLM reply composition, Gmail draft, persist draft row).

Design choices (see docs/background-email-processing.md):

- Every email uses its own short HANA transactions. A failure on one email
  is recorded on that email only; already processed emails stay committed.
- Emails of one run are processed in parallel by a bounded thread pool
  (``FETCH_MAX_WORKERS``). LLM calls dominate latency (seconds per call),
  so threads give a near-linear speed-up.
- Inbox fetches and scenario loads run as background runs. A HANA row per
  run key acts as a cross-instance lock and as the progress record polled
  by the UI.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.engine import Engine

from ..observability.llm_usage_logging import llm_call_context
from .agent import DocumentRequestAgent, build_failed_proposal
from .db import hana_connection
from .gmail_service import build_gmail_query, build_gmail_query_start, email_from_message
from .models import FetchSummary, GmailEmail
from .repositories import DocumentRequestRepository
from .settings import DocumentRequestSettings

logger = logging.getLogger(__name__)

# A run flagged "running" for longer than this is treated as crashed (for
# example the CF instance restarted mid-run) and a new run may take over.
RUN_STALE_AFTER = timedelta(minutes=30)

# Run key for scenario-email loads; Gmail runs use the mailbox id as key.
SCENARIO_RUN_KEY = "scenarios"

RepositoryScope = Callable[[], AbstractContextManager[DocumentRequestRepository]]
AgentFactory = Callable[[DocumentRequestRepository], DocumentRequestAgent]


@contextmanager
def repository_scope(engine: Engine) -> Iterator[DocumentRequestRepository]:
    """Open one HANA transaction and yield a repository bound to it.

    Commits when the block exits normally, rolls back on exception.

    Args:
        engine: Pooled SQLAlchemy engine.

    Yields:
        DocumentRequestRepository using the transactional connection.
    """
    with hana_connection(engine) as connection:
        yield DocumentRequestRepository(connection)


@dataclass(frozen=True)
class EmailOutcome:
    """Result of processing one email, used to update run counters.

    Attributes:
        kind: skipped_existing, extraction_failed, not_document_request, agent_failed, or processed.
        draft_id: Persisted review draft id, when one was created.
        proposal_status: Draft status (ready_for_review / needs_human_review) for processed emails.
    """

    kind: str
    draft_id: str | None = None
    proposal_status: str | None = None


def apply_outcome(summary: FetchSummary, outcome: EmailOutcome) -> None:
    """Add one email outcome to the run summary counters.

    Args:
        summary: Summary mutated in place.
        outcome: Outcome of one processed email.
    """
    if outcome.kind == "skipped_existing":
        summary.skipped_existing += 1
        return
    summary.fetched += 1
    if outcome.kind == "extraction_failed":
        summary.extraction_failed += 1
    elif outcome.kind == "not_document_request":
        summary.not_document_request += 1
    elif outcome.kind == "agent_failed":
        summary.agent_failed += 1
    if outcome.draft_id:
        summary.drafts_created += 1
        summary.created_draft_ids.append(outcome.draft_id)
    if outcome.proposal_status == "ready_for_review":
        summary.ready_for_review += 1
    elif outcome.proposal_status == "needs_human_review":
        summary.needs_human_review += 1


@dataclass
class EmailPipeline:
    """Process emails end to end with per-email transactions.

    Attributes:
        repository_scope: Factory returning a transactional repository context.
        agent_factory: Builds a DocumentRequestAgent bound to a repository.
        extractor: Service exposing ``extract(email)`` (LLM call 1).
        settings: Runtime settings (model names, worker count).
    """

    repository_scope: RepositoryScope
    agent_factory: AgentFactory
    extractor: Any
    settings: DocumentRequestSettings

    def process_email(self, email: GmailEmail, *, route: str, actor_type: str) -> EmailOutcome:
        """Process one email. Never raises: failures become an outcome.

        Args:
            email: Normalized source email.
            route: API route that triggered processing (for usage logging).
            actor_type: "batch" for background runs, "human" for manual input.

        Returns:
            EmailOutcome describing what happened.
        """
        with llm_call_context(route=route, method="POST", actor_type=actor_type, correlation_id=email.gmail_message_id):
            try:
                return self._process_email(email)
            except Exception as exc:
                logger.exception("Email %s failed", email.gmail_message_id)
                self._mark_email_failed(email.gmail_message_id, "agent_failed", str(exc))
                return EmailOutcome(kind="agent_failed")

    def _process_email(self, email: GmailEmail) -> EmailOutcome:
        """Run the pipeline steps for one email (may raise).

        Args:
            email: Normalized source email.

        Returns:
            EmailOutcome describing what happened.
        """
        # Transaction 1: claim the email so no other run processes it again.
        with self.repository_scope() as repo:
            if repo.email_exists(email.gmail_message_id):
                return EmailOutcome(kind="skipped_existing")
            repo.insert_gmail_email(email)

        # LLM call 1 runs outside any transaction (no DB connection held).
        extraction_result = self.extractor.extract(email)
        extraction = extraction_result.extraction

        # Transaction 2: store the extraction result.
        with self.repository_scope() as repo:
            extraction_id = repo.insert_extraction(
                gmail_message_id=email.gmail_message_id,
                model_name=self.settings.extractor_model,
                raw_model_output=extraction_result.raw_model_output,
                normalized_json=extraction.model_dump(),
                status="ok" if not extraction_result.validation_errors else "failed",
                validation_errors=extraction_result.validation_errors,
            )
            if extraction_result.validation_errors:
                repo.update_email_status(
                    email.gmail_message_id, "extraction_failed", "; ".join(extraction_result.validation_errors)
                )
                return EmailOutcome(kind="extraction_failed")
            if not extraction.is_document_request:
                repo.update_email_status(email.gmail_message_id, "not_document_request")
                return EmailOutcome(kind="not_document_request")

        # Transaction 3: agent (HANA lookups, LLM call 2, Gmail draft, draft row).
        try:
            with self.repository_scope() as repo:
                proposal, draft_id = self.agent_factory(repo).resolve_email(
                    email=email, extraction=extraction, extraction_id=extraction_id
                )
                repo.update_email_status(email.gmail_message_id, "processed")
            return EmailOutcome(kind="processed", draft_id=draft_id, proposal_status=proposal.status)
        except Exception as exc:
            logger.exception("Agent failed for email %s", email.gmail_message_id)
            # Transaction 3 rolled back; record a failed draft so a reviewer sees it.
            failed = build_failed_proposal(email=email, extraction=extraction, extraction_id=extraction_id, error=exc)
            with self.repository_scope() as repo:
                draft_id = repo.insert_document_email_draft(
                    proposal=failed,
                    gmail_draft_id=None,
                    gmail_draft_message_id=None,
                    raw_agent_output=failed.model_dump(mode="json", exclude={"selected_documents"}),
                )
                repo.update_email_status(email.gmail_message_id, "agent_failed", str(exc))
            return EmailOutcome(kind="agent_failed", draft_id=draft_id)

    def _mark_email_failed(self, gmail_message_id: str, status: str, error: str) -> None:
        """Best-effort status update after an unexpected failure.

        Args:
            gmail_message_id: Source email id.
            status: Email processing status to store.
            error: Error message.
        """
        try:
            with self.repository_scope() as repo:
                repo.update_email_status(gmail_message_id, status, error)
        except Exception:
            logger.exception("Could not record failure for email %s", gmail_message_id)

    def process_batch(
        self,
        emails: list[GmailEmail],
        summary: FetchSummary,
        *,
        route: str,
        actor_type: str,
        on_progress: Callable[[FetchSummary], None] | None = None,
    ) -> None:
        """Process emails in parallel and update ``summary`` as each finishes.

        Args:
            emails: Emails to process.
            summary: Summary mutated in place (thread-safe via a lock).
            route: API route for usage logging.
            actor_type: Actor type for usage logging.
            on_progress: Optional callback receiving the summary after each email.
        """
        lock = threading.Lock()

        def work(email: GmailEmail) -> None:
            outcome = self.process_email(email, route=route, actor_type=actor_type)
            with lock:
                apply_outcome(summary, outcome)
                if on_progress:
                    on_progress(summary)

        # ponytail: in-process thread pool; move to a message queue if volume outgrows one instance.
        with ThreadPoolExecutor(max_workers=self.settings.fetch_max_workers) as pool:
            list(pool.map(work, emails))


def try_start_run(repository_scope: RepositoryScope, run_key: str, started_at: datetime) -> bool:
    """Claim the run lock for ``run_key`` in its own committed transaction.

    Args:
        repository_scope: Transactional repository factory.
        run_key: Run identifier.
        started_at: Start timestamp of the new run.

    Returns:
        True when the run was started, False when another run is active.
    """
    with repository_scope() as repo:
        return repo.try_start_fetch_run(run_key, started_at, started_at - RUN_STALE_AFTER)


def _execute_run(
    pipeline: EmailPipeline,
    run_key: str,
    started_at: datetime,
    route: str,
    load_emails: Callable[[], list[GmailEmail]],
) -> None:
    """Load emails, process them, and record progress and the final state.

    Args:
        pipeline: Email pipeline.
        run_key: Run identifier (already claimed with try_start_run).
        started_at: Run start timestamp.
        route: API route for usage logging.
        load_emails: Callable returning the emails of this run.
    """
    summary = FetchSummary()

    def save_progress(current: FetchSummary) -> None:
        with pipeline.repository_scope() as repo:
            repo.update_fetch_progress(run_key, current.model_dump())

    try:
        emails = load_emails()
        summary.total = len(emails)
        save_progress(summary)
        pipeline.process_batch(emails, summary, route=route, actor_type="batch", on_progress=save_progress)
        status, error = "success", None
    except Exception as exc:
        logger.exception("Run %s failed", run_key)
        status, error = "failed", str(exc)
    with pipeline.repository_scope() as repo:
        repo.record_fetch_finished(run_key, started_at, datetime.now(timezone.utc), status, summary.model_dump(), error)


def run_gmail_fetch(pipeline: EmailPipeline, fetch_service: Any, run_key: str, started_at: datetime, route: str) -> None:
    """Background body of an inbox fetch run.

    Args:
        pipeline: Email pipeline.
        fetch_service: GmailFetchService for the mailbox.
        run_key: Gmail mailbox id (already claimed).
        started_at: Run start timestamp.
        route: API route for usage logging.
    """

    def load_emails() -> list[GmailEmail]:
        with pipeline.repository_scope() as repo:
            watermark = repo.get_last_successful_fetch_at(run_key)
        query_start = build_gmail_query_start(
            now=started_at, last_successful_fetch_at=watermark, fetch_max_days=pipeline.settings.fetch_max_days
        )
        refs = fetch_service.list_inbox_message_refs(build_gmail_query(query_start))
        return [email_from_message(message) for message in fetch_service.get_full_messages(refs)]

    _execute_run(pipeline, run_key, started_at, route, load_emails)


def run_scenario_seed(pipeline: EmailPipeline, emails: list[GmailEmail], started_at: datetime, route: str) -> None:
    """Background body of a scenario-email load: reset then reprocess samples.

    Args:
        pipeline: Email pipeline.
        emails: Scenario emails to (re)process.
        started_at: Run start timestamp.
        route: API route for usage logging.
    """

    def load_emails() -> list[GmailEmail]:
        with pipeline.repository_scope() as repo:
            repo.delete_emails_by_ids([email.gmail_message_id for email in emails])
        return emails

    _execute_run(pipeline, SCENARIO_RUN_KEY, started_at, route, load_emails)
