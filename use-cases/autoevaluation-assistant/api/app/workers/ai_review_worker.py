"""Run the assessment AI review worker.

Example commands:
    cd api
    python -m app.workers.ai_review_worker
    python -m app.workers.ai_review_worker --once
    python -m app.workers.ai_review_worker --poll-interval 10
    python -m app.workers.ai_review_worker --document-ingestion-concurrency 2
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from app.db import create_hana_engine, is_transient_database_error
from app.models.ai_review import QuestionReviewResult
from app.models.documents import DocumentIngestionJobResponse
from app.models.reports import AssessmentReportSource
from app.services.assessment_report_localization import invalid_snapshot_message
from app.services.assessment_report_pdf import render_assessment_report_pdf
from app.services.batch_review_graph import (
    run_document_ingestion_from_hana,
)
from app.services.customer_class_scope import scope_question_to_max_level
from app.services.ai_review_repository.hana import HanaAiReviewRepository

logger = logging.getLogger(__name__)


def _commit_if_supported(repository: Any) -> None:
    """Commit repository state when the repository exposes an explicit commit.

    Inputs:
        repository: Repository object that may expose ``commit``.

    Outputs:
        None. Durable repositories commit their current transaction, while
        repositories without a commit hook are ignored.
    """

    commit = getattr(repository, "commit", None)
    if callable(commit):
        commit()


def _assessment_report_file_name(source: AssessmentReportSource) -> str:
    """Build a safe, stable browser download filename for a report.

    Inputs:
        source: Immutable report source containing assessment and language IDs.

    Outputs:
        str: ASCII filename safe for the HTTP content-disposition header.
    """

    safe_assessment_id = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        source.assessment_id,
    ).strip("-.")
    if not safe_assessment_id:
        safe_assessment_id = "assessment"
    return (
        f"calification-report-{safe_assessment_id}-{source.language.lower()}.pdf"
    )


def _raw_report_language(source: Any) -> str:
    """Read a safe language from an unvalidated leased source mapping.

    Inputs:
        source: Raw decoded report source value returned after leasing.

    Outputs:
        str: ``it`` only for an exact Italian marker, otherwise ``en``.
    """

    if isinstance(source, dict) and source.get("language") == "it":
        return "it"
    return "en"


def _fail_snapshot_job(
    repository: Any,
    *,
    job_id: str,
    worker_id: str,
    error_code: str,
    error_message: str,
) -> None:
    """Persist and commit one safe leased snapshot compatibility failure.

    Inputs:
        repository: Report repository owning the active worker lease.
        job_id: Leased report job identity.
        worker_id: Current lease owner.
        error_code: Stable public machine code.
        error_message: Localized safe rerun instruction.

    Outputs:
        None. The normal 24-hour terminal retention window begins.

    Raises:
        DBAPIError: Durable write failures propagate to worker recovery.
    """

    repository.fail_assessment_report_job(
        job_id=job_id,
        worker_id=worker_id,
        error_code=error_code,
        error_message=error_message,
    )
    _commit_if_supported(repository)


def process_one_assessment_report_job(
    repository: Any,
    worker_id: str,
    render_pdf: Callable[[AssessmentReportSource], bytes] = render_assessment_report_pdf,
) -> bool:
    """Lease, validate, render, and persist one assessment report job.

    Inputs:
        repository: Repository exposing the assessment report lifecycle methods.
        worker_id: Identifier that owns the recoverable 15-minute lease.
        render_pdf: Injectable deterministic renderer accepting one current snapshot.

    Outputs:
        bool: ``True`` when one job was handled, otherwise ``False``.

    Notes:
        A crashed process leaves an expiring lease that another worker can recover.
    """

    lease_next = getattr(repository, "lease_next_assessment_report_job", None)
    if not callable(lease_next):
        return False
    job = lease_next(worker_id)
    if job is None:
        return False
    _commit_if_supported(repository)
    raw_source = job.get("source")
    language = _raw_report_language(raw_source)
    try:
        source = AssessmentReportSource.model_validate(raw_source)
    except Exception:
        _fail_snapshot_job(
            repository,
            job_id=job["job_id"],
            worker_id=worker_id,
            error_code="assessment_report_snapshot_invalid",
            error_message=invalid_snapshot_message(language),
        )
        logger.exception("Assessment report job %s has an invalid snapshot", job["job_id"])
        return True
    try:
        repository.update_assessment_report_job_progress(
            job_id=job["job_id"],
            worker_id=worker_id,
            status="rendering",
            progress_message="Rendering PDF report",
        )
        _commit_if_supported(repository)
        pdf_content = render_pdf(source)
        repository.complete_assessment_report_job(
            job_id=job["job_id"],
            worker_id=worker_id,
            file_name=_assessment_report_file_name(source),
            pdf_content=pdf_content,
        )
        _commit_if_supported(repository)
        logger.info("Completed assessment report job %s", job["job_id"])
    except DBAPIError:
        raise
    except Exception:
        repository.fail_assessment_report_job(
            job_id=job["job_id"],
            worker_id=worker_id,
            error_code="assessment_report_generation_failed",
            error_message="The report could not be generated. Please try again.",
        )
        _commit_if_supported(repository)
        logger.exception("Assessment report job %s failed", job["job_id"])
    return True


def process_one_document_ingestion_job(
    repository: Any,
    worker_id: str,
    run_ingestion: Callable[[dict[str, Any]], DocumentIngestionJobResponse],
) -> bool:
    """Lease and process exactly one pending document ingestion job.

    Inputs:
        repository: Repository exposing document ingestion lease and persistence
            methods.
        worker_id: Identifier for the active worker process.
        run_ingestion: Callable that indexes the leased document corpus job.

    Outputs:
        bool: True when one ingestion job was processed, False when none was
        available.
    """
    lease_next = getattr(repository, "lease_next_document_ingestion_job", None)
    if not callable(lease_next):
        return False
    job = lease_next(worker_id)
    if job is None:
        return False
    _commit_if_supported(repository)
    try:
        result = run_ingestion(job)
        repository.complete_document_ingestion_job(
            job_id=job["job_id"],
            worker_id=worker_id,
            indexed_chunk_count=result.indexed_chunk_count,
        )
        _commit_if_supported(repository)
    except DBAPIError:
        raise
    except Exception as exc:
        repository.save_document_ingestion_failure(
            job_id=job["job_id"],
            worker_id=worker_id,
            error_code="document_ingestion_failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        _commit_if_supported(repository)
        logger.exception("Document ingestion job %s failed", job["job_id"])
    return True


class _AdminDocumentIngestionRepositoryAdapter:
    """Expose admin ingestion methods through the generic ingestion interface.

    Inputs:
        repository: Repository exposing admin document persistence methods.

    Outputs:
        Adapter object accepted by ``run_document_ingestion_from_hana``.
    """

    def __init__(self, repository: Any) -> None:
        """Store the wrapped repository.

        Inputs:
            repository: Repository that owns admin document persistence.

        Outputs:
            None.
        """

        self.repository = repository

    def update_document_ingestion_progress(self, **kwargs: Any) -> None:
        """Forward ingestion progress updates to admin document tables.

        Inputs:
            kwargs: Progress fields accepted by the repository method.

        Outputs:
            None.
        """

        self.repository.update_admin_document_ingestion_progress(**kwargs)

    def save_document_extraction(self, **kwargs: Any) -> str:
        """Persist extracted source blocks in admin document tables.

        Inputs:
            kwargs: Extraction fields accepted by the repository method.

        Outputs:
            str: Generated extraction identifier.
        """

        return self.repository.save_admin_document_extraction(**kwargs)

    def save_document_chunks(
        self,
        assessment_id: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Persist embedded chunks in admin document tables.

        Inputs:
            assessment_id: Ignored synthetic corpus identifier from the generic
                ingestion flow.
            document_id: Admin document identifier.
            chunks: Embedded retrieval chunk dictionaries.

        Outputs:
            None.
        """

        _ = assessment_id
        self.repository.save_admin_document_chunks(
            document_id=document_id,
            chunks=chunks,
        )

    def commit(self) -> None:
        """Commit the wrapped repository when supported.

        Inputs:
            None.

        Outputs:
            None.
        """

        _commit_if_supported(self.repository)


def run_admin_document_ingestion_from_hana(
    repository: Any,
    job: dict[str, Any],
) -> DocumentIngestionJobResponse:
    """Extract and index uploaded documents for the global admin corpus.

    Inputs:
        repository: Repository exposing admin document ingestion methods.
        job: Leased admin ingestion job with documents.

    Outputs:
        DocumentIngestionJobResponse: Completed admin ingestion progress.
    """

    return run_document_ingestion_from_hana(
        repository=_AdminDocumentIngestionRepositoryAdapter(repository),
        job=job,
    )


def process_one_admin_document_ingestion_job(
    repository: Any,
    worker_id: str,
    run_ingestion: Callable[[dict[str, Any]], DocumentIngestionJobResponse],
) -> bool:
    """Lease and process one pending global admin document ingestion job.

    Inputs:
        repository: Repository exposing admin document ingestion lease and
            persistence methods.
        worker_id: Identifier for the active worker process.
        run_ingestion: Callable that indexes the leased admin corpus job.

    Outputs:
        bool: True when one ingestion job was processed, False when none was
        available.
    """

    lease_next = getattr(repository, "lease_next_admin_document_ingestion_job", None)
    if not callable(lease_next):
        return False
    job = lease_next(worker_id)
    if job is None:
        return False
    _commit_if_supported(repository)
    try:
        result = run_ingestion(job)
        repository.complete_admin_document_ingestion_job(
            job_id=job["job_id"],
            worker_id=worker_id,
            indexed_chunk_count=result.indexed_chunk_count,
        )
        _commit_if_supported(repository)
    except DBAPIError:
        raise
    except Exception as exc:
        repository.save_admin_document_ingestion_failure(
            job_id=job["job_id"],
            worker_id=worker_id,
            error_code="admin_document_ingestion_failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        _commit_if_supported(repository)
        logger.exception("Admin document ingestion job %s failed", job["job_id"])
    return True


def process_one_task(
    repository: Any,
    worker_id: str,
    run_task: Callable[[dict[str, Any], list[dict[str, Any]]], QuestionReviewResult],
) -> bool:
    """Lease and process exactly one pending AI review question task.

    Inputs:
        repository: AI review repository exposing task lease and result save
            methods.
        worker_id: Identifier for the active worker process and task lease.
        run_task: Callable that receives the leased task and returns a validated
            question review result.

    Outputs:
        bool: ``True`` when one task was leased, executed, and saved; ``False``
        when no task was available to lease.
    """

    task = repository.lease_next_task(worker_id)
    if task is None:
        return False

    return process_leased_task(
        repository=repository,
        worker_id=worker_id,
        task=task,
        run_task=lambda _repository, leased_task: run_task(leased_task, []),
    )


def _save_task_failure(
    repository: Any,
    task: dict[str, Any],
    worker_id: str,
    exc: Exception,
) -> None:
    """Persist one task failure while tolerating reset races.

    Inputs:
        repository: Repository used by the active worker thread.
        task: Leased task dictionary that failed processing.
        worker_id: Worker that owns the task lease.
        exc: Exception raised while processing the leased task.

    Outputs:
        None. The task is marked failed unless it was reset concurrently.
    """

    error_message = f"{type(exc).__name__}: {exc}"
    try:
        repository.save_question_failure(
            task_id=task["task_id"],
            worker_id=worker_id,
            error_message=error_message,
        )
    except KeyError:
        logger.info(
            "AI review task %s disappeared before failure persistence; assuming reset",
            task["task_id"],
        )


def process_leased_task(
    repository: Any,
    worker_id: str,
    task: dict[str, Any],
    run_task: Callable[[Any, dict[str, Any]], QuestionReviewResult],
) -> bool:
    """Process one already-leased question task.

    Inputs:
        repository: Repository used by this worker thread.
        worker_id: Worker that owns the task lease.
        task: Leased task dictionary.
        run_task: Callable receiving repository and task.

    Outputs:
        bool: True after the task is handled or safely ignored after reset.
    """

    try:
        result = run_task(repository, task)
    except DBAPIError:
        raise
    except Exception as exc:
        _save_task_failure(repository, task, worker_id, exc)
        logger.exception(
            "AI review task %s for question %s failed",
            task["task_id"],
            task["question_id"],
        )
        return True

    try:
        repository.save_question_result(task["task_id"], worker_id, result)
    except KeyError:
        logger.info(
            "AI review task %s disappeared before result persistence; assuming reset",
            task["task_id"],
        )
        return True
    logger.info(
        "Processed AI review task %s for question %s with status %s",
        task["task_id"],
        task["question_id"],
        result.overall_status,
    )
    return True


def _lease_tasks(repository: Any, worker_id: str, limit: int) -> list[dict[str, Any]]:
    """Lease pending question tasks up to a positive limit.

    Inputs:
        repository: Repository exposing ``lease_next_task``.
        worker_id: Worker identifier to own the leased tasks.
        limit: Maximum number of tasks to lease in one batch.

    Outputs:
        list[dict[str, Any]]: Leased task dictionaries, possibly empty.
    """

    tasks: list[dict[str, Any]] = []
    for _ in range(max(0, limit)):
        task = repository.lease_next_task(worker_id)
        if task is None:
            break
        tasks.append(task)
    return tasks


def process_task_batch(
    session_factory: Callable[[], Any],
    repository_factory: Callable[[Any], Any],
    worker_id: str,
    limit: int,
    run_task: Callable[[Any, dict[str, Any]], QuestionReviewResult],
) -> bool:
    """Lease and process multiple question tasks concurrently.

    Inputs:
        session_factory: Callable returning a context-managed session.
        repository_factory: Callable building a repository for a session.
        worker_id: Identifier of the active worker process.
        limit: Maximum question tasks to process in this batch.
        run_task: Callable receiving a repository and leased task.

    Outputs:
        bool: True when at least one task was leased, False when idle.
    """

    with session_factory() as session:
        lease_repository = repository_factory(session)
        tasks = _lease_tasks(lease_repository, worker_id, limit)
        _commit_if_supported(lease_repository)
    if not tasks:
        return False

    def _run_in_thread(task: dict[str, Any]) -> bool:
        """Run one leased task inside its own repository/session.

        Inputs:
            task: Leased task dictionary to process.

        Outputs:
            bool: True after the task is handled.
        """

        with session_factory() as session:
            repository = repository_factory(session)
            handled = process_leased_task(
                repository=repository,
                worker_id=worker_id,
                task=task,
                run_task=run_task,
            )
            _commit_if_supported(repository)
            return handled

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
        futures = [executor.submit(_run_in_thread, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse AI review worker command-line arguments.

    Inputs:
        argv: Optional argument sequence for tests or embedding. When omitted,
            arguments are read from the process command line.

    Outputs:
        argparse.Namespace: Parsed worker options including ``once`` and
        ``poll_interval``.
    """

    parser = argparse.ArgumentParser(description="Run assessment AI review worker.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one task and exit.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between idle polling attempts.",
    )
    parser.add_argument(
        "--document-ingestion-concurrency",
        type=int,
        default=1,
        help="Maximum document ingestion jobs to process per polling cycle.",
    )
    parser.add_argument(
        "--model-concurrency",
        type=int,
        default=4,
        help="Maximum GPT-5.4 assistant calls in flight per worker process.",
    )
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=4,
        help="Maximum embedding requests in flight per worker process.",
    )
    return parser.parse_args(argv)


def initialize_hana_schema(session_factory: Callable[[], Any]) -> None:
    """Create missing HANA schema objects before worker task processing.

    Inputs:
        session_factory: Callable returning a context-managed SQLAlchemy
        session bound to the worker's HANA engine.

    Outputs:
        None. Missing schema objects are created through the repository and the
        session is committed before tasks are leased.
    """

    with session_factory() as session:
        repository = HanaAiReviewRepository(session)
        repository.create_schema()
        session.commit()


def _find_question_for_task(repository: Any, task: dict[str, Any]) -> Any:
    """Find the framework question matching a leased batch task.

    Inputs:
        repository: Repository with question listing methods.
        task: Leased batch question task containing question_id and dimension.

    Outputs:
        AssessmentQuestion: Matching framework question.

    Raises:
        ValueError: Raised when the question cannot be found.
    """
    questions = repository.list_questions(
        dimension=task["dimension"],
    )
    for question in questions:
        if question.question_id == task["question_id"]:
            return scope_question_to_max_level(
                question,
                int(task.get("max_allowed_level", 5) or 5),
            )
    raise ValueError(
        f"Framework question {task['question_id']} not found in dimension "
        f"{task['dimension']}"
    )


def _run_corpus_question(repository: Any, task: dict[str, Any]) -> QuestionReviewResult:
    """Run the ReAct question review graph for one leased corpus task.

    Inputs:
        repository: AI review repository with corpus retrieval methods.
        task: Leased unified question task row.

    Outputs:
        QuestionReviewResult: Validated question review result.
    """
    from app.services.batch_question_react_graph import run_corpus_question_react_review

    question = _find_question_for_task(repository, task)
    return run_corpus_question_react_review(
        question=question,
        task=task,
        repository=repository,
        worker_id=task.get("lease_owner", ""),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the HANA-backed worker polling loop.

    Inputs:
        argv: Optional command-line arguments for tests or embedding. When
            omitted, arguments are read from the process command line.

    Outputs:
        None. The process runs until interrupted, or exits after one processing
        attempt when ``--once`` is supplied.
    """

    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    worker_id = f"worker-{uuid.uuid4()}"
    engine = create_hana_engine()
    session_factory = sessionmaker(bind=engine)
    schema_initialized = False

    while True:
        try:
            if not schema_initialized:
                initialize_hana_schema(session_factory)
                schema_initialized = True
            processed = False
            with session_factory() as session:
                repository = HanaAiReviewRepository(session)
                purge_reports = getattr(
                    repository,
                    "purge_expired_assessment_report_jobs",
                    None,
                )
                if callable(purge_reports):
                    purge_reports()
                for _ in range(max(1, args.document_ingestion_concurrency)):
                    if process_one_document_ingestion_job(
                        repository=repository,
                        worker_id=worker_id,
                        run_ingestion=lambda job: run_document_ingestion_from_hana(
                            repository=repository,
                            job=job,
                        ),
                    ):
                        processed = True
                    else:
                        break
                for _ in range(max(1, args.document_ingestion_concurrency)):
                    if process_one_admin_document_ingestion_job(
                        repository=repository,
                        worker_id=worker_id,
                        run_ingestion=lambda job: run_admin_document_ingestion_from_hana(
                            repository=repository,
                            job=job,
                        ),
                    ):
                        processed = True
                    else:
                        break
                if process_one_assessment_report_job(
                    repository=repository,
                    worker_id=worker_id,
                ):
                    processed = True
                session.commit()
            if not args.once or not processed:
                task_processed = process_task_batch(
                    session_factory=session_factory,
                    repository_factory=lambda session: HanaAiReviewRepository(session),
                    worker_id=worker_id,
                    limit=max(1, args.model_concurrency),
                    run_task=lambda repository, task: _run_corpus_question(
                        repository,
                        task,
                    ),
                )
                processed = task_processed or processed
        except DBAPIError as exc:
            if args.once or not is_transient_database_error(exc):
                raise
            retry_delay = max(0.0, args.poll_interval)
            logger.warning(
                "HANA is temporarily unavailable; retrying in %.1f seconds: %s",
                retry_delay,
                exc.orig,
            )
            engine.dispose()
            time.sleep(retry_delay)
            continue

        if args.once:
            return
        if not processed:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
