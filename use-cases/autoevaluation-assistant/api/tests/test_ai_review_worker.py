"""Tests for the AI review worker loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy.exc import DBAPIError

import app.workers.ai_review_worker as worker_module
from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.models.documents import DocumentIngestionJobResponse
from app.models.reports import AssessmentReportSource
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository
from app.workers.ai_review_worker import (
    process_one_assessment_report_job,
    process_task_batch,
    process_one_admin_document_ingestion_job,
    process_one_document_ingestion_job,
    process_one_task,
)
from tests.report_fixtures import make_compact_source


def _assessment_report_source(language: str = "en") -> AssessmentReportSource:
    """Create a minimal immutable report source for worker lifecycle tests.

    Inputs:
        language: Snapshot language, ``en`` or ``it``.

    Outputs:
        AssessmentReportSource: One-question, one-dimension score snapshot.
    """

    return make_compact_source(language)


def _create_review_result(question_id: str) -> QuestionReviewResult:
    """Create a minimal valid review result for worker tests.

    Inputs:
        question_id: Assessment question identifier that must match the leased
            task being completed.

    Outputs:
        QuestionReviewResult: Valid result payload accepted by the repository.
    """

    return QuestionReviewResult(
        question_id=question_id,
        model="gpt-5.4",
        overall_status="review_required",
        highest_supported_level=None,
        current_selected_answer_item_ids=[],
        verified_selected_answer_item_ids=[],
        level_results=[],
        evidence_refs=[],
        usage={},
        warnings=[],
    )


def _transient_database_error() -> DBAPIError:
    """Create the HANA route-loss error observed in the worker terminal.

    Inputs:
        None.

    Outputs:
        DBAPIError: SQLAlchemy wrapper carrying HANA error code ``-10807``.
    """

    return DBAPIError(
        "select top 1 job_id from worker_queue",
        {},
        Exception(-10807, "Connection down: No route to host"),
    )


def _insert_raw_report_job(
    repository: InMemoryAiReviewRepository,
    *,
    job_id: str,
    source: dict[str, Any],
    status: str = "pending",
    lease_expires_at: datetime | None = None,
) -> None:
    """Insert a raw legacy or malformed report row into the memory queue.

    Inputs:
        repository: In-memory queue to mutate.
        job_id: Stable report job identity.
        source: Raw decoded JSON mapping as persisted by an older API.
        status: Initial queue lifecycle status.
        lease_expires_at: Optional active lease deadline for reclaim tests.

    Outputs:
        None. The raw job is available to the normal lease implementation.
    """

    now = datetime.now(timezone.utc)
    repository.assessment_report_jobs[job_id] = {
        "job_id": job_id,
        "assessment_id": source.get("assessment_id", "assessment-legacy"),
        "customer_class": source.get("customer_class", "class_5"),
        "sector": source.get("sector"),
        "language": source.get("language", "en"),
        "status": status,
        "source": source,
        "progress_message": None,
        "lease_owner": "old-worker" if status in {"generating", "rendering"} else None,
        "lease_expires_at": lease_expires_at,
        "retry_count": 0,
        "file_name": None,
        "pdf_content": None,
        "error_code": None,
        "error_message": None,
        "expires_at": None,
        "created_at": now,
        "updated_at": now,
    }


def _semantically_invalid_report_source() -> dict[str, Any]:
    """Return current JSON with contradictory availability and peer values.

    Inputs:
        None.

    Outputs:
        dict[str, Any]: Raw source that the worker must reject after leasing.
    """

    payload = make_compact_source().model_dump(mode="json")
    payload["provenance"]["benchmark_available"] = False
    payload["provenance"]["unavailable_reason"] = "insufficient_peer_sample"
    payload["provenance"]["peer_sample_size"] = 2
    return payload


def _question(
    question_id: str,
    dimension: str,
    text: str,
) -> AssessmentQuestion:
    """Create a minimal assessment question for batch worker tests."""
    return AssessmentQuestion(
        question_id=question_id,
        dimension=dimension,
        section="Governance",
        question=text,
        explanation="Review uploaded evidence.",
        answer_items=[
            AnswerItem(
                answer_item_id=f"{question_id}-L1-001",
                question_id=question_id,
                level=1,
                item_index=1,
                text="Evidence exists.",
            )
        ],
    )


def _create_repository_with_one_task() -> InMemoryAiReviewRepository:
    """Create an in-memory repository containing one pending AI review task.

    Inputs:
        None.

    Outputs:
        InMemoryAiReviewRepository: Repository seeded with one Strategy task and
        one corpus-backed question task.
    """

    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    repository.create_corpus_question_job(
        assessment_id="demo-assessment",
        dimension="Strategy",
        question_ids=["Q.STR.01.01"],
        current_answers={"Q.STR.01.01": []},
    )
    return repository


def _create_repository_with_tasks(count: int) -> InMemoryAiReviewRepository:
    """Create an in-memory repository containing multiple pending review tasks.

    Inputs:
        count: Number of Strategy question tasks to create.

    Outputs:
        InMemoryAiReviewRepository: Repository seeded with pending question
        review tasks for batch worker tests.
    """

    question_ids = ["Q.STR.01.01", "Q.STR.02.01", "Q.STR.03.01"]
    if count > len(question_ids):
        raise ValueError("Batch worker test helper supports up to three tasks.")

    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question(question_id, "Strategy", f"Question {index}?")
        for index, question_id in enumerate(question_ids[:count], start=1)
    ]
    repository.create_corpus_question_job(
        assessment_id="demo-assessment",
        dimension="Strategy",
        question_ids=[
            question.question_id for question in repository.framework_questions
        ],
        current_answers={},
        customer_class="class_5",
    )
    return repository


def test_assessment_report_worker_renders_current_snapshot_without_generation() -> None:
    """Verify the worker validates once and calls only the deterministic renderer.

    Inputs:
        None. The test enqueues an in-memory source and injects a renderer.

    Outputs:
        None. Assertions cover renderer input, PDF content, and terminal state.
    """

    class ProgressRepository(InMemoryAiReviewRepository):
        """In-memory repository that records active report phases."""

        def __init__(self) -> None:
            """Initialize normal stores and an empty phase list."""
            super().__init__()
            self.report_phases: list[str] = []

        def update_assessment_report_job_progress(self, **kwargs: Any) -> None:
            """Capture the phase before delegating to normal persistence."""
            self.report_phases.append(str(kwargs["status"]))
            super().update_assessment_report_job_progress(**kwargs)

    repository = ProgressRepository()
    created = repository.create_assessment_report_job(_assessment_report_source())
    render_calls: list[int] = []

    def fake_render(source: AssessmentReportSource) -> bytes:
        """Record the validated source and return deterministic PDF bytes.

        Inputs:
            source: Parsed current report snapshot.

        Outputs:
            bytes: Deterministic PDF fixture content.
        """

        render_calls.append(source.schema_version)
        return b"%PDF-1.7\nworker-test"

    processed = process_one_assessment_report_job(
        repository=repository,
        worker_id="worker-report",
        render_pdf=fake_render,
    )

    assert processed is True
    assert render_calls == [4]
    assert repository.report_phases == ["rendering"]
    status = repository.get_assessment_report_job_status(created.job_id)
    assert status.status == "completed"
    assert status.download_ready is True
    assert status.file_name == "calification-report-assessment-1-en.pdf"
    download = repository.get_assessment_report_download(created.job_id)
    assert download.content == b"%PDF-1.7\nworker-test"


def test_process_one_assessment_report_job_persists_safe_failure() -> None:
    """Verify renderer failures become safe terminal job state.

    Inputs:
        None. The injected renderer raises a message that must not be exposed.

    Outputs:
        None. Assertions cover the stable error code and safe public message.
    """

    repository = InMemoryAiReviewRepository()
    created = repository.create_assessment_report_job(_assessment_report_source())

    def failing_render(source: AssessmentReportSource) -> bytes:
        """Raise a renderer exception containing internal detail."""
        _ = source
        raise RuntimeError("secret renderer detail")

    processed = process_one_assessment_report_job(
        repository=repository,
        worker_id="worker-report",
        render_pdf=failing_render,
    )

    assert processed is True
    status = repository.get_assessment_report_job_status(created.job_id)
    assert status.status == "failed"
    assert status.error_code == "assessment_report_generation_failed"
    assert status.error_message == (
        "The report could not be generated. Please try again."
    )
    assert "secret" not in (status.error_message or "")


def test_assessment_report_dbapi_error_propagates_with_active_lease() -> None:
    """Verify a renderer DBAPI error leaves the leased report recoverable.

    Inputs:
        None. The injected renderer raises a transient SQLAlchemy DBAPI error.

    Outputs:
        None. The outer worker boundary receives the error and the job remains
        active until its 15-minute lease can be reclaimed.
    """

    repository = InMemoryAiReviewRepository()
    created = repository.create_assessment_report_job(_assessment_report_source())

    def failing_render(source: AssessmentReportSource) -> bytes:
        """Raise a transient database error during report rendering."""
        _ = source
        raise _transient_database_error()

    with pytest.raises(DBAPIError):
        process_one_assessment_report_job(
            repository=repository,
            worker_id="worker-report",
            render_pdf=failing_render,
        )

    status = repository.get_assessment_report_job_status(created.job_id)
    assert status.status == "rendering"
    assert status.error_code is None
    leased = repository.assessment_report_jobs[created.job_id]
    assert leased["lease_owner"] == "worker-report"
    assert leased["lease_expires_at"] is not None


@pytest.mark.parametrize(
    "raw_source",
    [
        {"schema_version": 2, "assessment_id": "broken", "language": "en"},
        {"schema_version": 3, "assessment_id": "retired", "language": "en"},
        {"schema_version": 99, "assessment_id": "future", "language": "en"},
        {"schema_version": "unsafe", "language": "en"},
        _semantically_invalid_report_source(),
    ],
)
def test_assessment_report_worker_safely_fails_invalid_snapshot(
    raw_source: dict[str, Any],
) -> None:
    """Verify invalid snapshot details never escape into polling.

    Inputs:
        raw_source: Malformed or unsupported decoded snapshot mapping.

    Outputs:
        None. The leased job exposes only the stable localized invalid response.
    """

    repository = InMemoryAiReviewRepository()
    _insert_raw_report_job(
        repository,
        job_id="invalid-snapshot",
        source=raw_source,
    )

    assert process_one_assessment_report_job(
        repository,
        "worker-report",
        render_pdf=lambda _source: b"%PDF-1.7\ninvalid-should-not-render",
    ) is True

    status = repository.get_assessment_report_job_status("invalid-snapshot")
    assert status.status == "failed"
    assert status.error_code == "assessment_report_snapshot_invalid"
    assert status.error_message == (
        "The report snapshot is invalid. Generate the report again."
    )
    assert "validation" not in (status.error_message or "").lower()


def test_assessment_report_invalid_failure_dbapi_propagates_with_active_lease(
) -> None:
    """Verify a DBAPI failure while rejecting a snapshot keeps its lease recoverable.

    Inputs:
        None. The repository raises while persisting the invalid-snapshot state.

    Outputs:
        None. The DBAPI error escapes and the generating job remains leased.
    """

    class FailingRepository(InMemoryAiReviewRepository):
        """Raise a database error only while persisting terminal failure."""

        def fail_assessment_report_job(self, **_kwargs: Any) -> None:
            """Raise the representative DBAPI error instead of changing state."""

            raise _transient_database_error()

    repository = FailingRepository()
    _insert_raw_report_job(
        repository,
        job_id="legacy-db-error",
        source={"schema_version": 3, "assessment_id": "legacy", "language": "en"},
    )

    with pytest.raises(DBAPIError):
        process_one_assessment_report_job(repository, "worker-report")

    status = repository.get_assessment_report_job_status("legacy-db-error")
    assert status.status == "generating"
    assert status.error_code is None


def test_process_one_assessment_report_job_returns_false_when_queue_is_empty() -> None:
    """Verify an empty report queue leaves the worker idle."""

    repository = InMemoryAiReviewRepository()

    assert process_one_assessment_report_job(
        repository=repository,
        worker_id="worker-report",
        render_pdf=lambda _source: b"unused",
    ) is False


def test_process_one_task_marks_task_completed() -> None:
    """Verify the worker leases, runs, and completes one pending task.

    Inputs:
        None. The test creates an in-memory task and a fake graph runner.

    Outputs:
        None. Assertions verify processing succeeded and the repository stores
        the result.
    """

    repository = _create_repository_with_one_task()
    calls: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def fake_run_task(
        task: dict[str, Any], attachments: list[dict[str, Any]]
    ) -> QuestionReviewResult:
        """Return a deterministic result while recording worker inputs.

        Inputs:
            task: Leased task dictionary supplied by the worker.
            attachments: Attachment dictionaries loaded by the worker.

        Outputs:
            QuestionReviewResult: Minimal result for the leased question.
        """

        calls.append((task, attachments))
        return _create_review_result(task["question_id"])

    completed = process_one_task(
        repository=repository,
        worker_id="worker-test",
        run_task=fake_run_task,
    )

    assert completed is True
    assert len(calls) == 1
    task_id = calls[0][0]["task_id"]
    assert calls[0][1] == []
    assert repository.tasks[task_id]["status"] == "completed"
    assert repository.results[task_id].question_id == "Q.STR.01.01"


def test_process_one_task_returns_false_when_no_pending_task() -> None:
    """Verify the worker reports idle state when no task can be leased.

    Inputs:
        None. The test uses an empty in-memory repository.

    Outputs:
        None. Assertions verify no task was processed.
    """

    repository = InMemoryAiReviewRepository()

    def fake_run_task(
        task: dict[str, Any], attachments: list[dict[str, Any]]
    ) -> QuestionReviewResult:
        """Fail if the worker calls the graph runner without a leased task.

        Inputs:
            task: Unexpected task value.
            attachments: Unexpected attachment values.

        Outputs:
            QuestionReviewResult: This function never returns in the test.
        """

        raise AssertionError("run_task should not be called when no task exists")

    completed = process_one_task(
        repository=repository,
        worker_id="worker-test",
        run_task=fake_run_task,
    )

    assert completed is False


def test_process_one_task_saves_with_active_worker_id() -> None:
    """Verify result persistence uses the worker ID that holds the lease.

    Inputs:
        None. The test wraps the in-memory repository with a save spy.

    Outputs:
        None. Assertions verify the saved worker ID and completed task state.
    """

    class SpyRepository(InMemoryAiReviewRepository):
        """In-memory repository that records result save calls.

        Inputs:
            None. The repository starts empty and exposes ``save_calls`` after
            construction.

        Outputs:
            SpyRepository: Repository with normal persistence plus call capture.
        """

        def __init__(self) -> None:
            """Initialize repository state and an empty save call log.

            Inputs:
                None.

            Outputs:
                None. ``save_calls`` is ready to capture invocations.
            """

            super().__init__()
            self.save_calls: list[tuple[str, str, QuestionReviewResult]] = []

        def save_question_result(
            self,
            task_id: str,
            worker_id: str,
            result: QuestionReviewResult,
        ) -> None:
            """Record the save arguments and delegate to normal persistence.

            Inputs:
                task_id: Task identifier being completed.
                worker_id: Worker identifier passed by the worker process.
                result: Review result to persist.

            Outputs:
                None. The call is recorded and the task is completed.
            """

            self.save_calls.append((task_id, worker_id, result))
            super().save_question_result(task_id, worker_id, result)

    repository = SpyRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    repository.create_corpus_question_job(
        assessment_id="demo-assessment",
        dimension="Strategy",
        question_ids=["Q.STR.01.01"],
        current_answers={"Q.STR.01.01": []},
    )

    completed = process_one_task(
        repository=repository,
        worker_id="worker-active",
        run_task=lambda task, attachments: _create_review_result(task["question_id"]),
    )

    assert completed is True
    assert repository.save_calls[0][1] == "worker-active"


def test_process_one_task_marks_task_failed_when_run_task_raises() -> None:
    """Verify model failures are persisted instead of crashing the worker.

    Inputs:
        None. The test creates a pending task and a fake graph runner that
        raises the same kind of error a live model call would raise.

    Outputs:
        None. Assertions confirm the worker reports a processed task and the
        repository exposes the failure through job status polling.
    """

    repository = _create_repository_with_one_task()

    def failing_run_task(
        task: dict[str, Any], attachments: list[dict[str, Any]]
    ) -> QuestionReviewResult:
        """Raise a deterministic model failure for worker failure handling.

        Inputs:
            task: Leased task dictionary supplied by the worker.
            attachments: Attachment dictionaries loaded by the worker.

        Outputs:
            QuestionReviewResult: This fake never returns.

        Raises:
            RuntimeError: Always raised to mimic a failed model call.
        """

        raise RuntimeError("SAP Gen AI Hub Responses API question review failed.")

    completed = process_one_task(
        repository=repository,
        worker_id="worker-active",
        run_task=failing_run_task,
    )

    task_id = next(iter(repository.tasks))
    job_id = repository.tasks[task_id]["job_id"]
    status = repository.get_job_status(job_id)

    assert completed is True
    assert repository.tasks[task_id]["status"] == "failed"
    assert status.status == "failed"
    assert status.failed_count == 1
    assert "SAP Gen AI Hub Responses API" in (status.tasks[0].error_message or "")


def test_process_one_task_reraises_transient_database_failure() -> None:
    """Verify a leased question remains retryable after HANA connectivity loss.

    Inputs:
        None. A corpus task is leased and its runner raises the observed HANA
        route-loss error.

    Outputs:
        None. The assertion confirms the database error escapes to the outer
        worker recovery loop instead of being persisted as a task failure.
    """

    repository = _create_repository_with_one_task()

    def disconnected_run_task(
        _task: dict[str, Any],
        _attachments: list[dict[str, Any]],
    ) -> QuestionReviewResult:
        """Raise a transient database disconnect during task processing."""

        raise _transient_database_error()

    with pytest.raises(DBAPIError):
        process_one_task(
            repository=repository,
            worker_id="worker-active",
            run_task=disconnected_run_task,
        )

    task = next(iter(repository.tasks.values()))
    assert task["status"] == "in_progress"
    assert task["error_message"] is None


def test_process_one_task_tolerates_task_deleted_by_reset_before_save() -> None:
    """Verify reset races do not crash the long-running worker process.

    Inputs:
        None. The test uses a repository that clears the active Strategy review
        state immediately before result persistence.

    Outputs:
        None. Assertions confirm the worker treats the missing task as already
        reset by the user and returns without raising.
    """

    class ResetBeforeSaveRepository(InMemoryAiReviewRepository):
        """Repository that simulates a user reset during model execution.

        Inputs:
            None. The repository behaves normally except for result save.

        Outputs:
            ResetBeforeSaveRepository: In-memory repository with reset-on-save
            behavior for one worker race test.
        """

        def save_question_result(
            self,
            task_id: str,
            worker_id: str,
            result: QuestionReviewResult,
        ) -> None:
            """Delete review state before delegating to normal result save.

            Inputs:
                task_id: Task identifier being completed.
                worker_id: Worker identifier holding the task lease.
                result: Review result to persist.

            Outputs:
                None. Normal persistence raises because the reset removed the
                task, reproducing the race handled by the worker.
            """

            self.clear_dimension_review_state("demo-assessment", "Strategy")
            super().save_question_result(task_id, worker_id, result)

    repository = ResetBeforeSaveRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    repository.create_corpus_question_job(
        assessment_id="demo-assessment",
        dimension="Strategy",
        question_ids=["Q.STR.01.01"],
        current_answers={"Q.STR.01.01": []},
    )

    completed = process_one_task(
        repository=repository,
        worker_id="worker-active",
        run_task=lambda task, attachments: _create_review_result(task["question_id"]),
    )

    assert completed is True
    assert repository.tasks == {}
    assert repository.jobs == {}


def test_process_task_batch_processes_multiple_pending_tasks() -> None:
    """Verify one worker cycle can complete more than one question task."""

    repository = _create_repository_with_tasks(3)

    class SessionFactory:
        """Return the same in-memory repository for worker batch tests."""

        def __call__(self):
            """Return a context manager compatible with worker sessions."""

            return self

        def __enter__(self):
            """Enter the fake session context."""

            return self

        def __exit__(self, exc_type, exc, traceback):
            """Exit the fake session context without suppressing errors."""

            return False

    def fake_repository_factory(_session):
        """Return the shared in-memory repository."""

        return repository

    processed = process_task_batch(
        session_factory=SessionFactory(),
        repository_factory=fake_repository_factory,
        worker_id="worker-batch",
        limit=2,
        run_task=lambda repo, task: _create_review_result(task["question_id"]),
    )

    assert processed is True
    assert (
        sum(
            1
            for task in repository.tasks.values()
            if task["status"] == "completed"
        )
        == 2
    )
    assert (
        sum(
            1 for task in repository.tasks.values() if task["status"] == "pending"
        )
        == 1
    )


def test_process_task_batch_returns_false_when_no_tasks_exist() -> None:
    """Verify batch processing reports idle state when no task can be leased."""

    repository = InMemoryAiReviewRepository()

    class SessionFactory:
        """Return the same in-memory repository for worker batch tests."""

        def __call__(self):
            """Return a context manager compatible with worker sessions."""

            return self

        def __enter__(self):
            """Enter the fake session context."""

            return self

        def __exit__(self, exc_type, exc, traceback):
            """Exit the fake session context without suppressing errors."""

            return False

    processed = process_task_batch(
        session_factory=SessionFactory(),
        repository_factory=lambda _session: repository,
        worker_id="worker-batch",
        limit=4,
        run_task=lambda repo, task: _create_review_result(task["question_id"]),
    )

    assert processed is False


def test_process_task_batch_returns_false_when_limit_is_zero() -> None:
    """Verify batch processing honors zero as a maximum task limit."""

    repository = _create_repository_with_tasks(3)

    class SessionFactory:
        """Return the same in-memory repository for worker batch tests."""

        def __call__(self):
            """Return a context manager compatible with worker sessions."""

            return self

        def __enter__(self):
            """Enter the fake session context."""

            return self

        def __exit__(self, exc_type, exc, traceback):
            """Exit the fake session context without suppressing errors."""

            return False

    processed = process_task_batch(
        session_factory=SessionFactory(),
        repository_factory=lambda _session: repository,
        worker_id="worker-batch",
        limit=0,
        run_task=lambda repo, task: _create_review_result(task["question_id"]),
    )

    assert processed is False
    assert all(task["status"] == "pending" for task in repository.tasks.values())


def test_run_corpus_question_scopes_prompt_question_to_task_level() -> None:
    """Verify leased task limits scope the framework question answer catalog."""
    question_id = "Q.STR.03.01"
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        AssessmentQuestion(
            question_id=question_id,
            dimension="Strategy",
            section="Objectives",
            question="Are objectives defined and monitored?",
            answer_items=[
                AnswerItem(
                    answer_item_id=f"{question_id}-L1-001",
                    question_id=question_id,
                    level=1,
                    item_index=1,
                    text="Objectives are defined.",
                ),
                AnswerItem(
                    answer_item_id=f"{question_id}-L2-001",
                    question_id=question_id,
                    level=2,
                    item_index=1,
                    text="Objectives are monitored.",
                ),
                AnswerItem(
                    answer_item_id=f"{question_id}-L3-001",
                    question_id=question_id,
                    level=3,
                    item_index=1,
                    text="Objectives are monitored by top executives.",
                ),
            ],
        )
    ]

    question = worker_module._find_question_for_task(
        repository,
        {
            "dimension": "Strategy",
            "question_id": question_id,
            "max_allowed_level": 2,
        },
    )

    assert question.question_id == question_id
    assert [item.level for item in question.answer_items] == [1, 2]


def test_worker_main_initializes_hana_schema_before_processing(monkeypatch) -> None:
    """Verify the HANA worker creates missing schema objects before polling.

    Inputs:
        monkeypatch: Pytest fixture used to replace the live HANA engine,
        session factory, repository class, and task processor.

    Outputs:
        None. Assertions confirm ``create_schema`` runs before the worker tries
        to process a task.
    """
    events: list[str] = []

    class FakeSession:
        """Minimal context-managed session for worker startup tests.

        Inputs:
            None.

        Outputs:
            FakeSession: Session object exposing context manager and commit
            methods used by the worker.
        """

        def __enter__(self) -> "FakeSession":
            """Enter the fake session context.

            Inputs:
                None.

            Outputs:
                FakeSession: The current fake session object.
            """

            events.append("session_enter")
            return self

        def __exit__(self, *_args: object) -> None:
            """Exit the fake session context.

            Inputs:
                *_args: Exception context supplied by the context manager
                protocol.

            Outputs:
                None. The exit event is recorded for assertions if needed.
            """

            events.append("session_exit")

        def commit(self) -> None:
            """Record that the worker committed the session.

            Inputs:
                None.

            Outputs:
                None. The commit event is recorded in ``events``.
            """

            events.append("commit")

    class FakeSessionFactory:
        """Callable factory returning fake worker sessions.

        Inputs:
            None.

        Outputs:
            FakeSessionFactory: Factory object compatible with SQLAlchemy
            ``sessionmaker`` instances.
        """

        def __call__(self) -> FakeSession:
            """Return a new fake session.

            Inputs:
                None.

            Outputs:
                FakeSession: New fake context-managed session.
            """

            return FakeSession()

    class FakeRepository:
        """Fake HANA repository that records schema initialization.

        Inputs:
            session: Fake session supplied by the worker.

        Outputs:
            FakeRepository: Repository stand-in exposing ``create_schema``.
        """

        def __init__(self, session: FakeSession) -> None:
            """Store the fake session and record repository construction.

            Inputs:
                session: Fake session created by the fake session factory.

            Outputs:
                None. The construction event is recorded.
            """

            self.session = session
            events.append("repository")

        def create_schema(self) -> None:
            """Record schema initialization.

            Inputs:
                None.

            Outputs:
                None. The schema initialization event is recorded.
            """

            events.append("create_schema")

    def fake_sessionmaker(*_args: object, **_kwargs: object) -> FakeSessionFactory:
        """Return the fake worker session factory.

        Inputs:
            *_args: Positional arguments ignored from SQLAlchemy ``sessionmaker``.
            **_kwargs: Keyword arguments ignored from SQLAlchemy ``sessionmaker``.

        Outputs:
            FakeSessionFactory: Callable fake session factory.
        """

        return FakeSessionFactory()

    def fake_process_task_batch(**kwargs: object) -> bool:
        """Record processing and stop the ``--once`` worker loop.

        Inputs:
            **kwargs: Worker processing arguments supplied by ``main``.

        Outputs:
            bool: ``False`` to indicate no task was processed.
        """

        assert kwargs["limit"] == 2
        events.append("process")
        return False

    monkeypatch.setattr(worker_module, "create_hana_engine", lambda: object())
    monkeypatch.setattr(worker_module, "sessionmaker", fake_sessionmaker)
    monkeypatch.setattr(worker_module, "HanaAiReviewRepository", FakeRepository)
    monkeypatch.setattr(
        worker_module,
        "process_one_document_ingestion_job",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(worker_module, "process_task_batch", fake_process_task_batch)

    worker_module.main(["--once", "--model-concurrency", "2"])

    assert "create_schema" in events
    assert events.index("create_schema") < events.index("process")


def test_worker_main_clamps_non_positive_model_concurrency(monkeypatch) -> None:
    """Verify the worker CLI keeps question review active for non-positive concurrency."""

    captured_limits: list[int] = []

    class FakeSession:
        """Minimal context-managed session for worker clamp tests."""

        def __enter__(self) -> "FakeSession":
            """Enter the fake session context."""

            return self

        def __exit__(self, *_args: object) -> None:
            """Exit the fake session context."""

            return None

        def commit(self) -> None:
            """Accept worker commit calls."""

            return None

    class FakeSessionFactory:
        """Callable factory returning fake worker sessions."""

        def __call__(self) -> FakeSession:
            """Return a new fake session."""

            return FakeSession()

    class FakeRepository:
        """Fake repository exposing schema initialization for worker startup."""

        def __init__(self, session: FakeSession) -> None:
            """Store the session supplied by the worker."""

            self.session = session

        def create_schema(self) -> None:
            """Accept schema initialization calls."""

            return None

    def fake_sessionmaker(*_args: object, **_kwargs: object) -> FakeSessionFactory:
        """Return the fake session factory."""

        return FakeSessionFactory()

    def fake_process_task_batch(**kwargs: object) -> bool:
        """Capture the limit passed from ``main`` to the batch helper."""

        captured_limits.append(int(kwargs["limit"]))
        return False

    monkeypatch.setattr(worker_module, "create_hana_engine", lambda: object())
    monkeypatch.setattr(worker_module, "sessionmaker", fake_sessionmaker)
    monkeypatch.setattr(worker_module, "HanaAiReviewRepository", FakeRepository)
    monkeypatch.setattr(
        worker_module,
        "process_one_document_ingestion_job",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(worker_module, "process_task_batch", fake_process_task_batch)

    worker_module.main(["--once", "--model-concurrency", "0"])
    worker_module.main(["--once", "--model-concurrency", "-2"])

    assert captured_limits == [1, 1]


def test_worker_main_retries_after_transient_hana_disconnect(monkeypatch) -> None:
    """Verify a temporary HANA route loss does not terminate the worker daemon.

    Inputs:
        monkeypatch: Pytest fixture replacing HANA and worker processing seams.

    Outputs:
        None. Assertions confirm the engine pool is disposed, the configured
        retry wait is observed, and a second polling cycle is attempted.
    """

    class StopWorker(Exception):
        """Terminate the test after proving the worker entered a second cycle."""

    class FakeEngine:
        """Record connection-pool disposal requested by worker recovery."""

        def __init__(self) -> None:
            """Initialize the pool-disposal counter."""

            self.dispose_calls = 0

        def dispose(self) -> None:
            """Record one stale connection-pool disposal."""

            self.dispose_calls += 1

    class FakeSession:
        """Provide the context-manager and commit API used by the worker."""

        def __enter__(self) -> "FakeSession":
            """Return the active fake session."""

            return self

        def __exit__(self, *_args: object) -> None:
            """Close the fake session without suppressing exceptions."""

            return None

        def commit(self) -> None:
            """Accept worker transaction commits."""

            return None

    class FakeSessionFactory:
        """Create independent fake sessions for worker polling cycles."""

        def __call__(self) -> FakeSession:
            """Return one context-managed fake session."""

            return FakeSession()

    class FakeRepository:
        """Expose only the schema initialization needed before polling."""

        def __init__(self, session: FakeSession) -> None:
            """Store the fake session supplied by the worker."""

            self.session = session

        def create_schema(self) -> None:
            """Accept worker schema initialization."""

            return None

    engine = FakeEngine()
    poll_attempts = 0
    sleep_calls: list[float] = []

    def fake_sessionmaker(*_args: object, **_kwargs: object) -> FakeSessionFactory:
        """Return the deterministic fake worker session factory."""

        return FakeSessionFactory()

    def fail_once_then_stop(**_kwargs: object) -> bool:
        """Raise one observed HANA disconnect, then stop on the retry cycle."""

        nonlocal poll_attempts
        poll_attempts += 1
        if poll_attempts == 1:
            raise DBAPIError(
                "select top 1 job_id from document_ingestion_jobs",
                {},
                Exception(-10807, "Connection down: No route to host"),
            )
        raise StopWorker

    def record_sleep(seconds: float) -> None:
        """Record the worker's retry delay without blocking the test."""

        sleep_calls.append(seconds)

    def create_fake_engine() -> FakeEngine:
        """Return the fake engine whose disposal calls are asserted."""

        return engine

    monkeypatch.setattr(worker_module, "create_hana_engine", create_fake_engine)
    monkeypatch.setattr(worker_module, "sessionmaker", fake_sessionmaker)
    monkeypatch.setattr(worker_module, "HanaAiReviewRepository", FakeRepository)
    monkeypatch.setattr(
        worker_module,
        "process_one_document_ingestion_job",
        fail_once_then_stop,
    )
    monkeypatch.setattr(worker_module.time, "sleep", record_sleep)

    with pytest.raises(StopWorker):
        worker_module.main(["--poll-interval", "0.25"])

    assert poll_attempts == 2
    assert engine.dispose_calls == 1
    assert sleep_calls == [0.25]


def test_worker_main_retries_transient_schema_initialization_failure(
    monkeypatch,
) -> None:
    """Verify a worker started during a HANA outage waits and initializes later.

    Inputs:
        monkeypatch: Pytest fixture replacing schema and polling dependencies.

    Outputs:
        None. Assertions confirm continuous mode retries schema creation with a
        fresh pool, while proceeding to polling after connectivity returns.
    """

    class StopWorker(Exception):
        """Terminate the test after successful retry reaches normal polling."""

    class FakeEngine:
        """Record engine disposal after failed schema initialization."""

        def __init__(self) -> None:
            """Initialize the pool-disposal counter."""

            self.dispose_calls = 0

        def dispose(self) -> None:
            """Record one stale pool disposal."""

            self.dispose_calls += 1

    class FakeSession:
        """Provide the context-manager API for the first healthy poll cycle."""

        def __enter__(self) -> "FakeSession":
            """Return the active fake session."""

            return self

        def __exit__(self, *_args: object) -> None:
            """Close without suppressing the test stop signal."""

            return None

    class FakeSessionFactory:
        """Create fake sessions after schema initialization succeeds."""

        def __call__(self) -> FakeSession:
            """Return one fake polling session."""

            return FakeSession()

    engine = FakeEngine()
    schema_attempts = 0
    sleep_calls: list[float] = []

    def create_fake_engine() -> FakeEngine:
        """Return the fake engine used for recovery assertions."""

        return engine

    def fake_sessionmaker(*_args: object, **_kwargs: object) -> FakeSessionFactory:
        """Return a deterministic worker session factory."""

        return FakeSessionFactory()

    def initialize_fail_once(_session_factory: object) -> None:
        """Fail the first schema attempt and accept the retry."""

        nonlocal schema_attempts
        schema_attempts += 1
        if schema_attempts == 1:
            raise _transient_database_error()

    def create_repository(_session: FakeSession) -> object:
        """Return a minimal repository placeholder for the polling cycle."""

        return object()

    def stop_at_first_poll(**_kwargs: object) -> bool:
        """Stop after successful schema recovery enters normal polling."""

        raise StopWorker

    def record_sleep(seconds: float) -> None:
        """Record the schema-retry delay without blocking."""

        sleep_calls.append(seconds)

    monkeypatch.setattr(worker_module, "create_hana_engine", create_fake_engine)
    monkeypatch.setattr(worker_module, "sessionmaker", fake_sessionmaker)
    monkeypatch.setattr(worker_module, "initialize_hana_schema", initialize_fail_once)
    monkeypatch.setattr(worker_module, "HanaAiReviewRepository", create_repository)
    monkeypatch.setattr(
        worker_module,
        "process_one_document_ingestion_job",
        stop_at_first_poll,
    )
    monkeypatch.setattr(worker_module.time, "sleep", record_sleep)

    with pytest.raises(StopWorker):
        worker_module.main(["--poll-interval", "0.5"])

    assert schema_attempts == 2
    assert engine.dispose_calls == 1
    assert sleep_calls == [0.5]


def test_worker_parse_args_supports_document_ingestion_limits() -> None:
    """Verify worker CLI exposes document ingestion concurrency controls."""
    args = worker_module.parse_args(
        [
            "--document-ingestion-concurrency",
            "2",
            "--model-concurrency",
            "2",
            "--embedding-concurrency",
            "2",
        ]
    )

    assert args.document_ingestion_concurrency == 2
    assert args.model_concurrency == 2
    assert args.embedding_concurrency == 2


def test_process_one_document_ingestion_job_completes_indexing() -> None:
    """Verify the worker leases one ingestion job and marks it completed."""
    repository = InMemoryAiReviewRepository()
    job = repository.create_document_ingestion_job(
        assessment_id="assessment-1",
        documents=[
            {
                "file_name": "strategy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF",
            }
        ],
    )

    processed = process_one_document_ingestion_job(
        repository=repository,
        worker_id="worker-index",
        run_ingestion=lambda leased_job: DocumentIngestionJobResponse(
            job_id=leased_job["job_id"],
            assessment_id=leased_job["assessment_id"],
            status="completed",
            document_count=1,
            processed_document_count=1,
            indexed_chunk_count=5,
            error_message=None,
        ),
    )

    assert processed is True
    assert repository.document_ingestion_jobs[job.job_id]["status"] == "completed"
    assert repository.document_ingestion_jobs[job.job_id]["indexed_chunk_count"] == 5


def test_document_ingestion_reraises_transient_database_failure() -> None:
    """Verify document ingestion leases survive temporary HANA disconnects.

    Inputs:
        None. A document-ingestion job is leased before its runner raises the
        observed HANA route-loss error.

    Outputs:
        None. The database error reaches the outer retry loop and the job is not
        incorrectly persisted as a permanent ingestion failure.
    """

    repository = InMemoryAiReviewRepository()
    job = repository.create_document_ingestion_job(
        assessment_id="assessment-1",
        documents=[{"file_name": "strategy.pdf", "content": b"%PDF"}],
    )

    def disconnected_ingestion(
        _job: dict[str, Any],
    ) -> DocumentIngestionJobResponse:
        """Raise a transient database disconnect during document ingestion."""

        raise _transient_database_error()

    with pytest.raises(DBAPIError):
        process_one_document_ingestion_job(
            repository=repository,
            worker_id="worker-index",
            run_ingestion=disconnected_ingestion,
        )

    assert repository.document_ingestion_jobs[job.job_id]["status"] != "failed"


def test_process_one_admin_document_ingestion_job_completes_indexing() -> None:
    """Verify the worker indexes one global admin document ingestion job."""
    repository = InMemoryAiReviewRepository()
    job = repository.create_admin_document_ingestion_job(
        documents=[
            {
                "file_name": "policy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF",
            }
        ],
    )

    processed = process_one_admin_document_ingestion_job(
        repository=repository,
        worker_id="worker-admin-index",
        run_ingestion=lambda leased_job: DocumentIngestionJobResponse(
            job_id=leased_job["job_id"],
            assessment_id=leased_job["assessment_id"],
            status="completed",
            document_count=1,
            processed_document_count=1,
            indexed_chunk_count=7,
            error_message=None,
        ),
    )

    assert processed is True
    assert repository.admin_document_ingestion_jobs[job.job_id]["status"] == "completed"
    assert repository.admin_document_ingestion_jobs[job.job_id]["indexed_chunk_count"] == 7


def test_admin_ingestion_reraises_transient_database_failure() -> None:
    """Verify admin ingestion leases survive temporary HANA disconnects.

    Inputs:
        None. An admin-ingestion job is leased before its runner raises the
        observed HANA route-loss error.

    Outputs:
        None. The error reaches the outer retry loop and the job is not marked
        as a permanent admin-document failure.
    """

    repository = InMemoryAiReviewRepository()
    job = repository.create_admin_document_ingestion_job(
        documents=[{"file_name": "policy.pdf", "content": b"%PDF"}],
    )

    def disconnected_ingestion(
        _job: dict[str, Any],
    ) -> DocumentIngestionJobResponse:
        """Raise a transient database disconnect during admin ingestion."""

        raise _transient_database_error()

    with pytest.raises(DBAPIError):
        process_one_admin_document_ingestion_job(
            repository=repository,
            worker_id="worker-admin-index",
            run_ingestion=disconnected_ingestion,
        )

    assert repository.admin_document_ingestion_jobs[job.job_id]["status"] != "failed"
