"""Tests for AI review job repository behavior."""

import json
import time

import pytest

from app.db import build_hana_url
from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.ai_review_repository.hana import HanaAiReviewRepository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository
from app.services.document_extractors import ExtractedBlock, ExtractedDocument


def test_ai_review_repository_package_exports_explicit_modules() -> None:
    """Verify repository implementations are importable from focused modules.

    Inputs:
        None. Imports exercise the public package and explicit submodule paths.

    Outputs:
        None. Assertions confirm package exports and submodule classes point to
        the same repository implementations.
    """

    from app.services import ai_review_repository
    from app.services.ai_review_repository.hana import (
        HanaAiReviewRepository as ExplicitHanaRepository,
    )
    from app.services.ai_review_repository.memory import (
        InMemoryAiReviewRepository as ExplicitInMemoryRepository,
    )

    assert ai_review_repository.HanaAiReviewRepository is ExplicitHanaRepository
    assert (
        ai_review_repository.InMemoryAiReviewRepository
        is ExplicitInMemoryRepository
    )
    assert ai_review_repository.document_content_hash(b"evidence") == (
        "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
    )


class _FakeMappingsResult:
    """Small SQLAlchemy result stand-in for HANA adapter SQL boundary tests.

    Inputs:
        row: Mapping row returned by ``first()``.
        rowcount: Number exposed on write results.

    Outputs:
        Fake result object with the subset of SQLAlchemy methods used by the
        repository.
    """

    def __init__(
        self,
        row: dict[str, object] | list[dict[str, object]] | None = None,
        rowcount: int = 1,
    ) -> None:
        """Initialize a fake SQLAlchemy result.

        Inputs:
            row: Optional mapping row to return from ``first``.
            rowcount: Write row count exposed to repository validation.

        Outputs:
            None. Attributes are stored for fake query consumption.
        """

        self.row = row
        self.rowcount = rowcount

    def mappings(self) -> "_FakeMappingsResult":
        """Return self to mimic SQLAlchemy mapping result chaining.

        Inputs:
            None.

        Outputs:
            _FakeMappingsResult: The same fake result instance.
        """

        return self

    def first(self) -> dict[str, object] | None:
        """Return the configured first row.

        Inputs:
            None.

        Outputs:
            dict[str, object] | None: The fake row configured at construction.
        """

        if isinstance(self.row, list):
            return self.row[0] if self.row else None
        return self.row

    def all(self) -> list[dict[str, object]]:
        """Return all configured rows.

        Inputs:
            None.

        Outputs:
            list[dict[str, object]]: Fake result rows configured at construction.
        """

        if self.row is None:
            return []
        if isinstance(self.row, list):
            return self.row
        return [self.row]


class _SpySession:
    """Capture SQL statements sent through the HANA repository.

    Inputs:
        task_row: Row returned for the task validation select.
        existing_tables: Optional upper-case table names that exist for
            schema-introspection queries. ``None`` treats every table as
            present so existing SQL boundary tests stay concise.

    Outputs:
        Fake session object that records executed SQL and parameters.
    """

    def __init__(
        self,
        task_row: dict[str, object] | None,
        existing_tables: set[str] | None = None,
    ) -> None:
        """Initialize an empty SQL statement log.

        Inputs:
            task_row: Optional task row returned by the first select.
            existing_tables: Optional set of upper-case table names returned by
                ``sys.tables`` lookups. When omitted, all requested tables are
                considered present.

        Outputs:
            None. Executions are appended to ``calls``.
        """

        self.task_row = task_row
        self.existing_tables = existing_tables
        self.calls: list[
            tuple[str, dict[str, object] | list[dict[str, object]] | None]
        ] = []

    def execute(
        self,
        statement: object,
        parameters: dict[str, object] | list[dict[str, object]] | None = None,
    ) -> _FakeMappingsResult:
        """Record an executed SQL statement and return a fake result.

        Inputs:
            statement: SQLAlchemy text statement or any object coercible to text.
            parameters: Optional bind parameter dictionary.

        Outputs:
            _FakeMappingsResult: Fake select or write result expected by the
            repository method under test.
        """

        sql = str(statement)
        self.calls.append((sql, parameters))
        normalized_sql = " ".join(sql.lower().split())
        first_parameters = (
            parameters[0]
            if isinstance(parameters, list) and parameters
            else parameters
        )
        if "from sys.tables" in normalized_sql:
            table_name = str((first_parameters or {}).get("table_name", "")).upper()
            if self.existing_tables is None or table_name in self.existing_tables:
                return _FakeMappingsResult({"table_name": table_name})
            return _FakeMappingsResult(None)
        if "select count(*) as task_count" in normalized_sql:
            return _FakeMappingsResult(
                {"task_count": (self.task_row or {}).get("task_count", 0)}
            )
        if "select count(*) as job_count" in normalized_sql:
            return _FakeMappingsResult(
                {"job_count": (self.task_row or {}).get("job_count", 0)}
            )
        if "from ai_evidence_chunks" in normalized_sql:
            return _FakeMappingsResult(self.task_row)
        if "from joule_admin_document_chunks" in normalized_sql:
            return _FakeMappingsResult(self.task_row)
        if "from ai_question_tasks t" in normalized_sql and "join ai_dimension_jobs" in normalized_sql:
            return _FakeMappingsResult(self.task_row)
        if sql.startswith("select question_id"):
            return _FakeMappingsResult(self.task_row)
        return _FakeMappingsResult(rowcount=1)


class _SequencedSession:
    """Return configured mapping results in SQL execution order.

    Inputs:
        rows: Values consumed by successive repository queries.

    Outputs:
        _SequencedSession: Minimal SQLAlchemy session stand-in for polling tests.
    """

    def __init__(self, rows: list[object]) -> None:
        """Store ordered rows for successive ``execute`` calls.

        Inputs:
            rows: Values returned by successive mapping queries.

        Outputs:
            None. The values are copied into an internal queue.
        """

        self.rows = list(rows)

    def execute(
        self,
        _statement: object,
        _parameters: dict[str, object] | None = None,
    ) -> _FakeMappingsResult:
        """Return the next configured fake mapping result.

        Inputs:
            _statement: Ignored SQLAlchemy statement.
            _parameters: Ignored SQL bind parameters.

        Outputs:
            _FakeMappingsResult: Wrapper for the next configured row value.
        """

        return _FakeMappingsResult(self.rows.pop(0))


def _create_single_question_job(repository: InMemoryAiReviewRepository) -> str:
    """Create a one-question Strategy job and return its task ID.

    Inputs:
        repository: In-memory repository that should receive the test job.

    Outputs:
        str: The task ID created for question ``Q.STR.01.01``.
    """
    repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )
    return repository.list_pending_tasks(limit=1)[0]["task_id"]


def _assessment_question(question_id: str = "Q.STR.03.01") -> AssessmentQuestion:
    """Return a minimal assessment question for HANA adapter tests.

    Inputs:
        question_id: Question identifier to place in the fixture.

    Outputs:
        AssessmentQuestion: Valid question with one answer item.
    """

    return AssessmentQuestion(
        question_id=question_id,
        dimension="Strategy",
        section="Objectives",
        question="Are objectives monitored?",
        explanation="Review evidence.",
        answer_items=[
            AnswerItem(
                answer_item_id=f"{question_id}-L1-001",
                question_id=question_id,
                level=1,
                item_index=1,
                text="Objectives are monitored.",
            )
        ],
    )


def _question_result(question_id: str = "Q.STR.01.01") -> QuestionReviewResult:
    """Build a minimal validated question review result for repository tests.

    Inputs:
        question_id: Question identifier to include in the result.

    Outputs:
        QuestionReviewResult: A valid result object for the requested question.
    """
    return QuestionReviewResult(
        question_id=question_id,
        model="gpt-5.4",
        overall_status="supported",
        current_selected_answer_item_ids=["Q.STR.01.01-L1-001"],
        verified_selected_answer_item_ids=["Q.STR.01.01-L1-001"],
    )


def _legacy_result_json() -> str:
    """Return persisted review JSON using the retired ``deselect`` contract.

    Inputs:
        None.

    Outputs:
        str: Legacy result JSON shared by dimension and batch polling tests.
    """

    return json.dumps(
        {
            "question_id": "Q.STR.01.01",
            "model": "gpt-5.4",
            "overall_status": "unsupported",
            "current_selected_answer_item_ids": ["Q.STR.01.01-L1-001"],
            "verified_selected_answer_item_ids": ["Q.STR.01.01-L1-001"],
            "level_results": [
                {
                    "level": 1,
                    "level_status": "unsupported",
                    "level_reasoning": "The earlier review recommended removal.",
                    "answer_item_decisions": [
                        {
                            "answer_item_id": "Q.STR.01.01-L1-001",
                            "decision": "deselect",
                            "confidence": 0.82,
                            "rationale": "The evidence did not support this item.",
                            "evidence_ref_ids": [],
                        }
                    ],
                    "level_evidence_refs": [],
                }
            ],
            "evidence_refs": [],
            "warnings": [],
        }
    )


def _legacy_task_row() -> dict[str, object]:
    """Return a completed HANA task row containing a legacy result payload.

    Inputs:
        None.

    Outputs:
        dict[str, object]: Mapping returned by dimension or batch polling SQL.
    """

    return {
        "task_id": "task-legacy",
        "question_id": "Q.STR.01.01",
        "dimension": "Strategy",
        "status": "completed",
        "error_code": None,
        "error_message": None,
        "progress_message": None,
        "rag_call_count": 0,
        "query_count": 0,
        "retrieved_chunk_count": 0,
        "lease_owner": None,
        "lease_expires_at": None,
        "updated_at": None,
        "result_json": _legacy_result_json(),
    }


def test_hana_job_status_normalizes_legacy_deselect_results() -> None:
    """Verify polling remains compatible with persisted pre-guarantee results.

    Inputs:
        None. A sequenced fake HANA session returns one completed task whose
        stored JSON contains the retired ``deselect`` decision.

    Outputs:
        None. Assertions confirm the job stays pollable and exposes the legacy
        decision as the current neutral ``unsupported`` evidence status.
    """

    session = _SequencedSession(
        [
            {"job_id": "job-legacy", "language": "en", "status": "completed"},
            [_legacy_task_row()],
        ]
    )

    status = HanaAiReviewRepository(session=session).get_job_status("job-legacy")

    assert status.status == "completed"
    assert status.tasks[0].result is not None
    assert status.tasks[0].result_requires_rerun is True
    assert status.tasks[0].result.verified_selected_answer_item_ids == []
    decision = status.tasks[0].result.level_results[0].answer_item_decisions[0]
    assert decision.decision == "unsupported"
    assert "legacy" in status.tasks[0].result.warnings[0].lower()


def test_hana_batch_status_normalizes_legacy_deselect_results() -> None:
    """Verify batch polling applies the same persisted-result compatibility.

    Inputs:
        None. A sequenced fake HANA session returns the batch job, document
        progress, and one completed legacy question-result row.

    Outputs:
        None. Assertions confirm batch polling marks the result for rerun and
        prevents its historical verified IDs from being applied.
    """

    session = _SequencedSession(
        [
            {
                "job_id": "batch-job-legacy",
                "language": "en",
                "status": "completed",
                "indexed_chunk_count": 4,
                "active_question_id": None,
                "active_dimension": None,
            },
            {
                "document_count": 1,
                "extracted_document_count": 1,
                "embedded_document_count": 1,
            },
            [_legacy_task_row()],
        ]
    )

    status = HanaAiReviewRepository(session=session).get_batch_job_status(
        "batch-job-legacy"
    )

    assert status.status == "completed"
    assert status.tasks[0].result is not None
    assert status.tasks[0].result_requires_rerun is True
    assert status.tasks[0].result.verified_selected_answer_item_ids == []
    decision = status.tasks[0].result.level_results[0].answer_item_decisions[0]
    assert decision.decision == "unsupported"


def test_create_job_persists_question_tasks_and_attachments() -> None:
    """Verify creating a dimension job stores pending tasks and uploaded files.

    Inputs:
        None. The test creates a Strategy review job with current answers and
        one PDF attachment for a single question.

    Outputs:
        None. Assertions confirm the job task count, pending task question ID,
        and stored attachment file name.
    """
    repository = InMemoryAiReviewRepository()

    job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )

    tasks = repository.list_pending_tasks(limit=10)

    assert job.task_count == 1
    assert tasks[0]["question_id"] == "Q.STR.01.01"
    attachment = repository.get_task_attachments(tasks[0]["task_id"])[0]
    assert attachment["attachment_id"].startswith("attachment-")
    assert attachment["file_name"] == "strategy.pdf"


def test_create_job_persists_language_on_job_and_task() -> None:
    """Verify review job language survives repository creation and leasing.

    Inputs:
        None. The test creates an Italian Strategy job in memory.

    Outputs:
        None. Assertions confirm polling and worker lease payloads expose the
        per-job language needed for localized AI prompts.
    """
    repository = InMemoryAiReviewRepository()

    job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        language="it",
        current_answers={"Q.STR.01.01": []},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )
    task = repository.lease_next_task(worker_id="worker-1")

    assert job.language == "it"
    assert repository.get_job_status(job.job_id).language == "it"
    assert task is not None
    assert task["language"] == "it"


def test_lease_task_marks_task_in_progress() -> None:
    """Verify a worker lease marks the next pending task in progress.

    Inputs:
        None. The test creates a one-question Strategy job and leases it with a
        worker ID.

    Outputs:
        None. Assertions confirm the leased task status and lease owner.
    """
    repository = InMemoryAiReviewRepository()
    repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        current_answers={"Q.STR.01.01": []},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )

    task = repository.lease_next_task(worker_id="worker-1")

    assert task is not None
    assert task["status"] == "in_progress"
    assert task["lease_owner"] == "worker-1"


def test_hana_corpus_question_job_uses_local_pending_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify local corpus jobs are invisible to deployed legacy workers.

    Inputs:
        monkeypatch: Pytest fixture used to clear ``APP_ENV``.

    Outputs:
        None. Assertions confirm local HANA task rows use ``pending_local``.
    """

    monkeypatch.delenv("APP_ENV", raising=False)
    session = _SpySession(task_row=None)
    repository = HanaAiReviewRepository(session)
    repository.list_questions = lambda **_kwargs: [_assessment_question()]  # type: ignore[method-assign]

    repository.create_corpus_question_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        question_ids=["Q.STR.03.01"],
        current_answers={},
        language="en",
    )

    task_inserts = [
        params
        for sql, params in session.calls
        if "insert into ai_question_tasks" in sql.lower()
    ]
    assert task_inserts
    assert task_inserts[-1]["status"] == "pending_local"  # type: ignore[index]


def test_hana_corpus_question_job_keeps_production_pending_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify production corpus jobs keep the deployed worker queue status.

    Inputs:
        monkeypatch: Pytest fixture used to set ``APP_ENV``.

    Outputs:
        None. Assertions confirm production task rows use ``pending``.
    """

    monkeypatch.setenv("APP_ENV", "production")
    session = _SpySession(task_row=None)
    repository = HanaAiReviewRepository(session)
    repository.list_questions = lambda **_kwargs: [_assessment_question()]  # type: ignore[method-assign]

    repository.create_corpus_question_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        question_ids=["Q.STR.03.01"],
        current_answers={},
        language="en",
    )

    task_inserts = [
        params
        for sql, params in session.calls
        if "insert into ai_question_tasks" in sql.lower()
    ]
    assert task_inserts
    assert task_inserts[-1]["status"] == "pending"  # type: ignore[index]


def test_hana_lease_task_uses_runtime_pending_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify local workers lease ``pending_local`` tasks, not ``pending``.

    Inputs:
        monkeypatch: Pytest fixture used to clear ``APP_ENV``.

    Outputs:
        None. Assertions confirm HANA lease SQL binds ``pending_local``.
    """

    monkeypatch.delenv("APP_ENV", raising=False)
    session = _SpySession(
        {
            "task_id": "task-1",
            "job_id": "job-1",
            "assessment_id": "assessment-1",
            "question_id": "Q.STR.03.01",
            "dimension": "Strategy",
            "language": "en",
            "current_answers_json": "[]",
            "status": "pending_local",
        }
    )
    repository = HanaAiReviewRepository(session)

    task = repository.lease_next_task(worker_id="worker-local")

    assert task is not None
    assert task["task_id"] == "task-1"
    lease_calls = [
        params
        for sql, params in session.calls
        if "update ai_question_tasks" in sql.lower()
        and "pending_status" in str(params)
    ]
    assert lease_calls
    assert lease_calls[-1]["pending_status"] == "pending_local"  # type: ignore[index]


def test_update_question_progress_exposes_manual_task_progress() -> None:
    """Verify manual question progress is persisted for polling clients.

    Inputs:
        None. The test creates and leases one manual question task, then records
        progress telemetry that should mirror batch question progress fields.

    Outputs:
        None. Assertions confirm ``get_job_status`` exposes the progress
        message, RAG counters, and last update timestamp for the active task.
    """

    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    task = repository.lease_next_task(worker_id="worker-1")
    assert task is not None

    repository.update_question_progress(
        task_id=task_id,
        worker_id="worker-1",
        status="embedding_documents",
        progress_message="Embedding 12 evidence chunks.",
        rag_call_count=0,
        query_count=0,
        retrieved_chunk_count=0,
    )

    status = repository.get_job_status(repository.tasks[task_id]["job_id"])
    task_status = status.tasks[0]
    assert status.status == "in_progress"
    assert task_status.status == "embedding_documents"
    assert task_status.progress_message == "Embedding 12 evidence chunks."
    assert task_status.rag_call_count == 0
    assert task_status.query_count == 0
    assert task_status.retrieved_chunk_count == 0
    assert task_status.updated_at is not None


def test_hana_update_question_progress_uses_manual_task_progress_columns() -> None:
    """Verify the HANA manual progress updater writes task telemetry columns.

    Inputs:
        None. The test uses a spy SQLAlchemy session so the SQL boundary can be
        inspected without a HANA database.

    Outputs:
        None. Assertions confirm the update targets ``ai_question_tasks`` and
        guards writes with the active task lease.
    """

    session = _SpySession(task_row=None)
    repository = HanaAiReviewRepository(session)

    repository.update_question_progress(
        task_id="task-1",
        worker_id="worker-1",
        status="retrieving_evidence",
        progress_message="RAG call 1/2 retrieved 4 chunks.",
        rag_call_count=1,
        query_count=3,
        retrieved_chunk_count=4,
    )

    progress_calls = [
        (sql, params)
        for sql, params in session.calls
        if "update ai_question_tasks" in " ".join(sql.lower().split())
        and "progress_message" in sql.lower()
    ]
    assert progress_calls
    sql, params = progress_calls[-1]
    normalized_sql = " ".join(sql.lower().split())
    assert "lease_owner = :worker_id" in normalized_sql
    assert "lease_expires_at > current_timestamp" in normalized_sql
    assert params == {
        "task_id": "task-1",
        "worker_id": "worker-1",
        "status": "retrieving_evidence",
        "progress_message": "RAG call 1/2 retrieved 4 chunks.",
        "rag_call_count": 1,
        "query_count": 3,
        "retrieved_chunk_count": 4,
    }


def test_lease_task_does_not_duplicate_unexpired_in_progress_task() -> None:
    """Verify active leases are not assigned to another worker before expiry.

    Inputs:
        None. The test creates and leases one task, then immediately requests
        another lease from a different worker.

    Outputs:
        None. Assertions confirm no second worker receives the active task.
    """
    repository = InMemoryAiReviewRepository()
    _create_single_question_job(repository)

    first_task = repository.lease_next_task(worker_id="worker-1")
    second_task = repository.lease_next_task(worker_id="worker-2")

    assert first_task is not None
    assert second_task is None


def test_lease_task_reclaims_expired_in_progress_task() -> None:
    """Verify expired in-progress leases can be reclaimed by a new worker.

    Inputs:
        None. The test creates a task, leases it, forces the lease expiration
        into the past, and leases again.

    Outputs:
        None. Assertions confirm the same task is reassigned to the new worker.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")
    repository.tasks[task_id]["lease_expires_at"] = time.time() - 1

    reclaimed_task = repository.lease_next_task(worker_id="worker-2")

    assert reclaimed_task is not None
    assert reclaimed_task["task_id"] == task_id
    assert reclaimed_task["status"] == "in_progress"
    assert reclaimed_task["lease_owner"] == "worker-2"


def test_build_hana_url_preserves_reserved_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HANA credentials with reserved URL characters are safely encoded.

    Inputs:
        monkeypatch: Pytest helper used to set HANA environment variables.

    Outputs:
        None. Assertions confirm the URL object preserves the original password
        while rendering it with URL encoding.
    """
    monkeypatch.setenv("HANA_ADDRESS", "hana.example.test")
    monkeypatch.setenv("HANA_PORT", "443")
    monkeypatch.setenv("HANA_USER", "review_user")
    monkeypatch.setenv("HANA_PASSWORD", "p@ss/word")
    monkeypatch.setenv("HANA_ENCRYPT", "true")

    hana_url = build_hana_url()

    assert hana_url.password == "p@ss/word"
    assert "p%40ss%2Fword" in hana_url.render_as_string(hide_password=False)


def test_save_question_result_completes_known_matching_task() -> None:
    """Verify saving a matching result marks the task completed.

    Inputs:
        None. The test creates one task and saves a matching
        ``QuestionReviewResult``.

    Outputs:
        None. Assertions confirm the result is stored and the task is completed.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")
    result = _question_result()

    repository.save_question_result(
        task_id=task_id,
        worker_id="worker-1",
        result=result,
    )

    assert repository.results[task_id] == result
    assert repository.tasks[task_id]["status"] == "completed"


def test_save_question_result_rejects_unknown_task_id() -> None:
    """Verify saving a result for an unknown task fails.

    Inputs:
        None. The test attempts to save a valid result under a missing task ID.

    Outputs:
        None. Assertions confirm the repository rejects the unknown task.
    """
    repository = InMemoryAiReviewRepository()

    with pytest.raises(KeyError):
        repository.save_question_result(
            task_id="missing-task",
            worker_id="worker-1",
            result=_question_result(),
        )


def test_save_question_result_rejects_mismatched_question_id() -> None:
    """Verify task results must match the task question ID.

    Inputs:
        None. The test creates one task and tries to save a result for a
        different question.

    Outputs:
        None. Assertions confirm the mismatched result is rejected.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")

    with pytest.raises(ValueError):
        repository.save_question_result(
            task_id=task_id,
            worker_id="worker-1",
            result=_question_result(question_id="Q.STR.02.01"),
        )


def test_save_question_result_rejects_wrong_worker() -> None:
    """Verify stale workers cannot complete tasks leased by another worker.

    Inputs:
        None. The test leases a task to one worker and saves with another.

    Outputs:
        None. Assertions confirm the task remains in progress without a result.
    """

    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")

    with pytest.raises(ValueError):
        repository.save_question_result(
            task_id=task_id,
            worker_id="worker-2",
            result=_question_result(),
        )

    assert task_id not in repository.results
    assert repository.tasks[task_id]["status"] == "in_progress"


def test_save_question_result_rejects_expired_lease() -> None:
    """Verify expired leases cannot complete tasks.

    Inputs:
        None. The test leases a task and forces its expiration into the past.

    Outputs:
        None. Assertions confirm the stale lease is rejected.
    """

    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")
    repository.tasks[task_id]["lease_expires_at"] = time.time() - 1

    with pytest.raises(ValueError):
        repository.save_question_result(
            task_id=task_id,
            worker_id="worker-1",
            result=_question_result(),
        )

    assert task_id not in repository.results
    assert repository.tasks[task_id]["status"] == "in_progress"


def test_repository_persists_extraction_and_retrieval_metadata() -> None:
    """Verify in-memory repository stores extraction, chunk, and retrieval rows.

    Inputs:
        None. The test creates one review task and persists derived evidence
        metadata through the repository boundary.

    Outputs:
        None. Assertions confirm derived rows can be inspected by task ID.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    attachment = repository.get_task_attachments(task_id)[0]
    extracted = ExtractedDocument(
        file_name="strategy.pdf",
        document_type="pdf",
        metadata={"page_count": 1, "total_characters": 41},
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="Board reviews strategic objectives annually.",
                page=1,
            )
        ],
    )

    extraction_id = repository.save_attachment_extraction(
        task_id=task_id,
        attachment_id=attachment["attachment_id"],
        extracted=extracted,
        estimated_tokens=12,
        warnings=[],
    )
    repository.save_evidence_chunks(
        task_id=task_id,
        chunks=[
            {
                "chunk_id": "chunk-1",
                "attachment_id": attachment["attachment_id"],
                "file_name": "strategy.pdf",
                "document_type": "pdf",
                "source_block_ids": ["pdf-page-0001"],
                "location_json": {"page": 1},
                "chunk_text": "Board reviews strategic objectives annually.",
                "estimated_tokens": 12,
                "embedding_model": "text-embedding-3-large",
                "embedding": [0.1, 0.2, 0.3],
                "content_hash": "hash-1",
            }
        ],
    )
    repository.save_retrieval_round(
        task_id=task_id,
        round_number=1,
        queries=["board review objectives"],
        retrieved_chunk_ids=["chunk-1"],
        accepted_evidence_json={"accepted": ["chunk-1"]},
        evidence_gaps=["Need target evidence"],
        refined_queries=["objective targets"],
        stop_reason="needs_more_evidence",
    )
    attempt_id = repository.save_question_attempt(
        task_id=task_id,
        mode="rag",
        status="routed",
        model="gpt-5.4",
        usage_json={"estimated_input_tokens": 600_000},
        error_code=None,
        error_message=None,
    )

    assert extraction_id.startswith("extraction-")
    assert attempt_id.startswith("attempt-")
    assert repository.attempts[task_id][0]["usage_json"] == {
        "estimated_input_tokens": 600_000
    }
    assert attachment["attachment_id"].startswith("attachment-")
    assert repository.extractions[task_id][0]["attachment_id"] == (
        attachment["attachment_id"]
    )
    assert repository.extractions[task_id][0]["estimated_tokens"] == 12
    assert task_id not in repository.extracted_blocks
    assert repository.evidence_chunks[task_id][0]["attachment_id"] == (
        attachment["attachment_id"]
    )
    assert repository.evidence_chunks[task_id][0]["embedding"] == [0.1, 0.2, 0.3]
    assert repository.extractions[task_id][0]["quality_json"] == {
        "page_count": 1,
        "total_characters": 41,
    }
    assert repository.retrieval_rounds[task_id][0]["round_number"] == 1


def test_in_memory_save_evidence_chunks_replaces_retry_state() -> None:
    """Verify RAG chunk persistence is retry-safe in memory.

    Inputs:
        None. The test writes chunks and retrieval rounds for one task twice.

    Outputs:
        None. Assertions confirm the second write replaces deterministic chunks
        and clears stale retrieval rounds from a previous failed run.
    """

    repository = InMemoryAiReviewRepository()
    repository.save_evidence_chunks(
        task_id="task-1",
        chunks=[
            {
                "chunk_id": "chunk-old",
                "attachment_id": "attachment-1",
                "file_name": "old.pdf",
                "document_type": "pdf",
                "source_block_ids": ["pdf-page-0001"],
                "location_json": {"pages": [1]},
                "chunk_text": "Old evidence.",
                "estimated_tokens": 3,
                "embedding_model": "text-embedding-3-large",
                "embedding": [0.1],
                "content_hash": "old",
            }
        ],
    )
    repository.save_retrieval_round(
        task_id="task-1",
        round_number=1,
        queries=["old"],
        retrieved_chunk_ids=["chunk-old"],
        accepted_evidence_json={},
        evidence_gaps=[],
        refined_queries=[],
        stop_reason="old",
    )

    repository.save_evidence_chunks(
        task_id="task-1",
        chunks=[
            {
                "chunk_id": "chunk-new",
                "attachment_id": "attachment-2",
                "file_name": "new.pdf",
                "document_type": "pdf",
                "source_block_ids": ["pdf-page-0001"],
                "location_json": {"pages": [1]},
                "chunk_text": "New evidence.",
                "estimated_tokens": 3,
                "embedding_model": "text-embedding-3-large",
                "embedding": [0.2],
                "content_hash": "new",
            }
        ],
    )

    assert [chunk["chunk_id"] for chunk in repository.evidence_chunks["task-1"]] == [
        "chunk-new"
    ]
    assert "task-1" not in repository.retrieval_rounds


def test_hana_search_evidence_chunks_uses_vector_similarity() -> None:
    """Verify HANA retrieval uses vector SQL and parses location JSON."""

    session = _SpySession(
        [
            {
                "chunk_id": "chunk-1",
                "attachment_id": "attachment-1",
                "file_name": "strategy.pdf",
                "document_type": "pdf",
                "chunk_text": "Board reviews strategic objectives annually.",
                "location_json": "{\"pages\": [1]}",
                "similarity_score": 0.91,
            }
        ]
    )
    repository = HanaAiReviewRepository(session=session)

    rows = repository.search_evidence_chunks(
        task_id="task-1",
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    assert len(session.calls) == 1
    sql, parameters = session.calls[0]
    normalized_sql = " ".join(sql.lower().split())

    assert "from ai_evidence_chunks" in normalized_sql
    assert "chunk_id, attachment_id, file_name" in normalized_sql
    assert "cosine_similarity" in normalized_sql
    assert "to_real_vector(:query_vector)" in normalized_sql
    assert "where task_id = :task_id" in normalized_sql
    assert "embedding is not null" in normalized_sql
    assert "order by similarity_score desc" in normalized_sql
    assert parameters == {"task_id": "task-1", "query_vector": "[0.1,0.2,0.3]"}
    assert rows == [
        {
            "chunk_id": "chunk-1",
            "attachment_id": "attachment-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "chunk_text": "Board reviews strategic objectives annually.",
            "location_json": {"pages": [1]},
            "similarity_score": 0.91,
        }
    ]


def test_hana_search_admin_document_chunks_uses_isolated_vector_table() -> None:
    """Verify admin RAG vector search never reads assessment evidence chunks."""

    session = _SpySession(
        [
            {
                "chunk_id": "admin-chunk-1",
                "document_id": "admin-document-1",
                "file_name": "policy.pdf",
                "document_type": "pdf",
                "chunk_text": "Admin policy context.",
                "location_json": "{\"pages\": [2]}",
                "similarity_score": 0.88,
            }
        ]
    )
    repository = HanaAiReviewRepository(session=session)

    rows = repository.search_admin_document_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    assert len(session.calls) == 1
    sql, parameters = session.calls[0]
    normalized_sql = " ".join(sql.lower().split())

    assert "from joule_admin_document_chunks" in normalized_sql
    assert "from ai_document_chunks" not in normalized_sql
    assert "cosine_similarity" in normalized_sql
    assert "to_real_vector(:query_vector)" in normalized_sql
    assert "embedding is not null" in normalized_sql
    assert "order by similarity_score desc" in normalized_sql
    assert parameters == {"query_vector": "[0.1,0.2,0.3]"}
    assert rows == [
        {
            "chunk_id": "admin-chunk-1",
            "document_id": "admin-document-1",
            "file_name": "policy.pdf",
            "document_type": "pdf",
            "chunk_text": "Admin policy context.",
            "location_json": {"pages": [2]},
            "similarity_score": 0.88,
        }
    ]


def test_hana_create_admin_document_ingestion_job_uses_hana_safe_ids() -> None:
    """Verify generated admin ingestion IDs fit the HANA schema width."""

    session = _SpySession(None)
    repository = HanaAiReviewRepository(session=session)

    response = repository.create_admin_document_ingestion_job(
        documents=[
            {
                "file_name": "policy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF admin policy",
            }
        ]
    )

    insert_parameters = [
        parameters
        for sql, parameters in session.calls
        if "insert into joule_admin_document_ingestion_job" in sql.lower()
    ]
    flat_parameters = [
        parameters
        for parameters in insert_parameters
        if isinstance(parameters, dict)
    ]
    generated_ids = [
        value
        for parameters in flat_parameters
        for key, value in parameters.items()
        if key in {"job_id", "link_id"}
    ]

    assert len(response.job_id) <= 64
    assert generated_ids
    assert all(len(str(identifier)) <= 64 for identifier in generated_ids)


def test_hana_save_evidence_chunks_clears_retry_state_before_insert() -> None:
    """Verify HANA chunk persistence clears old RAG rows before inserting.

    Inputs:
        None. A fake SQLAlchemy session records generated SQL.

    Outputs:
        None. Assertions confirm old retrieval rounds and deterministic chunks
        are deleted for the task before new chunks are inserted.
    """

    session = _SpySession([])
    repository = HanaAiReviewRepository(session=session)

    repository.save_evidence_chunks(
        task_id="task-1",
        chunks=[
            {
                "chunk_id": "chunk-1",
                "attachment_id": "attachment-1",
                "file_name": "strategy.pdf",
                "document_type": "pdf",
                "source_block_ids": ["pdf-page-0001"],
                "location_json": {"pages": [1]},
                "chunk_text": "Board reviews strategic objectives annually.",
                "estimated_tokens": 12,
                "embedding_model": "text-embedding-3-large",
                "embedding": [0.1, 0.2, 0.3],
                "content_hash": "hash-1",
            }
        ],
    )

    normalized_sql = [" ".join(sql.lower().split()) for sql, _params in session.calls]
    assert normalized_sql[0] == "delete from ai_retrieval_rounds where task_id = :task_id"
    assert normalized_sql[1] == "delete from ai_evidence_chunks where task_id = :task_id"
    assert normalized_sql[2].startswith("insert into ai_evidence_chunks")
    assert session.calls[0][1] == {"task_id": "task-1"}
    assert session.calls[1][1] == {"task_id": "task-1"}
    assert isinstance(session.calls[2][1], list)
    assert len(session.calls[2][1]) == 1


def test_hana_save_attachment_extraction_skips_raw_block_rows() -> None:
    """Verify HANA extraction audit stores summaries without raw block rows.

    Inputs:
        None. A fake session records SQL generated for one extraction with two
        source blocks.

    Outputs:
        None. Assertions confirm only the extraction summary row is inserted;
        raw extracted blocks are not persisted because RAG chunks are the
        retained evidence corpus.
    """

    session = _SpySession([])
    repository = HanaAiReviewRepository(session=session)
    extracted = ExtractedDocument(
        file_name="strategy.pdf",
        document_type="pdf",
        metadata={"page_count": 2, "total_characters": 50},
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="First page text.",
                page=1,
            ),
            ExtractedBlock(
                block_id="pdf-page-0002",
                block_type="page",
                text="Second page text.",
                page=2,
            ),
        ],
    )

    repository.save_attachment_extraction(
        task_id="task-1",
        attachment_id="attachment-1",
        extracted=extracted,
        estimated_tokens=20,
        warnings=[],
    )

    normalized_sql = [" ".join(sql.lower().split()) for sql, _params in session.calls]
    assert any(sql.startswith("insert into ai_attachment_extractions") for sql in normalized_sql)
    assert not any("insert into ai_extracted_blocks" in sql for sql in normalized_sql)


def test_hana_save_evidence_chunks_batches_multiple_rows() -> None:
    """Verify HANA chunk persistence batches inserts for multiple chunks.

    Inputs:
        None. A fake session records SQL and parameters for two chunk rows.

    Outputs:
        None. Assertions confirm both chunk payloads are passed in one
        executemany-style insert call after stale task state is cleared.
    """

    session = _SpySession([])
    repository = HanaAiReviewRepository(session=session)
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "attachment_id": "attachment-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "source_block_ids": [f"pdf-page-{index:04d}"],
            "location_json": {"pages": [index]},
            "chunk_text": f"Chunk {index} text.",
            "estimated_tokens": 12,
            "embedding_model": "text-embedding-3-large",
            "embedding": [0.1, float(index)],
            "content_hash": f"hash-{index}",
        }
        for index in range(1, 3)
    ]

    repository.save_evidence_chunks(task_id="task-1", chunks=chunks)

    insert_calls = [
        (sql, params)
        for sql, params in session.calls
        if "insert into ai_evidence_chunks" in " ".join(sql.lower().split())
    ]
    assert len(insert_calls) == 1
    assert isinstance(insert_calls[0][1], list)
    assert [row["chunk_id"] for row in insert_calls[0][1]] == ["chunk-1", "chunk-2"]


def test_hana_save_batch_document_extraction_skips_raw_block_rows() -> None:
    """Verify batch extraction audit stores summaries without raw block rows.

    Inputs:
        None. A fake session records SQL generated for one batch document
        extraction with two source blocks.

    Outputs:
        None. Assertions confirm no ``ai_batch_extracted_blocks`` insert is
        issued because retained retrieval evidence is stored as chunks.
    """

    session = _SpySession([])
    repository = HanaAiReviewRepository(session=session)
    extracted = ExtractedDocument(
        file_name="strategy.pdf",
        document_type="pdf",
        metadata={"page_count": 2, "total_characters": 50},
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="First page text.",
                page=1,
            ),
            ExtractedBlock(
                block_id="pdf-page-0002",
                block_type="page",
                text="Second page text.",
                page=2,
            ),
        ],
    )

    repository.save_batch_document_extraction(
        job_id="batch-job-1",
        document_id="document-1",
        extracted=extracted,
        estimated_tokens=20,
        warnings=[],
    )

    normalized_sql = [" ".join(sql.lower().split()) for sql, _params in session.calls]
    assert any(sql.startswith("insert into ai_batch_document_extractions") for sql in normalized_sql)
    assert not any("insert into ai_batch_extracted_blocks" in sql for sql in normalized_sql)


def test_hana_save_batch_document_chunks_batches_multiple_rows() -> None:
    """Verify HANA batch chunk persistence batches inserts for multiple chunks.

    Inputs:
        None. A fake session records SQL and parameters for two batch document
        chunk rows.

    Outputs:
        None. Assertions confirm both rows are passed in one insert call after
        stale batch state is cleared.
    """

    session = _SpySession([])
    repository = HanaAiReviewRepository(session=session)
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "document_id": "document-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "source_block_ids": [f"pdf-page-{index:04d}"],
            "location_json": {"pages": [index]},
            "chunk_text": f"Chunk {index} text.",
            "estimated_tokens": 12,
            "embedding_model": "text-embedding-3-large",
            "embedding": [0.1, float(index)],
            "content_hash": f"hash-{index}",
            "chunk_kind": "raw",
        }
        for index in range(1, 3)
    ]

    repository.save_batch_document_chunks(
        job_id="batch-job-1",
        chunks=chunks,
        replace_existing=True,
    )

    insert_calls = [
        (sql, params)
        for sql, params in session.calls
        if "insert into ai_batch_document_chunks" in " ".join(sql.lower().split())
    ]
    assert len(insert_calls) == 1
    assert isinstance(insert_calls[0][1], list)
    assert [row["chunk_id"] for row in insert_calls[0][1]] == ["chunk-1", "chunk-2"]


def test_hana_search_evidence_chunks_clamps_top_k_to_hana_limit() -> None:
    """Verify HANA evidence search clamps requested result counts.

    Inputs:
        None. A fake SQLAlchemy session records the generated SQL.

    Outputs:
        None. Assertions confirm low and high ``top_k`` values are clamped to
        the supported HANA SQL range.
    """

    low_session = _SpySession([])
    low_repository = HanaAiReviewRepository(session=low_session)
    high_session = _SpySession([])
    high_repository = HanaAiReviewRepository(session=high_session)

    low_repository.search_evidence_chunks(
        task_id="task-1",
        query_embedding=[0.1],
        top_k=0,
    )
    high_repository.search_evidence_chunks(
        task_id="task-1",
        query_embedding=[0.1],
        top_k=500,
    )

    assert "select top 1" in low_session.calls[0][0].lower()
    assert "select top 50" in high_session.calls[0][0].lower()


def test_hana_save_batch_rag_tool_call_uses_retrieval_round_schema() -> None:
    """Verify ReAct RAG audit inserts match the batch retrieval DDL.

    Inputs:
        None. A fake SQLAlchemy session records generated SQL.

    Outputs:
        None. Assertions confirm the insert targets existing
        ``ai_batch_retrieval_rounds`` columns instead of the nonexistent
        ``round_id`` column.
    """

    session = _SpySession({})
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    retrieval_round_id = repository.save_batch_rag_tool_call(
        job_id="batch-job-1",
        task_id="batch-task-1",
        rag_call_number=1,
        queries_json=[{"query": "strategy", "answer_item_id": "A1", "level": 1}],
        retrieved_chunk_ids=["chunk-1"],
        matched_queries_json={"chunk-1": [{"query": "strategy"}]},
        query_count=1,
        retrieved_chunk_count=1,
        duration_ms=123,
        stop_reason="tool_result",
    )

    insert_sql, parameters = session.calls[-1]
    normalized_sql = " ".join(insert_sql.lower().split())
    assert retrieval_round_id.startswith("batch-retrieval-round-")
    assert "insert into ai_batch_retrieval_rounds" in normalized_sql
    assert "retrieval_round_id" in normalized_sql
    assert "(round_id" not in normalized_sql
    assert "queries_json" in normalized_sql
    assert "retrieved_chunk_ids_json" in normalized_sql
    assert parameters is not None
    assert parameters["retrieval_round_id"] == retrieval_round_id
    assert parameters["stop_reason"] == "tool_result"


def test_hana_save_question_result_requires_active_worker_lease() -> None:
    """Verify HANA result completion SQL validates the active worker lease.

    Inputs:
        None. A fake SQLAlchemy session returns a matching leased task row.

    Outputs:
        None. Assertions confirm the adapter uses active lease predicates and
        replaces any prior canonical result before inserting the new result.
    """

    session = _SpySession(
        {
            "question_id": "Q.STR.01.01",
            "status": "in_progress",
            "lease_owner": "worker-1",
        }
    )
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    repository.save_question_result(
        task_id="task-1",
        worker_id="worker-1",
        result=_question_result(),
    )

    executed_sql = "\n".join(sql.lower() for sql, _params in session.calls)
    assert "status in ('in_progress', 'extracting_documents'" in executed_sql
    assert "'finalizing_answer', 'retrying')" in executed_sql
    assert "lease_owner = :worker_id" in executed_sql
    assert "lease_expires_at > current_timestamp" in executed_sql
    assert "delete from ai_question_results" in executed_sql
    assert "where task_id = :task_id" in executed_sql
    statements = [sql.lower() for sql, _params in session.calls]
    update_index = next(
        index for index, sql in enumerate(statements) if "update ai_question_tasks" in sql
    )
    delete_index = next(
        index for index, sql in enumerate(statements) if "delete from ai_question_results" in sql
    )
    insert_index = next(
        index for index, sql in enumerate(statements) if "insert into ai_question_results" in sql
    )
    assert update_index < delete_index < insert_index


def test_pending_tasks_and_attachments_are_deep_copy_isolated() -> None:
    """Verify returned task and attachment data cannot mutate repository state.

    Inputs:
        None. The test mutates task and attachment data returned by read
        methods.

    Outputs:
        None. Assertions confirm repository internals still hold original data.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)

    pending_tasks = repository.list_pending_tasks(limit=1)
    pending_tasks[0]["current_selected_answer_item_ids"].append("mutated")
    attachments = repository.get_task_attachments(task_id)
    attachments[0]["file_name"] = "mutated.pdf"

    assert repository.tasks[task_id]["current_selected_answer_item_ids"] == [
        "Q.STR.01.01-L1-001"
    ]
    assert repository.attachments[task_id][0]["file_name"] == "strategy.pdf"


def test_save_question_failure_marks_task_failed_and_pollable() -> None:
    """Verify task failures are persisted for polling clients.

    Inputs:
        None. The test creates one task, leases it, and records a model failure.

    Outputs:
        None. Assertions confirm the task is marked failed and the job status
        exposes the error message instead of leaving the task in progress.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    leased_task = repository.lease_next_task(worker_id="worker-1")
    assert leased_task is not None
    job_id = leased_task["job_id"]

    repository.save_question_failure(
        task_id=task_id,
        worker_id="worker-1",
        error_message="SAP Gen AI Hub Responses API question review failed.",
    )

    status = repository.get_job_status(job_id)

    assert repository.tasks[task_id]["status"] == "failed"
    assert repository.tasks[task_id]["error_message"] == (
        "SAP Gen AI Hub Responses API question review failed."
    )
    assert status.status == "failed"
    assert status.failed_count == 1
    assert status.tasks[0].error_message == (
        "SAP Gen AI Hub Responses API question review failed."
    )


def test_save_question_failure_rejects_wrong_worker() -> None:
    """Verify stale workers cannot mark another worker's task as failed.

    Inputs:
        None. The test leases one task with ``worker-1`` and attempts failure
        persistence with a different worker ID.

    Outputs:
        None. Assertions confirm the repository rejects the stale worker and
        keeps the task in progress.
    """
    repository = InMemoryAiReviewRepository()
    task_id = _create_single_question_job(repository)
    repository.lease_next_task(worker_id="worker-1")

    with pytest.raises(ValueError):
        repository.save_question_failure(
            task_id=task_id,
            worker_id="worker-2",
            error_message="provider unavailable",
        )

    assert repository.tasks[task_id]["status"] == "in_progress"


def test_clear_dimension_review_state_removes_matching_review_data_only() -> None:
    """Verify dimension reset deletes persisted jobs, tasks, files, and results.

    Inputs:
        None. The test creates two review jobs in different dimensions and
        completes the Strategy task before clearing only Strategy state.

    Outputs:
        None. Assertions confirm matching jobs, tasks, attachments, and results
        are removed while other dimension state remains pollable.
    """
    repository = InMemoryAiReviewRepository()
    strategy_job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )
    risk_job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Risk & Control Governance",
        current_answers={"Q.RCG.01.01": []},
        attachments={
            "Q.RCG.01.01": [
                {
                    "file_name": "risk.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF risk evidence",
                }
            ]
        },
    )
    strategy_task = repository.lease_next_task(worker_id="worker-1")
    assert strategy_task is not None
    strategy_attachment = repository.get_task_attachments(strategy_task["task_id"])[0]
    extracted = ExtractedDocument(
        file_name="strategy.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="Board reviews strategic objectives annually.",
                page=1,
            )
        ],
        metadata={"page_count": 1, "total_characters": 41},
    )
    repository.save_attachment_extraction(
        task_id=strategy_task["task_id"],
        attachment_id=strategy_attachment["attachment_id"],
        extracted=extracted,
        estimated_tokens=12,
        warnings=[],
    )
    repository.save_evidence_chunks(
        task_id=strategy_task["task_id"],
        chunks=[
            {
                "chunk_id": "chunk-1",
                "attachment_id": strategy_attachment["attachment_id"],
                "file_name": "strategy.pdf",
                "document_type": "pdf",
                "source_block_ids": ["pdf-page-0001"],
                "location_json": {"page": 1},
                "chunk_text": "Board reviews strategic objectives annually.",
                "estimated_tokens": 12,
                "embedding_model": "text-embedding-3-large",
                "embedding": [0.1, 0.2, 0.3],
                "content_hash": "hash-1",
            }
        ],
    )
    repository.save_retrieval_round(
        task_id=strategy_task["task_id"],
        round_number=1,
        queries=["board review objectives"],
        retrieved_chunk_ids=["chunk-1"],
        accepted_evidence_json={"accepted": ["chunk-1"]},
        evidence_gaps=[],
        refined_queries=[],
        stop_reason="sufficient_evidence",
    )
    repository.save_question_attempt(
        task_id=strategy_task["task_id"],
        mode="rag",
        status="completed",
        model="gpt-5.4",
        usage_json={"estimated_input_tokens": 600_000},
        error_code=None,
        error_message=None,
    )
    repository.save_question_result(
        task_id=strategy_task["task_id"],
        worker_id="worker-1",
        result=_question_result(),
    )

    reset = repository.clear_dimension_review_state(
        assessment_id="assessment-1",
        dimension="Strategy",
    )

    assert reset.deleted_job_count == 1
    assert reset.deleted_task_count == 1
    assert strategy_job.job_id not in repository.jobs
    assert strategy_task["task_id"] not in repository.tasks
    assert strategy_task["task_id"] not in repository.attachments
    assert strategy_task["task_id"] not in repository.results
    assert strategy_task["task_id"] not in repository.extractions
    assert strategy_task["task_id"] not in repository.extracted_blocks
    assert strategy_task["task_id"] not in repository.evidence_chunks
    assert strategy_task["task_id"] not in repository.retrieval_rounds
    assert strategy_task["task_id"] not in repository.attempts
    with pytest.raises(KeyError):
        repository.get_job_status(strategy_job.job_id)
    assert repository.get_job_status(risk_job.job_id).task_count == 1


def test_clear_assessment_review_state_removes_all_dimension_review_data() -> None:
    """Verify assessment reset deletes review state across all dimensions.

    Inputs:
        None. The test creates two dimension jobs under the same assessment.

    Outputs:
        None. Assertions confirm both jobs and their tasks are removed, so a
        worker restart cannot resume stale work from another dimension.
    """
    repository = InMemoryAiReviewRepository()
    strategy_job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Strategy",
        current_answers={"Q.STR.01.01": []},
        attachments={
            "Q.STR.01.01": [
                {
                    "file_name": "strategy.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF strategy evidence",
                }
            ]
        },
    )
    risk_job = repository.create_dimension_job(
        assessment_id="assessment-1",
        dimension="Risk & Control Governance",
        current_answers={"Q.RCG.01.01": []},
        attachments={
            "Q.RCG.01.01": [
                {
                    "file_name": "risk.pdf",
                    "content_type": "application/pdf",
                    "content": b"%PDF risk evidence",
                }
            ]
        },
    )

    reset = repository.clear_assessment_review_state(assessment_id="assessment-1")

    assert reset.deleted_job_count == 2
    assert reset.deleted_task_count == 2
    assert reset.dimension == "all"
    with pytest.raises(KeyError):
        repository.get_job_status(strategy_job.job_id)
    with pytest.raises(KeyError):
        repository.get_job_status(risk_job.job_id)


def test_hana_clear_dimension_review_state_deletes_child_rows_before_jobs() -> None:
    """Verify HANA reset SQL removes all AI review rows scoped by dimension.

    Inputs:
        None. A fake SQLAlchemy session records the SQL executed by the HANA
        repository boundary.

    Outputs:
        None. Assertions confirm the repository counts affected jobs/tasks and
        deletes child tables before parent task and job rows.
    """
    session = _SpySession(
        {"task_count": 2, "job_count": 1},
        existing_tables={
            "AI_RETRIEVAL_ROUNDS",
            "AI_EVIDENCE_CHUNKS",
            "AI_EXTRACTED_BLOCKS",
            "AI_ATTACHMENT_EXTRACTIONS",
            "AI_APPLIED_SUGGESTIONS",
            "AI_EVIDENCE_REFS",
            "AI_QUESTION_CHUNK_RUNS",
            "AI_QUESTION_ATTEMPTS",
            "AI_QUESTION_RESULTS",
            "AI_TASK_ATTACHMENTS",
        },
    )
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    reset = repository.clear_dimension_review_state(
        assessment_id="assessment-1",
        dimension="Strategy",
    )

    assert reset.deleted_task_count == 2
    assert reset.deleted_job_count == 1
    statements = [sql.lower() for sql, _params in session.calls]
    executed_sql = "\n".join(statements)
    assert "delete from ai_retrieval_rounds" in executed_sql
    assert "delete from ai_evidence_chunks" in executed_sql
    assert "delete from ai_extracted_blocks" in executed_sql
    assert "delete from ai_attachment_extractions" in executed_sql
    assert "delete from ai_task_attachments" in executed_sql
    assert "delete from ai_question_results" in executed_sql
    assert "delete from ai_dimension_jobs" in executed_sql
    task_delete_index = next(
        index
        for index, sql in enumerate(statements)
        if "delete from ai_question_tasks" in sql
    )
    job_delete_index = next(
        index
        for index, sql in enumerate(statements)
        if "delete from ai_dimension_jobs" in sql
    )
    assert task_delete_index < job_delete_index


def test_hana_clear_assessment_review_state_deletes_batch_process_tables() -> None:
    """Verify complete HANA reset removes legacy and batch processing state.

    Inputs:
        None. A fake SQLAlchemy session records HANA SQL for assessment reset.

    Outputs:
        None. Assertions confirm batch parent jobs, question tasks, document
        rows, chunks, extractions, and retrieval audit rows are all targeted by
        the reset SQL.
    """
    session = _SpySession(
        {"task_count": 2, "job_count": 1},
        existing_tables={
            "AI_RETRIEVAL_ROUNDS",
            "AI_EVIDENCE_CHUNKS",
            "AI_EXTRACTED_BLOCKS",
            "AI_ATTACHMENT_EXTRACTIONS",
            "AI_APPLIED_SUGGESTIONS",
            "AI_EVIDENCE_REFS",
            "AI_QUESTION_CHUNK_RUNS",
            "AI_QUESTION_ATTEMPTS",
            "AI_QUESTION_RESULTS",
            "AI_TASK_ATTACHMENTS",
            "AI_BATCH_RETRIEVAL_ROUNDS",
            "AI_BATCH_QUESTION_RESULTS",
            "AI_BATCH_DOCUMENT_CHUNKS",
            "AI_BATCH_EXTRACTED_BLOCKS",
            "AI_BATCH_DOCUMENT_EXTRACTIONS",
            "AI_BATCH_JOB_DOCUMENTS",
            "AI_BATCH_QUESTION_TASKS",
            "AI_BATCH_DOCUMENTS",
            "AI_BATCH_JOBS",
        },
    )
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    reset = repository.clear_assessment_review_state(assessment_id="assessment-1")

    assert reset.deleted_task_count == 4
    assert reset.deleted_job_count == 2
    assert reset.dimension == "all"
    executed_sql = "\n".join(sql.lower() for sql, _params in session.calls)
    assert "delete from ai_batch_retrieval_rounds" in executed_sql
    assert "delete from ai_batch_question_results" in executed_sql
    assert "delete from ai_batch_document_chunks" in executed_sql
    assert "delete from ai_batch_extracted_blocks" in executed_sql
    assert "delete from ai_batch_document_extractions" in executed_sql
    assert "delete from ai_batch_job_documents" in executed_sql
    assert "delete from ai_batch_question_tasks" in executed_sql
    assert "delete from ai_batch_documents" in executed_sql
    assert "delete from ai_batch_jobs" in executed_sql


def test_hana_clear_dimension_review_state_skips_missing_child_tables() -> None:
    """Verify HANA reset works when a deployed schema lacks newer child tables.

    Inputs:
        None. A fake SQLAlchemy session reports only older task-scoped child
        tables as present through ``sys.tables``.

    Outputs:
        None. Assertions confirm reset still counts jobs/tasks, skips missing
        child deletes, and deletes parent task/job rows.
    """
    session = _SpySession(
        {"task_count": 2, "job_count": 1},
        existing_tables={
            "AI_QUESTION_ATTEMPTS",
            "AI_QUESTION_RESULTS",
            "AI_TASK_ATTACHMENTS",
        },
    )
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    reset = repository.clear_dimension_review_state(
        assessment_id="assessment-1",
        dimension="Strategy",
    )

    assert reset.deleted_task_count == 2
    assert reset.deleted_job_count == 1
    statements = [sql.lower() for sql, _params in session.calls]
    executed_sql = "\n".join(statements)
    assert "delete from ai_retrieval_rounds" not in executed_sql
    assert "delete from ai_evidence_chunks" not in executed_sql
    assert "delete from ai_extracted_blocks" not in executed_sql
    assert "delete from ai_attachment_extractions" not in executed_sql
    assert "delete from ai_question_attempts" in executed_sql
    assert "delete from ai_question_results" in executed_sql
    assert "delete from ai_task_attachments" in executed_sql
    assert "delete from ai_question_tasks" in executed_sql
    assert "delete from ai_dimension_jobs" in executed_sql


def test_hana_save_question_failure_requires_active_worker_lease() -> None:
    """Verify HANA task failure SQL validates the active worker lease.

    Inputs:
        None. A fake SQLAlchemy session returns a matching leased task row.

    Outputs:
        None. Assertions confirm the adapter updates failed state only through
        active lease predicates.
    """
    session = _SpySession(
        {
            "question_id": "Q.STR.01.01",
            "status": "in_progress",
            "lease_owner": "worker-1",
        }
    )
    repository = HanaAiReviewRepository(session)  # type: ignore[arg-type]

    repository.save_question_failure(
        task_id="task-1",
        worker_id="worker-1",
        error_message="provider unavailable",
    )

    executed_sql = "\n".join(sql.lower() for sql, _params in session.calls)
    assert "set status = 'failed'" in executed_sql
    assert "error_message = :error_message" in executed_sql
    assert "status in ('in_progress', 'extracting_documents'" in executed_sql
    assert "'finalizing_answer', 'retrying')" in executed_sql
    assert "lease_owner = :worker_id" in executed_sql
    assert "lease_expires_at > current_timestamp" in executed_sql
