"""Tests for deterministic report enqueue, persistence, and delivery contracts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from inspect import getsource
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.models.benchmarking import (
    AssessmentProfile,
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
    BenchmarkScopeScore,
)
from app.models.reports import (
    AssessmentReportDownload,
    AssessmentReportProvenance,
    AssessmentReportSource,
)
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository import hana_reports
from app.services.ai_review_repository.hana_reports import HanaAssessmentReportsMixin
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository
from app.services.assessment_report_context import build_assessment_report_source
from tests.report_fixtures import SNAPSHOT_TIME, make_compact_source


def _framework_question(language: str = "en") -> AssessmentQuestion:
    """Return one localized, scorable canonical framework question.

    Inputs:
        language: Framework language, ``en`` or ``it``.

    Outputs:
        AssessmentQuestion: One Strategy question with two answer levels.
    """

    return AssessmentQuestion(
        question_id="Q.STR.03.01",
        dimension="Strategy",
        section="Obiettivi" if language == "it" else "Objectives",
        topic_title=(
            "Definizione degli Obiettivi"
            if language == "it"
            else "Definition of Objectives"
        ),
        question=(
            "Gli obiettivi vengono monitorati?"
            if language == "it"
            else "Are objectives monitored?"
        ),
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.03.01-L1-001",
                question_id="Q.STR.03.01",
                level=1,
                item_index=1,
                text=(
                    "Gli obiettivi sono definiti."
                    if language == "it"
                    else "Objectives are defined."
                ),
            ),
            AnswerItem(
                answer_item_id="Q.STR.03.01-L2-001",
                question_id="Q.STR.03.01",
                level=2,
                item_index=1,
                text=(
                    "Gli obiettivi vengono monitorati."
                    if language == "it"
                    else "Objectives are monitored."
                ),
            ),
        ],
    )


class _LocalizedReportRepository(InMemoryAiReviewRepository):
    """Return language-specific framework fixtures for snapshot and API tests."""

    def list_all_questions(self, language: str = "en") -> list[AssessmentQuestion]:
        """Return one question localized to the requested language.

        Inputs:
            language: Framework language.

        Outputs:
            list[AssessmentQuestion]: One localized canonical question.
        """

        return [_framework_question(language)]

    def list_dimensions(self, language: str = "en") -> list[Any]:
        """Return dimensions derived from the localized question fixture.

        Inputs:
            language: Framework language.

        Outputs:
            list[Any]: One Strategy dimension compatible with score calculation.
        """

        self.framework_questions = [_framework_question(language)]
        return super().list_dimensions(language=language)


@pytest.fixture
def report_repository() -> _LocalizedReportRepository:
    """Create a profiled report repository with one selected answer.

    Inputs:
        None.

    Outputs:
        _LocalizedReportRepository: Repository ready for deterministic enqueue.
    """

    repository = _LocalizedReportRepository()
    repository.framework_questions = [_framework_question()]
    repository.save_assessment_responses(
        assessment_id="assessment-1",
        customer_class="class_3",
        answers={"Q.STR.03.01": ["Q.STR.03.01-L2-001"]},
        source="manual",
    )
    repository.upsert_assessment_profile(
        AssessmentProfile(
            assessment_id="assessment-1",
            display_name="assessment",
            source_company_id="company-current",
            customer_class="class_3",
            nace1="Electricity and gas",
        )
    )
    return repository


@pytest.fixture
def report_api_client(report_repository: _LocalizedReportRepository) -> TestClient:
    """Override the API repository with the localized in-memory implementation.

    Inputs:
        report_repository: Profile-aware in-memory repository fixture.

    Outputs:
        TestClient: Client wired to the report repository for this test.
    """

    app.dependency_overrides[get_ai_review_repository] = lambda: report_repository
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_ai_review_repository, None)


def _set_active_import(
    repository: _LocalizedReportRepository,
    *,
    import_id: str = "import-a",
    peer_count: int = 3,
) -> None:
    """Install one active import and deterministic exact-cohort peer rows.

    Inputs:
        repository: In-memory repository to mutate.
        import_id: Active version identity.
        peer_count: Number of eligible peers to create.

    Outputs:
        None. Active import metadata and score rows are stored in memory.
    """

    repository.active_benchmark_import = BenchmarkImportInfo(
        import_id=import_id,
        source_filename=f"{import_id}.xlsx",
        source_sha256=("a" if import_id == "import-a" else "b") * 64,
        scoring_version="assessment-v1",
        status="active",
        is_active=True,
        activated_at=SNAPSHOT_TIME,
    )
    repository.benchmark_peer_submissions = [
        BenchmarkPeerSubmission(
            source_company_id=f"peer-{index}",
            questionnaire_id=f"questionnaire-{index}",
            customer_class="class_3",
            nace1="Electricity and gas",
            release_status="REL",
            scores=[
                BenchmarkScopeScore(scope_type="overall", calculated_score=score),
                BenchmarkScopeScore(
                    scope_type="dimension",
                    dimension="Strategy",
                    calculated_score=score,
                ),
                BenchmarkScopeScore(
                    scope_type="topic",
                    dimension="Strategy",
                    question_id="Q.STR.03.01",
                    calculated_score=score,
                ),
            ],
        )
        for index, score in enumerate(range(40, 40 + peer_count * 10, 10), start=1)
    ]


def test_report_source_freezes_localized_metrics_and_provenance(
    report_repository: _LocalizedReportRepository,
) -> None:
    """Verify the report tree freezes trusted Italian metrics and provenance.

    Inputs:
        report_repository: Profile-aware localized repository with one response.

    Outputs:
        None. The source contains deterministic localized metrics and no peer
        identities or report-generation context.
    """

    _set_active_import(report_repository)

    source = build_assessment_report_source(
        repository=report_repository,
        assessment_id="assessment-1",
        language="it",
        generated_at=SNAPSHOT_TIME,
    )
    payload = source.model_dump(mode="json")

    assert source.schema_version == 4
    assert source.display_name == "assessment"
    assert source.source_company_id == "company-current"
    assert source.customer_class == "class_3"
    assert source.customer_class_label == "Media - Classe 3"
    assert source.nace1 == "Electricity and gas"
    assert source.provenance.import_id == "import-a"
    assert source.provenance.source_filename == "import-a.xlsx"
    assert source.provenance.source_sha256 == "a" * 64
    assert source.provenance.dataset_activated_at == SNAPSHOT_TIME
    assert source.provenance.scoring_version == "assessment-v1"
    assert source.provenance.peer_sample_size == 3
    assert source.overall.peer_average == 50.0
    assert source.overall.best_peer == 60.0
    assert source.dimensions[0].display_name == "Strategy"
    assert source.dimensions[0].topics[0].topic_title == (
        "Definizione degli Obiettivi"
    )
    assert source.dimensions[0].topics[0].answered is True
    serialized = json.dumps(payload)
    assert "peer-1" not in serialized
    assert "questionnaire-1" not in serialized
    assert "selected_answer_texts" not in serialized
    assert "unselected_answer_texts" not in serialized


def test_snapshot_reads_selected_responses_exactly_once(
    report_repository: _LocalizedReportRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one persisted response mapping for deterministic scoring.

    Inputs:
        report_repository: Profile-aware localized report repository.
        monkeypatch: Pytest helper used to count persisted response reads.

    Outputs:
        None. Snapshot construction must call the repository response loader once.
    """

    response_reads = 0
    original_loader = report_repository.get_assessment_responses

    def count_response_read(assessment_id: str) -> dict[str, list[str]]:
        """Count and delegate one persisted response read.

        Inputs:
            assessment_id: Assessment whose selected answer IDs are requested.

        Outputs:
            dict[str, list[str]]: Persisted selected answer IDs by question.
        """

        nonlocal response_reads
        response_reads += 1
        return original_loader(assessment_id)

    monkeypatch.setattr(
        report_repository,
        "get_assessment_responses",
        count_response_read,
    )

    build_assessment_report_source(
        report_repository,
        "assessment-1",
        "en",
        generated_at=SNAPSHOT_TIME,
    )

    assert response_reads == 1


@pytest.mark.parametrize(
    "changes",
    [
        ((("answered_question_count",), 2),),
        ((("provenance", "customer_class"), "class_4"),),
        ((("provenance", "nace1"), "Other cohort"),),
        ((("dimensions", 0, "topics", 0, "dimension"), "Other dimension"),),
        ((("overall", "company_score"), 100.0),),
        (
            (("provenance", "benchmark_available"), False),
            (("provenance", "unavailable_reason"), "insufficient_peer_sample"),
            (("provenance", "peer_sample_size"), 2),
        ),
    ],
    ids=(
        "answered-exceeds-applicable",
        "class-provenance-mismatch",
        "nace-provenance-mismatch",
        "topic-parent-mismatch",
        "positioning-score-mismatch",
        "unavailable-with-peer-values",
    ),
)
def test_report_model_rejects_cross_field_inconsistency(
    changes: tuple[tuple[tuple[str | int, ...], object], ...],
) -> None:
    """Verify structurally valid but contradictory report mappings are rejected.

    Inputs:
        changes: Nested raw-JSON mutations that break one snapshot invariant.

    Outputs:
        None. Parsing must raise rather than permit a misleading PDF.
    """

    payload = make_compact_source().model_dump(mode="json")
    for path, value in changes:
        target: Any = payload
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    with pytest.raises(ValueError):
        AssessmentReportSource.model_validate(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "benchmark_available": False,
            "unavailable_reason": "unknown_reason",
            "peer_sample_size": 2,
        },
        {
            "benchmark_available": False,
            "unavailable_reason": "no_active_dataset",
            "peer_sample_size": 0,
        },
        {
            "benchmark_available": False,
            "unavailable_reason": "insufficient_peer_sample",
            "peer_sample_size": 2,
            "import_id": None,
        },
    ],
    ids=(
        "arbitrary-reason",
        "no-active-dataset-retains-import",
        "insufficient-sample-lacks-import",
    ),
)
def test_report_provenance_rejects_reason_specific_contradictions(
    updates: dict[str, object],
) -> None:
    """Verify unavailable reasons agree with captured import provenance.

    Inputs:
        updates: Raw provenance changes forming one contradictory state.

    Outputs:
        None. The provenance model must reject the supplied state.
    """

    payload = make_compact_source().provenance.model_dump(mode="json")
    payload.update(updates)

    with pytest.raises(ValueError):
        AssessmentReportProvenance.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ("import_id", "source_filename", "source_sha256", "scoring_version"),
)
def test_report_provenance_rejects_blank_required_import_metadata(
    field_name: str,
) -> None:
    """Verify whitespace cannot satisfy captured import provenance.

    Inputs:
        field_name: Required string metadata field replaced with whitespace.

    Outputs:
        None. Provenance parsing must reject the blank captured field.
    """

    payload = make_compact_source().provenance.model_dump(mode="json")
    payload[field_name] = "   "

    with pytest.raises(ValueError):
        AssessmentReportProvenance.model_validate(payload)


def test_snapshot_and_enqueued_job_are_immutable_after_active_import_changes(
    report_repository: _LocalizedReportRepository,
) -> None:
    """Verify a later activation cannot alter a queued report's source tree."""

    _set_active_import(report_repository, import_id="import-a")
    source = build_assessment_report_source(
        report_repository,
        "assessment-1",
        "en",
        generated_at=SNAPSHOT_TIME,
    )
    created = report_repository.create_assessment_report_job(source)
    original_payload = source.model_dump(mode="json")

    _set_active_import(report_repository, import_id="import-b", peer_count=4)
    leased = report_repository.lease_next_assessment_report_job("worker-1")

    assert leased is not None
    assert leased["job_id"] == created.job_id
    assert leased["source"] == original_payload
    assert leased["source"]["provenance"]["import_id"] == "import-a"


@pytest.mark.parametrize(
    ("active", "peer_count", "reason"),
    [(False, 0, "no_active_dataset"), (True, 2, "insufficient_peer_sample")],
)
def test_unavailable_peer_states_still_enqueue_valid_company_only_snapshots(
    report_repository: _LocalizedReportRepository,
    active: bool,
    peer_count: int,
    reason: str,
) -> None:
    """Verify missing and undersized cohorts preserve company-only report data."""

    if active:
        _set_active_import(report_repository, peer_count=peer_count)

    source = build_assessment_report_source(
        report_repository,
        "assessment-1",
        "en",
        generated_at=SNAPSHOT_TIME,
    )

    assert source.provenance.benchmark_available is False
    assert source.provenance.unavailable_reason == reason
    assert source.provenance.peer_sample_size == peer_count
    assert source.overall.peer_average is None
    assert source.overall.best_peer is None
    assert source.overall.company_score == 25.0


def test_report_source_requires_profile_and_never_uses_legacy_fallback() -> None:
    """Verify request-provided class/sector cannot replace persisted context."""

    repository = _LocalizedReportRepository()
    repository.framework_questions = [_framework_question()]

    with pytest.raises(ValueError, match="profile"):
        build_assessment_report_source(repository, "assessment-1", "en")


def test_report_api_accepts_id_language_only_and_ignores_legacy_fields(
    report_api_client: TestClient,
    report_repository: _LocalizedReportRepository,
) -> None:
    """Verify the enqueue route is profile-authoritative and lifecycle-compatible."""

    _set_active_import(report_repository)
    created_response = report_api_client.post(
        "/api/assessment/calification-reports",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "language": "en",
            "customer_class": "class_5",
            "sector": "Legacy browser sector",
        },
    )

    assert created_response.status_code == 202
    created = created_response.json()
    job = report_repository.assessment_report_jobs[created["job_id"]]
    assert job["source"]["customer_class"] == "class_3"
    assert job["source"]["nace1"] == "Electricity and gas"
    assert job["source"]["provenance"]["import_id"] == "import-a"


def test_report_api_enqueues_polls_and_downloads(
    report_api_client: TestClient,
    report_repository: _LocalizedReportRepository,
) -> None:
    """Verify the existing 202, poll, and PDF download contract remains intact."""

    created = report_api_client.post(
        "/api/assessment/calification-reports",
        headers={"X-API-Key": "test-api-key"},
        json={"assessment_id": "assessment-1", "language": "en"},
    ).json()
    status_response = report_api_client.get(
        f"/api/assessment/calification-reports/{created['job_id']}",
        headers={"X-API-Key": "test-api-key"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["download_ready"] is False

    report_repository.lease_next_assessment_report_job("worker-1")
    report_repository.complete_assessment_report_job(
        created["job_id"],
        "worker-1",
        "assessment-report-assessment-1-en.pdf",
        b"%PDF-1.7\nfixture",
    )
    download = report_api_client.get(
        f"/api/assessment/calification-reports/{created['job_id']}/file",
        headers={"X-API-Key": "test-api-key"},
    )

    assert download.status_code == 200
    assert download.content == b"%PDF-1.7\nfixture"
    assert download.headers["content-type"] == "application/pdf"


def test_memory_lifecycle_leases_raw_json_and_preserves_retry_retention() -> None:
    """Verify raw leasing, reclaim count, completion, and 24-hour expiry."""

    repository = InMemoryAiReviewRepository()
    now = SNAPSHOT_TIME
    created = repository.create_assessment_report_job(make_compact_source(), now=now)
    first = repository.lease_next_assessment_report_job("worker-1", now=now)

    assert first is not None
    assert isinstance(first["source"], dict)
    assert first["source"]["schema_version"] == 4
    assert repository.lease_next_assessment_report_job(
        "worker-2", now=now + timedelta(minutes=14)
    ) is None
    reclaimed = repository.lease_next_assessment_report_job(
        "worker-2", now=now + timedelta(minutes=16)
    )
    assert reclaimed is not None
    assert reclaimed["retry_count"] == 1

    repository.complete_assessment_report_job(
        created.job_id,
        "worker-2",
        "assessment-report.pdf",
        b"%PDF-1.7\nfixture",
        now=now + timedelta(minutes=17),
    )
    status = repository.get_assessment_report_job_status(
        created.job_id, now=now + timedelta(minutes=18)
    )
    assert status.expires_at == now + timedelta(hours=24, minutes=17)


def test_memory_enqueue_persists_current_immutable_source_json() -> None:
    """Persist and lease the current source through existing fields.

    Inputs:
        None. A compact deterministic fixture is enqueued in memory.

    Outputs:
        None. Pending lifecycle and source data survive serialization.
    """

    repository = InMemoryAiReviewRepository()
    source = make_compact_source()

    job = repository.create_assessment_report_job(source, now=SNAPSHOT_TIME)
    leased = repository.lease_next_assessment_report_job(
        "worker-1",
        now=SNAPSHOT_TIME,
    )

    assert leased is not None
    assert leased["source"]["schema_version"] == 4
    assert "recommendation_context" not in leased["source"]
    assert job.status == "pending"


def test_completed_legacy_memory_pdf_download_never_parses_source() -> None:
    """Verify old completed BLOB delivery is independent of snapshot schema."""

    repository = InMemoryAiReviewRepository()
    repository.assessment_report_jobs["legacy-completed"] = {
        "job_id": "legacy-completed",
        "assessment_id": "assessment-legacy",
        "customer_class": "class_5",
        "sector": "Energy",
        "language": "en",
        "status": "completed",
        "source": {"malformed": object()},
        "progress_message": "Report ready",
        "lease_owner": None,
        "lease_expires_at": None,
        "retry_count": 0,
        "file_name": "legacy.pdf",
        "pdf_content": b"%PDF-1.7\nlegacy",
        "error_code": None,
        "error_message": None,
        "expires_at": SNAPSHOT_TIME + timedelta(hours=1),
        "created_at": SNAPSHOT_TIME,
        "updated_at": SNAPSHOT_TIME,
    }

    download = repository.get_assessment_report_download(
        "legacy-completed", now=SNAPSHOT_TIME
    )

    assert download == AssessmentReportDownload(
        job_id="legacy-completed",
        file_name="legacy.pdf",
        content=b"%PDF-1.7\nlegacy",
    )


class _FakeSqlResult:
    """Minimal SQLAlchemy result double for HANA report tests."""

    def __init__(self, row: dict[str, Any] | None = None, rowcount: int = 1) -> None:
        """Store one optional mapping row and affected-row count.

        Inputs:
            row: Optional selected mapping.
            rowcount: Affected row count for update/insert calls.

        Outputs:
            None. Test result state is initialized.
        """

        self.row = row
        self.rowcount = rowcount

    def mappings(self) -> "_FakeSqlResult":
        """Return this result for SQLAlchemy mapping-chain compatibility."""

        return self

    def first(self) -> dict[str, Any] | None:
        """Return the configured selected mapping."""

        return self.row


class _RecordingReportSession:
    """Capture report SQL and return configured select results."""

    def __init__(self, select_rows: list[dict[str, Any] | None] | None = None) -> None:
        """Initialize captured calls and queued select rows.

        Inputs:
            select_rows: Ordered mappings returned by select statements.

        Outputs:
            None. Recording state is initialized.
        """

        self.select_rows = list(select_rows or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(
        self,
        statement: object,
        parameters: dict[str, Any] | None = None,
    ) -> _FakeSqlResult:
        """Record SQL and return the next configured select result.

        Inputs:
            statement: SQLAlchemy text statement.
            parameters: Bound SQL parameters.

        Outputs:
            _FakeSqlResult: Select mapping or successful write result.
        """

        sql = str(statement)
        self.calls.append((sql, parameters or {}))
        if sql.lstrip().lower().startswith("select"):
            row = self.select_rows.pop(0) if self.select_rows else None
            return _FakeSqlResult(row=row)
        return _FakeSqlResult(rowcount=1)


class _HanaReportRepository(HanaAssessmentReportsMixin):
    """Expose the HANA report mixin with a recording session."""

    def __init__(self, session: _RecordingReportSession) -> None:
        """Store the recording SQL session.

        Inputs:
            session: Fake SQLAlchemy session.

        Outputs:
            None. Repository is ready for boundary tests.
        """

        self.session = session


class _SchemaAwareHanaReportRepository(_HanaReportRepository):
    """Record cached schema initialization before enqueue."""

    def __init__(self, session: _RecordingReportSession) -> None:
        """Initialize session and schema-call count."""

        super().__init__(session)
        self.schema_calls = 0

    def create_schema(self) -> None:
        """Record one successful schema initialization."""

        self.schema_calls += 1


def test_hana_enqueue_serializes_current_source_without_new_ddl(
    monkeypatch,
) -> None:
    """Verify the current source uses existing JSON and physical metadata.

    Inputs:
        monkeypatch: Pytest helper used to reset the module schema cache.

    Outputs:
        None. HANA enqueue serializes the source into the existing insert without
        introducing DDL or changing the legacy physical sector assignment.
    """

    monkeypatch.setattr(hana_reports, "_REPORT_SCHEMA_READY", False)
    session = _RecordingReportSession()
    repository = _SchemaAwareHanaReportRepository(session)

    repository.create_assessment_report_job(
        make_compact_source(),
        now=SNAPSHOT_TIME,
    )
    repository.create_assessment_report_job(
        make_compact_source(),
        now=SNAPSHOT_TIME,
    )

    assert repository.schema_calls == 1
    enqueue_sql, parameters = session.calls[0]
    payload = json.loads(parameters["source_json"])
    assert payload["schema_version"] == 4
    assert "recommendation_context" not in payload
    assert payload["provenance"]["import_id"] == "import-a"
    assert parameters["sector"] == "Electricity and gas"
    assert "insert into assessment_report_jobs" in enqueue_sql.lower()
    assert all(
        "alter table" not in sql.lower() and "create table" not in sql.lower()
        for sql, _params in session.calls
    )


def test_hana_lease_returns_decoded_raw_json_after_guarded_atomic_update() -> None:
    """Verify version validation is deferred until after the worker owns a lease."""

    raw_source = {"assessment_id": "legacy", "language": "it"}
    session = _RecordingReportSession(
        [
            {
                "job_id": "report-1",
                "source_json": json.dumps(raw_source),
                "status": "pending",
                "retry_count": 0,
            }
        ]
    )
    repository = _HanaReportRepository(session)

    leased = repository.lease_next_assessment_report_job("worker-1")

    assert leased is not None
    assert leased["source"] == raw_source
    update_sql, parameters = session.calls[1]
    assert "lease_expires_at = add_seconds(current_utctimestamp, 900)" in update_sql
    assert "case when status = 'pending' then 0 else 1 end" in update_sql
    assert parameters["worker_id"] == "worker-1"


def test_hana_completed_legacy_blob_download_is_source_agnostic() -> None:
    """Verify download SQL never reads or validates a legacy source snapshot."""

    session = _RecordingReportSession(
        [
            {
                "job_id": "legacy-completed",
                "status": "completed",
                "file_name": "legacy.pdf",
                "pdf_blob": memoryview(b"%PDF-1.7\nlegacy"),
                "expires_at": SNAPSHOT_TIME + timedelta(hours=1),
            }
        ]
    )
    repository = _HanaReportRepository(session)

    download = repository.get_assessment_report_download(
        "legacy-completed", now=SNAPSHOT_TIME
    )

    assert download.content == b"%PDF-1.7\nlegacy"
    select_sql, _parameters = session.calls[0]
    assert "source_json" not in select_sql
    assert "expires_at > current_utctimestamp" in select_sql


def test_report_context_contains_no_runtime_fake_benchmark_seeding() -> None:
    """Verify report snapshots cannot create synthetic runtime peer data."""

    source_text = getsource(build_assessment_report_source)

    assert "ensure_mock_score_benchmarks" not in source_text
    assert "list_score_benchmarks" not in source_text
