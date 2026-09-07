"""Public API tests for protected versioned assessment benchmark imports."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.benchmarking import BenchmarkImportInfo
from app.routers import imports as import_routes
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository
from app.services.benchmark_import.constants import MAX_SOURCE_BYTES
from app.services.benchmark_import.models import (
    BenchmarkRowError,
    BenchmarkValidationError,
    BenchmarkValidationSummary,
    BenchmarkWarning,
)
from app.services.framework_importer import load_framework_seed


class _BenchmarkImportRouteRepository(InMemoryAiReviewRepository):
    """Supply canonical framework rows and safe import history to API tests."""

    def __init__(self, repo_root: Path) -> None:
        """Load test-only canonical framework data and initialize history state."""

        super().__init__()
        seed = load_framework_seed(
            repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
            repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
        )
        self.framework_questions = seed.questions
        self.session = object()
        self.active_benchmark_import = BenchmarkImportInfo(
            import_id="import-active",
            source_filename="active.xlsx",
            source_sha256="d" * 64,
            scoring_version="assessment-v1",
            row_count=120,
            company_count=4,
            questionnaire_count=4,
            question_count=50,
            accepted_count=120,
            rejected_count=0,
            status="active",
            is_active=True,
        )
        self.benchmark_import_history = [
            self.active_benchmark_import,
            self.active_benchmark_import.model_copy(
                update={
                    "import_id": "import-old",
                    "source_filename": "old.xlsx",
                    "source_sha256": "e" * 64,
                    "status": "inactive",
                    "is_active": False,
                }
            ),
        ]
        self.requested_history_limits: list[int] = []

    def list_recent_benchmark_imports(self, limit: int) -> list[BenchmarkImportInfo]:
        """Return bounded safe metadata and record the requested route limit."""

        self.requested_history_limits.append(limit)
        return self.benchmark_import_history[:limit]


@pytest.fixture
def benchmark_import_repository(
    repo_root: Path,
) -> _BenchmarkImportRouteRepository:
    """Return a canonical-framework fake repository for import API calls."""

    return _BenchmarkImportRouteRepository(repo_root)


@pytest.fixture(autouse=True)
def override_benchmark_import_repository(
    benchmark_import_repository: _BenchmarkImportRouteRepository,
) -> Iterator[None]:
    """Install and remove the import API's repository dependency override."""

    app.dependency_overrides[get_ai_review_repository] = (
        lambda: benchmark_import_repository
    )
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_ai_review_repository, None)


def _upload(content: bytes, filename: str = "benchmark.xlsx") -> dict[str, tuple[str, bytes, str]]:
    """Return one multipart workbook upload mapping for the test client."""

    return {"workbook": (filename, content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}


def _unsafe_zip() -> bytes:
    """Return a ZIP container with a traversal path rejected by XLSX safety."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../unsafe.xml", "unsafe")
    return buffer.getvalue()


def test_benchmark_import_dry_run_parses_upload_against_repository_framework(
    api_client: TestClient,
    repo_root: Path,
) -> None:
    """Verify dry-run uses canonical repository questions and performs no write."""

    fixture = (
        repo_root
        / "api"
        / "tests"
        / "fixtures"
        / "benchmark"
        / "DB_extraction_clustering_NACE_synthetic.xlsx"
    )
    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        files=_upload(fixture.read_bytes(), fixture.name),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["status"] == "validated"
    assert payload["write_completed"] is False
    assert payload["company_count"] == 24
    assert payload["questionnaire_count"] == 24


def test_benchmark_import_write_reuses_parser_service_and_repository_session(
    api_client: TestClient,
    benchmark_import_repository: _BenchmarkImportRouteRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit write passes upload bytes, HANA framework, and session."""

    calls: list[dict[str, Any]] = []

    def fake_import(
        content: bytes,
        filename: str,
        framework_questions: list[object],
        *,
        write: bool,
        session: object | None,
    ) -> BenchmarkValidationSummary:
        """Record service arguments and return a safe imported summary."""

        calls.append(
            {
                "content": content,
                "filename": filename,
                "question_count": len(framework_questions),
                "write": write,
                "session": session,
            }
        )
        return BenchmarkValidationSummary(
            import_id="import-new",
            source_filename=filename,
            source_sha256="f" * 64,
            scoring_version="assessment-v1",
            success=True,
            status="active",
            write_completed=True,
        )

    monkeypatch.setattr(import_routes, "import_benchmark_workbook", fake_import)
    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        data={"write": "true"},
        files=_upload(b"uploaded-workbook"),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert calls == [
        {
            "content": b"uploaded-workbook",
            "filename": "benchmark.xlsx",
            "question_count": 50,
            "write": True,
            "session": benchmark_import_repository.session,
        }
    ]


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"not-an-xlsx", "malformed_xlsx"),
        (_unsafe_zip(), "unsafe_zip"),
        (b"x" * (MAX_SOURCE_BYTES + 1), "source_too_large"),
    ],
    ids=["malformed", "unsafe", "oversized"],
)
def test_benchmark_import_rejects_malformed_unsafe_and_oversized_uploads(
    api_client: TestClient,
    content: bytes,
    expected_code: str,
) -> None:
    """Verify untrusted workbook failures are structured HTTP 400 responses."""

    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        files=_upload(content),
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["validation"]["sampled_errors"][0]["code"] == expected_code


def test_benchmark_import_write_error_is_sanitized(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify database exceptions return safe HTTP 500 text without internals."""

    def fail_write(*_args: object, **_kwargs: object) -> None:
        """Raise a secret-bearing fake database error."""

        raise RuntimeError("password=secret internal SQL")

    monkeypatch.setattr(import_routes, "import_benchmark_workbook", fail_write)
    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        data={"write": "true"},
        files=_upload(b"uploaded-workbook"),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Assessment benchmark import failed while writing to HANA."
    )
    assert "secret" not in str(response.json()).lower()


def test_benchmark_import_success_removes_peer_ids_from_public_warnings(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify successful validation exposes warning metadata without raw peers."""

    sentinel_ids = (
        "SENTINEL-COMPANY-99",
        "SENTINEL-QUESTIONNAIRE-99",
        "SENTINEL-ANSWER-99",
    )
    internal_summaries: list[BenchmarkValidationSummary] = []

    def warning_result(*_args: object, **_kwargs: object) -> BenchmarkValidationSummary:
        """Return an internal warning summary containing sentinel peer IDs."""

        summary = BenchmarkValidationSummary(
            source_filename="benchmark.xlsx",
            source_sha256="a" * 64,
            scoring_version="assessment-v1",
            row_count=1,
            company_count=1,
            questionnaire_count=1,
            question_count=1,
            accepted_count=1,
            warnings=[
                BenchmarkWarning(
                    code="answer_text_mismatch",
                    message=f"Internal warning for {sentinel_ids[0]}",
                    count=1,
                    samples=[f"row 73: {sentinel_ids[1]} {sentinel_ids[2]}"],
                )
            ],
            success=True,
            status="validated",
        )
        internal_summaries.append(summary)
        return summary

    monkeypatch.setattr(import_routes, "import_benchmark_workbook", warning_result)
    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        files=_upload(b"uploaded-workbook"),
    )

    assert response.status_code == 200
    warning = response.json()["warnings"][0]
    assert warning["code"] == "answer_text_mismatch"
    assert warning["count"] == 1
    assert warning["row_numbers"] == [73]
    assert "samples" not in warning
    serialized = str(response.json())
    assert all(sentinel not in serialized for sentinel in sentinel_ids)
    assert all(
        sentinel in str(internal_summaries[0].model_dump())
        for sentinel in sentinel_ids
    )


def test_benchmark_import_failure_removes_peer_ids_from_public_summary(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTTP 400 summaries retain useful fields but never raw peer IDs."""

    sentinel_ids = (
        "SENTINEL-COMPANY-88",
        "SENTINEL-QUESTIONNAIRE-88",
        "SENTINEL-ANSWER-88",
    )
    internal_summaries: list[BenchmarkValidationSummary] = []

    def rejected_result(*_args: object, **_kwargs: object) -> None:
        """Raise an internal validation summary containing sentinel peer IDs."""

        summary = BenchmarkValidationSummary(
            source_filename="benchmark.xlsx",
            source_sha256="b" * 64,
            scoring_version="assessment-v1",
            row_count=1,
            company_count=1,
            questionnaire_count=1,
            question_count=1,
            rejected_count=1,
            warnings=[
                BenchmarkWarning(
                    code="profile_placeholder_normalized",
                    message=f"Internal warning for {sentinel_ids[0]}",
                    count=1,
                    samples=[f"company {sentinel_ids[0]}: Fatturato"],
                )
            ],
            sampled_errors=[
                BenchmarkRowError(
                    row_number=88,
                    code="conflicting_row",
                    message=(
                        f"Response {sentinel_ids[1]} {sentinel_ids[2]} "
                        "appears more than once"
                    ),
                )
            ],
            success=False,
            status="rejected",
        )
        internal_summaries.append(summary)
        raise BenchmarkValidationError(summary)

    monkeypatch.setattr(import_routes, "import_benchmark_workbook", rejected_result)
    response = api_client.post(
        "/api/import/assessment-benchmarks",
        headers={"X-API-Key": "test-api-key"},
        files=_upload(b"uploaded-workbook"),
    )

    assert response.status_code == 400
    validation = response.json()["detail"]["validation"]
    assert validation["sampled_errors"][0]["row_number"] == 88
    assert validation["sampled_errors"][0]["code"] == "conflicting_row"
    assert validation["warnings"][0]["code"] == "profile_placeholder_normalized"
    serialized = str(response.json())
    assert all(sentinel not in serialized for sentinel in sentinel_ids)
    assert all(
        sentinel in str(internal_summaries[0].model_dump())
        for sentinel in sentinel_ids
    )


def test_benchmark_import_history_is_bounded_safe_and_identity_free(
    api_client: TestClient,
    benchmark_import_repository: _BenchmarkImportRouteRepository,
) -> None:
    """Verify GET returns active/recent safe summaries without BLOBs or peers."""

    response = api_client.get(
        "/api/import/assessment-benchmarks?limit=1",
        headers={"X-API-Key": "test-api-key"},
    )
    unbounded = api_client.get(
        "/api/import/assessment-benchmarks?limit=51",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["active"]["import_id"] == "import-active"
    assert [item["import_id"] for item in response.json()["recent"]] == [
        "import-active"
    ]
    assert benchmark_import_repository.requested_history_limits == [1]
    assert unbounded.status_code == 422
    serialized = str(response.json()).lower()
    assert "source_workbook" not in serialized
    assert "source_company_id" not in serialized
    assert "questionnaire_id" not in serialized
