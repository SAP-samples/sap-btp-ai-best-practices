"""Upload transport and post-commit scheduling API tests."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.ecn.errors import SourceValidationError, UploadTooLargeError
from app.ecn.services.ingestion import IngestionResult
from app.ecn.settings import EcnSettings
from app.main import create_app
from app.persistence.records import AnalysisProgressResponse, UploadRecord
from app.routers.uploads import _schedule_automatic_analyses, build_bounded_upload_request
from fastapi.testclient import TestClient

REQUEST_ID = UUID("018f54fe-7b2d-7d7d-aef4-81940d349301")
UPLOAD_ID = "018f54fe-7b2d-7d7d-aef4-81940d349302"
JOB_ID = "018f54fe-7b2d-7d7d-aef4-81940d349303"
ECN_ROW_ID = "018f54fe-7b2d-7d7d-aef4-81940d349306"


@pytest.mark.asyncio
async def test_build_bounded_upload_request_validates_and_reads_files(
    async_upload,
) -> None:
    """Convert valid JSON/XLSX uploads to immutable byte records."""
    settings = EcnSettings(max_json_upload_bytes=10, max_workbook_upload_bytes=10)
    request = await build_bounded_upload_request(
        REQUEST_ID,
        async_upload(b"{}", filename="changes.json", content_type="application/json"),
        settings,
    )

    assert request.request_id == REQUEST_ID
    assert request.ecn_file.file_name == "changes.json"
    assert request.ecn_file.content == b"{}"
    assert request.bom_file is None
    assert request.actor_id is None


@pytest.mark.asyncio
async def test_build_bounded_upload_request_rejects_wrong_metadata_before_content(
    async_upload,
) -> None:
    """Reject a non-JSON ECN file with a typed source-validation error."""
    with pytest.raises(SourceValidationError) as raised:
        await build_bounded_upload_request(
            REQUEST_ID,
            async_upload(b"{}", filename="changes.txt", content_type="text/plain"),
            EcnSettings(),
        )
    assert raised.value.code == "UPLOAD_EXTENSION_INVALID"


@pytest.mark.asyncio
async def test_build_bounded_upload_request_enforces_stream_limit(async_upload) -> None:
    """Stop reading once the configured JSON byte bound is crossed."""
    with pytest.raises(UploadTooLargeError):
        await build_bounded_upload_request(
            REQUEST_ID,
            async_upload(
                b"12345", filename="changes.json", content_type="application/json"
            ),
            EcnSettings(max_json_upload_bytes=4),
        )


class IngestionServiceFake:
    """Return a pre-committed result and record the bounded request."""

    def __init__(self, created: bool) -> None:
        """Build one authoritative upload result."""
        self.requests = []
        self.result = IngestionResult(
            UploadRecord(
                upload_id=UPLOAD_ID,
                request_id=str(REQUEST_ID),
                ecn_artifact_id="018f54fe-7b2d-7d7d-aef4-81940d349304",
                bom_artifact_id="018f54fe-7b2d-7d7d-aef4-81940d349305",
                rule_artifact_id=None,
                bom_reused=False,
                imported_ecn_count=1,
                publication_order=1,
                created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
            ),
            created,
            (JOB_ID,) if created else (),
            (ECN_ROW_ID,) if created else (),
        )

    def ingest_upload(self, request):
        """Record the bounded request and return committed state."""
        self.requests.append(request)
        return self.result


class UploadRuntimeFake:
    """Record scheduled committed job batches."""

    def __init__(self) -> None:
        """Initialize an empty scheduling history."""
        self.batches = []

    def schedule_jobs(self, job_ids) -> None:
        """Record one post-commit batch."""
        self.batches.append(job_ids)


class ProjectionRuntimeFake:
    """Record post-commit graph projection batches without touching upload state."""

    def __init__(self) -> None:
        """Initialize an empty graph-work schedule history."""
        self.batches = []

    def project_many(self, ecn_row_ids) -> None:
        """Record only durable IDs emitted by the committed ingestion result."""
        self.batches.append(tuple(ecn_row_ids))


class ManufacturingRuntimeFake:
    """Record exact material batches passed to automatic drawing assessment."""

    def __init__(self) -> None:
        self.calls = []

    def schedule(self, ecn_row_ids, material_numbers) -> None:
        self.calls.append((ecn_row_ids, material_numbers))


class SnapshotServiceFake:
    """Return only the source-backed technical ECN objects needed for fixture selection."""

    def get(self, _ecn_row_id):
        return {
            "objects": [
                {"number": "QKX5525860", "technical": True, "origins": ["ecn"]},
                {"number": "BOM-ONLY", "technical": True, "origins": ["bom"]},
                {"number": None, "technical": True, "origins": ["ecn"]},
            ]
        }


@pytest.mark.asyncio
async def test_automatic_scheduling_starts_summary_and_exact_ecn_material_fixtures() -> None:
    """Post-commit automation excludes BOM-only and unnumbered objects from drawing lookup."""
    summary_runtime = UploadRuntimeFake()
    manufacturing_runtime = ManufacturingRuntimeFake()
    services = SimpleNamespace(
        runtime=summary_runtime,
        manufacturing_feasibility_runtime=manufacturing_runtime,
        snapshot_service=SnapshotServiceFake(),
    )

    await _schedule_automatic_analyses(services, (ECN_ROW_ID,), (JOB_ID,))

    assert summary_runtime.batches == [(JOB_ID,)]
    assert manufacturing_runtime.calls == [
        ((ECN_ROW_ID,), {ECN_ROW_ID: ("QKX5525860",)})
    ]


def _upload_client(monkeypatch, created: bool):
    """Return a test client and fakes for one upload result kind."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    ingestion = IngestionServiceFake(created)
    runtime = UploadRuntimeFake()
    projection = ProjectionRuntimeFake()
    progress = AnalysisProgressResponse(
        queued_count=1 if created else 0,
        running_count=0,
        completed_count=0 if created else 1,
        completed_with_gaps_count=0,
        failed_count=0,
        unfinished_count=1 if created else 0,
        total_count=1,
    )
    analysis_repository = SimpleNamespace(get_progress=lambda _upload_id: progress)
    services = SimpleNamespace(
        settings=EcnSettings(),
        ingestion_service=ingestion,
        analysis_repository=analysis_repository,
        runtime=runtime,
        rdf_projection_service=projection,
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield upload test services."""
        yield services

    return TestClient(create_app(service_lifespan_factory=lifespan)), ingestion, runtime, projection


@pytest.mark.parametrize("created", [True, False])
def test_upload_returns_committed_representation_and_automatically_starts_summary_work(
    monkeypatch, created: bool
) -> None:
    """Return 202 after ingestion and schedule only newly committed summary jobs."""
    client, ingestion, runtime, projection = _upload_client(monkeypatch, created)
    with client:
        response = client.post(
            "/api/uploads",
            headers={"X-API-Key": "test-api-key"},
            data={"request_id": str(REQUEST_ID)},
            files={
                "ecn_file": ("changes.json", b"{}", "application/json"),
            },
        )
    assert response.status_code == 202
    assert response.json()["created"] is created
    assert response.json()["upload_id"] == UPLOAD_ID
    assert len(ingestion.requests) == 1
    assert runtime.batches == ([(JOB_ID,)] if created else [])
    assert projection.batches == ([(ECN_ROW_ID,)] if created else [])


def test_upload_contract_rejects_legacy_bom_file(monkeypatch) -> None:
    """Keep ECN intake JSON-only after reusable sources move to Settings."""
    client, ingestion, runtime, projection = _upload_client(monkeypatch, True)
    with client:
        response = client.post(
            "/api/uploads",
            headers={"X-API-Key": "test-api-key"},
            data={"request_id": str(REQUEST_ID)},
            files={
                "ecn_file": ("changes.json", b"{}", "application/json"),
                "bom_file": ("bom.xlsx", b"xlsx", "application/octet-stream"),
            },
        )
    assert response.status_code == 400
    assert ingestion.requests == []
    assert runtime.batches == []
    assert projection.batches == []
