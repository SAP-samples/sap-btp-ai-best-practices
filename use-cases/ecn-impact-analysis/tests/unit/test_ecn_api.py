"""Authenticated ECN list, detail, and evidence route tests."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.ecn.api_models import EcnListResponse
from app.ecn.errors import PersistenceStateError
from app.ecn.rdf import ProjectionExpectedCounts, ProjectionRecord, ProjectionState
from app.main import create_app
from fastapi.testclient import TestClient
import pytest

ECN_ID = "018f54fe-7b2d-7d7d-aef4-81940d349401"


class RecordingEcnRepository:
    """Return empty reads while recording bounded query and evidence identities."""

    def __init__(self) -> None:
        """Initialize empty call histories."""
        self.queries = []
        self.evidence_calls: list[tuple[str, str]] = []
        self.bundle_calls: list[str] = []

    def list_ecns(self, query):
        """Return an empty page preserving the validated pagination values."""
        self.queries.append(query)
        return EcnListResponse(
            items=[],
            page=query.page,
            page_size=query.page_size,
            total_items=0,
            total_pages=0,
        )

    def get_ecn_detail(self, ecn_row_id: str):
        """Return no matching detail."""
        return

    def get_evidence(self, ecn_row_id: str, evidence_id: str):
        """Record a deterministic string evidence ID and return no match."""
        self.evidence_calls.append((ecn_row_id, evidence_id))

    def get_evidence_bundle(self, ecn_row_id: str):
        """Record the scoped bundle request and return one empty stored bundle."""
        from tests.fixtures.builders import make_evidence_bundle

        self.bundle_calls.append(ecn_row_id)
        return make_evidence_bundle(ecn_row_id=ecn_row_id)


class RecordingProjectionService:
    """Expose one bounded status and record trusted retry scheduling."""

    def __init__(self) -> None:
        """Initialize one failed record suitable for retry diagnostics."""
        self.record = ProjectionRecord(
            ecn_row_id=ECN_ID,
            source_sha256="a" * 64,
            projection_version="1.0.0",
            status=ProjectionState.FAILED,
            expected_counts=ProjectionExpectedCounts(1, 0, 1, 0, 1),
            verified_node_count=None,
            verified_triple_count=None,
            error_code="RDF_PROJECTION_FAILED",
            error_message="The RDF projection could not be rebuilt and verified.",
            last_attempt_at=None,
            current_at=None,
        )
        self.rebuilds: list[str] = []
        self.projects: list[str] = []

    def get_projection(self, ecn_row_id: str) -> ProjectionRecord | None:
        """Return the sole selected status and no unrelated ECN record."""
        return self.record if ecn_row_id == ECN_ID else None

    def request_rebuild(self, ecn_row_id: str) -> ProjectionRecord:
        """Record trusted retry input and move the fixture state to pending."""
        self.rebuilds.append(ecn_row_id)
        self.record = ProjectionRecord(
            **{**self.record.__dict__, "status": ProjectionState.PENDING}
        )
        return self.record

    def project(self, ecn_row_id: str) -> ProjectionRecord:
        """Record post-response graph work without accepting client graph inputs."""
        self.projects.append(ecn_row_id)
        return self.record

    def require_current(self, ecn_row_id: str) -> ProjectionRecord:
        """Reject graph-dependent work while the fixture remains non-current."""
        assert ecn_row_id == ECN_ID
        raise PersistenceStateError(
            "RDF_PROJECTION_FAILED",
            "The ECN RDF projection is not ready for graph-dependent work.",
            status_code=409,
        )


class RecordingAssistantService:
    """Record chat attempts that must not run while RDF remains non-current."""

    def __init__(self) -> None:
        """Initialize no invoked chat operations."""
        self.calls: list[tuple[object, ...]] = []

    async def ainvoke(self, *args: object, **_kwargs: object) -> dict[str, object]:
        """Record an unexpected chat invocation and return a harmless payload."""
        self.calls.append(args)
        return {"answer": "unexpected"}


def _client(monkeypatch):
    """Build a same-loop app with a recording repository and no production factories."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    repository = RecordingEcnRepository()
    services = SimpleNamespace(
        ecn_repository=repository,
        rdf_projection_service=RecordingProjectionService(),
        assistant_service=RecordingAssistantService(),
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield the test-only service namespace."""
        yield services

    app = create_app(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("engine used")),
        provider_factory=lambda _settings: (_ for _ in ()).throw(
            AssertionError("provider used")
        ),
        service_lifespan_factory=lifespan,
    )
    client = TestClient(app)
    client.services = services  # Lets tests read what the fake services recorded.
    return client, repository


def test_detail_returns_verified_outgoing_references(monkeypatch):
    """Return source relationships in the existing authenticated detail read."""
    from app.ecn.api_models import LinkedEcnReference
    from tests.unit.test_ecn_repository import ecn_detail_record

    client, repository = _client(monkeypatch)
    record = ecn_detail_record()
    repository.get_ecn_detail = lambda _row_id: record
    calls = []

    def resolve(selected):
        calls.append(selected)
        return [
            LinkedEcnReference(ecn_number="ECN12345", ecn_row_id=ECN_ID),
            LinkedEcnReference(ecn_number="ECN888"),
        ]

    repository.get_linked_ecns = resolve
    with client:
        response = client.get(f"/api/ecns/{record.ecn.ecn_row_id}", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    assert response.json()["linked_ecns"] == [
        {"ecn_number": "ECN12345", "ecn_row_id": ECN_ID},
        {"ecn_number": "ECN888", "ecn_row_id": None},
    ]
    assert calls == [record]


def test_list_requires_api_key_and_returns_exact_validated_page(monkeypatch) -> None:
    """Protect reads and pass the allowlisted query model to the repository."""
    client, repository = _client(monkeypatch)
    with client:
        denied = client.get("/api/ecns")
        response = client.get(
            "/api/ecns?page=2&page_size=10&sort=ecn_number&order=asc&has_warnings=true",
            headers={"X-API-Key": "test-api-key"},
        )

    assert denied.status_code == 403
    assert set(denied.json()) == {"code", "message", "details"}
    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "page": 2,
        "page_size": 10,
        "total_items": 0,
        "total_pages": 0,
    }
    assert repository.queries[0].sort == "ecn_number"
    assert repository.queries[0].has_warnings is True


@pytest.mark.parametrize("group", ["SUMMARIES_AVAILABLE", "IN_PROGRESS"])
def test_list_accepts_inbox_analysis_groups(monkeypatch, group) -> None:
    """Grouped filters reach the repository without weakening row-state responses."""
    client, repository = _client(monkeypatch)
    with client:
        response = client.get(
            "/api/ecns",
            params={"analysis_state": group, "query": "shaft", "page": 2},
            headers={"X-API-Key": "test-api-key"},
        )
    assert response.status_code == 200
    assert repository.queries[0].analysis_state.value == group
    assert repository.queries[0].query == "shaft"
    assert repository.queries[0].page == 2


def test_evidence_route_accepts_real_deterministic_non_uuid_ids(monkeypatch) -> None:
    """Pass `E-...` evidence identifiers instead of rejecting them as UUIDs."""
    client, repository = _client(monkeypatch)
    with client:
        response = client.get(
            f"/api/ecns/{ECN_ID}/evidence/E-0123456789abcdef",
            headers={"X-API-Key": "test-api-key"},
        )

    assert response.status_code == 404
    assert repository.evidence_calls == [(ECN_ID, "E-0123456789abcdef")]


def test_grouped_evidence_route_returns_one_bounded_business_section(monkeypatch) -> None:
    """Return a stable empty page without hydrating evidence in ECN detail."""
    client, repository = _client(monkeypatch)
    with client:
        response = client.get(
            f"/api/ecns/{ECN_ID}/evidence?section=data_quality&page=2&page_size=10",
            headers={"X-API-Key": "test-api-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "section": "data_quality",
        "items": [],
        "page": 2,
        "page_size": 10,
        "total_items": 0,
        "total_pages": 0,
    }
    assert repository.bundle_calls == [ECN_ID]


def test_unknown_sort_and_invalid_row_uuid_use_safe_422_envelope(monkeypatch) -> None:
    """Normalize framework validation failures without echoing submitted values."""
    client, _repository = _client(monkeypatch)
    headers = {"X-API-Key": "test-api-key"}
    with client:
        bad_sort = client.get("/api/ecns?sort=drop-table", headers=headers)
        bad_uuid = client.get("/api/ecns/not-a-uuid", headers=headers)

    for response in (bad_sort, bad_uuid):
        assert response.status_code == 422
        assert set(response.json()) == {"code", "message", "details"}
        assert "drop-table" not in response.text
        assert "not-a-uuid" not in response.text


def test_rdf_projection_status_and_retry_are_scoped_to_the_route_ecn(monkeypatch) -> None:
    """Expose safe status and schedule retry from the selected route identity alone."""
    client, _repository = _client(monkeypatch)
    headers = {"X-API-Key": "test-api-key"}
    with client:
        status = client.get(f"/api/ecns/{ECN_ID}/rdf-projection", headers=headers)
        retry = client.post(f"/api/ecns/{ECN_ID}/rdf-projection/retry", headers=headers)

    assert status.status_code == 200
    assert status.json()["status"] == "FAILED"
    assert status.json()["error_code"] == "RDF_PROJECTION_FAILED"
    assert retry.status_code == 202
    assert retry.json()["status"] == "PENDING"


def test_chat_is_gated_when_the_selected_rdf_projection_is_non_current(monkeypatch) -> None:
    """Do not invoke the assistant or silently use an incomplete graph projection."""
    client, _repository = _client(monkeypatch)
    with client:
        response = client.post(
            f"/api/ecns/{ECN_ID}/chat",
            headers={"X-API-Key": "test-api-key"},
            json={"assessment_id": ECN_ID, "message": "Who requested this?"},
        )

    assert response.status_code == 409
    assert response.json()["code"] == "RDF_PROJECTION_FAILED"


def test_global_chat_calls_the_assistant_without_an_ecn_and_validates_input(monkeypatch) -> None:
    """The global chat needs no open ECN, skips the per-ECN RDF gate, and rejects empty input."""
    client, _repository = _client(monkeypatch)
    with client:
        response = client.post(
            "/api/chat",
            headers={"X-API-Key": "test-api-key"},
            json={"message": "Which ECNs have high impact?", "history": [{"role": "user", "content": "hi"}]},
        )
        empty = client.post("/api/chat", headers={"X-API-Key": "test-api-key"}, json={"message": "  "})
        foreign = client.post(
            "/api/chat",
            headers={"X-API-Key": "test-api-key"},
            json={"message": "Hi", "assessment_id": ECN_ID},
        )

    assert response.status_code == 200
    assert empty.status_code == 422
    assert foreign.status_code == 422
    assert client.services.assistant_service.calls == [
        ("Which ECNs have high impact?", None, [{"role": "user", "content": "hi"}])
    ]
