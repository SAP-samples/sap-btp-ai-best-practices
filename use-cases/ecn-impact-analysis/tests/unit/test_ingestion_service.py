"""Unit tests for deterministic, atomic ECN upload orchestration.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_ingestion_service.py
"""

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from itertools import count
from threading import Event, Thread
from uuid import UUID

import pytest
from app.ecn.errors import PersistenceStateError
from app.ecn.models import ArtifactKind, ParsedEcnJson, ParseState, SourceArtifact
from app.ecn.reference_data.models import ReferenceDatasetRecord, ReferenceDatasetType
from app.ecn.services import ingestion as ingestion_module
from app.ecn.services.ingestion import (
    IngestionService,
    UploadBytesRequest,
    UploadFileBytes,
)
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.settings import EcnSettings
from app.persistence.records import UploadPublishResult, UploadRecord

from tests.fixtures.builders import make_bom_row, make_ecn, synthetic_uuid

NOW = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
DEFAULT_REQUEST_ID = UUID("00000000-0000-0000-0000-000000000001")


class _UploadRepository:
    """In-memory upload boundary that records only aggregate publications."""

    def __init__(self, latest_bom: SourceArtifact | None = None) -> None:
        """Initialize optional snapshot and empty idempotency indexes."""
        self.latest_bom = latest_bom
        self.by_request: dict[str, UploadRecord] = {}
        self.publications: list[object] = []

    def find_by_request_id(self, request_id: str) -> UploadRecord | None:
        """Return an already committed upload by idempotency key."""
        return self.by_request.get(request_id)

    def get_latest_bom_snapshot(self) -> SourceArtifact | None:
        """Return the latest valid immutable BOM artifact."""
        return self.latest_bom

    def publish_upload(self, publication: object) -> UploadPublishResult:
        """Record one atomic publication and return the Phase 01 envelope."""
        self.publications.append(publication)
        record = UploadRecord(
            upload_id=publication.upload_id,
            request_id=publication.request_id,
            ecn_artifact_id=publication.ecn_artifact.artifact_id,
            bom_artifact_id=publication.bom_artifact.artifact_id,
            rule_artifact_id=(
                publication.rule_artifact.artifact_id
                if publication.rule_artifact
                else None
            ),
            bom_reused=publication.bom_reused,
            imported_ecn_count=len(publication.ecns),
            publication_order=len(self.publications),
            created_at=publication.created_at,
        )
        self.by_request[publication.request_id] = record
        return UploadPublishResult(
            upload=record,
            created=True,
            projection_ecn_ids=tuple(item.ecn.ecn_row_id for item in publication.ecns),
        )


class _ArtifactRepository:
    """Supply current rules and persisted normalized BOM row batches."""

    def __init__(
        self,
        rows: tuple[object, ...] = (),
        rule_artifact: SourceArtifact | None = None,
        artifacts: tuple[SourceArtifact, ...] = (),
        load_started: Event | None = None,
        release_load: Event | None = None,
    ) -> None:
        """Initialize rows, artifacts, optional rules, and load barriers."""
        self.rows = rows
        self.rule_artifact = rule_artifact
        self.artifacts = {item.artifact_id: item for item in artifacts}
        self.load_started = load_started
        self.release_load = release_load

    def artifact_exists(self, artifact_id: str) -> bool:
        """Return whether an artifact remains in simulated persistence."""
        return artifact_id in self.artifacts

    def get_artifact(self, artifact_id: str) -> SourceArtifact | None:
        """Return one simulated immutable artifact by identity."""
        return self.artifacts.get(artifact_id)

    def get_current_rule_artifact(self) -> SourceArtifact | None:
        """Return optional current rule configuration."""
        return self.rule_artifact

    def iter_bom_rows(
        self, artifact_id: str, batch_size: int = 5_000
    ) -> object:
        """Yield stored rows in bounded batches."""
        assert artifact_id
        assert batch_size == 5_000
        if self.load_started is not None:
            self.load_started.set()
        if self.release_load is not None:
            assert self.release_load.wait(timeout=2)
        yield list(self.rows)


def _artifact(kind: ArtifactKind, artifact_id: str) -> SourceArtifact:
    """Return one deterministic source artifact for repository fixtures."""
    return SourceArtifact(
        artifact_id=artifact_id,
        artifact_kind=kind,
        file_name="source.xlsx",
        media_type="application/octet-stream",
        byte_size=1,
        sha256=hashlib.sha256(b"x").hexdigest(),
        content_bytes=b"x",
        normalized_content="{}",
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=NOW,
    )


def _request(
    request_id: UUID = DEFAULT_REQUEST_ID,
    with_bom: bool = True,
) -> UploadBytesRequest:
    """Return one deterministic upload request."""
    return UploadBytesRequest(
        request_id=request_id,
        ecn_file=UploadFileBytes("ecns.json", "application/json", b"[]"),
        bom_file=(
            UploadFileBytes("bom.xlsx", "application/octet-stream", b"xlsx")
            if with_bom
            else None
        ),
        actor_id="hashed-actor",
    )


def _one_row_workbook(
    content: bytes,
    artifact_id: str,
    maximum_rows: int,
    maximum_cells: int,
) -> object:
    """Return one within-limit normalized workbook for orchestration tests."""
    from app.ecn.models import BomKind, ParsedBomWorkbook

    assert content
    assert maximum_rows >= 1
    assert maximum_cells >= 12
    return ParsedBomWorkbook(
        rows=(make_bom_row(artifact_id=artifact_id),),
        sheet_roles={
            BomKind.ENRICHED_EBOM: "Enriched",
            BomKind.RAW_EBOM_130: "Raw",
            BomKind.MBOM_130: "MBOM",
        },
        row_count=1,
        cell_count=12,
        quality_qualifiers=(),
    )


def _service(
    upload_repository: _UploadRepository,
    artifact_repository: _ArtifactRepository,
    mutation_guard: HistoryMutationGuard | None = None,
    reference_repository: object | None = None,
    plant_settings_repository: object | None = None,
) -> IngestionService:
    """Build the service with deterministic IDs and clock."""
    numbers = count(1)
    return IngestionService(
        upload_repository=upload_repository,
        artifact_repository=artifact_repository,
        settings=EcnSettings(),
        preparation_gate=IngestionPreparationGate(),
        mutation_guard=mutation_guard or HistoryMutationGuard(),
        clock=lambda: NOW,
        uuid_factory=lambda: f"00000000-0000-0000-0000-{next(numbers):012d}",
        reference_repository=reference_repository,  # type: ignore[arg-type]
        plant_settings_repository=plant_settings_repository,  # type: ignore[arg-type]
    )


class _ReferenceRepository:
    """Return a fixed list of active Settings publications."""

    def __init__(self, records: tuple[ReferenceDatasetRecord, ...]) -> None:
        """Store immutable records in repository list order."""
        self.records = records

    def list(self) -> tuple[ReferenceDatasetRecord, ...]:
        """Return fixed reference metadata without source BLOBs."""
        return self.records


def _reference_record(
    dataset_type: ReferenceDatasetType,
    artifact_id: str,
    dataset_id: str,
) -> ReferenceDatasetRecord:
    """Build one active reference record for ingestion tests."""
    return ReferenceDatasetRecord(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        artifact_id=artifact_id,
        publication_order=1,
        file_name="source.xlsx",
        sha256="a" * 64,
        scope_code=None,
        scope_name=None,
        row_count=1,
        warning_counts={},
        parser_version="ecn-reference-v1",
        created_at=NOW,
        active=True,
    )


def test_first_upload_requires_bom_before_any_publication() -> None:
    """Reject JSON-only intake when no current snapshot exists."""
    uploads = _UploadRepository(latest_bom=None)
    service = _service(uploads, _ArtifactRepository())
    with pytest.raises(Exception) as caught:
        service.ingest_upload(_request(with_bom=False))
    assert getattr(caught.value, "code", None) == "BOM_SNAPSHOT_REQUIRED"
    assert uploads.publications == []


def test_production_json_upload_requires_active_settings_bom() -> None:
    """Return the explicit Settings prerequisite error before publication."""
    uploads = _UploadRepository()
    service = _service(
        uploads,
        _ArtifactRepository(),
        reference_repository=_ReferenceRepository(()),
    )
    with pytest.raises(Exception) as caught:
        service.ingest_upload(_request(with_bom=False))
    assert getattr(caught.value, "code", None) == "NO_ACTIVE_BOM"
    assert uploads.publications == []


def test_production_json_upload_pins_all_active_settings_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse active BOM rows and pin immutable provenance for every source type."""
    bom_id = synthetic_uuid("settings-bom")
    bom_dataset_id = synthetic_uuid("settings-bom-dataset")
    material_dataset_id = synthetic_uuid("settings-material-dataset")
    bom = _artifact(ArtifactKind.BOM_WORKBOOK, bom_id)
    references = _ReferenceRepository(
        (
            _reference_record(ReferenceDatasetType.BOM, bom_id, bom_dataset_id),
            _reference_record(
                ReferenceDatasetType.MATERIAL_MASTER,
                synthetic_uuid("settings-material"),
                material_dataset_id,
            ),
        )
    )
    artifacts = _ArtifactRepository(
        rows=(make_bom_row(artifact_id=bom_id),), artifacts=(bom,)
    )
    uploads = _UploadRepository()
    service = _service(uploads, artifacts, reference_repository=references)
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    service.ingest_upload(_request(with_bom=False))
    publication = uploads.publications[0]
    assert publication.bom_artifact is bom
    assert {(link.dataset_type, link.dataset_id) for link in publication.dataset_links} == {
        (ReferenceDatasetType.BOM, bom_dataset_id),
        (ReferenceDatasetType.MATERIAL_MASTER, material_dataset_id),
    }


def test_upload_freezes_the_current_plant_code_mapping_into_every_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapping and excluded rule states read at import are frozen in the evidence context."""
    from app.ecn.plant_settings import PlantMapping, PlantSettings

    class PlantSettingsRepository:
        """Return fixed Settings values as HANA would."""

        def get(self):
            """Return one mapping row and one excluded state."""
            return PlantSettings((PlantMapping("JB", "PL10", "Plant A"),), ("To be Obsolete",))

    bom_id = synthetic_uuid("settings-bom")
    bom = _artifact(ArtifactKind.BOM_WORKBOOK, bom_id)
    references = _ReferenceRepository((_reference_record(ReferenceDatasetType.BOM, bom_id, synthetic_uuid("bom-dataset")),))
    uploads = _UploadRepository()
    service = _service(uploads, _ArtifactRepository(rows=(make_bom_row(artifact_id=bom_id),), artifacts=(bom,)),
                       reference_repository=references, plant_settings_repository=PlantSettingsRepository())
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"),
    )
    service.ingest_upload(_request(with_bom=False))
    [ecn] = uploads.publications[0].ecns
    assert ecn.evidence.context.plant_assignment == {
        "mappings": [{"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}],
        "excluded_rule_states": ["To be Obsolete"],
    }


def test_reused_snapshot_references_old_artifact_without_copying_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use normalized HANA rows and pin the new ECNs to the reused artifact."""
    bom_id = synthetic_uuid("bom-existing")
    stored_bom = _artifact(ArtifactKind.BOM_WORKBOOK, bom_id)
    row = make_bom_row(artifact_id=bom_id)
    uploads = _UploadRepository(latest_bom=stored_bom)
    service = _service(
        uploads,
        _ArtifactRepository(rows=(row,), artifacts=(stored_bom,)),
    )
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    result = service.ingest_upload(_request(with_bom=False))
    publication = uploads.publications[0]
    assert result.created is True
    assert result.projection_ecn_ids == tuple(
        item.ecn.ecn_row_id for item in publication.ecns
    )
    assert publication.bom_artifact is stored_bom
    assert publication.bom_reused is True
    assert publication.new_bom_rows == ()
    assert publication.ecns[0].evidence.context.bom_artifact_id == bom_id


def test_new_snapshot_replaces_default_only_after_valid_atomic_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the prior snapshot current when replacement parsing fails."""
    prior = _artifact(ArtifactKind.BOM_WORKBOOK, synthetic_uuid("bom-prior"))
    uploads = _UploadRepository(latest_bom=prior)
    service = _service(uploads, _ArtifactRepository())
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    monkeypatch.setattr(
        ingestion_module,
        "parse_bom_workbook",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("invalid replacement")
        ),
    )
    with pytest.raises(ValueError, match="invalid replacement"):
        service.ingest_upload(_request(with_bom=True))
    assert uploads.publications == []
    assert uploads.latest_bom is prior


def test_replayed_request_skips_all_preparation_and_creates_no_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the committed upload for the same request ID without duplicate work."""
    uploads = _UploadRepository()
    existing = UploadRecord(
        upload_id=synthetic_uuid("upload-existing"),
        request_id=str(_request().request_id),
        ecn_artifact_id=synthetic_uuid("ecn-a"),
        bom_artifact_id=synthetic_uuid("bom-a"),
        rule_artifact_id=None,
        bom_reused=False,
        imported_ecn_count=2,
        publication_order=1,
        created_at=NOW,
    )
    uploads.by_request[existing.request_id] = existing
    service = _service(uploads, _ArtifactRepository())
    monkeypatch.setattr(
        service,
        "prepare_upload",
        lambda request: (_ for _ in ()).throw(AssertionError("prepared")),
    )
    result = service.ingest_upload(_request())
    assert result.upload is existing
    assert result.created is False
    assert result.queued_job_ids == ()
    assert result.projection_ecn_ids == ()
    assert uploads.publications == []


def test_prepare_upload_keeps_repeated_and_missing_business_ids_as_occurrences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use row IDs/source ordinals rather than nullable business number as identity."""
    parsed = [
        make_ecn(
            ecn_row_id=synthetic_uuid("row-a"),
            source_ordinal=0,
            ecn_number="ECN1",
        ),
        make_ecn(
            ecn_row_id=synthetic_uuid("row-b"),
            source_ordinal=1,
            ecn_number="ECN1",
        ),
        make_ecn(
            ecn_row_id=synthetic_uuid("row-c"),
            source_ordinal=2,
            ecn_number=None,
        ),
    ]
    uploads = _UploadRepository()
    service = _service(uploads, _ArtifactRepository())
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=tuple(replace(item, artifact_id=artifact_id) for item in parsed),
            normalized_content="[]",
        ),
    )
    monkeypatch.setattr(ingestion_module, "parse_bom_workbook", _one_row_workbook)
    publication = service.prepare_upload(_request())
    assert [item.ecn.ecn_row_id for item in publication.ecns] == [
        synthetic_uuid("row-a"),
        synthetic_uuid("row-b"),
        synthetic_uuid("row-c"),
    ]
    assert [item.ecn.ecn_number for item in publication.ecns] == [
        "ECN1",
        "ECN1",
        None,
    ]
    assert len({item.job_id for item in publication.ecns}) == 3


def test_absent_rules_create_section_gap_without_failing_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish usable BOM evidence when setup configuration was not imported."""
    uploads = _UploadRepository()
    service = _service(uploads, _ArtifactRepository(rule_artifact=None))
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    monkeypatch.setattr(ingestion_module, "parse_bom_workbook", _one_row_workbook)
    result = service.ingest_upload(_request())
    gaps = uploads.publications[0].ecns[0].evidence.sections.configuration_scope
    assert result.created is True
    assert any(item.values.get("code") == "RULE_ARTIFACT_MISSING" for item in gaps)


def test_repository_race_loss_returns_no_jobs_to_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use repository result as authority and discard speculative job IDs."""
    uploads = _UploadRepository()
    service = _service(uploads, _ArtifactRepository())
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    monkeypatch.setattr(ingestion_module, "parse_bom_workbook", _one_row_workbook)
    winner = UploadRecord(
        upload_id=synthetic_uuid("race-winner-upload"),
        request_id=str(_request().request_id),
        ecn_artifact_id=synthetic_uuid("race-winner-ecn"),
        bom_artifact_id=synthetic_uuid("race-winner-bom"),
        rule_artifact_id=None,
        bom_reused=False,
        imported_ecn_count=1,
        publication_order=9,
        created_at=NOW,
    )
    monkeypatch.setattr(
        uploads,
        "publish_upload",
        lambda publication: UploadPublishResult(upload=winner, created=False),
    )
    result = service.ingest_upload(_request())
    assert result.upload is winner
    assert result.created is False
    assert result.queued_job_ids == ()


def test_preparation_gate_serializes_memory_heavy_work_without_blocking_others(
) -> None:
    """Allow one preparation while code outside the gate remains responsive."""
    gate = IngestionPreparationGate()
    first_inside = Event()
    release_first = Event()
    second_started = Event()
    second_inside = Event()
    unrelated_ran = Event()

    def first_worker() -> None:
        """Hold the only preparation permit until the test releases it."""
        with gate.hold():
            first_inside.set()
            assert release_first.wait(timeout=2)

    def second_worker() -> None:
        """Attempt a second preparation and record eventual admission."""
        second_started.set()
        with gate.hold():
            second_inside.set()

    first = Thread(target=first_worker)
    second = Thread(target=second_worker)
    first.start()
    assert first_inside.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    unrelated = Thread(target=unrelated_ran.set)
    unrelated.start()
    try:
        assert unrelated_ran.wait(timeout=2)
        assert not second_inside.wait(timeout=0.05)
    finally:
        release_first.set()
    assert second_inside.wait(timeout=2)
    first.join(timeout=2)
    second.join(timeout=2)
    unrelated.join(timeout=2)
    assert not first.is_alive()
    assert not second.is_alive()
    assert not unrelated.is_alive()


def test_reset_during_reused_snapshot_preparation_rejects_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a precise conflict if reset deletes the prepared reused snapshot."""
    bom_id = synthetic_uuid("bom-reset-race")
    stored_bom = _artifact(ArtifactKind.BOM_WORKBOOK, bom_id)
    uploads = _UploadRepository(latest_bom=stored_bom)
    load_started = Event()
    release_load = Event()
    artifacts = _ArtifactRepository(
        rows=(make_bom_row(artifact_id=bom_id),),
        artifacts=(stored_bom,),
        load_started=load_started,
        release_load=release_load,
    )
    guard = HistoryMutationGuard()
    service = _service(uploads, artifacts, mutation_guard=guard)
    monkeypatch.setattr(
        ingestion_module,
        "parse_ecn_json",
        lambda content, artifact_id: ParsedEcnJson(
            ecns=(make_ecn(artifact_id=artifact_id),), normalized_content="[]"
        ),
    )
    errors: list[PersistenceStateError] = []

    def ingest() -> None:
        """Capture the deterministic service error from the interleaving."""
        try:
            service.ingest_upload(_request(with_bom=False))
        except PersistenceStateError as exc:
            errors.append(exc)

    worker = Thread(target=ingest)
    worker.start()
    assert load_started.wait(timeout=2)
    with guard.hold():
        artifacts.artifacts.clear()
        uploads.latest_bom = None
        release_load.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert getattr(errors[0], "code", None) == "BOM_SNAPSHOT_INVALIDATED"
    assert uploads.publications == []
