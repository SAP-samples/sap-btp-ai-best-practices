"""Opt-in real-data HANA plus SAP Gen AI Hub end-to-end verification.

Run from the repository root with real credentials in ``api/.env``::

    RUN_FULL_LIVE_TESTS=1 PYTHONPATH=api .venv/bin/python -m pytest -q -s \
        tests/integration/test_generation_api_live.py

Do not source ``api/.env`` in a shell. The application loads it with
``python-dotenv``, which preserves shell-significant characters in secrets.

The test creates UUID-suffixed HANA tables, uploads the repository's real ECN,
BOM, and rule fixtures through the FastAPI surface, performs one staged
analysis for the golden ECN occurrence, verifies persisted state
and token telemetry, then drops only its isolated tables.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from app.dependencies import ServiceContainer
from app.ecn.llm.provider import create_impact_provider
from app.ecn.rules.importer import normalize_rule_workbook
from app.ecn.rules.models import RuleArtifactMetadata
from app.ecn.services.generation import AnalysisGenerationService, SystemClock
from app.ecn.services.history import HistoryService
from app.ecn.services.ingestion import IngestionService
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.services.retry import RetryService
from app.ecn.services.runtime import GenerationRuntime
from app.ecn.services.snapshots import SnapshotService
from app.ecn.settings import load_ecn_settings
from app.main import create_app
from app.observability import emit_llm_usage_event
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.codec import normalize_row_keys
from app.persistence.ecn_repository import EcnRepository
from app.persistence.hana import create_hana_engine
from app.persistence.history_repository import HistoryRepository
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.schema import current_schema, ensure_schema, table_exists
from app.persistence.table_contracts import (
    PersistenceTableNames,
    build_table_contracts,
    quote_identifier,
)
from app.persistence.upload_repository import UploadRepository
from sqlalchemy import text

pytestmark = [pytest.mark.hana_live, pytest.mark.llm_live]

DATA = Path(__file__).resolve().parents[2] / "data"
ECN_PATH = DATA / "ECN.json"
BOM_PATH = DATA / "EBOM MBOM_Q900V_04.09.2026.xlsx"
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class RecordingScheduler:
    """Record upload scheduling without starting all 35 live analyses."""

    def __init__(self) -> None:
        """Initialize an empty committed batch history."""
        self.job_batches: list[tuple[str, ...]] = []

    def schedule_jobs(self, job_ids: tuple[str, ...]) -> None:
        """Record exact post-publication job identities."""
        self.job_batches.append(job_ids)


def _rule_path() -> Path:
    """Return the real non-lock conditional-rule workbook fixture."""
    candidates = sorted(
        path
        for path in DATA.glob("R*glesconditionnelleslocales*.xlsx")
        if not path.name.startswith("~$")
    )
    if not candidates:
        pytest.skip("real conditional-rule workbook is unavailable")
    return candidates[0]


def _drop_isolated_tables(engine: object, names: PersistenceTableNames) -> None:
    """Drop only this test's seven suffixed HANA tables in FK-safe order."""
    with engine.connect() as connection:  # type: ignore[attr-defined]
        schema = current_schema(connection)
        for contract in reversed(build_table_contracts(names)):
            if table_exists(connection, schema, contract.name):
                connection.execute(
                    text(f"DROP TABLE {quote_identifier(contract.name)}")
                )
                connection.commit()


@pytest.mark.skipif(
    os.getenv("RUN_FULL_LIVE_TESTS") != "1",
    reason="set RUN_FULL_LIVE_TESTS=1 for the isolated HANA plus LLM check",
)
@pytest.mark.asyncio
async def test_real_upload_one_generation_and_readback() -> None:
    """Upload 35 real ECNs, generate one analysis, and verify HANA/API readback."""
    for path in (ECN_PATH, BOM_PATH, _rule_path()):
        if not path.exists():
            pytest.skip(
                f"required ignored development fixture is unavailable: {path.name}"
            )

    suffix = "_T" + uuid4().hex[:10].upper()
    names = PersistenceTableNames().with_suffix(suffix)
    engine = create_hana_engine()
    provider = None
    generation_runtime = None
    try:
        ensure_schema(engine, names)
        settings = load_ecn_settings()
        assert settings.llm_model_name == "gpt-5.4"
        # Keep the recurring live smoke representative but materially cheaper
        # than the production ceiling; omitted evidence is explicitly surfaced
        # as gaps by the same projection and validation contracts.
        settings = settings.model_copy(
            update={
                "max_prompt_evidence_items": 75,
                "max_prompt_characters": 60_000,
            }
        )
        upload_repository = UploadRepository(engine, names)
        artifact_repository = ArtifactRepository(engine, names)
        ecn_repository = EcnRepository(engine, names)
        reference_repository = ReferenceDatasetRepository(engine, names)
        analysis_repository = AnalysisRepository(engine, names)
        history_repository = HistoryRepository(engine, names)
        mutation_guard = HistoryMutationGuard()

        rule_path = _rule_path()
        rule_normalization = normalize_rule_workbook(
            rule_path.read_bytes(),
            RuleArtifactMetadata(
                artifact_id=str(uuid4()),
                file_name=rule_path.name,
                media_type=XLSX_MEDIA_TYPE,
                created_at=datetime.now(timezone.utc),
            ),
        )
        assert len(rule_normalization.rules) == 4_575
        artifact_repository.publish_rule_artifact(rule_normalization.artifact)

        ingestion_service = IngestionService(
            upload_repository=upload_repository,
            artifact_repository=artifact_repository,
            settings=settings,
            preparation_gate=IngestionPreparationGate(),
            mutation_guard=mutation_guard,
            clock=lambda: datetime.now(timezone.utc),
            uuid_factory=lambda: str(uuid4()),
            reference_repository=reference_repository,
        )
        history_service = HistoryService(history_repository, mutation_guard)
        retry_service = RetryService(analysis_repository, mutation_guard)
        scheduler = RecordingScheduler()
        services = ServiceContainer(
            settings=settings,
            ingestion_service=ingestion_service,
            history_service=history_service,
            retry_service=retry_service,
            ecn_repository=ecn_repository,
            analysis_repository=analysis_repository,
            runtime=scheduler,  # type: ignore[arg-type] - intentionally records all 35 jobs.
            engine=engine,
        )

        @asynccontextmanager
        async def lifespan(_app):
            """Yield the isolated live service container on this event loop."""
            yield services

        app = create_app(service_lifespan_factory=lifespan)
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://live-test"
            ) as client,
        ):
            response = await client.post(
                "/api/uploads",
                headers={"X-API-Key": os.environ["API_KEY"]},
                data={"request_id": str(uuid4())},
                files={
                    "ecn_file": (
                        ECN_PATH.name,
                        ECN_PATH.read_bytes(),
                        "application/json",
                    ),
                    "bom_file": (
                        BOM_PATH.name,
                        BOM_PATH.read_bytes(),
                        XLSX_MEDIA_TYPE,
                    ),
                },
            )
            assert response.status_code == 202, response.text
            upload = response.json()
            assert upload["created"] is True
            assert upload["imported_ecn_count"] == 35
            assert upload["ready_count"] == 35 and upload["queued_count"] == 0
            assert scheduler.job_batches == []
            # Explicitly select the golden original source ordinal; upload itself makes no calls.
            from app.persistence.workspace_repository import WorkspaceRepository
            with engine.connect() as connection:
                golden = connection.execute(text(
                    f'SELECT "ECN_ROW_ID" FROM {quote_identifier(names.ecn)} WHERE "UPLOAD_ID" = :id AND "SOURCE_ORDINAL" = 3'
                ), {"id": upload["upload_id"]}).scalar_one()
            _, selected = WorkspaceRepository(engine, names).create_batch(str(uuid4()), [golden])
            golden_job_id = selected[0]
            provider = create_impact_provider(settings)
            clock = SystemClock()
            generation_service = AnalysisGenerationService(
                analysis_repository,
                provider,
                emit_llm_usage_event,
                clock,
                uuid4,
                snapshot_service=SnapshotService(
                    ecn_repository, artifact_repository, reference_repository, engine, names,
                ),
            )
            generation_runtime = GenerationRuntime(
                generation_service,
                analysis_repository,
                mutation_guard,
                maximum_concurrency=1,
                provider_max_concurrency=4,
                clock=clock,
                uuid_factory=uuid4,
            )
            result = await generation_runtime.process_job(golden_job_id)
            assert result.state == "COMPLETED", result

            progress = await client.get(
                "/api/analysis-progress",
                headers={"X-API-Key": os.environ["API_KEY"]},
                params={"upload_id": upload["upload_id"]},
            )
            assert progress.status_code == 200, progress.text
            counts = progress.json()
            assert counts["ready_count"] == 34 and counts["queued_count"] == 0
            assert counts["completed_count"] + counts["completed_with_gaps_count"] == 1

            ecns = await client.get(
                "/api/ecns",
                headers={"X-API-Key": os.environ["API_KEY"]},
                params={"ecn_number": "ECN126772"},
            )
            assert ecns.status_code == 200, ecns.text
            assert ecns.json()["total_items"] == 1
            ecn_row_id = ecns.json()["items"][0]["ecn_row_id"]
            detail = await client.get(
                f"/api/ecns/{ecn_row_id}",
                headers={"X-API-Key": os.environ["API_KEY"]},
            )
            assert detail.status_code == 200, detail.text
            assert detail.json()["analysis"] is not None

        with engine.connect() as connection:
            attempt = normalize_row_keys(
                connection.execute(
                    text(
                        f'SELECT "PROVIDER_OUTCOME", "VALIDATION_OUTCOME", '
                        f'"USAGE_AVAILABLE", "TOTAL_TOKENS" '
                        f"FROM {quote_identifier(names.llm_attempt)} "
                        f'WHERE "JOB_ID" = :job_id'
                    ),
                    {"job_id": golden_job_id},
                )
                .mappings()
                .one()
            )
        assert attempt["PROVIDER_OUTCOME"] == "SUCCESS"
        assert attempt["VALIDATION_OUTCOME"] == "VALID"
        assert int(attempt["USAGE_AVAILABLE"]) == 1
        assert int(attempt["TOTAL_TOKENS"]) > 0
    finally:
        if generation_runtime is not None:
            await generation_runtime.close()
        elif provider is not None:
            await provider.close()
        try:
            _drop_isolated_tables(engine, names)
        finally:
            engine.dispose()
