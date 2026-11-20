"""Real selected-only import acceptance in disposable HANA tables.

Run deterministic acceptance:
    RUN_SNAPSHOT_LIVE=1 PYTHONPATH=api .venv/bin/python -m pytest -q -s --tb=short tests/integration/test_snapshot_workspace_live.py
Add RUN_SNAPSHOT_LLM=1 for one summary and one assistant turn (up to four calls).
Credentials load from api/.env; never source it in the shell. No deployment or
unsuffixed application tables are modified. Test-owned tables are removed in finally.
"""

import asyncio
import gc
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.ecn.reference_data.models import ReferenceDatasetType
from app.ecn.reference_data.service import (
    ReferenceDatasetService,
    ReferenceUploadRequest,
)
from app.ecn.api_models import EcnListQuery
from app.ecn.services.ingestion import (
    IngestionService,
    UploadBytesRequest,
    UploadFileBytes,
)
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.services.snapshots import SnapshotService
from app.ecn.settings import load_ecn_settings
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.ecn_repository import EcnRepository
from app.persistence.hana import create_hana_engine, to_hana_timestamp
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.schema import current_schema, ensure_schema, table_exists
from app.persistence.table_contracts import (
    PersistenceTableNames,
    build_table_contracts,
    quote_identifier,
)
from app.persistence.upload_repository import UploadRepository
from app.persistence.workspace_repository import WorkspaceRepository
from sqlalchemy import text

DATA = Path(__file__).resolve().parents[2] / "data"
MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
pytestmark = [
    pytest.mark.hana_live,
    pytest.mark.skipif(
        os.getenv("RUN_SNAPSHOT_LIVE") != "1",
        reason="Explicit isolated live acceptance only",
    ),
]


def report_stage(message):
    """Report progress and peak process memory without exposing source payloads."""
    peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)
    print(f"{message} [peak {peak} MiB]", flush=True)


def test_full_export_staged_snapshots_and_reviews():
    """Import selected-only batches, then verify HANA facts and review versions."""
    settings = load_ecn_settings()
    names = PersistenceTableNames().with_suffix("_S" + uuid4().hex[:10].upper())
    engine = create_hana_engine()
    snapshots = None
    try:
        report_stage("Initializing isolated HANA tables")
        ensure_schema(engine, names)
        ensure_schema(engine, names)
        # Exercise additive migration against actual HANA with old nullable layout columns absent.
        with engine.begin() as connection:
            for table, column in [
                (names.material_master_row, "MATERIAL_GROUP_DESCRIPTION"),
                (names.purchase_info_row, "SUPPLIER_NAME"),
            ]:
                connection.execute(
                    text(f'ALTER TABLE {quote_identifier(table)} DROP ("{column}")')
                )
        ensure_schema(engine, names)
        artifacts = ArtifactRepository(engine, names)
        references = ReferenceDatasetRepository(engine, names)
        reference_service = ReferenceDatasetService(
            references,
            settings,
            lambda: datetime.now(timezone.utc),
            lambda: str(uuid4()),
        )
        sources = [
            (ReferenceDatasetType.BOM, DATA / "EBOM MBOM_Q900V_04.09.2026.xlsx"),
            (
                ReferenceDatasetType.CONDITIONAL_RULES,
                next(DATA.glob("R*glesconditionnelleslocales*.xlsx")),
            ),
            (
                ReferenceDatasetType.MATERIAL_MASTER,
                DATA / "new_16-09/SAP AI4U - Material Master2.XLSX",
            ),
            (
                ReferenceDatasetType.PURCHASE_INFO_RECORDS,
                DATA / "new_16-09/SAP AI4U - Purchase Info Record2.XLSX",
            ),
            (
                ReferenceDatasetType.STOCK,
                DATA / "new_16-09/SAP AI4U - Stock Information.XLSX",
            ),
        ]
        for kind, path in sources:
            report_stage(f"Publishing test snapshot: {kind.value}")
            reference_service.publish(
                ReferenceUploadRequest(kind, path.name, MEDIA, path.read_bytes())
            )
        uploads = UploadRepository(engine, names)
        ecns = EcnRepository(engine, names)
        jobs = AnalysisRepository(engine, names)
        guard = HistoryMutationGuard()
        ingestion = IngestionService(
            upload_repository=uploads,
            artifact_repository=artifacts,
            settings=settings,
            preparation_gate=IngestionPreparationGate(),
            mutation_guard=guard,
            clock=lambda: datetime.now(timezone.utc),
            uuid_factory=lambda: str(uuid4()),
            reference_repository=references,
        )
        workspace = WorkspaceRepository(engine, names, guard)
        snapshots = SnapshotService(ecns, artifacts, references, engine, names)
        report_stage("Selecting records from the full export in memory")
        source_path = DATA / "new_16-09/ECN_full.json"
        source_records = json.loads(source_path.read_text(encoding="utf-8"))
        assert isinstance(source_records, list) and len(source_records) == 557
        positions = {
            str(record.get("Number")): index
            for index, record in enumerate(source_records)
        }
        selected_positions = list(range(25, 50))
        selected_positions.extend(
            positions[number]
            for number in ("ECN346863", "ECN345274")
            if positions[number] not in selected_positions
        )
        selected_records = [source_records[index] for index in selected_positions]
        report_stage("Importing selected-only batches of at most 25 ECNs; no AI")
        for offset in range(0, len(selected_records), 25):
            chunk = selected_records[offset : offset + 25]
            result = ingestion.ingest_upload(
                UploadBytesRequest(
                    uuid4(),
                    UploadFileBytes(
                        source_path.name,
                        "application/json",
                        json.dumps(chunk, ensure_ascii=False).encode("utf-8"),
                    ),
                    None,
                    None,
                )
            )
            assert result.created and result.upload.imported_ecn_count == len(chunk)
        listed = ecns.list_ecns(EcnListQuery(page_size=100))
        ids = {item.ecn_number: str(item.ecn_row_id) for item in listed.items}
        golden = ids["ECN346863"]
        snapshot = snapshots.get(golden)
        assert snapshot["counts"]["technical_objects"] == 9
        stock_object = next(
            o for o in snapshot["objects"] if o["number"] == "QKX7702260"
        )
        assert stock_object["origins"] == ["bom"]
        assert any(
            r["dataset_type"] == "STOCK" for r in stock_object["reference_records"]
        )
        assert any(r["row"] == 21008 for r in stock_object["locations"])
        assert all(j["STATE"] == "READY" for j in workspace.jobs(list(ids.values())))
        report_stage("Checking review attribution and versioned reassessment")
        review = {
            "REVIEW_ID": str(uuid4()),
            "ECN_ROW_ID": golden,
            "EVIDENCE_HASH": snapshot["evidence_hash"],
            "SNAPSHOT_VERSION": snapshot["version"],
            "REVIEWER": "Acceptance test",
            "ACTOR_ID": "test",
            "OUTCOME": "ACCEPTED",
            "COMMENTS": "Snapshot reviewed; source facts preserved.",
            "CREATED_AT": to_hana_timestamp(datetime.now(timezone.utc)),
        }
        workspace.save_review(review)
        workspace.save_review(review)
        assert len(workspace.reviews(golden)) == 1
        from app.routers.workspace_actions import _reassess

        services = SimpleNamespace(
            ecn_repository=ecns,
            artifact_repository=artifacts,
            ingestion_service=ingestion,
        )
        fresh = _reassess(services, golden, uuid4())
        assert fresh["ecn_row_id"] != golden and fresh["model_calls"] == 0
        assert len(workspace.versions(golden)) == 2
        assert snapshots.get(fresh["ecn_row_id"])["counts"] == snapshot["counts"]
        if os.getenv("RUN_SNAPSHOT_LLM") == "1":
            asyncio.run(
                _bounded_ai(settings, jobs, workspace, snapshots, ecns, guard, golden)
            )
        gc.collect()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        print(
            json.dumps(
                {
                    "source_records": 557,
                    "imported_selected_records": len(ids),
                    "stock_bom_object": "QKX7702260",
                    "peak_rss_mib": round(peak, 1),
                    "isolated_cleanup": True,
                }
            ),
            flush=True,
        )
        assert peak < float(os.getenv("SNAPSHOT_MEMORY_BUDGET_MIB", "4096")), (
            "The acceptance run exceeded its configured memory budget"
        )
    finally:
        if snapshots:
            snapshots.clear()
        with engine.connect() as connection:
            schema = current_schema(connection)
            for contract in reversed(build_table_contracts(names)):
                if table_exists(connection, schema, contract.name):
                    connection.execute(
                        text(f"DROP TABLE {quote_identifier(contract.name)}")
                    )
                    connection.commit()
        engine.dispose()
        report_stage("Isolated tables removed")


async def _bounded_ai(settings, jobs, workspace, snapshots, ecns, guard, ecn_id):
    """Run only the selected summary and one read-only assistant question with telemetry."""
    from app.ecn.assistant.runtime import AgentRuntime
    from app.ecn.llm.provider import create_impact_provider
    from app.ecn.services.generation import AnalysisGenerationService, SystemClock
    from app.ecn.services.runtime import GenerationRuntime
    from app.observability import emit_llm_usage_event

    provider = create_impact_provider(settings)
    clock = SystemClock()
    service = AnalysisGenerationService(
        jobs, provider, emit_llm_usage_event, clock, uuid4, snapshot_service=snapshots
    )
    runtime = GenerationRuntime(service, jobs, guard, 2, clock, uuid4)
    assistant = AgentRuntime(
        snapshots, ecns, settings.llm_model_name, runtime.provider_capacity
    )
    try:
        print("Generating one explicitly selected real summary", flush=True)
        _, selected = workspace.create_batch(str(uuid4()), [ecn_id])
        assert len(selected) == 1
        result = await runtime.process_job(selected[0])
        assert result.state == "COMPLETED", result
        print("Asking one read-only stock/hierarchy question", flush=True)
        answer = await assistant.ainvoke(
            "Inspect QKX7702260. Explain how it relates to this ECN, its observed BOM hierarchy and stock context. Cite source rows and distinguish JSON references from BOM rows carrying the ECN.",
            ecn_id,
            [],
        )
        assert answer["answer"] and 1 <= answer["model_calls"] <= 10
        print(
            json.dumps(
                {
                    "summary": "completed",
                    "assistant_calls": answer["model_calls"],
                    "consulted_objects": answer["consulted_objects"],
                    "answer": answer["answer"],
                }
            ),
            flush=True,
        )
    finally:
        await assistant.close()
        await runtime.close()
