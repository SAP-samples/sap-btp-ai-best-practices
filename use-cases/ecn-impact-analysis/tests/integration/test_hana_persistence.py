"""Live-HANA integration tests for schema initialization and repository round-trips.

These tests are explicitly gated behind the ``RUN_HANA_INTEGRATION_TESTS``
environment variable and the ``hana_live`` pytest marker.  They are SKIPPED by
default so that the CI and unit-test suites do not require live HANA credentials.

To run the live tests in an isolated namespace::

    RUN_HANA_INTEGRATION_TESTS=1 \\
        HANA_ADDRESS=<host> HANA_PORT=<port> HANA_USER=<user> \\
        HANA_PASSWORD=<pass> HANA_ENCRYPT=true \\
        PYTHONPATH=api python -m pytest -q tests/integration/test_hana_persistence.py

Default-skipped run (CI)::

    PYTHONPATH=api python -m pytest -q tests/integration/test_hana_persistence.py
    # Expected: 1 skipped (or 2 skipped when both tests are counted)

Each test creates a UUID-suffixed set of seven tables, runs its assertions,
and drops only those seven suffixed tables in the ``finally`` block.  The tests
never touch unsuffixed application tables and never call history reset on real data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.persistence.hana import create_hana_engine
from app.persistence.schema import current_schema, ensure_schema, table_exists
from app.persistence.table_contracts import (
    PersistenceTableNames,
    build_table_contracts,
    quote_identifier,
)


pytestmark = pytest.mark.hana_live

# ---------------------------------------------------------------------------
# Integration-test helpers
# ---------------------------------------------------------------------------

# Valid pinned generation config that satisfies _PinnedPayload strict validation.
# Required for claim_job_and_start_attempt to succeed in the live round-trip test.
_VALID_MODEL_CONFIG = json.dumps(
    {
        "max_prompt_characters": 250000,
        "max_prompt_evidence_items": 500,
        "request_timeout_seconds": 180,
        "strict_structured_output": True,
    },
    sort_keys=True,
    separators=(",", ":"),
)

_INTEGRATION_UTC_FINISH = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def _make_upload_publication(ecn_count: int = 1) -> object:
    """Build a valid ``UploadPublication`` with correct model_config_json for live tests.

    This builder generates fresh UUIDs so each test run is isolated.  The
    ``model_config_json`` uses the valid ``_VALID_MODEL_CONFIG`` constant so
    that ``claim_job_and_start_attempt`` can parse and validate the pinned config.

    Args:
        ecn_count: Number of ECN publications to include.

    Returns:
        A frozen ``UploadPublication`` instance ready for ``publish_upload``.
    """
    from app.ecn.api_models import ImpactCounts
    from app.ecn.models import (
        AnalysisSectionKey,
        ArtifactKind,
        DETERMINISTIC_LOGIC_VERSION,
        EvidenceItem,
        EvidenceKind,
        EvidenceSource,
        EvidenceTier,
        PARSER_VERSION,
        ParseState,
        RULE_EVALUATOR_VERSION,
    )
    from app.persistence.records import EcnPublication, UploadPublication
    from tests.fixtures.builders import (
        make_bom_row,
        make_evidence_bundle,
        make_evidence_bundle_sections,
        make_evidence_context,
        make_normalized_ecn,
        make_source_artifact,
    )

    upload_id = str(uuid4())
    request_id = str(uuid4())
    ecn_artifact_id = str(uuid4())
    bom_artifact_id = str(uuid4())

    ecn_artifact = make_source_artifact(
        artifact_id=ecn_artifact_id,
        artifact_kind=ArtifactKind.ECN_JSON,
        parse_state=ParseState.VALID,
    )
    bom_artifact = make_source_artifact(
        artifact_id=bom_artifact_id,
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        parse_state=ParseState.VALID,
    )
    bom_row = make_bom_row(artifact_id=bom_artifact_id)

    ecn_pubs = []
    for i in range(ecn_count):
        ecn_row_id = str(uuid4())
        job_id = str(uuid4())
        ecn = make_normalized_ecn(
            ecn_row_id=ecn_row_id,
            artifact_id=ecn_artifact_id,
            source_ordinal=i,
        )
        context = make_evidence_context(
            upload_id=upload_id,
            ecn_artifact_id=ecn_artifact_id,
            bom_artifact_id=bom_artifact_id,
        )
        evidence_item = EvidenceItem(
            evidence_id=str(uuid4()),
            section_key=AnalysisSectionKey.EXECUTIVE_SUMMARY,
            kind=EvidenceKind.SOURCE_FACT,
            tier=EvidenceTier.DIRECT_ECN_RAW,
            summary="Synthetic ECN source fact for the live round-trip.",
            sources=(
                EvidenceSource(
                    artifact_id=ecn_artifact_id,
                    artifact_kind=ArtifactKind.ECN_JSON,
                    json_path="$[0]",
                    sheet_name=None,
                    bom_row_id=None,
                    source_row_number=None,
                ),
            ),
            values={"source_ordinal": i},
            related_ids=(ecn_row_id,),
            quality_qualifiers=(),
        )
        evidence = make_evidence_bundle(
            ecn_row_id=ecn_row_id,
            context=context,
            sections=make_evidence_bundle_sections(
                executive_summary=(evidence_item,),
            ),
        )
        counts = ImpactCounts(
            affected_item_count=0,
            raw_ecn_match_count=0,
            ebom_match_count=0,
            ebom_context_count=0,
            mbom_match_count=0,
            warning_count=0,
        )
        ecn_pubs.append(
            EcnPublication(
                ecn=ecn,
                change_candidates=(),
                evidence=evidence,
                counts=counts,
                job_id=job_id,
                prompt_version="v1",
                output_schema_version="v1",
                model_name="gpt-4",
                model_config_json=_VALID_MODEL_CONFIG,
            )
        )

    created_at = datetime(2026, 9, 14, tzinfo=timezone.utc)
    return UploadPublication(
        upload_id=upload_id,
        request_id=request_id,
        ecn_artifact=ecn_artifact,
        bom_artifact=bom_artifact,
        new_bom_rows=(bom_row,),
        rule_artifact=None,
        bom_reused=False,
        ecns=tuple(ecn_pubs),
        actor_id=None,
        parser_version=PARSER_VERSION,
        deterministic_logic_version=DETERMINISTIC_LOGIC_VERSION,
        rule_evaluator_version=RULE_EVALUATOR_VERSION,
        created_at=created_at,
    )


def _make_attempt_seed() -> object:
    """Build a minimal ``AttemptSeed`` with fresh UUIDs for the integration test.

    Returns:
        A frozen ``AttemptSeed`` instance.
    """
    from app.persistence.records import AttemptSeed

    return AttemptSeed(
        attempt_id=str(uuid4()),
        correlation_id="integration-test-correlation",
        started_at=datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc),
        provider="generative-ai-hub",
        model_deployment="deployment-a",
        llm_endpoint="responses",
    )


def _make_provider_outcome_success() -> object:
    """Build a SUCCESS ``ProviderOutcome`` with token usage for the integration test.

    Returns:
        A frozen ``ProviderOutcome`` instance.
    """
    from app.observability.llm_usage_logging import TokenUsage
    from app.persistence.records import ProviderCallOutcome, ProviderOutcome

    return ProviderOutcome(
        provider_outcome=ProviderCallOutcome.SUCCESS,
        finished_at=_INTEGRATION_UTC_FINISH,
        latency_ms=200,
        usage=TokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=50,
            total_tokens=150,
        ),
        usage_available=True,
        model_deployment="deployment-a",
        error_code=None,
    )


def _make_analysis_completion(claimed: object) -> object:
    """Build a valid ``AnalysisCompletion`` from a claimed job for the round-trip test.

    Uses NOT_APPLICABLE sections with a fake evidence ID so the Pydantic model
    is valid without requiring real evidence IDs from the bundle.  The
    evidence_hash, version strings, and model deployment are all drawn from
    the claimed job's fields so that all precondition checks pass.

    Args:
        claimed: A ``ClaimedJob`` returned by ``claim_job_and_start_attempt``.

    Returns:
        A frozen ``AnalysisCompletion`` instance.
    """
    from app.ecn.analysis_models import (
        AnalysisResponse,
        AnalysisSection,
        AnalysisSections,
        SectionStatus,
        derive_completeness,
    )
    from app.persistence.records import AnalysisCompletion

    section = AnalysisSection(
        status=SectionStatus.NOT_APPLICABLE,
        assessment_markdown="Integration test assessment",
        evidence_ids=["int-test-ev-1"],
        missing_evidence=[],
    )
    response = AnalysisResponse(
        sections=AnalysisSections(
            executive_summary=section,
            change_intent=section,
            affected_materials=section,
            ebom_mbom_impact=section,
            configuration_scope=section,
            uncertainties_and_missing_evidence=section,
        )
    )

    return AnalysisCompletion(
        analysis_id=str(uuid4()),
        job_id=claimed.job_id,  # type: ignore[attr-defined]
        ecn_row_id=claimed.ecn_row_id,  # type: ignore[attr-defined]
        successful_attempt_id=claimed.attempt.attempt_id,  # type: ignore[attr-defined]
        response=response,
        completeness=derive_completeness(response),
        evidence_hash=claimed.evidence.evidence_hash,  # type: ignore[attr-defined]
        prompt_version=claimed.generation_config.prompt_version,  # type: ignore[attr-defined]
        model_name=claimed.generation_config.model_name,  # type: ignore[attr-defined]
        model_deployment=claimed.attempt.model_deployment,  # type: ignore[attr-defined]
        deterministic_logic_version=claimed.evidence.context.deterministic_logic_version,  # type: ignore[attr-defined]
        rule_evaluator_version=claimed.evidence.context.rule_evaluator_version,  # type: ignore[attr-defined]
        output_schema_version=claimed.generation_config.output_schema_version,  # type: ignore[attr-defined]
        created_at=_INTEGRATION_UTC_FINISH,
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RUN_HANA_INTEGRATION_TESTS") != "1",
    reason="set RUN_HANA_INTEGRATION_TESTS=1 for isolated live HANA checks",
)
def test_live_hana_creates_and_revalidates_isolated_schema() -> None:
    """Create all seven tables in an isolated UUID-suffixed namespace, then re-run.

    Asserts:
    * ``ensure_schema`` returns a non-empty schema string on first run.
    * ``ensure_schema`` returns the same schema on the second (idempotent) run.
    * All seven suffixed tables exist in ``SYS.TABLES`` after both runs.

    Cleanup (``finally`` block):
    * Drops only the seven UUID-suffixed tables created by this test in
      reverse dependency order.
    * Never touches unsuffixed production tables.
    * Never calls history reset.
    """
    # Generate a unique 10-character hex suffix to isolate this test run.
    suffix = "_T" + uuid4().hex[:10].upper()
    names = PersistenceTableNames().with_suffix(suffix)
    engine = create_hana_engine()
    try:
        # First run: all seven tables should be created.
        schema = ensure_schema(engine, names)
        assert schema, f"ensure_schema returned an empty schema: {schema!r}"

        # Second run: all tables exist and are compatible; no DDL should be needed.
        schema_retry = ensure_schema(engine, names)
        assert schema_retry == schema, (
            f"Idempotent re-run returned different schema: {schema_retry!r} != {schema!r}"
        )

        # Verify all seven tables are physically present.
        with engine.connect() as connection:
            verification_schema = current_schema(connection)
            for contract in build_table_contracts(names):
                assert table_exists(connection, verification_schema, contract.name), (
                    f"Expected table {contract.name!r} not found in schema "
                    f"{verification_schema!r}"
                )
    finally:
        # Clean up in reverse dependency order (FK-safe drop sequence).
        # Only the seven UUID-suffixed names created by this test are dropped.
        with engine.connect() as connection:
            cleanup_schema = current_schema(connection)
            for contract in reversed(build_table_contracts(names)):
                if table_exists(connection, cleanup_schema, contract.name):
                    connection.execute(
                        text(f"DROP TABLE {quote_identifier(contract.name)}")
                    )
                    connection.commit()
        engine.dispose()


@pytest.mark.skipif(
    os.getenv("RUN_HANA_INTEGRATION_TESTS") != "1",
    reason="set RUN_HANA_INTEGRATION_TESTS=1 for isolated live HANA checks",
)
def test_live_hana_repository_round_trip() -> None:
    """Publish, claim, record success, complete, and verify progress in an isolated namespace.

    This test exercises the full analysis job lifecycle against a live HANA
    instance using a UUID-suffixed isolated schema:

    1. ``UploadRepository.publish_upload`` — stores the upload and one ECN/job.
    2. ``find_by_request_id`` — verifies idempotency lookup.
    3. ``get_ecn_detail`` / ``get_evidence`` — reads provenance and scoped evidence.
    4. ``EcnRepository.list_queued_job_ids`` — retrieves the QUEUED job.
    5. ``AnalysisRepository.claim_job_and_start_attempt`` — atomically claims it.
    6. ``record_provider_outcome`` — records a SUCCESS result with token usage.
    7. ``complete_job`` — inserts the analysis and closes the job as COMPLETED.
    8. A duplicate completion is rejected and progress remains exactly complete.

    Cleanup (``finally`` block):
    * Drops only the seven UUID-suffixed tables in reverse dependency order.
    * Never touches unsuffixed production tables.
    * Never calls ``clear_history_if_idle`` on real data.
    """
    from app.ecn.errors import PersistenceStateError
    from app.persistence.analysis_repository import AnalysisRepository
    from app.persistence.ecn_repository import EcnRepository
    from app.persistence.records import ClaimJobStatus
    from app.persistence.upload_repository import UploadRepository

    suffix = "_T" + uuid4().hex[:10].upper()
    names = PersistenceTableNames().with_suffix(suffix)
    engine = create_hana_engine()
    try:
        # Set up the isolated schema.
        ensure_schema(engine, names)

        upload_repository = UploadRepository(engine, names)
        ecn_repository = EcnRepository(engine, names)
        analysis_repository = AnalysisRepository(engine, names)

        # Step 1: Publish a synthetic upload with one ECN and one job.
        publication = _make_upload_publication(ecn_count=1)
        publication_result = upload_repository.publish_upload(publication)  # type: ignore[arg-type]
        assert publication_result.created is True

        # Step 2: Idempotency lookup.
        found = upload_repository.find_by_request_id(
            publication_result.upload.request_id
        )
        assert found == publication_result.upload

        # Step 3: Read the full detail and one evidence item before analysis.
        ecn_publication = publication.ecns[0]  # type: ignore[attr-defined]
        detail = ecn_repository.get_ecn_detail(ecn_publication.ecn.ecn_row_id)
        assert detail is not None
        assert detail.upload == publication_result.upload
        assert (
            detail.ecn_artifact.artifact_id
            == publication.ecn_artifact.artifact_id  # type: ignore[attr-defined]
        )
        assert (
            detail.bom_artifact.artifact_id
            == publication.bom_artifact.artifact_id  # type: ignore[attr-defined]
        )
        assert detail.analysis is None
        evidence_item = ecn_publication.evidence.sections.executive_summary[0]
        assert (
            ecn_repository.get_evidence(
                ecn_publication.ecn.ecn_row_id,
                evidence_item.evidence_id,
            )
            == evidence_item
        )

        # Step 4: Explicitly queue this test's selected Ready assessment.
        from app.persistence.workspace_repository import WorkspaceRepository
        WorkspaceRepository(engine, names).create_batch(str(uuid4()), [ecn_publication.ecn.ecn_row_id])
        # List only the explicitly queued job IDs for this upload.
        job_ids = ecn_repository.list_queued_job_ids(
            publication_result.upload.upload_id
        )
        assert len(job_ids) == 1
        job_id = job_ids[0]

        # Step 5: Claim the job atomically.
        seed = _make_attempt_seed()
        claim_result = analysis_repository.claim_job_and_start_attempt(
            job_id, seed  # type: ignore[arg-type]
        )
        assert claim_result.status is ClaimJobStatus.CLAIMED
        claimed = claim_result.claimed_job
        assert claimed is not None

        # Step 6: Record a SUCCESS provider outcome with token usage.
        analysis_repository.record_provider_outcome(
            claimed.attempt.attempt_id,
            _make_provider_outcome_success(),  # type: ignore[arg-type]
        )

        # Step 7: Complete the job with a valid analysis record.
        completion = _make_analysis_completion(claimed)
        analysis_repository.complete_job(job_id, completion)  # type: ignore[arg-type]

        completed_detail = ecn_repository.get_ecn_detail(
            ecn_publication.ecn.ecn_row_id
        )
        assert completed_detail is not None
        assert completed_detail.analysis is not None
        assert str(completed_detail.analysis.analysis_id) == completion.analysis_id

        # Step 8: Completion is immutable and progress remains exactly complete.
        with pytest.raises(PersistenceStateError) as duplicate_completion:
            analysis_repository.complete_job(job_id, completion)  # type: ignore[arg-type]
        assert duplicate_completion.value.code == "ANALYSIS_STATE_CONFLICT"

        progress = analysis_repository.get_progress(
            upload_id=publication_result.upload.upload_id
        )
        assert progress.completed_count == 1, (
            f"Expected completed_count=1, got {progress.completed_count}"
        )
        assert progress.total_count == 1
        assert progress.unfinished_count == 0

    finally:
        # Clean up in reverse dependency order (FK-safe drop sequence).
        # Only the seven UUID-suffixed names created by this test are dropped.
        with engine.connect() as connection:
            cleanup_schema = current_schema(connection)
            for contract in reversed(build_table_contracts(names)):
                if table_exists(connection, cleanup_schema, contract.name):
                    connection.execute(
                        text(f"DROP TABLE {quote_identifier(contract.name)}")
                    )
                    connection.commit()
        engine.dispose()


@pytest.mark.skipif(
    os.getenv("RUN_HANA_INTEGRATION_TESTS") != "1",
    reason="set RUN_HANA_INTEGRATION_TESTS=1 for isolated live HANA checks",
)
def test_live_hana_plant_column_upgrade_filter_and_sort() -> None:
    """Upgrade a pre-plant ECN table, then filter and sort the list by plant.

    Steps:
    1. Create an isolated schema and drop ``AFFECTED_PLANTS_JSON`` to mimic a
       table created before the plant column existed.
    2. Re-run ``ensure_schema``; it must add the column back (additive ALTER).
    3. Publish three ECNs, store plant labels on two, leave one NULL.
    4. ``plant=pl10`` matches only the PL10 row (case-insensitive); the NULL
       row never matches; ``sort=affected_plants`` puts NULL last.

    Run:
        RUN_HANA_INTEGRATION_TESTS=1 HANA_ADDRESS=... HANA_PORT=... HANA_USER=... \\
            HANA_PASSWORD=... PYTHONPATH=api .venv/bin/python -m pytest -q \\
            tests/integration/test_hana_persistence.py -k plant_column
    """
    from sqlalchemy import inspect

    from app.ecn.api_models import EcnListQuery
    from app.ecn.services.snapshots import plant_labels_json
    from app.persistence.ecn_repository import EcnRepository
    from app.persistence.upload_repository import UploadRepository

    suffix = "_T" + uuid4().hex[:10].upper()
    names = PersistenceTableNames().with_suffix(suffix)
    engine = create_hana_engine()
    ecn_table = quote_identifier(names.ecn)
    try:
        ensure_schema(engine, names)
        with engine.connect() as connection:
            connection.execute(text(f'ALTER TABLE {ecn_table} DROP ("AFFECTED_PLANTS_JSON")'))
            connection.commit()

        ensure_schema(engine, names)
        with engine.connect() as connection:
            columns = [c["name"].upper() for c in inspect(connection).get_columns(names.ecn, schema=current_schema(connection))]
        assert columns[-1] == "AFFECTED_PLANTS_JSON"

        publication = _make_upload_publication(ecn_count=3)
        UploadRepository(engine, names).publish_upload(publication)  # type: ignore[arg-type]
        ids = [item.ecn.ecn_row_id for item in publication.ecns]  # type: ignore[attr-defined]
        with engine.connect() as connection:
            for ecn_row_id, labels in ((ids[0], ["PL10 Plant A"]), (ids[1], ["DE01 Plant C"])):
                connection.execute(
                    text(f'UPDATE {ecn_table} SET "AFFECTED_PLANTS_JSON" = :plants WHERE "ECN_ROW_ID" = :id'),
                    {"plants": plant_labels_json(labels), "id": ecn_row_id},
                )
            connection.commit()

        repository = EcnRepository(engine, names)
        filtered = repository.list_ecns(EcnListQuery(plant="pl10"))
        assert [str(item.ecn_row_id) for item in filtered.items] == [ids[0]]
        assert filtered.items[0].affected_plants == ["PL10 Plant A"]
        assert repository.list_matching_ecn_ids(EcnListQuery(plant="pl10")) == [ids[0]]

        ordered = repository.list_ecns(EcnListQuery(sort="affected_plants", order="asc"))
        assert [str(item.ecn_row_id) for item in ordered.items] == [ids[1], ids[0], ids[2]]
        assert ordered.items[-1].affected_plants is None
    finally:
        with engine.connect() as connection:
            cleanup_schema = current_schema(connection)
            for contract in reversed(build_table_contracts(names)):
                if table_exists(connection, cleanup_schema, contract.name):
                    connection.execute(text(f"DROP TABLE {quote_identifier(contract.name)}"))
                    connection.commit()
        engine.dispose()
