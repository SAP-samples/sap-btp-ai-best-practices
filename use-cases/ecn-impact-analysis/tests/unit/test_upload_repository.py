"""Unit tests for UploadRepository: lookups, BOM snapshots, and atomic publication.

Tests are grouped into three suites following the Task 6 TDD brief:

1. Lookups — ``find_by_request_id``, ``get_upload``, ``get_latest_bom_snapshot``.
2. Preflight invariants — validation failures that block the transaction.
3. Atomic publication — insert ordering, PUBLICATION_ORDER allocation, BOM reuse,
   idempotency races, count correctness, and transaction boundary assertions.

No live HANA connection is required; all assertions run against recording fakes.

Example run (full suite)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_upload_repository.py

Example run (lookup / replay subset)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_upload_repository.py \\
        -k 'latest or request or upload'
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.ecn.errors import PersistenceStateError
from app.ecn.models import ArtifactKind, ParseState
from tests.fixtures.builders import (
    make_source_artifact,
    source_artifact_row,
    upload_publication,
    upload_row,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_engine():
    """Return a fresh RecordingEngine for each test.

    Returns:
        A ``RecordingEngine`` with zero counters and an empty unified queue.
    """
    from tests.fixtures.persistence import RecordingEngine

    return RecordingEngine()


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def insert_target(sql: str) -> str:
    """Return the table name from an ``INSERT INTO`` statement, or empty string.

    Used to assert insert ordering without coupling to the full SQL text.

    Args:
        sql: A SQL string that may or may not start with ``INSERT INTO``.

    Returns:
        The unquoted table name, e.g. ``"ECN_IMPACT_UPLOAD"``, or ``""``
        when the statement is not an INSERT.
    """
    match = re.search(r"^INSERT INTO \"([^\"]+)\"", sql.strip())
    return match.group(1) if match else ""


# ===========================================================================
# Suite 1: Upload lookups and latest BOM snapshot
# ===========================================================================


def test_find_by_request_id_returns_none_when_not_found(recording_engine) -> None:
    """find_by_request_id returns None when no row matches the request ID."""
    from app.persistence.upload_repository import UploadRepository

    result = UploadRepository(recording_engine).find_by_request_id(
        "00000000-0000-4000-8000-000000000007"
    )
    assert result is None
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


def test_find_by_request_id_validates_uuid_before_querying(recording_engine) -> None:
    """find_by_request_id rejects non-canonical UUIDs before opening a connection."""
    from app.persistence.upload_repository import UploadRepository

    with pytest.raises(ValueError):
        UploadRepository(recording_engine).find_by_request_id("not-a-uuid")
    assert recording_engine.connect_count == 0


def test_find_by_request_id_binds_the_id_value(recording_engine) -> None:
    """find_by_request_id passes request_id as a bound parameter, not interpolated."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row())
    UploadRepository(recording_engine).find_by_request_id(
        "00000000-0000-4000-8000-000000000007"
    )
    statement, params = recording_engine.executions[0]
    assert "00000000-0000-4000-8000-000000000007" not in statement
    assert params["request_id"] == "00000000-0000-4000-8000-000000000007"


def test_find_by_request_id_returns_upload_record(recording_engine) -> None:
    """find_by_request_id returns a reconstructed UploadRecord when found."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row())
    result = UploadRepository(recording_engine).find_by_request_id(
        "00000000-0000-4000-8000-000000000007"
    )
    assert result is not None
    assert result.upload_id == "00000000-0000-4000-8000-000000000002"
    assert result.request_id == "00000000-0000-4000-8000-000000000007"
    assert result.created_at.tzinfo is timezone.utc


def test_get_upload_returns_none_when_not_found(recording_engine) -> None:
    """get_upload returns None when no row matches the upload ID."""
    from app.persistence.upload_repository import UploadRepository

    result = UploadRepository(recording_engine).get_upload(
        "00000000-0000-4000-8000-000000000002"
    )
    assert result is None
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


def test_get_upload_validates_uuid_before_querying(recording_engine) -> None:
    """get_upload rejects non-canonical UUIDs before opening a connection."""
    from app.persistence.upload_repository import UploadRepository

    with pytest.raises(ValueError):
        UploadRepository(recording_engine).get_upload("bad-uuid")
    assert recording_engine.connect_count == 0


def test_get_upload_returns_upload_record(recording_engine) -> None:
    """get_upload returns a reconstructed UploadRecord when found."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row(UPLOAD_ID="00000000-0000-4000-8000-000000000002"))
    result = UploadRepository(recording_engine).get_upload(
        "00000000-0000-4000-8000-000000000002"
    )
    assert result is not None
    assert result.upload_id == "00000000-0000-4000-8000-000000000002"
    assert result.created_at.tzinfo is timezone.utc


def test_get_upload_binds_upload_id_as_parameter(recording_engine) -> None:
    """get_upload passes upload_id as a bound parameter, not interpolated."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row())
    UploadRepository(recording_engine).get_upload(
        "00000000-0000-4000-8000-000000000002"
    )
    statement, params = recording_engine.executions[0]
    assert "00000000-0000-4000-8000-000000000002" not in statement
    assert params["upload_id"] == "00000000-0000-4000-8000-000000000002"


def test_latest_bom_uses_successful_publication_order(recording_engine) -> None:
    """get_latest_bom_snapshot orders by PUBLICATION_ORDER DESC with LIMIT 1."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(source_artifact_row())
    result = UploadRepository(recording_engine).get_latest_bom_snapshot()
    assert result is not None
    statement, params = recording_engine.executions[-1]
    assert 'ORDER BY U."PUBLICATION_ORDER" DESC' in statement
    assert "LIMIT 1" in statement
    assert params == {"artifact_kind": "BOM_WORKBOOK"}


def test_latest_bom_returns_none_when_no_upload_exists(recording_engine) -> None:
    """get_latest_bom_snapshot returns None when no upload has a BOM artifact."""
    from app.persistence.upload_repository import UploadRepository

    result = UploadRepository(recording_engine).get_latest_bom_snapshot()
    assert result is None
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


def test_latest_bom_uses_connect_not_begin(recording_engine) -> None:
    """get_latest_bom_snapshot uses engine.connect() (read-only query)."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(source_artifact_row())
    UploadRepository(recording_engine).get_latest_bom_snapshot()
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


def test_latest_bom_ties_resolved_by_upload_id_desc(recording_engine) -> None:
    """get_latest_bom_snapshot includes UPLOAD_ID DESC as a deterministic tie-breaker."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(source_artifact_row())
    UploadRepository(recording_engine).get_latest_bom_snapshot()
    statement, _ = recording_engine.executions[-1]
    # The tie-breaker column after PUBLICATION_ORDER ensures determinism.
    assert "PUBLICATION_ORDER" in statement


# ===========================================================================
# Suite 2: Idempotent-replay (no transaction)
# ===========================================================================


def test_same_request_returns_existing_without_a_write(recording_engine) -> None:
    """publish_upload returns existing record without begin() when request_id matches."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row())
    result = UploadRepository(recording_engine).publish_upload(upload_publication())
    assert result.created is False
    assert result.upload.request_id == upload_publication().request_id
    assert recording_engine.begin_count == 0


def test_same_request_replay_uses_connect_for_preflight(recording_engine) -> None:
    """publish_upload opens exactly one connect() for the preflight lookup on replay."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(upload_row())
    UploadRepository(recording_engine).publish_upload(upload_publication())
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


# ===========================================================================
# Suite 3: Preflight invariants
# ===========================================================================


def test_ecn_artifact_must_be_ecn_json_kind(recording_engine) -> None:
    """publish_upload raises ValueError before querying when ECN artifact has wrong kind."""
    from app.persistence.upload_repository import UploadRepository

    bad_ecn = make_source_artifact(artifact_kind=ArtifactKind.BOM_WORKBOOK)
    pub = upload_publication(ecn_artifact=bad_ecn)
    with pytest.raises(ValueError, match="ECN_JSON"):
        UploadRepository(recording_engine).publish_upload(pub)
    assert recording_engine.connect_count == 0
    assert recording_engine.begin_count == 0


def test_ecn_artifact_must_be_valid_parse_state(recording_engine) -> None:
    """publish_upload raises ValueError when ECN artifact parse state is INVALID."""
    from app.persistence.upload_repository import UploadRepository

    bad_ecn = make_source_artifact(
        artifact_kind=ArtifactKind.ECN_JSON,
        parse_state=ParseState.INVALID,
        parse_error_code="PARSE_FAILED",
    )
    pub = upload_publication(ecn_artifact=bad_ecn)
    with pytest.raises(ValueError, match="VALID"):
        UploadRepository(recording_engine).publish_upload(pub)
    assert recording_engine.connect_count == 0


def test_bom_artifact_must_be_bom_workbook_kind(recording_engine) -> None:
    """publish_upload raises ValueError when BOM artifact has wrong kind."""
    from app.persistence.upload_repository import UploadRepository

    bad_bom = make_source_artifact(artifact_kind=ArtifactKind.ECN_JSON)
    pub = upload_publication(bom_artifact=bad_bom)
    with pytest.raises(ValueError, match="BOM_WORKBOOK"):
        UploadRepository(recording_engine).publish_upload(pub)
    assert recording_engine.connect_count == 0


def test_bom_rows_required_when_not_reused(recording_engine) -> None:
    """publish_upload raises ValueError when bom_reused=False but new_bom_rows is empty."""
    from app.persistence.upload_repository import UploadRepository
    from app.persistence.records import UploadPublication

    pub = upload_publication(bom_reused=False)
    # Replace new_bom_rows with empty tuple to trigger the invariant check.
    pub_invalid = UploadPublication(
        upload_id=pub.upload_id,
        request_id=pub.request_id,
        ecn_artifact=pub.ecn_artifact,
        bom_artifact=pub.bom_artifact,
        new_bom_rows=(),
        rule_artifact=pub.rule_artifact,
        bom_reused=False,
        ecns=pub.ecns,
        actor_id=pub.actor_id,
        parser_version=pub.parser_version,
        deterministic_logic_version=pub.deterministic_logic_version,
        rule_evaluator_version=pub.rule_evaluator_version,
        created_at=pub.created_at,
    )
    with pytest.raises(ValueError, match="bom_rows"):
        UploadRepository(recording_engine).publish_upload(pub_invalid)
    assert recording_engine.connect_count == 0


def test_bom_rows_must_be_empty_when_reused(recording_engine) -> None:
    """publish_upload raises ValueError when bom_reused=True but new_bom_rows is non-empty."""
    from app.persistence.upload_repository import UploadRepository
    from app.persistence.records import UploadPublication
    from tests.fixtures.builders import make_bom_row

    pub = upload_publication(bom_reused=True)
    bom_row_obj = make_bom_row(artifact_id=pub.bom_artifact.artifact_id)
    pub_invalid = UploadPublication(
        upload_id=pub.upload_id,
        request_id=pub.request_id,
        ecn_artifact=pub.ecn_artifact,
        bom_artifact=pub.bom_artifact,
        new_bom_rows=(bom_row_obj,),
        rule_artifact=pub.rule_artifact,
        bom_reused=True,
        ecns=pub.ecns,
        actor_id=pub.actor_id,
        parser_version=pub.parser_version,
        deterministic_logic_version=pub.deterministic_logic_version,
        rule_evaluator_version=pub.rule_evaluator_version,
        created_at=pub.created_at,
    )
    with pytest.raises(ValueError, match="bom_rows"):
        UploadRepository(recording_engine).publish_upload(pub_invalid)
    assert recording_engine.connect_count == 0


def test_actor_id_must_not_exceed_64_chars(recording_engine) -> None:
    """publish_upload raises ValueError when actor_id exceeds 64 characters."""
    from app.persistence.upload_repository import UploadRepository

    pub = upload_publication(actor_id="x" * 65)
    with pytest.raises(ValueError, match="actor_id"):
        UploadRepository(recording_engine).publish_upload(pub)
    assert recording_engine.connect_count == 0


def test_analysis_job_id_must_be_a_canonical_uuid(recording_engine) -> None:
    """publish_upload rejects an invalid analysis-job ID before database access."""
    from app.persistence.upload_repository import UploadRepository

    pub = upload_publication()
    bad_item = replace(pub.ecns[0], job_id="not-a-canonical-uuid")
    bad_pub = replace(pub, ecns=(bad_item,))

    with pytest.raises(ValueError, match="canonical UUID"):
        UploadRepository(recording_engine).publish_upload(bad_pub)
    assert recording_engine.connect_count == 0


def test_evidence_must_belong_to_its_normalized_ecn(recording_engine) -> None:
    """publish_upload rejects evidence linked to a different normalized ECN."""
    from app.persistence.upload_repository import UploadRepository

    pub = upload_publication()
    bad_evidence = replace(
        pub.ecns[0].evidence,
        ecn_row_id="00000000-0000-4000-8000-000000000099",
    )
    bad_item = replace(pub.ecns[0], evidence=bad_evidence)
    bad_pub = replace(pub, ecns=(bad_item,))

    with pytest.raises(ValueError, match="evidence ecn_row_id"):
        UploadRepository(recording_engine).publish_upload(bad_pub)
    assert recording_engine.connect_count == 0


def test_evidence_rule_artifact_id_must_match_publication(recording_engine) -> None:
    """publish_upload rejects a context that names an unsupplied rule artifact."""
    from app.persistence.upload_repository import UploadRepository

    pub = upload_publication()
    bad_context = replace(
        pub.ecns[0].evidence.context,
        rule_artifact_id="00000000-0000-4000-8000-000000000099",
    )
    bad_evidence = replace(pub.ecns[0].evidence, context=bad_context)
    bad_item = replace(pub.ecns[0], evidence=bad_evidence)
    bad_pub = replace(pub, ecns=(bad_item,))

    with pytest.raises(ValueError, match="rule_artifact_id"):
        UploadRepository(recording_engine).publish_upload(bad_pub)
    assert recording_engine.connect_count == 0


def test_ecn_source_ordinals_must_be_unique(recording_engine) -> None:
    """publish_upload raises ValueError when two ECNs share the same source_ordinal."""
    from app.persistence.upload_repository import UploadRepository
    from app.persistence.records import UploadPublication, EcnPublication
    from tests.fixtures.builders import make_evidence_bundle, make_evidence_context
    from app.ecn.api_models import ImpactCounts

    pub = upload_publication(ecn_count=2)
    # Reuse ordinal 0 for both ECNs to violate uniqueness.
    ecn0 = pub.ecns[0]
    from app.ecn.models import NormalizedEcn
    duplicate_ecn = NormalizedEcn(
        ecn_row_id="00000000-0000-4000-8000-000000000021",
        artifact_id=pub.ecn_artifact.artifact_id,
        source_ordinal=0,  # duplicate ordinal
        ecn_number=None,
        title=None,
        description_summary=None,
        description_text=None,
        affected_objects=(),
        resulting_objects=(),
        source_json={},
        normalization_qualifiers=(),
    )
    ctx = make_evidence_context(
        upload_id=pub.upload_id,
        ecn_artifact_id=pub.ecn_artifact.artifact_id,
        bom_artifact_id=pub.bom_artifact.artifact_id,
    )
    evidence = make_evidence_bundle(ecn_row_id="00000000-0000-4000-8000-000000000021", context=ctx)
    counts = ImpactCounts(
        affected_item_count=0, raw_ecn_match_count=0, ebom_match_count=0,
        ebom_context_count=0, mbom_match_count=0, warning_count=0,
    )
    dup_item = EcnPublication(
        ecn=duplicate_ecn,
        change_candidates=(),
        evidence=evidence,
        counts=counts,
        job_id="00000000-0000-4000-8000-000000000031",
        prompt_version="v1",
        output_schema_version="v1",
        model_name="gpt-4",
        model_config_json="{}",
    )
    pub_dup = UploadPublication(
        upload_id=pub.upload_id,
        request_id=pub.request_id,
        ecn_artifact=pub.ecn_artifact,
        bom_artifact=pub.bom_artifact,
        new_bom_rows=pub.new_bom_rows,
        rule_artifact=None,
        bom_reused=False,
        ecns=(ecn0, dup_item),
        actor_id=None,
        parser_version=pub.parser_version,
        deterministic_logic_version=pub.deterministic_logic_version,
        rule_evaluator_version=pub.rule_evaluator_version,
        created_at=pub.created_at,
    )
    with pytest.raises(ValueError, match="ordinal"):
        UploadRepository(recording_engine).publish_upload(pub_dup)
    assert recording_engine.connect_count == 0


# ===========================================================================
# Suite 4: Atomic publication
# ===========================================================================


def test_new_upload_is_one_transaction_and_first_order_is_one(recording_engine) -> None:
    """publish_upload uses exactly one begin() and allocates PUBLICATION_ORDER=1."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)   # preflight find_by_request_id → no existing
    recording_engine.queue_scalar(1)       # COALESCE(MAX(PUBLICATION_ORDER), 0) + 1
    publication = upload_publication(bom_reused=False, ecn_count=2)
    result = UploadRepository(recording_engine).publish_upload(publication)
    assert result.created is True
    assert result.upload.publication_order == 1
    assert recording_engine.begin_count == 1
    inserted_tables = [t for sql, _ in recording_engine.executions if (t := insert_target(sql))]
    assert inserted_tables == [
        "ECN_IMPACT_SOURCE_ARTIFACT",  # ECN JSON artifact
        "ECN_IMPACT_SOURCE_ARTIFACT",  # new BOM workbook artifact
        "ECN_IMPACT_BOM_ROW",
        "ECN_IMPACT_UPLOAD",
        "ECN_IMPACT_ECN",  # Large NCLOB rows are bound independently within this transaction.
        "ECN_IMPACT_ECN",
        "ECN_IMPACT_RDF_PROJECTION",  # Derived PENDING status shares the source transaction.
        "ECN_IMPACT_ANALYSIS_JOB",
    ]


def test_new_upload_inserts_pending_rdf_statuses_after_ecns_before_jobs(recording_engine) -> None:
    """Every new ECN receives one rebuildable PENDING status in its source transaction."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    publication = upload_publication(bom_reused=False, ecn_count=2)

    result = UploadRepository(recording_engine).publish_upload(publication)

    inserted_tables = [t for sql, _ in recording_engine.executions if (t := insert_target(sql))]
    ecn_positions = [index for index, table in enumerate(inserted_tables) if table == "ECN_IMPACT_ECN"]
    rdf_position = inserted_tables.index("ECN_IMPACT_RDF_PROJECTION")
    job_position = inserted_tables.index("ECN_IMPACT_ANALYSIS_JOB")
    status_params = next(
        params
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "RDF_PROJECTION" in sql
    )

    assert result.projection_ecn_ids == tuple(item.ecn.ecn_row_id for item in publication.ecns)
    assert max(ecn_positions) < rdf_position < job_position
    assert [row["STATUS"] for row in status_params] == ["PENDING", "PENDING"]
    assert all(len(row["SOURCE_SHA256"]) == 64 for row in status_params)
    assert all(row["PROJECTION_VERSION"] for row in status_params)


def test_reused_bom_references_existing_artifact_without_copying_rows(recording_engine) -> None:
    """publish_upload with bom_reused=True inserts 1 artifact and no BOM rows."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(8)
    publication = upload_publication(bom_reused=True, ecn_count=1)
    UploadRepository(recording_engine).publish_upload(publication)
    statements = [sql for sql, _ in recording_engine.executions]
    # Only the ECN JSON artifact is inserted; BOM artifact already exists.
    assert sum(
        "SOURCE_ARTIFACT" in sql and sql.startswith("INSERT") for sql in statements
    ) == 1
    assert not any(sql.startswith("INSERT") and "BOM_ROW" in sql for sql in statements)


def test_publication_order_is_bound_from_coalesce_result(recording_engine) -> None:
    """publish_upload pins publication_order from the database-allocated scalar."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(42)
    result = UploadRepository(recording_engine).publish_upload(upload_publication())
    assert result.created is True
    assert result.upload.publication_order == 42


def test_publication_creates_one_job_per_ecn(recording_engine) -> None:
    """publish_upload inserts exactly one ANALYSIS_JOB row per ECN occurrence."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    publication = upload_publication(bom_reused=False, ecn_count=3)
    UploadRepository(recording_engine).publish_upload(publication)
    # Find the job INSERT execution and verify its params list length.
    job_params = [
        params
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "ANALYSIS_JOB" in sql
    ]
    assert len(job_params) == 1
    assert isinstance(job_params[0], list)
    assert len(job_params[0]) == 3


def test_publication_pins_active_reference_datasets_in_same_transaction(
    recording_engine,
) -> None:
    """Write Settings provenance links after the upload and before ECN rows."""
    from dataclasses import replace

    from app.ecn.reference_data.models import ReferenceDatasetPin, ReferenceDatasetType
    from app.persistence.upload_repository import UploadRepository

    publication = replace(
        upload_publication(),
        dataset_links=(
            ReferenceDatasetPin(
                ReferenceDatasetType.BOM,
                "00000000-0000-4000-8000-000000000071",
            ),
            ReferenceDatasetPin(
                ReferenceDatasetType.MATERIAL_MASTER,
                "00000000-0000-4000-8000-000000000072",
            ),
        ),
    )
    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    UploadRepository(recording_engine).publish_upload(publication)
    link_inserts = [
        (sql, params)
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "UPLOAD_DATASET" in sql
    ]
    assert len(link_inserts) == 1
    assert [row["DATASET_TYPE"] for row in link_inserts[0][1]] == [
        "BOM",
        "MATERIAL_MASTER",
    ]
    assert recording_engine.begin_count == 1


def test_all_six_impact_counts_are_written(recording_engine) -> None:
    """publish_upload stores all six ImpactCounts fields in the ECN insert params."""
    from app.persistence.upload_repository import UploadRepository
    from app.ecn.api_models import ImpactCounts
    from app.persistence.records import EcnPublication, UploadPublication
    from tests.fixtures.builders import make_normalized_ecn, make_evidence_bundle, make_evidence_context

    pub = upload_publication(ecn_count=1)
    ctx = make_evidence_context(
        upload_id=pub.upload_id,
        ecn_artifact_id=pub.ecn_artifact.artifact_id,
        bom_artifact_id=pub.bom_artifact.artifact_id,
    )
    evidence = make_evidence_bundle(
        ecn_row_id="00000000-0000-4000-8000-000000000020",
        context=ctx,
    )
    ecn_item = make_normalized_ecn(
        ecn_row_id="00000000-0000-4000-8000-000000000020",
        artifact_id=pub.ecn_artifact.artifact_id,
        source_ordinal=0,
    )
    counts = ImpactCounts(
        affected_item_count=10,
        raw_ecn_match_count=5,
        ebom_match_count=3,
        ebom_context_count=7,
        mbom_match_count=2,
        warning_count=1,
    )
    ecn_pub = EcnPublication(
        ecn=ecn_item,
        change_candidates=(),
        evidence=evidence,
        counts=counts,
        job_id="00000000-0000-4000-8000-000000000030",
        prompt_version="v1",
        output_schema_version="v1",
        model_name="gpt-4",
        model_config_json="{}",
    )
    full_pub = UploadPublication(
        upload_id=pub.upload_id,
        request_id=pub.request_id,
        ecn_artifact=pub.ecn_artifact,
        bom_artifact=pub.bom_artifact,
        new_bom_rows=pub.new_bom_rows,
        rule_artifact=None,
        bom_reused=False,
        ecns=(ecn_pub,),
        actor_id=None,
        parser_version=pub.parser_version,
        deterministic_logic_version=pub.deterministic_logic_version,
        rule_evaluator_version=pub.rule_evaluator_version,
        created_at=pub.created_at,
    )

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    UploadRepository(recording_engine).publish_upload(full_pub)

    ecn_params_list = [
        params
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "ECN_IMPACT_ECN" in sql
    ]
    assert len(ecn_params_list) == 1
    row = ecn_params_list[0][0]
    assert row["AFFECTED_ITEM_COUNT"] == 10
    assert row["RAW_ECN_MATCH_COUNT"] == 5
    assert row["EBOM_MATCH_COUNT"] == 3
    assert row["EBOM_CONTEXT_COUNT"] == 7
    assert row["MBOM_MATCH_COUNT"] == 2
    assert row["WARNING_COUNT"] == 1


def test_upload_insert_contains_imported_ecn_count(recording_engine) -> None:
    """publish_upload stores IMPORTED_ECN_COUNT = len(ecns) in the upload row."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    UploadRepository(recording_engine).publish_upload(upload_publication(ecn_count=4))
    upload_params = [
        params
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "ECN_IMPACT_UPLOAD" in sql
    ]
    assert len(upload_params) == 1
    assert upload_params[0]["IMPORTED_ECN_COUNT"] == 4


def test_repeated_file_hash_with_new_request_id_inserts_new_ecn_rows(recording_engine) -> None:
    """A new request_id always inserts new ECN rows even when source hashes repeat."""
    from app.persistence.upload_repository import UploadRepository

    # Two separate publications with different request IDs but same content.
    pub_a = upload_publication(request_id="00000000-0000-4000-8000-000000000007")
    pub_b = upload_publication(
        request_id="00000000-0000-4000-8000-0000000000aa",
        upload_id="00000000-0000-4000-8000-0000000000bb",
    )

    for pub in (pub_a, pub_b):
        recording_engine.queue_mapping(None)
        recording_engine.queue_scalar(1)
        result = UploadRepository(recording_engine).publish_upload(pub)
        assert result.created is True

    ecn_inserts = [
        sql for sql, _ in recording_engine.executions
        if sql.startswith("INSERT") and "ECN_IMPACT_ECN" in sql
    ]
    assert len(ecn_inserts) == 2


def test_integrity_error_on_request_id_returns_existing(recording_engine) -> None:
    """publish_upload handles REQUEST_ID race: rolls back and returns the existing upload."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)   # preflight → no record yet
    recording_engine.queue_scalar(1)       # COALESCE MAX
    recording_engine.queue_exception(
        IntegrityError("Unique constraint", params=None, orig=None)
    )
    # Post-rollback lookup returns the winner's row.
    recording_engine.queue_mapping(upload_row())

    result = UploadRepository(recording_engine).publish_upload(upload_publication())
    assert result.created is False
    assert result.upload.upload_id == "00000000-0000-4000-8000-000000000002"
    assert recording_engine.begin_count == 1


def test_integrity_error_on_publication_order_raises_conflict(recording_engine) -> None:
    """publish_upload raises PersistenceStateError for non-request-id uniqueness violations."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)   # preflight → no existing
    recording_engine.queue_scalar(1)       # COALESCE MAX
    recording_engine.queue_exception(
        IntegrityError("Unique constraint on PUBLICATION_ORDER", params=None, orig=None)
    )
    # Post-rollback find_by_request_id returns None → not a request race.
    recording_engine.queue_mapping(None)

    with pytest.raises(PersistenceStateError) as exc_info:
        UploadRepository(recording_engine).publish_upload(upload_publication())
    assert "CONFLICT" in exc_info.value.code


def test_no_connection_held_after_publish_upload(recording_engine) -> None:
    """publish_upload leaves no open connection after returning."""
    from app.persistence.upload_repository import UploadRepository

    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    UploadRepository(recording_engine).publish_upload(upload_publication())
    # If a connection were held, it would have been entered but not exited.
    # Asserting connect_count == 1 (preflight) and begin_count == 1 is sufficient.
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 1


def test_publication_result_has_created_true_and_correct_fields(recording_engine) -> None:
    """publish_upload result carries upload fields from the publication record."""
    from app.persistence.upload_repository import UploadRepository

    publication = upload_publication(bom_reused=False, ecn_count=1)
    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(5)
    result = UploadRepository(recording_engine).publish_upload(publication)
    assert result.created is True
    assert result.upload.upload_id == publication.upload_id
    assert result.upload.request_id == publication.request_id
    assert result.upload.ecn_artifact_id == publication.ecn_artifact.artifact_id
    assert result.upload.bom_artifact_id == publication.bom_artifact.artifact_id
    assert result.upload.publication_order == 5
    assert result.upload.bom_reused is False
    assert result.upload.imported_ecn_count == 1


def test_ecn_insert_stores_canonical_source_json(recording_engine) -> None:
    """publish_upload stores SOURCE_JSON as canonical JSON text in the ECN row."""
    from app.persistence.upload_repository import UploadRepository
    import json

    publication = upload_publication(ecn_count=1)
    recording_engine.queue_mapping(None)
    recording_engine.queue_scalar(1)
    UploadRepository(recording_engine).publish_upload(publication)

    ecn_params_list = [
        params
        for sql, params in recording_engine.executions
        if sql.startswith("INSERT") and "ECN_IMPACT_ECN" in sql
    ]
    assert len(ecn_params_list) == 1
    source_json_str = ecn_params_list[0][0]["SOURCE_JSON"]
    # Must be valid JSON; default ECN has an empty source_json dict.
    assert json.loads(source_json_str) == {}


# ===========================================================================
# Regression guards: dialect-case-insensitive row mappers
# ===========================================================================
# These tests prove that upload_record_from_row and artifact_from_row work
# correctly when the row keys are returned in LOWERCASE (as sqlalchemy-hana
# does against a real HANA instance) rather than UPPERCASE (as the recording
# fake uses).  A failure here means normalize_row_keys is not applied at the
# right boundary and the live integration test would regress.


def test_upload_record_from_row_tolerates_lowercase_keys() -> None:
    """upload_record_from_row returns the correct record when row keys are lowercase.

    Simulates the sqlalchemy-hana dialect returning lowercase column identifiers
    instead of the uppercase names used in the SELECT projection.  This is the
    exact class of failure that caused test_live_hana_repository_round_trip to
    fail with NoSuchColumnError on 'UPLOAD_ID'.
    """

    from app.persistence.codec import upload_record_from_row
    from app.persistence.records import UploadRecord

    # Build a normal uppercase-keyed row (as produced by the RecordingEngine fakes)
    # and lowercase ALL keys to simulate the HANA driver casing.
    uppercase_row: dict[str, object] = {
        "UPLOAD_ID": "00000000-0000-4000-8000-000000000002",
        "REQUEST_ID": "00000000-0000-4000-8000-000000000007",
        "PUBLICATION_ORDER": 1,
        "ECN_ARTIFACT_ID": "00000000-0000-4000-8000-000000000001",
        "BOM_ARTIFACT_ID": "00000000-0000-4000-8000-000000000006",
        "RULE_ARTIFACT_ID": None,
        "BOM_REUSED": 0,
        "IMPORTED_ECN_COUNT": 2,
        "CREATED_AT": datetime(2026, 9, 14),  # naive UTC as HANA returns it
    }
    lowercase_row = {k.lower(): v for k, v in uppercase_row.items()}

    record = upload_record_from_row(lowercase_row)

    assert isinstance(record, UploadRecord)
    assert record.upload_id == "00000000-0000-4000-8000-000000000002"
    assert record.request_id == "00000000-0000-4000-8000-000000000007"
    assert record.publication_order == 1
    assert record.ecn_artifact_id == "00000000-0000-4000-8000-000000000001"
    assert record.bom_artifact_id == "00000000-0000-4000-8000-000000000006"
    assert record.rule_artifact_id is None
    assert record.bom_reused is False
    assert record.imported_ecn_count == 2
    assert record.created_at.tzinfo is not None  # from_hana_timestamp restores UTC


def test_artifact_from_row_tolerates_lowercase_keys() -> None:
    """artifact_from_row returns the correct record when row keys are lowercase.

    Covers both the plain (empty prefix) call path used by UploadRepository and
    ArtifactRepository, and confirms that the normalization does not alter any
    column values (especially the CONTENT_BYTES BLOB and NORMALIZED_CONTENT NCLOB).
    """
    from hashlib import sha256

    from app.ecn.models import ArtifactKind, ParseState
    from app.persistence.codec import artifact_from_row

    content = b"[{\"ecn\": 1}]"
    uppercase_row: dict[str, object] = {
        "ARTIFACT_ID": "00000000-0000-4000-8000-000000000001",
        "ARTIFACT_KIND": "ECN_JSON",
        "FILE_NAME": "ecn.json",
        "MEDIA_TYPE": "application/json",
        "BYTE_SIZE": len(content),
        "SHA256": sha256(content).hexdigest(),
        "CONTENT_BYTES": content,
        "NORMALIZED_CONTENT": "[{}]",
        "PARSE_STATE": "VALID",
        "PARSE_ERROR_CODE": None,
        "CREATED_AT": datetime(2026, 9, 14),
    }
    lowercase_row = {k.lower(): v for k, v in uppercase_row.items()}

    artifact = artifact_from_row(lowercase_row)

    assert artifact.artifact_id == "00000000-0000-4000-8000-000000000001"
    assert artifact.artifact_kind is ArtifactKind.ECN_JSON
    assert artifact.file_name == "ecn.json"
    assert artifact.content_bytes == content
    assert artifact.parse_state is ParseState.VALID
    assert artifact.parse_error_code is None
    assert artifact.created_at.tzinfo is not None
