"""Unit tests for persistence records, canonical codecs, and ArtifactRepository.

Tests are grouped into three suites that follow the TDD steps in the Task 5
brief:

1. Codec and record correctness — canonical JSON, LOB boundary, UTC handling,
   token-usage compatibility, and discriminator invariants.
2. Artifact lookup and rule publication — SQL text, transaction boundaries,
   and bound-parameter assertions via RecordingEngine.
3. Chunked BOM-row reading — connection independence per page, stable ordering,
   offset arithmetic, and error propagation.

No live HANA connection is required; all assertions run against recording fakes.

Example run (full suite)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_artifact_repository.py

Example run (codec only)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_artifact_repository.py \\
        -k 'canonical or lob or token or round_trip or attempt_seed or discriminator'
"""

from __future__ import annotations

from dataclasses import fields
from datetime import date, timezone
from hashlib import sha256

import pytest
from app.ecn.errors import PersistenceStateError
from app.ecn.models import (
    ArtifactKind,
    BomKind,
    BomRowClassification,
    ParseState,
)
from app.observability.llm_usage_logging import TokenUsage
from app.persistence.codec import (
    artifact_from_row,
    artifact_to_row,
    bom_row_from_row,
    bom_row_to_row,
    canonical_json,
    evidence_bundle_from_json,
    read_lob_bytes,
    read_lob_text,
    usage_to_columns,
)
from app.persistence.records import (
    AttemptSeed,
    AttemptStart,
    ClaimJobResult,
    ClaimJobStatus,
    RetryQueueResult,
    RetryQueueStatus,
)

from tests.fixtures.builders import (
    analysis_payload,
    bom_row_mapping,
    make_bom_row,
    make_evidence_bundle,
    make_source_artifact,
    source_artifact,
    source_artifact_row,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_engine():
    """Return a fresh RecordingEngine for each test.

    Returns:
        A ``RecordingEngine`` with empty counters and an empty row queue.
    """
    from tests.fixtures.persistence import RecordingEngine

    return RecordingEngine()


# ---------------------------------------------------------------------------
# Helpers used in multiple tests
# ---------------------------------------------------------------------------


class ReadableLob:
    """Expose the minimum hdbcli-like LOB interface used by the codec.

    Args:
        value: The string to return from the ``read()`` call.
    """

    def __init__(self, value: str) -> None:
        """Store the string that the fake LOB returns from ``read``."""
        self.value = value

    def read(self) -> str:
        """Return the stored value, mimicking an hdbcli NCLOB object.

        Returns:
            The stored string value.
        """
        return self.value


class ReadableBytesLob:
    """LOB fake whose ``read()`` returns bytes instead of a string.

    Args:
        value: The bytes to return from the ``read()`` call.
    """

    def __init__(self, value: bytes) -> None:
        """Store the bytes that the fake LOB returns from ``read``."""
        self.value = value

    def read(self) -> bytes:
        """Return the stored bytes, mimicking an hdbcli BLOB object.

        Returns:
            The stored bytes value.
        """
        return self.value


# ===========================================================================
# Suite 1: Codec and record correctness
# ===========================================================================


def test_canonical_json_is_unicode_stable_and_key_sorted() -> None:
    """canonical_json emits sorted keys with no ASCII escaping."""
    assert canonical_json({"z": 1, "a": "é"}) == '{"a":"é","z":1}'
    assert read_lob_text(ReadableLob("content")) == "content"


def test_current_four_field_token_usage_maps_without_inventing_future_counts() -> None:
    """cached_input_tokens maps to CACHE_READ; absent fields remain None."""
    columns = usage_to_columns(
        TokenUsage(
            input_tokens=11, cached_input_tokens=3, output_tokens=5, total_tokens=16
        ),
        usage_available=True,
    )
    assert columns.input_tokens == 11
    assert columns.cache_read_input_tokens == 3
    assert columns.cache_write_input_tokens is None
    assert columns.input_total_tokens is None
    assert columns.output_tokens == 5
    assert columns.total_tokens == 16


def test_unavailable_usage_is_null_even_when_legacy_defaults_are_zero() -> None:
    """usage_available=False produces all-None columns regardless of object."""
    columns = usage_to_columns(TokenUsage(), usage_available=False)
    assert columns.usage_available is False
    assert all(
        value is None
        for name, value in vars(columns).items()
        if name != "usage_available"
    )


def test_attempt_seed_cannot_override_the_persisted_model() -> None:
    """AttemptSeed lacks model_name; AttemptStart adds it after pinning."""
    assert "model_name" not in {field.name for field in fields(AttemptSeed)}
    assert "model_name" in {field.name for field in fields(AttemptStart)}


def test_result_discriminators_reject_impossible_payload_shapes() -> None:
    """ClaimJobResult and RetryQueueResult enforce claimed/queued invariants."""
    assert ClaimJobResult(ClaimJobStatus.NOT_FOUND, None).claimed_job is None
    assert RetryQueueResult(RetryQueueStatus.ECN_NOT_FOUND, None).job_id is None
    with pytest.raises(ValueError, match="requires the claimed job"):
        ClaimJobResult(ClaimJobStatus.CLAIMED, None)
    with pytest.raises(ValueError, match="requires a job ID"):
        RetryQueueResult(RetryQueueStatus.QUEUED, None)


# --- LOB boundary tests -----------------------------------------------------


def test_read_lob_text_accepts_str_bytes_and_readable() -> None:
    """read_lob_text handles all three accepted LOB shapes."""
    assert read_lob_text("hello") == "hello"
    assert read_lob_text(b"bytes") == "bytes"
    assert read_lob_text(memoryview(b"buffer")) == "buffer"
    assert read_lob_text(ReadableLob("from read")) == "from read"
    assert read_lob_text(ReadableBytesLob(b"binary")) == "binary"


def test_read_lob_bytes_accepts_bytes_str_and_readable() -> None:
    """read_lob_bytes handles all three accepted LOB shapes."""
    assert read_lob_bytes(b"raw") == b"raw"
    assert read_lob_bytes(memoryview(b"hana-driver-buffer")) == b"hana-driver-buffer"
    assert read_lob_bytes("text") == b"text"
    assert read_lob_bytes(ReadableBytesLob(b"from read")) == b"from read"
    assert read_lob_bytes(ReadableLob("readable text")) == b"readable text"


def test_read_lob_text_rejects_unsupported_type() -> None:
    """read_lob_text raises PersistenceStateError for integers and other types."""
    with pytest.raises(PersistenceStateError):
        read_lob_text(42)  # type: ignore[arg-type]
    with pytest.raises(PersistenceStateError):
        read_lob_text(["not", "a", "lob"])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [b"\xff", ReadableBytesLob(b"\xff")])
def test_read_lob_text_wraps_invalid_utf8_as_safe_persistence_error(
    value: object,
) -> None:
    """Convert invalid NCLOB UTF-8 into the stable safe persistence error."""
    with pytest.raises(PersistenceStateError) as caught:
        read_lob_text(value)
    assert caught.value.code == "LOB_READ_ERROR"
    assert "\\xff" not in caught.value.public_message


def test_read_lob_bytes_rejects_unsupported_type() -> None:
    """read_lob_bytes raises PersistenceStateError for non-LOB types."""
    with pytest.raises(PersistenceStateError):
        read_lob_bytes(3.14)  # type: ignore[arg-type]


# --- Token usage edge cases -------------------------------------------------


def test_usage_to_columns_raises_when_available_but_usage_is_none() -> None:
    """usage_available=True with usage=None is a caller contract violation."""
    with pytest.raises(ValueError):
        usage_to_columns(None, usage_available=True)


def test_usage_to_columns_zero_available_counts_persist() -> None:
    """Explicit zero counts with usage_available=True are stored, not treated as absent."""
    columns = usage_to_columns(
        TokenUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0, total_tokens=0),
        usage_available=True,
    )
    assert columns.usage_available is True
    assert columns.input_tokens == 0
    assert columns.output_tokens == 0


# --- Round-trip: SourceArtifact ---------------------------------------------


def test_source_artifact_round_trip_via_row_mapper() -> None:
    """artifact_to_row / artifact_from_row losslessly round-trips a SourceArtifact."""
    original = make_source_artifact(
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        file_name="bom.xlsx",
        content_bytes=b"\x00\x01\x02",
        sha256=sha256(b"\x00\x01\x02").hexdigest(),
        normalized_content=None,
        parse_state=ParseState.INVALID,
        parse_error_code="PARSE_FAILED",
    )
    row = artifact_to_row(original)
    # CREATED_AT must be naive UTC for the DB; restore tz for reconstruction.
    assert row["CREATED_AT"].tzinfo is None
    # Reconstruct from row; CREATED_AT is the naive value from to_hana_timestamp.
    restored = artifact_from_row(row)
    assert restored.artifact_id == original.artifact_id
    assert restored.artifact_kind is ArtifactKind.BOM_WORKBOOK
    assert restored.content_bytes == b"\x00\x01\x02"
    assert restored.normalized_content is None
    assert restored.parse_state is ParseState.INVALID
    assert restored.parse_error_code == "PARSE_FAILED"
    assert restored.created_at.tzinfo is timezone.utc


def test_artifact_from_row_accepts_lob_objects_for_blob_and_nclob() -> None:
    """artifact_from_row calls read_lob_bytes/text on LOB-shaped column values."""
    # BYTE_SIZE must match the actual content length so SourceArtifact.__post_init__
    # passes its invariant check.
    row = source_artifact_row(
        CONTENT_BYTES=ReadableBytesLob(b"raw"),
        NORMALIZED_CONTENT=ReadableLob("text"),
        BYTE_SIZE=3,
        SHA256=sha256(b"raw").hexdigest(),
    )
    restored = artifact_from_row(row)
    assert restored.content_bytes == b"raw"
    assert restored.normalized_content == "text"


def test_artifact_from_row_with_prefix_reads_prefixed_columns() -> None:
    """artifact_from_row prefix parameter maps prefixed column names."""
    row = {f"a.{k}": v for k, v in source_artifact_row().items()}
    restored = artifact_from_row(row, prefix="a.")
    assert restored.artifact_id == "00000000-0000-4000-8000-000000000001"


# --- Round-trip: BomRow -----------------------------------------------------


def test_bom_row_round_trip_with_scalar_fields() -> None:
    """bom_row_to_row / bom_row_from_row round-trips plain text and enum fields."""
    from decimal import Decimal

    original = make_bom_row(
        material_number="MAT-001",
        bom_kind=BomKind.MBOM_130,
        row_classification=BomRowClassification.CONFIRMED_DOCUMENT,
        quantity=Decimal("1.5"),
        valid_from_date=date(2026, 1, 1),
        valid_to_date=date(2099, 12, 31),
        is_documentation=True,
        is_migration_date=False,
    )
    row_dict = bom_row_to_row(original)
    restored = bom_row_from_row(row_dict)
    assert restored.material_number == "MAT-001"
    assert restored.bom_kind is BomKind.MBOM_130
    assert restored.row_classification is BomRowClassification.CONFIRMED_DOCUMENT
    assert restored.quantity == Decimal("1.5")
    assert restored.valid_from_date == date(2026, 1, 1)
    assert restored.valid_to_date == date(2099, 12, 31)
    assert restored.is_documentation is True
    assert restored.is_migration_date is False


def test_bom_row_round_trip_with_quality_and_row_json() -> None:
    """ROW_JSON and QUALITY_JSON encode and decode without loss."""
    original = make_bom_row(
        row_json=(
            {"column_index": 0, "raw_header": "Part", "normalized_header": "material_number", "value": "ABC"},
        ),
        quality_qualifiers=("INFERRED_LEVEL", "MISSING_PARENT"),
        quality_details={"INFERRED_LEVEL": {"reason": "no dotted level present"}},
    )
    row_dict = bom_row_to_row(original)
    restored = bom_row_from_row(row_dict)
    assert len(restored.row_json) == 1
    assert restored.row_json[0]["value"] == "ABC"
    assert "INFERRED_LEVEL" in restored.quality_qualifiers
    assert restored.quality_details["INFERRED_LEVEL"]["reason"] == "no dotted level present"


def test_bom_row_from_row_rejects_malformed_quality_json() -> None:
    """Malformed QUALITY_JSON raises PersistenceStateError."""
    row = bom_row_mapping(1, QUALITY_JSON="{not valid json")
    with pytest.raises(PersistenceStateError):
        bom_row_from_row(row)


def test_bom_row_from_row_rejects_quality_json_missing_keys() -> None:
    """QUALITY_JSON without 'details' and 'qualifiers' raises PersistenceStateError."""
    row = bom_row_mapping(1, QUALITY_JSON='{"wrong_key": 1}')
    with pytest.raises(PersistenceStateError):
        bom_row_from_row(row)


# --- Round-trip: EvidenceBundle JSON ----------------------------------------


def test_evidence_bundle_json_round_trip() -> None:
    """evidence_bundle_from_json reverses canonical_json for an empty bundle."""
    bundle = make_evidence_bundle()
    json_str = canonical_json(bundle)
    restored = evidence_bundle_from_json(json_str)
    assert restored.ecn_row_id == bundle.ecn_row_id
    assert restored.evidence_hash == bundle.evidence_hash
    assert restored.context.upload_id == bundle.context.upload_id
    assert restored.context.rule_artifact_id is None
    assert restored.context.analysis_as_of_date is None


def test_evidence_bundle_json_round_trip_with_date() -> None:
    """analysis_as_of_date round-trips through canonical JSON as a date."""
    from tests.fixtures.builders import make_evidence_bundle, make_evidence_context

    ctx = make_evidence_context(analysis_as_of_date=date(2026, 6, 1))
    bundle = make_evidence_bundle(context=ctx)
    json_str = canonical_json(bundle)
    restored = evidence_bundle_from_json(json_str)
    assert restored.context.analysis_as_of_date == date(2026, 6, 1)


def test_evidence_bundle_round_trip_preserves_frozen_plant_assignment() -> None:
    """Keep the Plant Code Mapping frozen at import in canonical HANA JSON readback."""
    from tests.fixtures.builders import make_evidence_bundle, make_evidence_context

    frozen = {"mappings": [{"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}],
              "excluded_rule_states": ["To be Obsolete"]}
    context = make_evidence_context(plant_assignment=frozen)
    restored = evidence_bundle_from_json(canonical_json(make_evidence_bundle(context=context)))

    assert restored.context.plant_assignment == frozen


def test_evidence_bundle_from_json_rejects_malformed_payload() -> None:
    """evidence_bundle_from_json raises PersistenceStateError for bad JSON."""
    with pytest.raises(PersistenceStateError):
        evidence_bundle_from_json("{not json}")


# --- Round-trip: AnalysisResponse -------------------------------------------


def test_analysis_response_json_round_trip() -> None:
    """analysis_response_from_json reverses canonical_json of a full response."""
    from app.ecn.analysis_models import AnalysisResponse, SectionStatus
    from app.persistence.codec import analysis_response_from_json

    payload = analysis_payload("SUFFICIENT")
    response = AnalysisResponse.model_validate(payload)
    # Serialize via model_dump(mode="json") then canonical_json to produce
    # a deterministic string, then parse back.
    json_str = canonical_json(response.model_dump(mode="json"))
    restored = analysis_response_from_json(json_str)
    assert restored.sections.executive_summary.status is SectionStatus.SUFFICIENT


def test_analysis_response_from_json_rejects_malformed() -> None:
    """analysis_response_from_json raises PersistenceStateError for invalid payload."""
    from app.persistence.codec import analysis_response_from_json

    with pytest.raises(PersistenceStateError):
        analysis_response_from_json('{"sections": "bad"}')


# ===========================================================================
# Suite 2: Artifact lookup and rule publication
# ===========================================================================


def test_publish_rule_artifact_uses_one_short_transaction(recording_engine) -> None:
    """publish_rule_artifact opens exactly one begin() transaction."""
    from app.persistence.artifact_repository import ArtifactRepository

    artifact = source_artifact(artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK)
    repository = ArtifactRepository(recording_engine)
    assert repository.publish_rule_artifact(artifact) == artifact
    assert recording_engine.begin_count == 1
    statement, params = recording_engine.executions[0]
    assert statement.startswith('INSERT INTO "ECN_IMPACT_SOURCE_ARTIFACT"')
    assert params["CONTENT_BYTES"] == artifact.content_bytes


def test_current_rule_order_has_deterministic_tie_breaker(recording_engine) -> None:
    """get_current_rule_artifact orders by CREATED_AT then ARTIFACT_ID."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mapping(source_artifact_row())
    ArtifactRepository(recording_engine).get_current_rule_artifact()
    statement, params = recording_engine.executions[-1]
    assert 'ORDER BY "CREATED_AT" DESC, "ARTIFACT_ID" DESC' in statement
    assert params == {"artifact_kind": "CONDITIONAL_RULE_WORKBOOK"}


def test_get_artifact_returns_none_when_not_found(recording_engine) -> None:
    """get_artifact returns None when the row queue is empty (no row found)."""
    from app.persistence.artifact_repository import ArtifactRepository

    result = ArtifactRepository(recording_engine).get_artifact(
        "00000000-0000-4000-8000-000000000001"
    )
    assert result is None


def test_get_artifact_validates_uuid_before_querying(recording_engine) -> None:
    """get_artifact rejects non-canonical UUID strings before opening a connection."""
    from app.persistence.artifact_repository import ArtifactRepository

    with pytest.raises(ValueError):
        ArtifactRepository(recording_engine).get_artifact("not-a-uuid")
    assert recording_engine.connect_count == 0


def test_get_artifact_returns_reconstructed_artifact(recording_engine) -> None:
    """get_artifact reconstructs a SourceArtifact from the queued row."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mapping(source_artifact_row())
    artifact = ArtifactRepository(recording_engine).get_artifact(
        "00000000-0000-4000-8000-000000000001"
    )
    assert artifact is not None
    assert artifact.artifact_id == "00000000-0000-4000-8000-000000000001"
    assert artifact.created_at.tzinfo is timezone.utc


def test_publish_rule_artifact_rejects_wrong_kind(recording_engine) -> None:
    """publish_rule_artifact raises ValueError for non-rule artifact kinds."""
    from app.persistence.artifact_repository import ArtifactRepository

    ecn_artifact = source_artifact(artifact_kind=ArtifactKind.ECN_JSON)
    with pytest.raises(ValueError, match="conditional-rule"):
        ArtifactRepository(recording_engine).publish_rule_artifact(ecn_artifact)
    assert recording_engine.begin_count == 0


def test_publish_rule_artifact_rejects_invalid_parse_state(recording_engine) -> None:
    """publish_rule_artifact rejects rule artifacts with INVALID parse state."""
    from app.persistence.artifact_repository import ArtifactRepository

    bad = source_artifact(
        artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
        parse_state=ParseState.INVALID,
        parse_error_code="MISSING_SHEET",
    )
    with pytest.raises(ValueError, match="valid"):
        ArtifactRepository(recording_engine).publish_rule_artifact(bad)
    assert recording_engine.begin_count == 0


def test_get_artifact_uses_connect_not_begin(recording_engine) -> None:
    """get_artifact uses engine.connect() (not begin()) for read queries."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mapping(source_artifact_row())
    ArtifactRepository(recording_engine).get_artifact(
        "00000000-0000-4000-8000-000000000001"
    )
    assert recording_engine.connect_count == 1
    assert recording_engine.begin_count == 0


def test_get_artifact_parameterizes_the_id_value(recording_engine) -> None:
    """get_artifact passes the artifact_id as a bound parameter, not interpolated."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mapping(source_artifact_row())
    ArtifactRepository(recording_engine).get_artifact(
        "00000000-0000-4000-8000-000000000001"
    )
    statement, params = recording_engine.executions[0]
    # The UUID must not appear as a literal in the SQL text.
    assert "00000000-0000-4000-8000-000000000001" not in statement
    assert params["artifact_id"] == "00000000-0000-4000-8000-000000000001"


def test_artifact_exists_uses_lightweight_parameterized_projection(
    recording_engine,
) -> None:
    """Check identity existence without selecting either persisted LOB column."""
    from app.persistence.artifact_repository import ArtifactRepository

    artifact_id = "00000000-0000-4000-8000-000000000001"
    recording_engine.queue_mapping({"PRESENT": 1})
    assert ArtifactRepository(recording_engine).artifact_exists(artifact_id) is True
    statement, params = recording_engine.executions[0]
    assert statement.startswith("SELECT 1")
    assert "CONTENT_BYTES" not in statement
    assert "NORMALIZED_CONTENT" not in statement
    assert artifact_id not in statement
    assert params == {"artifact_id": artifact_id}


def test_artifact_exists_returns_false_without_materializing_an_artifact(
    recording_engine,
) -> None:
    """Return false from an empty lightweight existence result."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mapping(None)
    assert (
        ArtifactRepository(recording_engine).artifact_exists(
            "00000000-0000-4000-8000-000000000001"
        )
        is False
    )


def test_get_current_rule_artifact_returns_none_for_empty_result(recording_engine) -> None:
    """get_current_rule_artifact returns None when no rule artifact exists."""
    from app.persistence.artifact_repository import ArtifactRepository

    result = ArtifactRepository(recording_engine).get_current_rule_artifact()
    assert result is None


# ===========================================================================
# Suite 3: Chunked BOM-row reading
# ===========================================================================


def test_iter_bom_rows_uses_fresh_bounded_pages(recording_engine) -> None:
    """iter_bom_rows opens one connection per page and drains the queue."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([bom_row_mapping(1), bom_row_mapping(2)])
    recording_engine.queue_mappings([bom_row_mapping(3)])
    batches = list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010", batch_size=2
        )
    )
    assert [len(batch) for batch in batches] == [2, 1]
    assert recording_engine.connect_count == 2
    for statement, params in recording_engine.executions:
        assert 'ORDER BY "BOM_KIND", "SHEET_NAME", "SOURCE_ROW_NUMBER", "BOM_ROW_ID"' in statement
        assert params["batch_size"] == 2


def test_iter_bom_rows_second_page_uses_correct_offset(recording_engine) -> None:
    """iter_bom_rows increments offset by exactly the previous batch size."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([bom_row_mapping(1), bom_row_mapping(2)])
    recording_engine.queue_mappings([bom_row_mapping(3)])
    list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010", batch_size=2
        )
    )
    _, first_params = recording_engine.executions[0]
    _, second_params = recording_engine.executions[1]
    assert first_params["offset"] == 0
    assert second_params["offset"] == 2


def test_iter_bom_rows_stops_when_no_rows_returned(recording_engine) -> None:
    """iter_bom_rows yields nothing when the first page is empty."""
    from app.persistence.artifact_repository import ArtifactRepository

    batches = list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010"
        )
    )
    assert batches == []
    assert recording_engine.connect_count == 1


def test_iter_bom_rows_rejects_non_positive_batch_size(recording_engine) -> None:
    """iter_bom_rows raises ValueError for batch_size <= 0."""
    from app.persistence.artifact_repository import ArtifactRepository

    with pytest.raises(ValueError, match="batch_size"):
        list(ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010", batch_size=0
        ))
    with pytest.raises(ValueError, match="batch_size"):
        list(ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010", batch_size=-1
        ))


def test_iter_bom_rows_validates_uuid_before_querying(recording_engine) -> None:
    """iter_bom_rows rejects non-canonical UUIDs before opening any connection."""
    from app.persistence.artifact_repository import ArtifactRepository

    with pytest.raises(ValueError):
        list(ArtifactRepository(recording_engine).iter_bom_rows("bad-uuid"))
    assert recording_engine.connect_count == 0


def test_iter_bom_rows_parameterizes_artifact_id(recording_engine) -> None:
    """iter_bom_rows passes artifact_id as a bound parameter, not interpolated."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([bom_row_mapping(1)])
    list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010"
        )
    )
    statement, params = recording_engine.executions[0]
    assert "00000000-0000-4000-8000-000000000010" not in statement
    assert params["artifact_id"] == "00000000-0000-4000-8000-000000000010"


def test_iter_bom_rows_materializes_nclobs_in_page_query(recording_engine) -> None:
    """Avoid hdbcli's per-cell LOB network reads for the four NCLOB columns."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([])
    list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010"
        )
    )

    statement, _ = recording_engine.executions[0]
    assert "SELECT *" not in statement
    for column in (
        "HIERARCHY_PATH",
        "DESCRIPTION_TEXT",
        "ROW_JSON",
        "QUALITY_JSON",
    ):
        assert f'TO_NVARCHAR("{column}") AS "{column}"' in statement


def test_iter_bom_rows_reconstructs_bom_row_objects(recording_engine) -> None:
    """iter_bom_rows yields lists of BomRow domain objects."""
    from app.ecn.models import BomRow
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([bom_row_mapping(1)])
    batches = list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010"
        )
    )
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert isinstance(batches[0][0], BomRow)


def test_snapshot_bom_rows_fetches_only_matches_and_their_ancestors(
    recording_engine,
) -> None:
    """Snapshot reads filter in HANA and then follow only trusted parent row IDs."""
    from app.persistence.artifact_repository import ArtifactRepository

    artifact_id = "00000000-0000-4000-8000-000000000010"
    parent_id = "00000000-0000-4000-8000-000000000001"
    child = bom_row_mapping(
        2,
        ARTIFACT_ID=artifact_id,
        PARENT_BOM_ROW_ID=parent_id,
        COMPONENT_NUMBER="PART-1",
        DOTTED_LEVEL="5",
    )
    parent = bom_row_mapping(
        1,
        ARTIFACT_ID=artifact_id,
        BOM_ROW_ID=parent_id,
        COMPONENT_NUMBER="PARENT-1",
        DOTTED_LEVEL="4",
    )
    recording_engine.queue_mappings([child])
    recording_engine.queue_mappings([])
    recording_engine.queue_mappings([parent])

    rows = ArtifactRepository(recording_engine).snapshot_bom_rows(
        artifact_id,
        component_numbers=("PART-1",),
        change_number="ECN-1",
    )

    assert [row.bom_row_id for row in rows] == [parent_id, child["BOM_ROW_ID"]]
    assert len(recording_engine.executions) == 3
    location_sql, location_params = recording_engine.executions[0]
    change_sql, change_params = recording_engine.executions[1]
    parent_sql, parent_params = recording_engine.executions[2]
    assert '"COMPONENT_NUMBER" IN' in location_sql
    assert location_params["component_numbers"] == ("PART-1",)
    assert location_params["artifact_id"] == artifact_id
    assert '"CHANGE_NUMBER" = :change_number' in change_sql
    assert change_params["change_number"] == "ECN-1"
    assert '"BOM_ROW_ID" IN' in parent_sql
    assert parent_params["parent_ids"] == (parent_id,)
    assert all(
        "SELECT *" not in statement for statement, _ in recording_engine.executions
    )


def test_iter_bom_rows_full_page_followed_by_empty_opens_two_connections(recording_engine) -> None:
    """A full first page triggers a second query even when the second is empty."""
    from app.persistence.artifact_repository import ArtifactRepository

    recording_engine.queue_mappings([bom_row_mapping(1)])
    # Second call returns empty — signals end of stream.
    batches = list(
        ArtifactRepository(recording_engine).iter_bom_rows(
            "00000000-0000-4000-8000-000000000010", batch_size=1
        )
    )
    assert [len(b) for b in batches] == [1]
    assert recording_engine.connect_count == 2
