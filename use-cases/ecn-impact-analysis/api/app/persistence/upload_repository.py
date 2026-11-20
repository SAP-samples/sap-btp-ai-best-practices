"""Atomic upload-snapshot publisher and upload/BOM-snapshot reader.

This module owns the SQL that reads and writes the tables touched during
upload publication: ``ECN_IMPACT_UPLOAD``, ``ECN_IMPACT_SOURCE_ARTIFACT``,
``ECN_IMPACT_BOM_ROW``, ``ECN_IMPACT_ECN``, ``ECN_IMPACT_RDF_PROJECTION``,
and ``ECN_IMPACT_ANALYSIS_JOB``.

Design rules enforced here
--------------------------
* The repository NEVER retains an open ``Connection`` across method calls.
  Every public method opens and closes its own ``engine.connect()`` or
  ``engine.begin()`` context.
* ``publish_upload`` runs a single ``engine.begin()`` transaction covering all
  insert groups.  ``PUBLICATION_ORDER`` is allocated inside that transaction
  with ``COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1``; the caller is responsible
  for serializing concurrent publications at the service level.
* Idempotency is enforced by a ``UNIQUE`` constraint on ``REQUEST_ID``.  A
  duplicate found before the transaction (common case) returns the existing
  record without opening a transaction.  A race that resolves inside the
  transaction as an ``IntegrityError`` is handled by rolling back and
  re-querying; only a ``REQUEST_ID`` conflict is treated as idempotent — any
  other uniqueness violation (e.g. ``PUBLICATION_ORDER``) raises
  ``PersistenceStateError("UPLOAD_PUBLICATION_CONFLICT")``.
* BOM-snapshot reuse is a reference, not a copy.  When ``bom_reused=True``,
  neither the BOM artifact row nor any BOM-row rows are inserted.  The upload
  row references the existing artifact by ID.
* Every value is bound; the only text interpolated into SQL is a table
  identifier validated by ``quote_identifier`` from the allowlist
  ``^ECN_IMPACT_[A-Z0-9_]{1,110}$``.

Usage::

    from app.persistence.upload_repository import UploadRepository
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    repo = UploadRepository(engine)
    result = repo.publish_upload(publication)
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from app.ecn.errors import PersistenceStateError
from app.ecn.models import ArtifactKind, ParseState, SourceArtifact, canonical_uuid
from app.ecn.rdf import ProjectionState, project_ecn_json
from app.persistence.bulk import insert_chunks
from app.persistence.codec import (
    artifact_from_row,
    artifact_to_row,
    bom_row_to_row,
    ecn_publication_to_row,
    job_publication_to_row,
    upload_publication_to_row,
    upload_record_from_row,
)
from app.persistence.records import (
    UploadPublication,
    UploadPublishResult,
    UploadRecord,
)
from app.persistence.hana import to_hana_timestamp
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier


# ---------------------------------------------------------------------------
# SQL helpers — read queries
# ---------------------------------------------------------------------------


def _upload_by_request_sql(table_name: str) -> str:
    """Return the parameterized SELECT for an upload row by its idempotency key.

    Selects only the nine columns exposed by ``UploadRecord``; write-only
    columns (parser versions, created_by) are intentionally excluded.

    Args:
        table_name: Physical upload table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f'SELECT '
        f'"UPLOAD_ID", "REQUEST_ID", "PUBLICATION_ORDER", "ECN_ARTIFACT_ID", '
        f'"BOM_ARTIFACT_ID", "RULE_ARTIFACT_ID", "BOM_REUSED", "IMPORTED_ECN_COUNT", '
        f'"CREATED_AT" '
        f'FROM {t} '
        f'WHERE "REQUEST_ID" = :request_id'
    )


def _upload_by_id_sql(table_name: str) -> str:
    """Return the parameterized SELECT for an upload row by its primary key.

    Args:
        table_name: Physical upload table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f'SELECT '
        f'"UPLOAD_ID", "REQUEST_ID", "PUBLICATION_ORDER", "ECN_ARTIFACT_ID", '
        f'"BOM_ARTIFACT_ID", "RULE_ARTIFACT_ID", "BOM_REUSED", "IMPORTED_ECN_COUNT", '
        f'"CREATED_AT" '
        f'FROM {t} '
        f'WHERE "UPLOAD_ID" = :upload_id'
    )


def _latest_bom_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized SELECT for the latest valid BOM workbook artifact.

    Joins ``ECN_IMPACT_UPLOAD`` to ``ECN_IMPACT_SOURCE_ARTIFACT`` on the BOM
    artifact FK so that only artifacts that appear in a successful upload are
    considered.  Independently published rule artifacts are excluded because
    they do not have a corresponding upload row.

    ``PUBLICATION_ORDER DESC, UPLOAD_ID DESC`` provides a deterministic
    ordering even when two uploads share the same publication order (which
    should not happen in production but is safe to handle).

    Args:
        tables: Active ``PersistenceTableNames`` providing both physical names.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    u = quote_identifier(tables.upload)
    a = quote_identifier(tables.source_artifact)
    return (
        f'SELECT '
        f'A."ARTIFACT_ID", A."ARTIFACT_KIND", A."FILE_NAME", A."MEDIA_TYPE", '
        f'A."BYTE_SIZE", A."SHA256", A."CONTENT_BYTES", A."NORMALIZED_CONTENT", '
        f'A."PARSE_STATE", A."PARSE_ERROR_CODE", A."CREATED_AT" '
        f'FROM {a} A '
        f'JOIN {u} U ON U."BOM_ARTIFACT_ID" = A."ARTIFACT_ID" '
        f'WHERE A."ARTIFACT_KIND" = :artifact_kind '
        f'ORDER BY U."PUBLICATION_ORDER" DESC, U."UPLOAD_ID" DESC '
        f'LIMIT 1'
    )


# ---------------------------------------------------------------------------
# SQL helpers — insert statements
# ---------------------------------------------------------------------------


def _artifact_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized INSERT for one source-artifact row.

    Args:
        tables: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(tables.source_artifact)
    return (
        f"INSERT INTO {t} ("
        '"ARTIFACT_ID", "ARTIFACT_KIND", "FILE_NAME", "MEDIA_TYPE", "BYTE_SIZE", '
        '"SHA256", "CONTENT_BYTES", "NORMALIZED_CONTENT", "PARSE_STATE", '
        '"PARSE_ERROR_CODE", "CREATED_AT"'
        ") VALUES ("
        ":ARTIFACT_ID, :ARTIFACT_KIND, :FILE_NAME, :MEDIA_TYPE, :BYTE_SIZE, "
        ":SHA256, :CONTENT_BYTES, :NORMALIZED_CONTENT, :PARSE_STATE, "
        ":PARSE_ERROR_CODE, :CREATED_AT"
        ")"
    )


def _bom_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized INSERT for BOM-row batch inserts.

    Args:
        tables: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(tables.bom_row)
    cols = (
        '"BOM_ROW_ID", "ARTIFACT_ID", "BOM_KIND", "SHEET_NAME", "SOURCE_ROW_NUMBER", '
        '"PARENT_BOM_ROW_ID", "DOTTED_LEVEL", "HIERARCHY_PATH", "MAIN_VALUE", '
        '"SUB_VALUE", "VC_VALUE", "LEVEL_4_VALUE", "LEVEL_5_VALUE", "LEVEL_6_VALUE", '
        '"ITEM_VALUE", "POSITION_VALUE", "MATERIAL_NUMBER", "PARENT_MATERIAL_NUMBER", '
        '"COMPONENT_NUMBER", "CHANGE_NUMBER", "CHANGE_NUMBER_TO", "DESCRIPTION_TEXT", '
        '"QUANTITY", "UNIT", "PLANT", "ALTERNATIVE", "VALID_FROM_RAW", "VALID_FROM_DATE", '
        '"VALID_TO_RAW", "VALID_TO_DATE", "ROW_CLASSIFICATION", "IS_DOCUMENTATION", '
        '"IS_MIGRATION_DATE", "ROW_JSON", "QUALITY_JSON"'
    )
    params = (
        ":BOM_ROW_ID, :ARTIFACT_ID, :BOM_KIND, :SHEET_NAME, :SOURCE_ROW_NUMBER, "
        ":PARENT_BOM_ROW_ID, :DOTTED_LEVEL, :HIERARCHY_PATH, :MAIN_VALUE, "
        ":SUB_VALUE, :VC_VALUE, :LEVEL_4_VALUE, :LEVEL_5_VALUE, :LEVEL_6_VALUE, "
        ":ITEM_VALUE, :POSITION_VALUE, :MATERIAL_NUMBER, :PARENT_MATERIAL_NUMBER, "
        ":COMPONENT_NUMBER, :CHANGE_NUMBER, :CHANGE_NUMBER_TO, :DESCRIPTION_TEXT, "
        ":QUANTITY, :UNIT, :PLANT, :ALTERNATIVE, :VALID_FROM_RAW, :VALID_FROM_DATE, "
        ":VALID_TO_RAW, :VALID_TO_DATE, :ROW_CLASSIFICATION, :IS_DOCUMENTATION, "
        ":IS_MIGRATION_DATE, :ROW_JSON, :QUALITY_JSON"
    )
    return f"INSERT INTO {t} ({cols}) VALUES ({params})"


def _upload_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized INSERT for one upload row.

    Args:
        tables: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(tables.upload)
    return (
        f"INSERT INTO {t} ("
        '"UPLOAD_ID", "REQUEST_ID", "PUBLICATION_ORDER", "ECN_ARTIFACT_ID", '
        '"BOM_ARTIFACT_ID", "RULE_ARTIFACT_ID", "BOM_REUSED", "IMPORTED_ECN_COUNT", '
        '"CREATED_BY", "CREATED_AT", "PARSER_VERSION", '
        '"DETERMINISTIC_LOGIC_VERSION", "RULE_EVALUATOR_VERSION"'
        ") VALUES ("
        ":UPLOAD_ID, :REQUEST_ID, :PUBLICATION_ORDER, :ECN_ARTIFACT_ID, "
        ":BOM_ARTIFACT_ID, :RULE_ARTIFACT_ID, :BOM_REUSED, :IMPORTED_ECN_COUNT, "
        ":CREATED_BY, :CREATED_AT, :PARSER_VERSION, "
        ":DETERMINISTIC_LOGIC_VERSION, :RULE_EVALUATOR_VERSION"
        ")"
    )


def _upload_dataset_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the bound insert for pinned ECN-to-reference provenance."""
    table = quote_identifier(tables.upload_dataset)
    return (
        f'INSERT INTO {table} ('
        '"UPLOAD_ID", "DATASET_TYPE", "DATASET_ID", "PINNED_AT"'
        ") VALUES (:UPLOAD_ID, :DATASET_TYPE, :DATASET_ID, :PINNED_AT)"
    )


def _ecn_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized INSERT for ECN-occurrence batch inserts.

    Args:
        tables: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(tables.ecn)
    cols = (
        '"ECN_ROW_ID", "UPLOAD_ID", "SOURCE_ORDINAL", "ECN_NUMBER", "TITLE", '
        '"DESCRIPTION_SUMMARY", "DESCRIPTION_TEXT", "AFFECTED_ITEM_COUNT", '
        '"RAW_ECN_MATCH_COUNT", "EBOM_MATCH_COUNT", "EBOM_CONTEXT_COUNT", '
        '"MBOM_MATCH_COUNT", "WARNING_COUNT", "SOURCE_JSON", "CHANGE_CANDIDATES_JSON", '
        '"EVIDENCE_JSON", "EVIDENCE_HASH", "PARSER_VERSION", '
        '"DETERMINISTIC_LOGIC_VERSION", "RULE_EVALUATOR_VERSION", "CREATED_AT"'
    )
    vals = (
        ":ECN_ROW_ID, :UPLOAD_ID, :SOURCE_ORDINAL, :ECN_NUMBER, :TITLE, "
        ":DESCRIPTION_SUMMARY, :DESCRIPTION_TEXT, :AFFECTED_ITEM_COUNT, "
        ":RAW_ECN_MATCH_COUNT, :EBOM_MATCH_COUNT, :EBOM_CONTEXT_COUNT, "
        ":MBOM_MATCH_COUNT, :WARNING_COUNT, :SOURCE_JSON, :CHANGE_CANDIDATES_JSON, "
        ":EVIDENCE_JSON, :EVIDENCE_HASH, :PARSER_VERSION, "
        ":DETERMINISTIC_LOGIC_VERSION, :RULE_EVALUATOR_VERSION, :CREATED_AT"
    )
    return f"INSERT INTO {t} ({cols}) VALUES ({vals})"


def _job_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized INSERT for analysis-job batch inserts.

    Args:
        tables: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(tables.analysis_job)
    cols = (
        '"JOB_ID", "ECN_ROW_ID", "STATE", "ATTEMPT_COUNT", "CREATED_AT", '
        '"STARTED_AT", "FINISHED_AT", "PROMPT_VERSION", "OUTPUT_SCHEMA_VERSION", '
        '"MODEL_NAME", "MODEL_CONFIG_JSON", "ERROR_CODE", "ERROR_MESSAGE"'
    )
    vals = (
        ":JOB_ID, :ECN_ROW_ID, :STATE, :ATTEMPT_COUNT, :CREATED_AT, "
        ":STARTED_AT, :FINISHED_AT, :PROMPT_VERSION, :OUTPUT_SCHEMA_VERSION, "
        ":MODEL_NAME, :MODEL_CONFIG_JSON, :ERROR_CODE, :ERROR_MESSAGE"
    )
    return f"INSERT INTO {t} ({cols}) VALUES ({vals})"


def _rdf_projection_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the parameterized insert for one committed RDF projection status.

    Args:
        tables: Active application table-name contract.

    Returns:
        SQL that inserts only derived, rebuildable projection metadata.
    """
    table = quote_identifier(tables.rdf_projection)
    columns = (
        '"ECN_ROW_ID", "SOURCE_SHA256", "PROJECTION_VERSION", "STATUS", '
        '"EXPECTED_OBJECT_COUNT", "EXPECTED_ARRAY_COUNT", "EXPECTED_SCALAR_COUNT", '
        '"EXPECTED_NULL_COUNT", "EXPECTED_NODE_COUNT", "VERIFIED_NODE_COUNT", '
        '"VERIFIED_TRIPLE_COUNT", "ERROR_CODE", "ERROR_MESSAGE", "LAST_ATTEMPT_AT", '
        '"CURRENT_AT"'
    )
    values = (
        ':ECN_ROW_ID, :SOURCE_SHA256, :PROJECTION_VERSION, :STATUS, '
        ':EXPECTED_OBJECT_COUNT, :EXPECTED_ARRAY_COUNT, :EXPECTED_SCALAR_COUNT, '
        ':EXPECTED_NULL_COUNT, :EXPECTED_NODE_COUNT, :VERIFIED_NODE_COUNT, '
        ':VERIFIED_TRIPLE_COUNT, :ERROR_CODE, :ERROR_MESSAGE, :LAST_ATTEMPT_AT, '
        ':CURRENT_AT'
    )
    return f"INSERT INTO {table} ({columns}) VALUES ({values})"


def _pending_projection_rows(publication: UploadPublication) -> list[dict[str, object]]:
    """Build the relational ``PENDING`` baselines from authoritative ECN JSON.

    Args:
        publication: Fully validated upload aggregate whose ECNs will commit.

    Returns:
        One derived status row per selected ECN in source order.
    """
    rows: list[dict[str, object]] = []
    for item in publication.ecns:
        projection = project_ecn_json(
            item.ecn.ecn_row_id,
            publication.ecn_artifact.artifact_id,
            item.ecn.source_json,
        )
        rows.append(
            {
                "ECN_ROW_ID": item.ecn.ecn_row_id,
                "SOURCE_SHA256": projection.source_sha256,
                "PROJECTION_VERSION": projection.projection_version,
                "STATUS": ProjectionState.PENDING.value,
                "EXPECTED_OBJECT_COUNT": projection.counts.object_count,
                "EXPECTED_ARRAY_COUNT": projection.counts.array_count,
                "EXPECTED_SCALAR_COUNT": projection.counts.scalar_count,
                "EXPECTED_NULL_COUNT": projection.counts.null_count,
                "EXPECTED_NODE_COUNT": projection.counts.node_count,
                "VERIFIED_NODE_COUNT": None,
                "VERIFIED_TRIPLE_COUNT": None,
                "ERROR_CODE": None,
                "ERROR_MESSAGE": None,
                "LAST_ATTEMPT_AT": None,
                "CURRENT_AT": None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------


def _validate_publication(publication: UploadPublication) -> None:
    """Validate all preconditions before opening a transaction.

    Called by ``publish_upload`` before any database access.  All checks are
    pure Python: no connection is required.

    Invariants enforced (corresponds to the Task 6 brief invariants 1–7):

    1. ``upload_id`` and ``request_id`` are canonical UUIDs.
    2. ECN artifact is ``VALID/ECN_JSON``; BOM artifact is ``VALID/BOM_WORKBOOK``.
    3. When ``bom_reused=False``, ``new_bom_rows`` is non-empty and every row
       points to the supplied BOM artifact.  When ``bom_reused=True``,
       ``new_bom_rows`` must be empty.
    4. A non-null rule artifact has kind ``CONDITIONAL_RULE_WORKBOOK`` and
       state ``VALID``.
    5. Every ECN item points to the ECN artifact; source ordinals are unique
       and nonnegative; job IDs are canonical UUIDs; evidence belongs to the
       ECN and its context IDs match every publication artifact.
    6. ``IMPORTED_ECN_COUNT`` equals ``len(publication.ecns)`` (implicit).
    7. ``actor_id``, when present, is at most 64 characters.

    Args:
        publication: The full upload command record to validate.

    Raises:
        ValueError: When any invariant is violated.
    """
    canonical_uuid(publication.upload_id)
    canonical_uuid(publication.request_id)
    linked_types = [link.dataset_type for link in publication.dataset_links]
    if len(linked_types) != len(set(linked_types)):
        raise ValueError("dataset_links must contain at most one active dataset per type")

    # Invariant 2: artifact kinds and parse states.
    if publication.ecn_artifact.artifact_kind is not ArtifactKind.ECN_JSON:
        raise ValueError(
            "ecn_artifact must have artifact_kind ECN_JSON; "
            f"got {publication.ecn_artifact.artifact_kind.value!r}"
        )
    if publication.ecn_artifact.parse_state is not ParseState.VALID:
        raise ValueError(
            "ecn_artifact must have parse_state VALID; "
            f"got {publication.ecn_artifact.parse_state.value!r}"
        )
    if publication.bom_artifact.artifact_kind is not ArtifactKind.BOM_WORKBOOK:
        raise ValueError(
            "bom_artifact must have artifact_kind BOM_WORKBOOK; "
            f"got {publication.bom_artifact.artifact_kind.value!r}"
        )
    if publication.bom_artifact.parse_state is not ParseState.VALID:
        raise ValueError(
            "bom_artifact must have parse_state VALID; "
            f"got {publication.bom_artifact.parse_state.value!r}"
        )

    # Invariant 3: BOM rows vs. reuse flag.
    if publication.bom_reused:
        if publication.new_bom_rows:
            raise ValueError(
                "new_bom_rows must be empty when bom_reused=True"
            )
    else:
        if not publication.new_bom_rows:
            raise ValueError(
                "new_bom_rows must contain at least one bom_rows entry when bom_reused=False"
            )
        bom_id = publication.bom_artifact.artifact_id
        for row in publication.new_bom_rows:
            if row.artifact_id != bom_id:
                raise ValueError(
                    f"all new_bom_rows must point to the BOM artifact {bom_id!r}; "
                    f"found row with artifact_id {row.artifact_id!r}"
                )

    # Invariant 4: optional rule artifact.
    if publication.rule_artifact is not None:
        if publication.rule_artifact.artifact_kind is not ArtifactKind.CONDITIONAL_RULE_WORKBOOK:
            raise ValueError(
                "rule_artifact must have artifact_kind CONDITIONAL_RULE_WORKBOOK"
            )
        if publication.rule_artifact.parse_state is not ParseState.VALID:
            raise ValueError(
                "rule_artifact must have parse_state VALID"
            )

    # Invariant 5: ECN items.
    ecn_art_id = publication.ecn_artifact.artifact_id
    expected_rule_artifact_id = (
        publication.rule_artifact.artifact_id
        if publication.rule_artifact is not None
        else None
    )
    seen_ordinals: set[int] = set()
    for item in publication.ecns:
        canonical_uuid(item.job_id)
        if item.ecn.artifact_id != ecn_art_id:
            raise ValueError(
                f"all ECN items must point to ecn_artifact {ecn_art_id!r}; "
                f"found item with artifact_id {item.ecn.artifact_id!r}"
            )
        ordinal = item.ecn.source_ordinal
        if ordinal < 0:
            raise ValueError(
                f"source_ordinal must be nonnegative; got {ordinal}"
            )
        if ordinal in seen_ordinals:
            raise ValueError(
                f"source_ordinal {ordinal} appears more than once in ecns"
            )
        seen_ordinals.add(ordinal)
        if item.evidence.ecn_row_id != item.ecn.ecn_row_id:
            raise ValueError(
                f"evidence ecn_row_id {item.evidence.ecn_row_id!r} does not match "
                f"normalized ECN row ID {item.ecn.ecn_row_id!r}"
            )
        # Evidence context must agree with the upload and every artifact ID.
        ctx = item.evidence.context
        if ctx.upload_id != publication.upload_id:
            raise ValueError(
                f"evidence context upload_id {ctx.upload_id!r} does not match "
                f"publication upload_id {publication.upload_id!r}"
            )
        if ctx.ecn_artifact_id != ecn_art_id:
            raise ValueError(
                f"evidence context ecn_artifact_id {ctx.ecn_artifact_id!r} does not match "
                f"ecn_artifact {ecn_art_id!r}"
            )
        if ctx.bom_artifact_id != publication.bom_artifact.artifact_id:
            raise ValueError(
                f"evidence context bom_artifact_id {ctx.bom_artifact_id!r} does not match "
                f"bom_artifact {publication.bom_artifact.artifact_id!r}"
            )
        if ctx.rule_artifact_id != expected_rule_artifact_id:
            raise ValueError(
                f"evidence context rule_artifact_id {ctx.rule_artifact_id!r} does not match "
                f"rule_artifact {expected_rule_artifact_id!r}"
            )

    # Invariant 7: actor_id length.
    if publication.actor_id is not None and len(publication.actor_id) > 64:
        raise ValueError(
            f"actor_id must be at most 64 characters; "
            f"got {len(publication.actor_id)} characters"
        )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _upload_record_from_publication(
    publication: UploadPublication, publication_order: int
) -> UploadRecord:
    """Build an ``UploadRecord`` from the publication and its allocated order.

    Used in the success path of ``publish_upload`` to return the newly
    created record without re-querying the database.

    Args:
        publication: The publication command record.
        publication_order: The ``COALESCE(MAX(...), 0) + 1`` value allocated
            inside the transaction.

    Returns:
        A frozen ``UploadRecord`` matching the insert that just completed.
    """
    return UploadRecord(
        upload_id=publication.upload_id,
        request_id=publication.request_id,
        ecn_artifact_id=publication.ecn_artifact.artifact_id,
        bom_artifact_id=publication.bom_artifact.artifact_id,
        rule_artifact_id=(
            publication.rule_artifact.artifact_id
            if publication.rule_artifact is not None
            else None
        ),
        bom_reused=publication.bom_reused,
        imported_ecn_count=len(publication.ecns),
        publication_order=publication_order,
        created_at=publication.created_at,
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class UploadRepository:
    """Publish immutable upload snapshots and retrieve their provenance.

    All public methods open a short-lived connection or transaction per
    operation.  No connection is retained between calls.

    ``publish_upload`` atomically inserts the ECN artifact, an optional new
    BOM artifact and its rows, the upload row, all ECN occurrence rows, and
    one QUEUED analysis job per ECN occurrence — all in one ``engine.begin()``
    transaction.  When the BOM is reused, no artifact or BOM-row inserts
    occur; the upload row references the existing artifact ID.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the HANA instance.
        table_names: Optional ``PersistenceTableNames`` override; defaults to
            the production names.  Pass a suffixed instance for isolated live
            tests.

    Example::

        repo = UploadRepository(engine)
        result = repo.publish_upload(publication)
        if result.created:
            print(f"Created upload {result.upload.upload_id}")
    """

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames = PersistenceTableNames(),
    ) -> None:
        """Bind the engine and table-name set.

        Args:
            engine: SQLAlchemy ``Engine``; never stored beyond this reference.
            table_names: Physical table identifiers already validated by
                ``PersistenceTableNames.__post_init__``.
        """
        self._engine = engine
        self._tables = table_names

    def find_by_request_id(self, request_id: str) -> UploadRecord | None:
        """Return the successful upload already assigned to an idempotency key.

        Opens one short ``engine.connect()`` context per call.  The UUID is
        validated before the connection is opened so that an invalid argument
        never reaches the database.

        Args:
            request_id: A canonical UUID string for the idempotency key.

        Returns:
            The ``UploadRecord`` when found, or ``None`` when absent.

        Raises:
            ValueError: When ``request_id`` is not a canonical UUID.
        """
        canonical_uuid(request_id)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_upload_by_request_sql(self._tables.upload)),
                {"request_id": request_id},
            ).mappings().one_or_none()
        return None if row is None else upload_record_from_row(row)

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        """Return one upload batch by its canonical internal UUID.

        Opens one short ``engine.connect()`` context.

        Args:
            upload_id: A canonical UUID string for the upload batch.

        Returns:
            The ``UploadRecord`` when found, or ``None`` when absent.

        Raises:
            ValueError: When ``upload_id`` is not a canonical UUID.
        """
        canonical_uuid(upload_id)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_upload_by_id_sql(self._tables.upload)),
                {"upload_id": upload_id},
            ).mappings().one_or_none()
        return None if row is None else upload_record_from_row(row)

    def get_latest_bom_snapshot(self) -> SourceArtifact | None:
        """Return the BOM artifact from the greatest successful publication order.

        Joins ``ECN_IMPACT_UPLOAD`` to ``ECN_IMPACT_SOURCE_ARTIFACT`` so that
        only BOM artifacts linked to a completed upload are returned.
        Independently published rule artifacts and artifacts that failed to
        produce an upload row are excluded.

        Returns:
            The ``SourceArtifact`` for the most recently published BOM workbook,
            or ``None`` when no upload with a BOM artifact exists.
        """
        with self._engine.connect() as connection:
            row = connection.execute(
                text(_latest_bom_sql(self._tables)),
                {"artifact_kind": ArtifactKind.BOM_WORKBOOK.value},
            ).mappings().one_or_none()
        return None if row is None else artifact_from_row(row)

    def publish_upload(self, publication: UploadPublication) -> UploadPublishResult:
        """Atomically append one successful upload, occurrences, evidence, and queued jobs.

        The insert groups executed inside the single transaction are:

        1. ECN JSON source artifact.
        2. BOM workbook source artifact (only when ``bom_reused=False``).
        3. BOM rows (only when ``bom_reused=False`` and rows exist).
        4. Upload row with allocated ``PUBLICATION_ORDER``.
        5. ECN occurrence rows (one per item in ``publication.ecns``).
        6. Derived RDF ``PENDING`` status rows (one per item in ``publication.ecns``).
        7. Analysis job rows (one per item in ``publication.ecns``).

        Idempotency: if the request ID already exists before the transaction
        opens, the existing record is returned immediately with
        ``created=False``.  If a ``REQUEST_ID`` uniqueness race resolves as an
        ``IntegrityError`` inside the transaction, the transaction rolls back
        and the existing record is returned.  Any other uniqueness violation
        (e.g. a ``PUBLICATION_ORDER`` collision from a programming error)
        raises ``PersistenceStateError("UPLOAD_PUBLICATION_CONFLICT")``.

        Args:
            publication: A fully validated ``UploadPublication`` carrying all
                artifacts, BOM rows, normalized ECNs, and pinned model config.

        Returns:
            An ``UploadPublishResult`` with ``created=True`` when a new record
            was stored, or ``created=False`` when an existing record matched
            the idempotency key.

        Raises:
            ValueError: When any preflight invariant is violated.
            PersistenceStateError: When a non-idempotent uniqueness conflict
                occurs inside the database.
        """
        _validate_publication(publication)

        # Fast-path: check for an existing record before opening a transaction.
        existing = self.find_by_request_id(publication.request_id)
        if existing is not None:
            return UploadPublishResult(existing, created=False)

        try:
            with self._engine.begin() as connection:
                order = connection.execute(
                    text(
                        f'SELECT COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1 '
                        f'FROM {quote_identifier(self._tables.upload)}'
                    )
                ).scalar_one()

                # Reassessment reuses its immutable source artifact instead of inserting it.
                if not publication.ecn_reused:
                    connection.execute(
                        text(_artifact_insert_sql(self._tables)),
                        artifact_to_row(publication.ecn_artifact),
                    )

                # 2–3. BOM artifact and rows — only when the BOM is new.
                if not publication.bom_reused:
                    connection.execute(
                        text(_artifact_insert_sql(self._tables)),
                        artifact_to_row(publication.bom_artifact),
                    )
                    if publication.new_bom_rows:
                        insert_chunks(
                            connection,
                            text(_bom_insert_sql(self._tables)),
                            (bom_row_to_row(row) for row in publication.new_bom_rows),
                        )

                # 4. Upload row.
                connection.execute(
                    text(_upload_insert_sql(self._tables)),
                    upload_publication_to_row(publication, publication_order=int(order)),
                )

                # 5. Pin every Settings source active when this ECN was published.
                if publication.dataset_links:
                    connection.execute(
                        text(_upload_dataset_insert_sql(self._tables)),
                        [
                            {
                                "UPLOAD_ID": publication.upload_id,
                                "DATASET_TYPE": link.dataset_type.value,
                                "DATASET_ID": link.dataset_id,
                                "PINNED_AT": to_hana_timestamp(publication.created_at),
                            }
                            for link in publication.dataset_links
                        ],
                    )

                # 6. ECN occurrence rows.
                if publication.ecns:
                    insert_chunks(
                        connection,
                        text(_ecn_insert_sql(self._tables)),
                        (ecn_publication_to_row(publication, item) for item in publication.ecns),
                        chunk_size=1,
                    )

                # 7. Derived RDF status rows must commit with authoritative ECNs.
                if publication.ecns:
                    connection.execute(
                        text(_rdf_projection_insert_sql(self._tables)),
                        _pending_projection_rows(publication),
                    )

                # 8. Analysis job rows — one per ECN occurrence.
                if publication.ecns:
                    connection.execute(
                        text(_job_insert_sql(self._tables)),
                        [job_publication_to_row(publication, item) for item in publication.ecns],
                    )

        except IntegrityError as exc:
            # Distinguish a REQUEST_ID race from any other uniqueness violation.
            raced = self.find_by_request_id(publication.request_id)
            if raced is not None:
                return UploadPublishResult(raced, created=False)
            raise PersistenceStateError(
                "UPLOAD_PUBLICATION_CONFLICT",
                "The upload could not be published because its immutable identity conflicts "
                "with an existing record",
                status_code=409,
            ) from exc

        return UploadPublishResult(
            upload=_upload_record_from_publication(publication, int(order)),
            created=True,
            projection_ecn_ids=tuple(item.ecn.ecn_row_id for item in publication.ecns),
        )
