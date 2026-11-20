"""Append-only source-artifact repository and paginated BOM-row reader.

This module owns the SQL that reads and writes the two storage-heavy tables:
``ECN_IMPACT_SOURCE_ARTIFACT`` (immutable raw uploads) and
``ECN_IMPACT_BOM_ROW`` (normalized BOM rows keyed to an artifact).

Design rules enforced here
--------------------------
* The repository NEVER retains an open ``Connection`` across method calls.
  Every public method opens and closes its own ``engine.connect()`` or
  ``engine.begin()`` context.
* ``iter_bom_rows`` opens ONE connection per page, materialises the page
  fully, closes the connection, and only then yields.  The generator never
  holds a connection between yields so that callers can process pages at
  their own pace without exhausting the connection pool.
* Every value is bound; the only text interpolated into SQL is a table
  identifier validated by ``quote_identifier`` from the allowlist
  ``^ECN_IMPACT_[A-Z0-9_]{1,110}$``.
* ``artifact_exists`` uses an ID-only projection and never materialises the
  persisted workbook BLOB or normalized content payload.
* Duplicate-key ``IntegrityError`` from ``publish_rule_artifact`` is
  converted to ``PersistenceStateError("ARTIFACT_ALREADY_EXISTS")`` so that
  driver-specific error text never surfaces in API responses.

Usage::

    from app.persistence.artifact_repository import ArtifactRepository
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    repo = ArtifactRepository(engine)
    artifact = repo.get_artifact("00000000-0000-4000-8000-000000000001")
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from app.ecn.errors import PersistenceStateError
from app.ecn.models import (
    ArtifactKind,
    BomKind,
    BomRow,
    BomRowClassification,
    ParseState,
    SourceArtifact,
    canonical_uuid,
)
from app.persistence.codec import artifact_from_row, artifact_to_row, bom_row_from_row
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


_BOM_SELECT_COLUMNS = (
    '"BOM_ROW_ID", "ARTIFACT_ID", "BOM_KIND", "SHEET_NAME", '
    '"SOURCE_ROW_NUMBER", "PARENT_BOM_ROW_ID", "DOTTED_LEVEL", '
    'TO_NVARCHAR("HIERARCHY_PATH") AS "HIERARCHY_PATH", '
    '"MAIN_VALUE", "SUB_VALUE", "VC_VALUE", "LEVEL_4_VALUE", '
    '"LEVEL_5_VALUE", "LEVEL_6_VALUE", "ITEM_VALUE", "POSITION_VALUE", '
    '"MATERIAL_NUMBER", "PARENT_MATERIAL_NUMBER", "COMPONENT_NUMBER", '
    '"CHANGE_NUMBER", "CHANGE_NUMBER_TO", '
    'TO_NVARCHAR("DESCRIPTION_TEXT") AS "DESCRIPTION_TEXT", '
    '"QUANTITY", "UNIT", "PLANT", "ALTERNATIVE", "VALID_FROM_RAW", '
    '"VALID_FROM_DATE", "VALID_TO_RAW", "VALID_TO_DATE", '
    '"ROW_CLASSIFICATION", "IS_DOCUMENTATION", "IS_MIGRATION_DATE", '
    'TO_NVARCHAR("ROW_JSON") AS "ROW_JSON", '
    'TO_NVARCHAR("QUALITY_JSON") AS "QUALITY_JSON"'
)


def _artifact_insert_sql(table_name: str) -> str:
    """Return the parameterized INSERT statement for one source-artifact row.

    Only the quoted table identifier is interpolated; all column values are
    passed as bound parameters with their exact column-name keys.

    Args:
        table_name: The physical table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f'INSERT INTO {t} ('
        '"ARTIFACT_ID", "ARTIFACT_KIND", "FILE_NAME", "MEDIA_TYPE", "BYTE_SIZE", '
        '"SHA256", "CONTENT_BYTES", "NORMALIZED_CONTENT", "PARSE_STATE", '
        '"PARSE_ERROR_CODE", "CREATED_AT"'
        ") VALUES ("
        ":ARTIFACT_ID, :ARTIFACT_KIND, :FILE_NAME, :MEDIA_TYPE, :BYTE_SIZE, "
        ":SHA256, :CONTENT_BYTES, :NORMALIZED_CONTENT, :PARSE_STATE, "
        ":PARSE_ERROR_CODE, :CREATED_AT"
        ")"
    )


def _current_rule_sql(table_name: str) -> str:
    """Return the parameterized SELECT for the most recent conditional-rule artifact.

    Ordering by ``CREATED_AT DESC, ARTIFACT_ID DESC`` provides a deterministic
    tie-breaker when multiple rules share the same creation timestamp.

    Args:
        table_name: The physical table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f'SELECT * FROM {t} '
        'WHERE "ARTIFACT_KIND" = :artifact_kind '
        'ORDER BY "CREATED_AT" DESC, "ARTIFACT_ID" DESC '
        "LIMIT 1"
    )


def _artifact_exists_sql(table_name: str) -> str:
    """Return a lightweight parameterized artifact-existence query.

    The projection deliberately excludes both storage-heavy payload columns,
    allowing callers to validate an identity while holding a short lifecycle
    mutation guard without reading the workbook BLOB or normalized content.

    Args:
        table_name: The physical table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f'SELECT 1 AS "PRESENT" FROM {t} '
        'WHERE "ARTIFACT_ID" = :artifact_id LIMIT 1'
    )


def _bom_page_sql(table_name: str) -> str:
    """Return the parameterized SELECT for one BOM-row page.

    The stable four-column order (BOM_KIND, SHEET_NAME, SOURCE_ROW_NUMBER,
    BOM_ROW_ID) ensures that pagination across pages does not miss or repeat
    rows.  ``LIMIT :batch_size OFFSET :offset`` are bound parameters so that
    no values are interpolated into the SQL text. NCLOB values are projected
    through ``TO_NVARCHAR`` so hdbcli transfers them inline with each result
    page instead of performing one additional LOB round trip per cell.

    Args:
        table_name: The physical table name validated by ``quote_identifier``.

    Returns:
        A SQL string suitable for use with SQLAlchemy ``text()``.
    """
    t = quote_identifier(table_name)
    return (
        f"SELECT {_BOM_SELECT_COLUMNS} FROM {t} "
        'WHERE "ARTIFACT_ID" = :artifact_id '
        'ORDER BY "BOM_KIND", "SHEET_NAME", "SOURCE_ROW_NUMBER", "BOM_ROW_ID" '
        "LIMIT :batch_size OFFSET :offset"
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ArtifactRepository:
    """Persist and retrieve immutable source artifacts and normalized BOM rows.

    All public methods open a short-lived connection or transaction per
    operation.  No connection is retained between calls.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the HANA instance.
        table_names: Optional ``PersistenceTableNames`` override; defaults to
            the production names.  Pass a suffixed instance for isolated live
            tests.

    Example::

        repo = ArtifactRepository(engine)
        artifact = repo.get_artifact("00000000-0000-4000-8000-000000000001")
        for page in repo.iter_bom_rows(artifact.artifact_id):
            process(page)
    """

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames | None = None,
    ) -> None:
        """Bind the engine and table-name set; validate no credentials stored.

        Args:
            engine: SQLAlchemy ``Engine``; never stored beyond this reference.
            table_names: Optional physical table identifiers already validated
                by ``PersistenceTableNames.__post_init__``. Production names
                are created when omitted.
        """
        self._engine = engine
        self._tables = table_names or PersistenceTableNames()

    def get_artifact(self, artifact_id: str) -> SourceArtifact | None:
        """Return one artifact by its canonical internal UUID.

        Opens one short ``engine.connect()`` context, executes a single
        ``SELECT``, and closes the connection before returning.

        Args:
            artifact_id: A canonical UUID string (rejected before querying if
                not in canonical form).

        Returns:
            The ``SourceArtifact`` when found, or ``None`` when absent.

        Raises:
            ValueError: When ``artifact_id`` is not a canonical UUID.
        """
        canonical_uuid(artifact_id)
        statement = text(
            f'SELECT * FROM {quote_identifier(self._tables.source_artifact)} '
            'WHERE "ARTIFACT_ID" = :artifact_id'
        )
        with self._engine.connect() as connection:
            row = (
                connection.execute(statement, {"artifact_id": artifact_id})
                .mappings()
                .one_or_none()
            )
        return None if row is None else artifact_from_row(row)

    def artifact_exists(self, artifact_id: str) -> bool:
        """Return whether an artifact exists without reading payload columns.

        Opens one short ``engine.connect()`` context and projects only a
        constant. This is suitable for identity revalidation in a bounded
        critical section because neither ``CONTENT_BYTES`` nor
        ``NORMALIZED_CONTENT`` is selected.

        Args:
            artifact_id: A canonical UUID string (rejected before querying if
                not in canonical form).

        Returns:
            ``True`` when the artifact identity exists, otherwise ``False``.

        Raises:
            ValueError: When ``artifact_id`` is not a canonical UUID.
        """
        canonical_uuid(artifact_id)
        statement = text(_artifact_exists_sql(self._tables.source_artifact))
        with self._engine.connect() as connection:
            row = (
                connection.execute(statement, {"artifact_id": artifact_id})
                .mappings()
                .one_or_none()
            )
        return row is not None

    def get_current_rule_artifact(self) -> SourceArtifact | None:
        """Return the most recently stored conditional-rule workbook artifact.

        Queries with ``ORDER BY "CREATED_AT" DESC, "ARTIFACT_ID" DESC`` and
        ``LIMIT 1`` to provide a deterministic result.  Opens one short
        ``engine.connect()`` context.

        Returns:
            The latest ``CONDITIONAL_RULE_WORKBOOK`` artifact, or ``None``
            when no rule artifact has been published.
        """
        statement = text(_current_rule_sql(self._tables.source_artifact))
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    statement,
                    {"artifact_kind": ArtifactKind.CONDITIONAL_RULE_WORKBOOK.value},
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else artifact_from_row(row)

    def publish_rule_artifact(self, artifact: SourceArtifact) -> SourceArtifact:
        """Append one validated conditional-rule workbook artifact to the store.

        The artifact must have ``artifact_kind=CONDITIONAL_RULE_WORKBOOK`` and
        ``parse_state=VALID``; any other combination is rejected before opening
        a transaction.  Uses a single ``engine.begin()`` context for the
        insert.

        Args:
            artifact: A validated ``CONDITIONAL_RULE_WORKBOOK`` artifact.

        Returns:
            The artifact unchanged (for chaining convenience).

        Raises:
            ValueError: When the artifact kind or parse state is wrong.
            PersistenceStateError: When the database rejects the insert due to
                a duplicate artifact ID (code ``ARTIFACT_ALREADY_EXISTS``).
        """
        if artifact.artifact_kind is not ArtifactKind.CONDITIONAL_RULE_WORKBOOK:
            raise ValueError(
                "rule publication requires a conditional-rule workbook artifact"
            )
        if artifact.parse_state is not ParseState.VALID:
            raise ValueError("only a valid rule artifact may be published")
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text(_artifact_insert_sql(self._tables.source_artifact)),
                    artifact_to_row(artifact),
                )
        except IntegrityError as exc:
            raise PersistenceStateError(
                "ARTIFACT_ALREADY_EXISTS",
                "An artifact with this identifier already exists",
                status_code=409,
            ) from exc
        return artifact

    def iter_bom_rows(
        self, artifact_id: str, batch_size: int = 5_000
    ) -> Iterator[list[BomRow]]:
        """Yield stable BOM-row pages while closing the HANA connection between pages.

        Each page opens and closes its own ``engine.connect()`` context so that
        long-running iteration does not hold a connection open for the duration
        of the generator.  The page is fully materialised before the connection
        is closed, and only then yielded.

        The stable ordering (BOM_KIND, SHEET_NAME, SOURCE_ROW_NUMBER, BOM_ROW_ID)
        guarantees that paginating through large workbooks produces the same
        total set regardless of row-level inserts between pages.

        Args:
            artifact_id: A canonical UUID string for the parent BOM artifact.
            batch_size: Number of rows per page; must be positive.

        Yields:
            Lists of ``BomRow`` domain objects, one list per page.  The last
            page may contain fewer rows than ``batch_size``.

        Raises:
            ValueError: When ``artifact_id`` is not a canonical UUID or when
                ``batch_size`` is not positive.
            PersistenceStateError: When any row's ``ROW_JSON`` or
                ``QUALITY_JSON`` column is malformed.
        """
        canonical_uuid(artifact_id)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        offset = 0
        sql = text(_bom_page_sql(self._tables.bom_row))
        while True:
            with self._engine.connect() as connection:
                rows = (
                    connection.execute(
                        sql,
                        {
                            "artifact_id": artifact_id,
                            "batch_size": batch_size,
                            "offset": offset,
                        },
                    )
                    .mappings()
                    .all()
                )
            # Connection is closed here; decode outside the connection context.
            if not rows:
                return
            batch = [bom_row_from_row(row) for row in rows]
            yield batch
            if len(batch) < batch_size:
                return
            offset += len(batch)

    def change_touched_components(self, artifact_id: str, ecn_number: str | None) -> tuple[str, ...]:
        """Return component numbers whose BOM rows the ECN starts or ends.

        Enriched eBOM rows carry the ECN number itself; MBOM rows carry a
        derived code embedding the ECN digits (``MPXA3452740A``). HANA narrows
        MBOM candidates with ``LIKE`` on the digits; ``change_code_matches``
        then applies the exact rule so near-miss codes are dropped.

        Args:
            artifact_id: Canonical UUID of the pinned BOM artifact.
            ecn_number: ECN number of the assessment.

        Returns:
            Sorted unique component numbers, empty when the ECN has no digits.
        """
        from app.ecn.evidence.bom_changes import ECN_DIGITS, change_code_matches

        canonical_uuid(artifact_id)
        digits = ECN_DIGITS.search(ecn_number or "")
        if not ecn_number:
            return ()
        from app.ecn.settings import load_ecn_settings
        pattern = load_ecn_settings().mbom_change_number_pattern
        statement = text(
            'SELECT "COMPONENT_NUMBER", "CHANGE_NUMBER", "CHANGE_NUMBER_TO", "BOM_KIND" '
            f"FROM {quote_identifier(self._tables.bom_row)} "
            'WHERE "ARTIFACT_ID" = :artifact_id AND "COMPONENT_NUMBER" IS NOT NULL AND ('
            '"CHANGE_NUMBER" = :ecn OR "CHANGE_NUMBER_TO" = :ecn OR '
            '("BOM_KIND" = :mbom AND ("CHANGE_NUMBER" LIKE :pattern OR "CHANGE_NUMBER_TO" LIKE :pattern)))'
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement, {
                "artifact_id": artifact_id, "ecn": ecn_number, "pattern": f"%{digits['digits']}%" if digits else ecn_number,
                "enriched": BomKind.ENRICHED_EBOM.value, "mbom": BomKind.MBOM_130.value,
            }).all()
        return tuple(sorted({
            row[0] for row in rows
            if change_code_matches(ecn_number, row[1], row[3], pattern) or change_code_matches(ecn_number, row[2], row[3], pattern)
        }))

    def snapshot_bom_rows(
        self,
        artifact_id: str,
        component_numbers: Sequence[str],
        change_number: str | None,
    ) -> tuple[BomRow, ...]:
        """Return snapshot-relevant BOM rows and their ancestor chain.

        Matching is performed in HANA rather than by materializing the complete
        artifact. Enriched eBOM and MBOM rows are selected by component number;
        raw eBOM rows are selected by the ECN change number. Trusted parent row
        IDs are then followed until no unseen ancestor remains. Bind lists are
        capped at 500 values per query.

        Args:
            artifact_id: Canonical UUID of the pinned BOM artifact.
            component_numbers: Technical object numbers referenced by the ECN.
            change_number: ECN number used to select raw change rows, if known.

        Returns:
            A stable tuple of matching rows and ancestors, without duplicates.

        Raises:
            ValueError: When ``artifact_id`` is not a canonical UUID.
            PersistenceStateError: When a selected row cannot be decoded.
        """
        canonical_uuid(artifact_id)
        numbers = tuple(
            dict.fromkeys(
                number.strip() for number in component_numbers if number.strip()
            )
        )
        table = quote_identifier(self._tables.bom_row)
        location_statement = text(
            f"SELECT {_BOM_SELECT_COLUMNS} FROM {table} "
            'WHERE "ARTIFACT_ID" = :artifact_id '
            'AND "COMPONENT_NUMBER" IN :component_numbers '
            'AND "BOM_KIND" IN :bom_kinds'
        ).bindparams(
            bindparam("component_numbers", expanding=True),
            bindparam("bom_kinds", expanding=True),
        )
        change_statement = text(
            f"SELECT {_BOM_SELECT_COLUMNS} FROM {table} "
            'WHERE "ARTIFACT_ID" = :artifact_id '
            'AND "CHANGE_NUMBER" = :change_number '
            'AND "BOM_KIND" = :bom_kind '
            'AND "ROW_CLASSIFICATION" = :row_classification'
        )
        parent_statement = text(
            f"SELECT {_BOM_SELECT_COLUMNS} FROM {table} "
            'WHERE "ARTIFACT_ID" = :artifact_id AND "BOM_ROW_ID" IN :parent_ids'
        ).bindparams(bindparam("parent_ids", expanding=True))
        selected: dict[str, BomRow] = {}

        with self._engine.connect() as connection:
            for offset in range(0, len(numbers), 500):
                rows = connection.execute(
                    location_statement,
                    {
                        "artifact_id": artifact_id,
                        "component_numbers": numbers[offset : offset + 500],
                        "bom_kinds": (
                            BomKind.ENRICHED_EBOM.value,
                            BomKind.RAW_EBOM_130.value,
                            BomKind.MBOM_130.value,
                        ),
                    },
                ).mappings().all()
                selected.update(
                    (row.bom_row_id, row) for row in map(bom_row_from_row, rows)
                )

            if change_number and change_number.strip():
                rows = connection.execute(
                    change_statement,
                    {
                        "artifact_id": artifact_id,
                        "change_number": change_number.strip(),
                        "bom_kind": BomKind.RAW_EBOM_130.value,
                        "row_classification": (
                            BomRowClassification.MATERIAL_LIKE_OBSERVED.value
                        ),
                    },
                ).mappings().all()
                selected.update(
                    (row.bom_row_id, row) for row in map(bom_row_from_row, rows)
                )

            pending = {
                row.parent_bom_row_id
                for row in selected.values()
                if row.parent_bom_row_id and row.parent_bom_row_id not in selected
            }
            while pending:
                parent_ids = tuple(sorted(pending)[:500])
                rows = connection.execute(
                    parent_statement,
                    {"artifact_id": artifact_id, "parent_ids": parent_ids},
                ).mappings().all()
                parents = tuple(map(bom_row_from_row, rows))
                selected.update((row.bom_row_id, row) for row in parents)
                pending.difference_update(parent_ids)
                pending.update(
                    row.parent_bom_row_id
                    for row in parents
                    if row.parent_bom_row_id and row.parent_bom_row_id not in selected
                )

        return tuple(
            sorted(
                selected.values(),
                key=lambda row: (
                    row.bom_kind.value,
                    row.sheet_name,
                    row.source_row_number,
                    row.bom_row_id,
                ),
            )
        )
