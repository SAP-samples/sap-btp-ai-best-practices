"""Guarded dependency-ordered history deletion for the ECN impact pipeline.

This module implements ``HistoryRepository``, which owns the ``clear_history_if_idle``
operation and its scoped counterpart ``delete_ecns_if_idle`` (selected ECN line
items only, see ``docs/ecn-scoped-delete.md``).  The operation atomically checks for active (queued or running) analysis
jobs and, only when none exist, deletes all historical data in the FK-safe reverse
dependency order.

Design rules enforced here
--------------------------
* One transaction: the active-job count check and all dependency-ordered DELETE statements
  share a single ``engine.begin()`` context so that a concurrent publication
  cannot race the deletion.
* Exact dependency order:
  analysis → llm_attempt → analysis_job → rdf_projection →
  manufacturing_feasibility_assessment → ecn → upload → bom_row →
  source_artifact (ECN_JSON and BOM_WORKBOOK kinds only)
* External-reference preflight: before any DELETE, tables the app does not own
  that still reference the ECNs in scope are detected
  (``app.persistence.external_references``) and the deletion is refused with
  ``ExternalEcnReferenceError``.  Such tables are never modified here.
* No DROP or TRUNCATE: table definitions are always preserved.
* Conditional-rule artifact preservation: the ``ECN_IMPACT_SOURCE_ARTIFACT``
  delete filters by ``ARTIFACT_KIND IN ('ECN_JSON', 'BOM_WORKBOOK')`` so that
  ``CONDITIONAL_RULE_WORKBOOK`` rows are retained.
* Parameterized artifact kinds: the artifact-kind values are bound parameters
  (not literals) so the SQL fragment is consistent with the rest of the module.
* Process-guard precondition (caller responsibility): the calling code must
  serialize ``clear_history_if_idle`` against upload publication via a
  process-wide mutex or equivalent mechanism.  The active-job check inside
  the transaction provides only a database-level race guard.
* Short transactions: no ``Connection`` is retained across methods.
* Every value is a bound parameter; the only text interpolated into SQL is an
  allowlisted table identifier validated by ``quote_identifier``.

SQL safety rules
----------------
* Every user or caller-supplied value is a bound parameter.
* The only text interpolated into SQL is a table identifier validated against
  ``^ECN_IMPACT_[A-Z0-9_]{1,110}$`` by ``quote_identifier``.

Usage::

    from app.persistence.history_repository import HistoryRepository
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    repo = HistoryRepository(engine)
    counts = repo.clear_history_if_idle()
    if counts is None:
        print("Active work blocked deletion")
    else:
        print(f"Deleted {counts.jobs} jobs and {counts.analyses} analyses")
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from app.persistence.codec import normalize_row_keys
from app.persistence.external_references import (
    ExternalEcnReferenceError,
    find_external_ecn_references,
)
from app.persistence.records import EcnDeletionCounts, HistoryDeletionCounts
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier


# ---------------------------------------------------------------------------
# SQL fragment builders
# ---------------------------------------------------------------------------


def _active_job_count_sql(analysis_job_table: str) -> str:
    """Return the SELECT that counts QUEUED and RUNNING jobs.

    Returns exactly one integer row via ``scalar_one()``.  A non-zero result
    means active work is in progress and deletion must be aborted.

    Bind parameters use lowercase names (``queued``, ``running``) to match
    the scalar dict passed at call time.

    Args:
        analysis_job_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL string returning a single integer count.
    """
    t = quote_identifier(analysis_job_table)
    return (
        f'SELECT COUNT(1) FROM {t} '
        f'WHERE "STATE" IN (:queued, :running)'
    )


def _delete_all_sql(table_name: str) -> str:
    """Return an unconditional DELETE for all rows in the given table.

    Args:
        table_name: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterless DELETE SQL string.
    """
    t = quote_identifier(table_name)
    return f"DELETE FROM {t}"


def _delete_legacy_bom_rows_sql(tables: PersistenceTableNames) -> str:
    """Delete BOM rows not owned by an immutable Settings publication."""
    rows = quote_identifier(tables.bom_row)
    datasets = quote_identifier(tables.reference_dataset)
    return (
        f'DELETE FROM {rows} WHERE "ARTIFACT_ID" NOT IN ('
        f'SELECT "ARTIFACT_ID" FROM {datasets})'
    )


def _delete_source_history_sql(
    source_artifact_table: str,
    reference_dataset_table: str,
) -> str:
    """Delete ECN artifacts and only legacy, unreferenced BOM artifacts.

    CONDITIONAL_RULE_WORKBOOK artifacts are excluded so that conditional-rule
    workbook data is preserved across history resets.  The artifact kinds are
    bound parameters so the SQL fragment follows the same bind-parameter
    discipline as the rest of the module.

    Bind parameters:
    - ``:ecn_kind``  — ``"ECN_JSON"`` artifact kind value.
    - ``:bom_kind``  — ``"BOM_WORKBOOK"`` artifact kind value.

    Args:
        source_artifact_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized DELETE SQL string with ``ecn_kind`` and ``bom_kind`` binds.
    """
    t = quote_identifier(source_artifact_table)
    datasets = quote_identifier(reference_dataset_table)
    return (
        f'DELETE FROM {t} WHERE "ARTIFACT_KIND" = :ecn_kind OR ('
        f'"ARTIFACT_KIND" = :bom_kind AND "ARTIFACT_ID" NOT IN ('
        f'SELECT "ARTIFACT_ID" FROM {datasets}))'
    )


def _delete_all(connection: object, table_name: str) -> int:
    """Execute an unconditional DELETE and return the rowcount.

    Args:
        connection: An open SQLAlchemy connection-like object supporting ``execute``.
        table_name: Physical table name validated by ``quote_identifier``.

    Returns:
        Number of rows deleted.
    """
    result = connection.execute(text(_delete_all_sql(table_name)))  # type: ignore[attr-defined]
    return int(result.rowcount)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class HistoryRepository:
    """Delete all historical analysis data atomically only when no work is active.

    The public ``clear_history_if_idle`` method performs a two-phase operation
    inside one transaction:

    1. Count QUEUED and RUNNING jobs; abort if any exist.
    2. Delete data in FK-safe reverse dependency order, preserving rule artifacts
       and all table definitions.

    All public methods open a short-lived ``engine.begin()`` context per call.
    No ``Connection`` is retained as instance state between calls.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the HANA instance.
        table_names: Optional ``PersistenceTableNames`` override; defaults to
            the production names.  Pass a suffixed instance for isolated live
            tests.

    Example::

        repo = HistoryRepository(engine)
        counts = repo.clear_history_if_idle()
        if counts is None:
            raise Conflict("cannot reset while jobs are active")
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

    def clear_history_if_idle(self) -> HistoryDeletionCounts | None:
        """Delete historical analysis data atomically only when no work is active.

        Executes inside a single ``engine.begin()`` transaction.  First checks
        for QUEUED or RUNNING jobs; returns ``None`` immediately (without any
        DELETE) when any active work is found.  When no active work exists,
        executes dependency-ordered DELETE statements required
        to satisfy FK constraints:

        1. ``ECN_IMPACT_ANALYSIS``
        2. ``ECN_IMPACT_LLM_ATTEMPT``
        3. ``ECN_IMPACT_ANALYSIS_JOB``
        4. ``ECN_IMPACT_RDF_PROJECTION``
        4b. ``ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT``
        5. ``ECN_IMPACT_ECN``
        6. ``ECN_IMPACT_UPLOAD``
        7. ``ECN_IMPACT_BOM_ROW``
        8. ``ECN_IMPACT_SOURCE_ARTIFACT`` (ECN_JSON and BOM_WORKBOOK kinds only)

        CONDITIONAL_RULE_WORKBOOK artifacts and all table definitions are
        preserved.

        Caller precondition: the calling code must serialize this operation
        against upload publication via a process-wide mutex or equivalent
        mechanism so that no new publication races the active-job check.

        Returns:
            A ``HistoryDeletionCounts`` record with per-table row counts when
            deletion succeeded, or ``None`` when active work blocked deletion.

        Raises:
            ExternalEcnReferenceError: If a table the app does not own still
                references ECN rows; nothing is deleted.
        """
        with self._engine.begin() as connection:
            active = int(
                connection.execute(
                    text(_active_job_count_sql(self._tables.analysis_job)),
                    {"queued": "QUEUED", "running": "RUNNING"},
                ).scalar_one()
            )
            if active:
                return None
            # Refuse before any DELETE when foreign tables still point at ECNs.
            blocking = find_external_ecn_references(connection, self._tables, None)
            if blocking:
                raise ExternalEcnReferenceError(blocking)

            # Delete in FK-safe dependency order (leaf tables first).
            _delete_all(connection, self._tables.review)
            _delete_all(connection, self._tables.batch_job)
            _delete_all(connection, self._tables.generation_batch)
            analyses = _delete_all(connection, self._tables.analysis)
            _delete_all(connection, self._tables.provider_call)
            _delete_all(connection, self._tables.analysis_stage)
            attempts = _delete_all(connection, self._tables.llm_attempt)
            jobs = _delete_all(connection, self._tables.analysis_job)
            _delete_all(connection, self._tables.rdf_projection)
            _delete_all(connection, self._tables.manufacturing_feasibility_assessment)
            ecns = _delete_all(connection, self._tables.ecn)
            _delete_all(connection, self._tables.upload_dataset)
            uploads = _delete_all(connection, self._tables.upload)
            bom_rows = connection.execute(
                text(_delete_legacy_bom_rows_sql(self._tables))
            ).rowcount
            source_artifacts = connection.execute(
                text(
                    _delete_source_history_sql(
                        self._tables.source_artifact,
                        self._tables.reference_dataset,
                    )
                ),
                {"ecn_kind": "ECN_JSON", "bom_kind": "BOM_WORKBOOK"},
            ).rowcount

        return HistoryDeletionCounts(
            analyses=analyses,
            attempts=attempts,
            jobs=jobs,
            ecns=ecns,
            uploads=uploads,
            bom_rows=int(bom_rows),
            source_artifacts=int(source_artifacts),
        )

    def delete_ecns_if_idle(
        self, ecn_row_ids: Sequence[str]
    ) -> EcnDeletionCounts | None:
        """Delete selected ECN line items (all assessment versions) atomically.

        Scoped counterpart of ``clear_history_if_idle``: same single-transaction
        active-job guard, limited to the selected ECNs.  See ``_delete_ecn_scope``.

        Args:
            ecn_row_ids: Selected ECN row IDs; unknown IDs are ignored.

        Returns:
            ``EcnDeletionCounts`` after commit, or ``None`` when a selected ECN
            has a queued or running job and nothing was deleted.
        """
        return _delete_ecn_scope(self._engine, self._tables, ecn_row_ids)


# ---------------------------------------------------------------------------
# Scoped ECN line-item deletion
# ---------------------------------------------------------------------------


def _expanded(sql: str, *names: str):
    """Return a text clause whose named binds each expand a Python sequence.

    Args:
        sql: SQL containing ``IN :name`` placeholders.
        names: Bind names that receive tuples of IDs.

    Returns:
        A ``TextClause`` ready for ``connection.execute``.
    """
    return text(sql).bindparams(*(bindparam(name, expanding=True) for name in names))


def _ecn_scope_sql(tables: PersistenceTableNames) -> str:
    """Select every assessment version sharing a selected row's source position.

    A reassessment reuses the selected row's ECN_JSON artifact and keeps its
    ``SOURCE_ORDINAL``, so the pair ``(ECN_ARTIFACT_ID, SOURCE_ORDINAL)``
    identifies one line item across all of its versions.  Separate imports of
    the same ECN get distinct artifacts and are therefore separate line items.

    Returns:
        SQL returning ``ECN_ROW_ID``, ``UPLOAD_ID`` and ``ECN_ARTIFACT_ID`` rows.
    """
    ecn = quote_identifier(tables.ecn)
    upload = quote_identifier(tables.upload)
    return (
        f'SELECT DISTINCT V."ECN_ROW_ID", V."UPLOAD_ID", VU."ECN_ARTIFACT_ID" '
        f'FROM {ecn} S JOIN {upload} SU ON S."UPLOAD_ID" = SU."UPLOAD_ID" '
        f'JOIN {upload} VU ON VU."ECN_ARTIFACT_ID" = SU."ECN_ARTIFACT_ID" '
        f'JOIN {ecn} V ON V."UPLOAD_ID" = VU."UPLOAD_ID" '
        f'AND V."SOURCE_ORDINAL" = S."SOURCE_ORDINAL" '
        f'WHERE S."ECN_ROW_ID" IN :ids'
    )


def _scoped_delete_statements(tables: PersistenceTableNames) -> list[tuple[str, str]]:
    """Return ``(count_key, sql)`` pairs in FK-safe order for the ECN scope.

    Every statement binds ``:ids`` (expanded ECN row IDs); the orphan cleanup
    statements additionally bind ``:uploads``, ``:artifacts`` or ``:batches``
    and delete a parent only when no remaining child references it.

    Returns:
        Ordered statements; ``count_key`` is ``""`` when the rowcount is not reported.
    """
    q = quote_identifier
    t = tables
    jobs = f'SELECT "JOB_ID" FROM {q(t.analysis_job)} WHERE "ECN_ROW_ID" IN :ids'
    attempts = f'SELECT "ATTEMPT_ID" FROM {q(t.llm_attempt)} WHERE "JOB_ID" IN ({jobs})'
    return [
        ("reviews", f'DELETE FROM {q(t.review)} WHERE "ECN_ROW_ID" IN :ids'),
        ("", f'DELETE FROM {q(t.batch_job)} WHERE "JOB_ID" IN ({jobs})'),
        ("analyses", f'DELETE FROM {q(t.analysis)} WHERE "ECN_ROW_ID" IN :ids'),
        ("", f'DELETE FROM {q(t.provider_call)} WHERE "ATTEMPT_ID" IN ({attempts})'),
        ("", f'DELETE FROM {q(t.analysis_stage)} WHERE "JOB_ID" IN ({jobs})'),
        ("attempts", f'DELETE FROM {q(t.llm_attempt)} WHERE "JOB_ID" IN ({jobs})'),
        ("jobs", f'DELETE FROM {q(t.analysis_job)} WHERE "ECN_ROW_ID" IN :ids'),
        ("", f'DELETE FROM {q(t.rdf_projection)} WHERE "ECN_ROW_ID" IN :ids'),
        (
            "feasibility_assessments",
            f'DELETE FROM {q(t.manufacturing_feasibility_assessment)} WHERE "ECN_ROW_ID" IN :ids',
        ),
        ("ecns", f'DELETE FROM {q(t.ecn)} WHERE "ECN_ROW_ID" IN :ids'),
        (
            "",
            f'DELETE FROM {q(t.upload_dataset)} WHERE "UPLOAD_ID" IN :uploads '
            f'AND "UPLOAD_ID" NOT IN (SELECT "UPLOAD_ID" FROM {q(t.ecn)})',
        ),
        (
            "uploads",
            f'DELETE FROM {q(t.upload)} WHERE "UPLOAD_ID" IN :uploads '
            f'AND "UPLOAD_ID" NOT IN (SELECT "UPLOAD_ID" FROM {q(t.ecn)})',
        ),
        (
            "source_artifacts",
            f'DELETE FROM {q(t.source_artifact)} WHERE "ARTIFACT_ID" IN :artifacts '
            f'AND "ARTIFACT_KIND" = :ecn_kind '
            f'AND "ARTIFACT_ID" NOT IN (SELECT "ECN_ARTIFACT_ID" FROM {q(t.upload)})',
        ),
        (
            "",
            f'DELETE FROM {q(t.generation_batch)} WHERE "BATCH_ID" IN :batches '
            f'AND "BATCH_ID" NOT IN (SELECT "BATCH_ID" FROM {q(t.batch_job)})',
        ),
    ]


def _delete_ecn_scope(
    engine: Engine, tables: PersistenceTableNames, ecn_row_ids: Sequence[str]
) -> EcnDeletionCounts | None:
    """Delete selected ECN line items atomically only when none of them is active.

    One ``engine.begin()`` transaction:

    1. Expand the selection to every assessment version (see ``_ecn_scope_sql``).
    2. Return ``None`` without deleting if any expanded ECN has a QUEUED or
       RUNNING job.
    3. Raise ``ExternalEcnReferenceError`` without deleting if a table the app
       does not own still references an expanded ECN.
    4. Delete reviews, batch membership, analyses, provider calls, stages,
       attempts, jobs, RDF status, manufacturing feasibility assessments and
       ECN rows.
    5. Delete uploads, upload dataset pins, ECN_JSON artifacts and generation
       batches only when nothing else still references them.

    Reference datasets and their rows are never touched.  IDs that no longer
    exist are ignored, so repeating a request is harmless.

    Args:
        engine: SQLAlchemy engine connected to HANA.
        tables: Physical table names.
        ecn_row_ids: Selected ECN row IDs.

    Returns:
        ``EcnDeletionCounts`` after commit, or ``None`` when active work blocked it.

    Raises:
        ExternalEcnReferenceError: If a table the app does not own still
            references a selected ECN; nothing is deleted.
    """
    empty = EcnDeletionCounts((), 0, 0, 0, 0, 0, 0, 0)
    if not ecn_row_ids:
        return empty
    # ponytail: one expanding IN list per statement; chunk if selections exceed HANA's parameter limit.
    with engine.begin() as connection:
        # sqlalchemy-hana may return lowercase column keys; normalize before reading.
        scope = [
            normalize_row_keys(row)
            for row in connection.execute(
                _expanded(_ecn_scope_sql(tables), "ids"), {"ids": tuple(ecn_row_ids)}
            ).mappings().all()
        ]
        if not scope:
            return empty
        ids = tuple(sorted({str(row["ECN_ROW_ID"]) for row in scope}))
        params = {
            "ids": ids,
            "uploads": tuple(sorted({str(row["UPLOAD_ID"]) for row in scope})),
            "artifacts": tuple(sorted({str(row["ECN_ARTIFACT_ID"]) for row in scope})),
        }
        active = int(
            connection.execute(
                _expanded(
                    _active_job_count_sql(tables.analysis_job) + ' AND "ECN_ROW_ID" IN :ids',
                    "ids",
                ),
                {"queued": "QUEUED", "running": "RUNNING", "ids": ids},
            ).scalar_one()
        )
        if active:
            return None
        # Refuse before any DELETE when foreign tables still point at these ECNs.
        blocking = find_external_ecn_references(connection, tables, ids)
        if blocking:
            raise ExternalEcnReferenceError(blocking)
        batch_rows = connection.execute(
            _expanded(
                f'SELECT DISTINCT "BATCH_ID" FROM {quote_identifier(tables.batch_job)} '
                f'WHERE "JOB_ID" IN (SELECT "JOB_ID" FROM '
                f'{quote_identifier(tables.analysis_job)} WHERE "ECN_ROW_ID" IN :ids)',
                "ids",
            ),
            {"ids": ids},
        ).mappings().all()
        params["batches"] = tuple(
            str(normalize_row_keys(row)["BATCH_ID"]) for row in batch_rows
        )
        params["ecn_kind"] = "ECN_JSON"

        counts: dict[str, int] = {}
        for key, sql in _scoped_delete_statements(tables):
            names = [name for name in ("ids", "uploads", "artifacts", "batches") if f":{name}" in sql]
            bound = {name: params[name] for name in names}
            if any(not values for values in bound.values()):
                continue  # nothing to clean up (e.g. never part of a generation batch)
            if ":ecn_kind" in sql:
                bound["ecn_kind"] = params["ecn_kind"]
            rowcount = int(connection.execute(_expanded(sql, *names), bound).rowcount)
            if key:
                counts[key] = rowcount

    return EcnDeletionCounts(ecn_row_ids=ids, **counts)

