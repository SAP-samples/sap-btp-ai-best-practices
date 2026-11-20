"""Unit tests for HistoryRepository: guarded dependency-ordered history deletion.

These tests verify the ``clear_history_if_idle`` behaviour using the
``RecordingEngine`` fake so no live HANA connection is required.  Every test
asserts real behavioral semantics: the active-work guard, exact DELETE order,
conditional-rule artifact preservation, bind-parameter discipline, and
rollback on any failure.

Key scenarios covered
---------------------

* ``test_active_work_blocks_history_deletion`` — a non-zero active-job count
  short-circuits and returns ``None`` without executing any DELETE.
* ``test_idle_reset_deletes_dependencies_and_preserves_rules`` — seven DELETEs
  in exact FK-safe order; source-artifact DELETE uses ``ecn_kind``/``bom_kind``
  bind params so CONDITIONAL_RULE_WORKBOOK rows are preserved.
* ``test_idle_reset_returns_deletion_counts`` — the returned
  ``HistoryDeletionCounts`` carries the exact rowcounts reported by each DELETE.
* ``test_active_check_and_deletes_share_one_transaction`` — active check and all
  deletes execute inside exactly one ``engine.begin()`` context.
* ``test_all_table_definitions_untouched`` — no DROP or TRUNCATE appears anywhere
  in the execution log.
* ``test_rollback_counted_on_delete_exception`` — a queued exception inside the
  transaction increments ``rollback_count``.
* ``test_delete_source_history_sql_bind_params_are_subset_of_params_dict`` —
  bind-parameter discipline guard for the source-artifact DELETE statement.

Example run::

    PYTHONPATH=api python -m pytest -q tests/unit/test_history_repository.py
"""

from __future__ import annotations

import re as _re

import pytest

from tests.fixtures.persistence import RecordingEngine
from sqlalchemy.exc import SQLAlchemyError


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_engine() -> RecordingEngine:
    """Return a fresh RecordingEngine for each test.

    Returns:
        A new ``RecordingEngine`` with empty queue and counters.
    """
    return RecordingEngine()


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def delete_target(sql: str) -> str:
    """Extract the unquoted table name from a ``DELETE FROM "TABLE_NAME"`` statement.

    Args:
        sql: A SQL string that may start with ``DELETE FROM``.

    Returns:
        The table name without surrounding quotes, or an empty string when
        the statement is not a DELETE.
    """
    match = _re.search(r'DELETE FROM "([^"]+)"', sql)
    return match.group(1) if match else ""


# ===========================================================================
# HistoryRepository tests
# ===========================================================================


def test_active_work_blocks_history_deletion(recording_engine) -> None:
    """A non-zero active-job count returns None and executes no DELETE statements.

    The active count is checked inside the same transaction as the deletes, so
    when it is non-zero the transaction still commits (nothing was modified).
    """
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(1)  # one queued or running job
    result = HistoryRepository(recording_engine).clear_history_if_idle()
    assert result is None
    assert not any(sql.startswith("DELETE") for sql, _ in recording_engine.executions)


def test_idle_reset_deletes_projection_status_before_ecns_and_preserves_rules(
    recording_engine,
) -> None:
    """Dependency-ordered deletes preserve rule artifacts and remove RDF state first.

    The exact target order is:
      ECN_IMPACT_ANALYSIS → ECN_IMPACT_LLM_ATTEMPT → ECN_IMPACT_ANALYSIS_JOB →
      ECN_IMPACT_RDF_PROJECTION → ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT → ECN_IMPACT_ECN → ECN_IMPACT_UPLOAD → ECN_IMPACT_BOM_ROW →
      ECN_IMPACT_SOURCE_ARTIFACT

    The last DELETE (source artifacts) must use ``:ecn_kind`` and ``:bom_kind``
    bind params (not CONDITIONAL_RULE_WORKBOOK), so conditional-rule workbook
    rows are preserved.
    """
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(0)  # no active jobs
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_rowcounts([0, 0, 0, 2, 9, 0, 3, 1, 4, 0, 8, 1, 20, 2, 0])
    counts = HistoryRepository(recording_engine).clear_history_if_idle()
    assert counts is not None

    targets = [
        delete_target(sql)
        for sql, _ in recording_engine.executions
        if sql.startswith("DELETE")
    ]
    assert targets == [
        "ECN_IMPACT_REVIEW",
        "ECN_IMPACT_BATCH_JOB",
        "ECN_IMPACT_GENERATION_BATCH",
        "ECN_IMPACT_ANALYSIS",
        "ECN_IMPACT_PROVIDER_CALL",
        "ECN_IMPACT_ANALYSIS_STAGE",
        "ECN_IMPACT_LLM_ATTEMPT",
        "ECN_IMPACT_ANALYSIS_JOB",
        "ECN_IMPACT_RDF_PROJECTION",
        "ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT",
        "ECN_IMPACT_ECN",
        "ECN_IMPACT_UPLOAD_DATASET",
        "ECN_IMPACT_UPLOAD",
        "ECN_IMPACT_BOM_ROW",
        "ECN_IMPACT_SOURCE_ARTIFACT",
    ]

    # Last DELETE params must not contain CONDITIONAL_RULE_WORKBOOK.
    last_params: dict = recording_engine.executions[-1][1]
    assert "CONDITIONAL_RULE_WORKBOOK" not in last_params.values()

    # Last DELETE SQL must use the ecn_kind/bom_kind bind names.
    artifact_sqls = [
        sql for sql, _ in recording_engine.executions if "SOURCE_ARTIFACT" in sql
    ]
    assert artifact_sqls, "Expected at least one SOURCE_ARTIFACT DELETE"
    artifact_sql = artifact_sqls[-1]
    assert '"ARTIFACT_KIND" = :ecn_kind' in artifact_sql
    assert '"ARTIFACT_KIND" = :bom_kind' in artifact_sql
    assert "ECN_IMPACT_REFERENCE_DATASET" in artifact_sql


def test_idle_reset_returns_deletion_counts(recording_engine) -> None:
    """The returned HistoryDeletionCounts carries the exact per-table rowcounts."""
    from app.persistence.history_repository import HistoryRepository
    from app.persistence.records import HistoryDeletionCounts

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    # Rowcounts include projection status and feasibility rows before their ECN parent.
    recording_engine.queue_rowcounts([0, 0, 0, 5, 9, 0, 8, 3, 0, 0, 7, 6, 2, 30, 4])
    counts = HistoryRepository(recording_engine).clear_history_if_idle()
    assert isinstance(counts, HistoryDeletionCounts)
    assert counts.analyses == 5
    assert counts.attempts == 8
    assert counts.jobs == 3
    assert counts.ecns == 7
    assert counts.uploads == 2
    assert counts.bom_rows == 30
    assert counts.source_artifacts == 4


def test_active_check_and_deletes_share_one_transaction(recording_engine) -> None:
    """The active-job check and all seven DELETEs occur inside a single begin() context."""
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_rowcounts([0] * 13)
    HistoryRepository(recording_engine).clear_history_if_idle()
    assert recording_engine.begin_count == 1


def test_zero_active_jobs_produces_non_none_result(recording_engine) -> None:
    """When active count is zero, clear_history_if_idle returns a non-None counts object."""
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_rowcounts([0] * 14)
    result = HistoryRepository(recording_engine).clear_history_if_idle()
    assert result is not None


def test_all_table_definitions_untouched(recording_engine) -> None:
    """No DROP or TRUNCATE statement is issued during history deletion."""
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_rowcounts([1] * 9)
    HistoryRepository(recording_engine).clear_history_if_idle()
    for sql, _ in recording_engine.executions:
        upper = sql.upper()
        assert not upper.startswith("DROP"), f"Unexpected DROP statement: {sql!r}"
        assert not upper.startswith("TRUNCATE"), f"Unexpected TRUNCATE statement: {sql!r}"


def test_rollback_counted_on_exception_during_delete(recording_engine) -> None:
    """An exception raised during a DELETE increments rollback_count."""
    from app.persistence.history_repository import HistoryRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_exception(SQLAlchemyError("simulated delete failure"))
    with pytest.raises(SQLAlchemyError):
        HistoryRepository(recording_engine).clear_history_if_idle()
    assert recording_engine.rollback_count == 1


def test_delete_source_history_sql_bind_params_are_subset_of_params_dict(
    recording_engine,
) -> None:
    """Every bind token in the source-history DELETE SQL must exist in its params dict.

    This is the bind-parameter discipline guard for the conditional source-artifact
    delete, ensuring that case/spelling drift between the SQL and the params dict
    never silently becomes a live-only runtime error.
    """
    from app.persistence.history_repository import _delete_source_history_sql
    from app.persistence.table_contracts import PersistenceTableNames

    names = PersistenceTableNames()
    sql = _delete_source_history_sql(
        names.source_artifact, names.reference_dataset
    )
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    params = {"ecn_kind": "ECN_JSON", "bom_kind": "BOM_WORKBOOK"}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"SQL bind tokens not found in params dict: {sorted(missing)}"
    )


def test_suffixed_table_names_used_in_delete_statements(recording_engine) -> None:
    """HistoryRepository uses the injected table names, not the default production names."""
    from app.persistence.history_repository import HistoryRepository
    from app.persistence.table_contracts import PersistenceTableNames

    suffix = "_TSUFFIX"
    names = PersistenceTableNames().with_suffix(suffix)
    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])  # external-reference preflight: no foreign tables
    recording_engine.queue_rowcounts([0] * 12)
    HistoryRepository(recording_engine, names).clear_history_if_idle()
    delete_sqls = [sql for sql, _ in recording_engine.executions if sql.startswith("DELETE")]
    assert len(delete_sqls) == 15
    for sql in delete_sqls:
        assert suffix in sql, f"Expected suffix {suffix!r} in DELETE: {sql!r}"
