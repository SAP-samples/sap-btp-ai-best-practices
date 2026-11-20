"""Guarded application service for idle-only analysis history deletion."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from sqlalchemy.exc import DBAPIError

from app.ecn.errors import PersistenceStateError, SourceValidationError
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.external_references import (
    ExternalEcnReferenceError,
    describe_external_references,
)
from app.persistence.history_repository import HistoryRepository
from app.persistence.records import EcnDeletionCounts, HistoryDeletionCounts


logger = logging.getLogger(__name__)

# HANA SQL error code for "failed on update or delete by foreign key constraint violation".
_HANA_FOREIGN_KEY_VIOLATION = 462


def _is_foreign_key_violation(exc: DBAPIError) -> bool:
    """Return whether a wrapped DB-API error is a HANA foreign-key violation.

    Args:
        exc: SQLAlchemy wrapper around the ``hdbcli`` error.

    Returns:
        ``True`` when the original error code is 462.
    """
    original = exc.orig
    code = getattr(original, "errorcode", None)
    if code is None and getattr(original, "args", None):
        code = original.args[0]
    return code == _HANA_FOREIGN_KEY_VIOLATION


def _external_reference_conflict(code: str, exc: ExternalEcnReferenceError) -> PersistenceStateError:
    """Build the 409 error that names the tables blocking a deletion.

    Args:
        code: Stable API error code for the calling operation.
        exc: Repository error listing the blocking tables.

    Returns:
        A ``PersistenceStateError`` with HTTP status 409.
    """
    return PersistenceStateError(
        code,
        "Nothing was deleted. These ECNs are still referenced by tables this application "
        f"does not manage: {describe_external_references(exc.references)}. "
        "Remove that data first (see docs/ecn-scoped-delete.md).",
        status_code=409,
    )


def _foreign_key_conflict(code: str) -> PersistenceStateError:
    """Build the 409 error for a foreign-key violation the preflight did not catch.

    Args:
        code: Stable API error code for the calling operation.

    Returns:
        A ``PersistenceStateError`` with HTTP status 409.
    """
    return PersistenceStateError(
        code,
        "Nothing was deleted. Other data in the database still references these ECNs. "
        "The server log names the table.",
        status_code=409,
    )


class HistoryService:
    """Validate and serialize destructive history reset operations."""

    def __init__(
        self,
        repository: HistoryRepository,
        mutation_guard: HistoryMutationGuard,
        clear_projection_graph: Callable[[], None] | None = None,
        delete_projection_graphs: Callable[[Sequence[str]], None] | None = None,
    ) -> None:
        """Bind the atomic repository operation and shared mutation guard.

        Args:
            repository: HANA repository owning active check and ordered deletes.
            mutation_guard: Process-local guard shared with final publication.
            clear_projection_graph: Optional derived-graph cleanup invoked only
                after the authoritative relational reset commits.
            delete_projection_graphs: Optional per-ECN graph cleanup invoked
                only after a scoped ECN deletion commits.
        """
        self._repository = repository
        self._guard = mutation_guard
        self._clear_projection_graph = clear_projection_graph
        self._delete_projection_graphs = delete_projection_graphs

    def reset_history(self, confirmation: str) -> HistoryDeletionCounts:
        """Clear idle analysis history while preserving rules and tables.

        Args:
            confirmation: Exact case-sensitive destructive phrase.

        Returns:
            Repository-reported deletion counts after a committed reset.

        Raises:
            SourceValidationError: If the confirmation is not exact.
            PersistenceStateError: If queued or running jobs prevent deletion,
                or other data still references the ECNs (409, nothing deleted).
        """
        if confirmation != "RESET HISTORY":
            raise SourceValidationError(
                "HISTORY_RESET_CONFIRMATION_INVALID",
                "Type RESET HISTORY exactly to clear analysis history.",
                status_code=400,
            )
        with self._guard.hold():
            try:
                counts = self._repository.clear_history_if_idle()
            except ExternalEcnReferenceError as exc:
                raise _external_reference_conflict(
                    "HISTORY_RESET_EXTERNAL_REFERENCES", exc
                ) from exc
            except DBAPIError as exc:
                if not _is_foreign_key_violation(exc):
                    raise
                # The transaction was rolled back by engine.begin(); keep the detail server-side.
                logger.exception("History reset blocked by a foreign key")
                raise _foreign_key_conflict("HISTORY_RESET_REFERENCED") from exc
            if counts is None:
                raise PersistenceStateError(
                    "HISTORY_RESET_ACTIVE_JOBS",
                    "Analysis history cannot be reset while jobs are queued or running.",
                    status_code=409,
                )
            if self._clear_projection_graph is not None:
                try:
                    self._clear_projection_graph()
                except Exception:  # noqa: BLE001 - derived cleanup cannot undo committed source reset.
                    logger.exception("RDF projection graph cleanup failed after history reset")
            return counts

    def delete_ecns(
        self, confirmation: str, ecn_row_ids: Sequence[str]
    ) -> tuple[EcnDeletionCounts, bool]:
        """Delete selected ECN line items as if they had never been imported.

        Scoped counterpart of ``reset_history``: every assessment version of
        each selected ECN and all of its summaries, reviews and graph triples
        are removed; reference datasets stay untouched.

        Args:
            confirmation: Exact case-sensitive phrase ``DELETE``.
            ecn_row_ids: Resolved ECN row IDs from the list selection.

        Returns:
            Committed deletion counts and whether graph cleanup succeeded.

        Raises:
            SourceValidationError: If the confirmation is not exact.
            PersistenceStateError: If a selected ECN has a queued or running job,
                or other data still references a selected ECN (409, nothing deleted).
        """
        if confirmation != "DELETE":
            raise SourceValidationError(
                "ECN_DELETE_CONFIRMATION_INVALID",
                "Type DELETE exactly to delete the selected ECNs.",
                status_code=400,
            )
        with self._guard.hold():
            try:
                counts = self._repository.delete_ecns_if_idle(ecn_row_ids)
            except ExternalEcnReferenceError as exc:
                raise _external_reference_conflict("ECN_DELETE_EXTERNAL_REFERENCES", exc) from exc
            except DBAPIError as exc:
                if not _is_foreign_key_violation(exc):
                    raise
                # The transaction was rolled back by engine.begin(); keep the detail server-side.
                logger.exception("ECN deletion blocked by a foreign key")
                raise _foreign_key_conflict("ECN_DELETE_REFERENCED") from exc
            if counts is None:
                raise PersistenceStateError(
                    "ECN_DELETE_ACTIVE_JOBS",
                    "Selected ECNs cannot be deleted while their summaries are queued or running.",
                    status_code=409,
                )
            graph_ok = True
            if self._delete_projection_graphs is not None and counts.ecn_row_ids:
                try:
                    self._delete_projection_graphs(counts.ecn_row_ids)
                except Exception:  # noqa: BLE001 - graph cleanup cannot undo the committed delete.
                    graph_ok = False
                    logger.exception("RDF graph cleanup failed after ECN deletion")
            return counts, graph_ok
