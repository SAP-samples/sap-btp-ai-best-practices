"""Post-commit lifecycle orchestration for rebuildable ECN RDF projections.

The relational ECN JSON and its durable projection status remain authoritative.
This service only invokes fixed repository operations after those facts commit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from threading import Lock
from typing import Literal

from app.ecn.errors import PersistenceNotFoundError, PersistenceStateError
from app.ecn.rdf import (
    PROJECTION_VERSION,
    GraphExpansionPage,
    GraphFactPage,
    GraphFieldPage,
    ProjectionBatchResult,
    ProjectionRecord,
    ProjectionState,
    ReplacementSourceObjectPage,
    project_ecn_json,
)
from app.persistence.rdf_repository import RdfRepository


logger = logging.getLogger(__name__)


def _as_ids(ecn_row_ids: str | Sequence[str]) -> tuple[str, ...]:
    """Return one identity or a sequence of identities as a tuple."""
    return (ecn_row_ids,) if isinstance(ecn_row_ids, str) else tuple(ecn_row_ids)


class RdfProjectionService:
    """Rebuild, verify, reconcile, and gate per-ECN derived RDF projections.

    Args:
        repository: Fixed HANA graph/status boundary.
        clock: Timezone-aware clock used for durable lifecycle timestamps.
    """

    def __init__(
        self,
        repository: RdfRepository,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind the fixed repository and lifecycle timestamp source."""
        self._repository = repository
        self._clock = clock
        # ponytail: one process-wide graph writer; use a database lock only if
        # the API is intentionally scaled beyond the single configured instance.
        self._mutation_lock = Lock()

    def project(self, ecn_row_id: str) -> ProjectionRecord:
        """Rebuild and verify one committed ECN projection from relational source.

        Args:
            ecn_row_id: Trusted selected ECN occurrence identity.

        Returns:
            The verified ``CURRENT`` record, or its durable ``FAILED`` record.

        Raises:
            PersistenceNotFoundError: If the authoritative ECN source is absent.
            PersistenceStateError: If a durable failure cannot itself be recorded.
        """
        with self._mutation_lock:
            source = self._repository.load_source(ecn_row_id)
            if source is None:
                raise PersistenceNotFoundError(
                    "RDF_PROJECTION_NOT_FOUND",
                    "The selected ECN source does not exist.",
                    status_code=404,
                )
            if self._repository.get_projection(ecn_row_id) is None:
                raise PersistenceStateError(
                    "RDF_PROJECTION_STATE_MISSING",
                    "The selected ECN has no projection state record.",
                    status_code=500,
                )
            attempted_at = self._clock()
            try:
                self._repository.mark_pending(ecn_row_id, attempted_at)
                projection = project_ecn_json(
                    ecn_row_id,
                    source.artifact_id,
                    source.source_json,
                )
                self._repository.replace_graph_projection(projection)
                verification = self._repository.verify_graph_projection(projection)
                if not verification.matches(projection):
                    raise PersistenceStateError(
                        "RDF_PROJECTION_VERIFICATION_FAILED",
                        "The RDF projection could not be verified after rebuild.",
                        status_code=500,
                    )
                self._repository.mark_current(
                    ecn_row_id,
                    verification.counts.node_count,
                    verification.counts.triple_count,
                    self._clock(),
                )
                return self._required_record(ecn_row_id)
            except Exception:  # noqa: BLE001 - graph drivers expose varied failures.
                logger.exception(
                    "RDF projection failed for selected ECN",
                    extra={"ecn_row_id": ecn_row_id},
                )
                try:
                    self._repository.mark_failed(
                        ecn_row_id,
                        "RDF_PROJECTION_FAILED",
                        "The RDF projection could not be rebuilt and verified.",
                        self._clock(),
                    )
                except Exception as record_exc:  # noqa: BLE001 - preserve the durable-state fault.
                    raise PersistenceStateError(
                        "RDF_PROJECTION_FAILURE_UNRECORDED",
                        "The RDF projection failure could not be recorded.",
                        status_code=500,
                    ) from record_exc
                return self._required_record(ecn_row_id)

    def project_many(self, ecn_row_ids: Sequence[str]) -> ProjectionBatchResult:
        """Attempt trusted ECNs sequentially without stopping on one failure.

        Args:
            ecn_row_ids: Selected ECN occurrence IDs returned by trusted code.

        Returns:
            Exact requested/current/failed counts for the submitted IDs.
        """
        current = 0
        for ecn_row_id in ecn_row_ids:
            try:
                current += self.project(ecn_row_id).status is ProjectionState.CURRENT
            except Exception:  # noqa: BLE001 - one source must not halt recovery.
                logger.exception(
                    "RDF projection batch item failed",
                    extra={"ecn_row_id": ecn_row_id},
                )
        return ProjectionBatchResult(len(ecn_row_ids), current, len(ecn_row_ids) - current)

    def reconcile(self) -> ProjectionBatchResult:
        """Rebuild only durable statuses selected by the repository at startup."""
        return self.project_many(self._repository.list_reconcilable(PROJECTION_VERSION))

    def get_projection(self, ecn_row_id: str) -> ProjectionRecord | None:
        """Return one durable projection status without starting graph work.

        Args:
            ecn_row_id: Trusted selected ECN occurrence identity.

        Returns:
            The current durable status record, or ``None`` when the ECN has no
            committed projection state.
        """
        return self._repository.get_projection(ecn_row_id)

    def find_facts(
        self,
        ecn_row_id: str | Sequence[str],
        query: str,
        search_in: Literal["predicate", "value", "both"],
        node_type: str | None,
        role: Literal["affected", "resulting"] | None,
        page: int,
    ) -> GraphFactPage:
        """Gate and delegate one bounded source-fact search for one or several ECNs.

        Args:
            ecn_row_id: Trusted ECN occurrence identity, or a sequence of them
                searched in one query; every one must be ``CURRENT`` (filter
                with ``current_ids`` first).
            query: Bounded lexical discovery term.
            search_in: Closed predicate/value search mode.
            node_type: Optional known projected node-type filter.
            role: Optional source-role filter.
            page: One-based bounded result page.

        Returns:
            One fixed-size source-fact page from a verified current projection.
        """
        self.require_all_current(_as_ids(ecn_row_id))
        return self._repository.find_facts(
            ecn_row_id, query, search_in, node_type, role, page
        )

    def list_fields(
        self,
        ecn_row_id: str | Sequence[str],
        query: str,
        page: int,
    ) -> GraphFieldPage:
        """Gate and list exact projected source-field names for one or several ECNs.

        Args:
            ecn_row_id: Trusted ECN occurrence identity, or a sequence of them;
                every one must be ``CURRENT``.
            query: Optional case-insensitive field-name fragment.
            page: One-based bounded result page.

        Returns:
            One fixed-size catalogue page from a verified current projection.
        """
        self.require_all_current(_as_ids(ecn_row_id))
        return self._repository.list_fields(ecn_row_id, query, page)

    def current_ids(self, ecn_row_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        """Split ECNs into those whose projection is ``CURRENT`` and the rest.

        Cross-ECN searches use this to search every ready ECN and report the
        others, instead of failing the whole search for one pending graph.

        Args:
            ecn_row_ids: Trusted ECN occurrence identities.

        Returns:
            ``(current, not_current)`` identity lists, in input order.
        """
        # ponytail: one status read per ECN; read all statuses in one SQL
        # query if the stored ECN count makes this noticeable.
        current: list[str] = []
        not_current: list[str] = []
        for ecn_row_id in ecn_row_ids:
            record = self._repository.get_projection(ecn_row_id)
            ready = record is not None and record.status is ProjectionState.CURRENT
            (current if ready else not_current).append(ecn_row_id)
        return current, not_current

    def expand_nodes(
        self,
        ecn_row_id: str,
        node_refs: Sequence[str],
        direction: Literal["outgoing", "incoming", "both"],
        predicates: Sequence[str],
        depth: int,
        page: int,
    ) -> GraphExpansionPage:
        """Gate and delegate one membership-checked local graph expansion.

        Args:
            ecn_row_id: Trusted selected ECN occurrence identity.
            node_refs: Root or trusted graph nodes issued in the current turn.
            direction: Closed local traversal direction.
            predicates: Bounded source-field or structural relation filters.
            depth: One or two local hops.
            page: One-based bounded result page.

        Returns:
            One fixed-size local expansion page from a verified current projection.
        """
        self.require_current(ecn_row_id)
        return self._repository.expand_nodes(
            ecn_row_id, node_refs, direction, predicates, depth, page
        )

    def replacement_source_objects(
        self, ecn_row_id: str, page: int
    ) -> ReplacementSourceObjectPage:
        """Gate and delegate one fixed source-input page for replacement candidates.

        Args:
            ecn_row_id: Trusted selected ECN occurrence identity.
            page: One-based source-object page within the fixed 2,000-object cap.

        Returns:
            Path-proven affected/resulting source objects from a verified current
            projection. The result is not a replacement mapping.
        """
        self.require_current(ecn_row_id)
        return self._repository.replacement_source_objects(ecn_row_id, page)

    def request_rebuild(self, ecn_row_id: str) -> ProjectionRecord:
        """Durably mark one existing ECN pending without starting graph work.

        Args:
            ecn_row_id: Route-selected ECN occurrence identity.

        Returns:
            The newly persisted ``PENDING`` record for post-commit scheduling.

        Raises:
            PersistenceNotFoundError: If the selected relational source is absent.
            PersistenceStateError: If the source lacks the required status record.
        """
        if self._repository.load_source(ecn_row_id) is None:
            raise PersistenceNotFoundError(
                "RDF_PROJECTION_NOT_FOUND",
                "The selected ECN source does not exist.",
                status_code=404,
            )
        if self._repository.get_projection(ecn_row_id) is None:
            raise PersistenceStateError(
                "RDF_PROJECTION_STATE_MISSING",
                "The selected ECN has no projection state record.",
                status_code=500,
            )
        self._repository.mark_pending(ecn_row_id, self._clock())
        return self._required_record(ecn_row_id)

    def require_current(self, ecn_row_id: str) -> ProjectionRecord:
        """Return a verified projection or reject a graph-dependent operation.

        Args:
            ecn_row_id: Trusted selected ECN occurrence identity.

        Returns:
            The selected ``CURRENT`` projection record.

        Raises:
            PersistenceStateError: With a clear 409-compatible state code when
                no verified projection is available.
        """
        record = self._repository.get_projection(ecn_row_id)
        if record is not None and record.status is ProjectionState.CURRENT:
            return record
        status = record.status if record is not None else ProjectionState.PENDING
        code = f"RDF_PROJECTION_{status.value}"
        raise PersistenceStateError(
            code,
            "The ECN RDF projection is not ready for graph-dependent work.",
            status_code=409,
        )

    def require_all_current(self, ecn_row_ids: Sequence[str]) -> None:
        """Reject a batch as soon as one selected ECN lacks verified RDF.

        Args:
            ecn_row_ids: Trusted selected ECN occurrence identities.
        """
        for ecn_row_id in ecn_row_ids:
            self.require_current(ecn_row_id)

    def delete_ecns(self, ecn_row_ids: Sequence[str]) -> None:
        """Remove graph triples of ECNs deleted from the relational tables.

        Args:
            ecn_row_ids: ECN row IDs whose relational rows were committed as deleted.
        """
        with self._mutation_lock:
            self._repository.delete_ecn_graphs(ecn_row_ids)

    def clear_all(self) -> None:
        """Clear only the fixed application graph after relational history reset."""
        with self._mutation_lock:
            self._repository.clear_application_graph()

    def _required_record(self, ecn_row_id: str) -> ProjectionRecord:
        """Read the status record just written by a successful durable transition."""
        record = self._repository.get_projection(ecn_row_id)
        if record is None:
            raise PersistenceStateError(
                "RDF_PROJECTION_STATE_MISSING",
                "The selected ECN has no projection state record.",
                status_code=500,
            )
        return record
