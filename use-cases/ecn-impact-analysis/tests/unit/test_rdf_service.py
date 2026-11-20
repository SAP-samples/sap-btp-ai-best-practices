"""Unit tests for the post-commit RDF projection lifecycle service.

The fakes here model only the fixed repository boundary.  They prove graph
work is recoverable and never changes the authoritative relational source.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, Lock

import pytest

from app.ecn.errors import PersistenceStateError
from app.ecn.rdf import (
    GraphExpansionPage,
    GraphFactPage,
    GraphFieldPage,
    PROJECTION_VERSION,
    ProjectionExpectedCounts,
    ProjectionRecord,
    ProjectionSource,
    ProjectionState,
    ProjectionVerification,
    ReplacementSourceObjectPage,
    project_ecn_json,
)


FIRST_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
SECOND_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3302"
ARTIFACT_ID = "11111111-2222-3333-4444-555555555555"
NOW = datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)


def _expected(projection: object) -> ProjectionExpectedCounts:
    """Extract the five relational baseline counts from one RDF projection."""
    counts = projection.counts  # type: ignore[attr-defined]
    return ProjectionExpectedCounts(
        counts.object_count,
        counts.array_count,
        counts.scalar_count,
        counts.null_count,
        counts.node_count,
    )


class _Repository:
    """In-memory fixed RDF repository with configurable write failures."""

    def __init__(self) -> None:
        """Initialize source, status, operation, and failure observations."""
        self.sources: dict[str, ProjectionSource] = {}
        self.statuses: dict[str, ProjectionRecord] = {}
        self.operations: list[str] = []
        self.failed_replacements: set[str] = set()
        self.reconcilable: tuple[str, ...] = ()
        self.graph_cleared = False
        self.graph_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def add(self, ecn_row_id: str, source_json: object) -> None:
        """Add one authoritative source and its durable pending status."""
        projection = project_ecn_json(ecn_row_id, ARTIFACT_ID, source_json)
        self.sources[ecn_row_id] = ProjectionSource(ecn_row_id, ARTIFACT_ID, source_json)
        self.statuses[ecn_row_id] = ProjectionRecord(
            ecn_row_id=ecn_row_id,
            source_sha256=projection.source_sha256,
            projection_version=projection.projection_version,
            status=ProjectionState.PENDING,
            expected_counts=_expected(projection),
            verified_node_count=None,
            verified_triple_count=None,
            error_code=None,
            error_message=None,
            last_attempt_at=None,
            current_at=None,
        )

    def get_projection(self, ecn_row_id: str) -> ProjectionRecord | None:
        """Return the exact durable projection record, if present."""
        return self.statuses.get(ecn_row_id)

    def load_source(self, ecn_row_id: str) -> ProjectionSource | None:
        """Return the authoritative source without mutating it."""
        return self.sources.get(ecn_row_id)

    def mark_pending(self, ecn_row_id: str, attempted_at: datetime) -> None:
        """Persist a pending state for the selected source only."""
        prior = self.statuses[ecn_row_id]
        self.operations.append(f"pending:{ecn_row_id}")
        self.statuses[ecn_row_id] = ProjectionRecord(
            **{
                **prior.__dict__,
                "status": ProjectionState.PENDING,
                "last_attempt_at": attempted_at,
                "current_at": None,
                "error_code": None,
                "error_message": None,
                "verified_node_count": None,
                "verified_triple_count": None,
            }
        )

    def replace_graph_projection(self, projection: object) -> None:
        """Record selected projection replacement or raise an injected failure."""
        ecn_row_id = projection.ecn_row_id  # type: ignore[attr-defined]
        self.operations.append(f"replace:{ecn_row_id}")
        if ecn_row_id in self.failed_replacements:
            raise PersistenceStateError(
                "RDF_SPARQL_FAILED", "Graph write failed.", status_code=500
            )

    def verify_graph_projection(self, projection: object) -> ProjectionVerification:
        """Return exact structural read-back facts for the supplied projection."""
        self.operations.append(f"verify:{projection.ecn_row_id}")  # type: ignore[attr-defined]
        return ProjectionVerification(
            source_sha256=projection.source_sha256,  # type: ignore[attr-defined]
            projection_version=projection.projection_version,  # type: ignore[attr-defined]
            counts=projection.counts,  # type: ignore[attr-defined]
            invalid_array_member_count=0,
            cross_ecn_membership_count=0,
        )

    def mark_current(
        self, ecn_row_id: str, nodes: int, triples: int, current_at: datetime
    ) -> None:
        """Persist exact verification counts only after a successful read-back."""
        prior = self.statuses[ecn_row_id]
        self.operations.append(f"current:{ecn_row_id}")
        self.statuses[ecn_row_id] = ProjectionRecord(
            **{
                **prior.__dict__,
                "status": ProjectionState.CURRENT,
                "verified_node_count": nodes,
                "verified_triple_count": triples,
                "last_attempt_at": current_at,
                "current_at": current_at,
            }
        )

    def mark_failed(
        self, ecn_row_id: str, code: str, message: str, attempted_at: datetime
    ) -> None:
        """Persist a terminal safe failure while retaining the source unchanged."""
        prior = self.statuses[ecn_row_id]
        self.operations.append(f"failed:{ecn_row_id}")
        self.statuses[ecn_row_id] = ProjectionRecord(
            **{
                **prior.__dict__,
                "status": ProjectionState.FAILED,
                "error_code": code,
                "error_message": message,
                "last_attempt_at": attempted_at,
                "current_at": None,
            }
        )

    def list_reconcilable(self, version: str) -> tuple[str, ...]:
        """Return the configured durable non-current/stale selection."""
        assert version == PROJECTION_VERSION
        return self.reconcilable

    def clear_application_graph(self) -> None:
        """Record application-graph cleanup after the relational reset commits."""
        self.graph_cleared = True

    def find_facts(self, *args, **kwargs) -> GraphFactPage:
        """Record graph discovery through the service's fixed repository boundary."""
        self.graph_calls.append(("find", args, kwargs))
        return GraphFactPage((), page=1, page_size=10, total_matches=0, has_next=False)

    def list_fields(self, *args, **kwargs) -> GraphFieldPage:
        """Record field discovery through the service's fixed repository boundary."""
        self.graph_calls.append(("fields", args, kwargs))
        return GraphFieldPage((), page=1, page_size=50, total_matches=0, has_next=False)

    def expand_nodes(self, *args, **kwargs) -> GraphExpansionPage:
        """Record local expansion through the service's fixed repository boundary."""
        self.graph_calls.append(("expand", args, kwargs))
        return GraphExpansionPage((), page=1, page_size=100, total_matches=0, has_next=False)

    def replacement_source_objects(self, *args, **kwargs) -> ReplacementSourceObjectPage:
        """Record fixed candidate inputs through the repository boundary."""
        self.graph_calls.append(("replacement", args, kwargs))
        return ReplacementSourceObjectPage(
            (), page=1, page_size=100, total_matches=0, has_next=False
        )


class _ConcurrentWriteRejectingRepository(_Repository):
    """Model the shared HANA graph rejecting overlapping mutation attempts."""

    def __init__(self) -> None:
        """Initialize deterministic overlap coordination and observations."""
        super().__init__()
        self._counter_lock = Lock()
        self._first_write_started = Event()
        self._second_write_started = Event()
        self._active_writes = 0
        self.peak_active_writes = 0

    def replace_graph_projection(self, projection: object) -> None:
        """Reject a second write while holding the first long enough to expose overlap."""
        with self._counter_lock:
            self._active_writes += 1
            self.peak_active_writes = max(
                self.peak_active_writes, self._active_writes
            )
            first = not self._first_write_started.is_set()
            if first:
                self._first_write_started.set()
            else:
                self._second_write_started.set()
        try:
            if first:
                self._second_write_started.wait(timeout=0.2)
            if self.peak_active_writes > 1:
                raise PersistenceStateError(
                    "RDF_CONCURRENT_WRITE",
                    "The shared graph rejected overlapping writes.",
                    status_code=500,
                )
            super().replace_graph_projection(projection)
        finally:
            with self._counter_lock:
                self._active_writes -= 1


def test_project_marks_current_only_after_exact_graph_readback() -> None:
    """Projection transitions pending to current only after replace and verify succeed."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"requester": "Ada", "items": ["one"]})

    result = RdfProjectionService(repository, clock=lambda: NOW).project(FIRST_ID)

    assert result.status is ProjectionState.CURRENT
    assert repository.operations == [
        f"pending:{FIRST_ID}",
        f"replace:{FIRST_ID}",
        f"verify:{FIRST_ID}",
        f"current:{FIRST_ID}",
    ]
    assert repository.sources[FIRST_ID].source_json == {
        "requester": "Ada",
        "items": ["one"],
    }


def test_project_many_isolates_failed_graph_writes_and_continues() -> None:
    """One graph failure becomes FAILED and does not prevent later selected ECNs."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"id": "first"})
    repository.add(SECOND_ID, {"id": "second"})
    repository.failed_replacements.add(FIRST_ID)

    result = RdfProjectionService(repository, clock=lambda: NOW).project_many(
        (FIRST_ID, SECOND_ID)
    )

    assert (result.requested, result.current, result.failed) == (2, 1, 1)
    assert repository.statuses[FIRST_ID].status is ProjectionState.FAILED
    assert repository.statuses[FIRST_ID].error_code == "RDF_PROJECTION_FAILED"
    assert repository.statuses[SECOND_ID].status is ProjectionState.CURRENT
    assert repository.sources[FIRST_ID].source_json == {"id": "first"}


def test_concurrent_batches_serialize_shared_graph_mutations() -> None:
    """Import, retry, and startup batches cannot overlap shared HANA graph writes."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _ConcurrentWriteRejectingRepository()
    repository.add(FIRST_ID, {"id": "first"})
    repository.add(SECOND_ID, {"id": "second"})
    service = RdfProjectionService(repository, clock=lambda: NOW)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(service.project_many, ((FIRST_ID,), (SECOND_ID,)))
        )

    assert [(result.current, result.failed) for result in results] == [(1, 0), (1, 0)]
    assert repository.peak_active_writes == 1


def test_rebuild_is_post_commit_and_current_gate_never_falls_back() -> None:
    """Retry marks only its source pending while non-current gates use explicit 409 codes."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"id": "first"})
    repository.statuses[FIRST_ID] = ProjectionRecord(
        **{**repository.statuses[FIRST_ID].__dict__, "status": ProjectionState.FAILED}
    )
    service = RdfProjectionService(repository, clock=lambda: NOW)

    pending = service.request_rebuild(FIRST_ID)
    assert pending.status is ProjectionState.PENDING
    assert repository.operations == [f"pending:{FIRST_ID}"]
    with pytest.raises(PersistenceStateError) as raised:
        service.require_current(FIRST_ID)
    assert raised.value.code == "RDF_PROJECTION_PENDING"


def test_reconcile_uses_only_repository_selected_durable_records() -> None:
    """Startup recovery projects exactly missing, failed, stale, or old-version records."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"id": "first"})
    repository.reconcilable = (FIRST_ID,)

    result = RdfProjectionService(repository, clock=lambda: NOW).reconcile()

    assert (result.requested, result.current, result.failed) == (1, 1, 0)
    assert repository.statuses[FIRST_ID].status is ProjectionState.CURRENT


def test_require_all_current_and_clear_all_are_explicit_boundaries() -> None:
    """Bulk gates reject incomplete projection state and history cleanup owns one graph."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"id": "first"})
    repository.add(SECOND_ID, {"id": "second"})
    service = RdfProjectionService(repository, clock=lambda: NOW)

    with pytest.raises(PersistenceStateError) as raised:
        service.require_all_current((FIRST_ID, SECOND_ID))
    assert raised.value.code == "RDF_PROJECTION_PENDING"
    service.clear_all()
    assert repository.graph_cleared is True


def test_history_clear_waits_for_inflight_graph_mutation() -> None:
    """Derived graph cleanup cannot race an import, retry, or startup rebuild."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"id": "first"})
    write_started = Event()
    release_write = Event()
    clear_started = Event()
    replace = repository.replace_graph_projection

    def blocking_replace(projection: object) -> None:
        """Hold one graph write until the test checks cleanup exclusion."""
        write_started.set()
        assert release_write.wait(timeout=1)
        replace(projection)

    def observe_clear() -> None:
        """Record cleanup admission before applying the fake graph clear."""
        clear_started.set()
        repository.graph_cleared = True

    repository.replace_graph_projection = blocking_replace  # type: ignore[method-assign]
    repository.clear_application_graph = observe_clear  # type: ignore[method-assign]
    service = RdfProjectionService(repository, clock=lambda: NOW)

    with ThreadPoolExecutor(max_workers=2) as executor:
        projection = executor.submit(service.project, FIRST_ID)
        assert write_started.wait(timeout=1)
        cleanup = executor.submit(service.clear_all)
        try:
            assert clear_started.wait(timeout=0.05) is False
        finally:
            release_write.set()
        assert projection.result().status is ProjectionState.CURRENT
        cleanup.result()

    assert clear_started.is_set()


def test_graph_reads_require_current_then_delegate_only_closed_parameters() -> None:
    """Graph search and expansion cannot bypass durable current-state gating."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"requester": "Ada"})
    service = RdfProjectionService(repository, clock=lambda: NOW)

    with pytest.raises(PersistenceStateError) as raised:
        service.find_facts(FIRST_ID, "requester", "predicate", None, None, 1)
    assert raised.value.code == "RDF_PROJECTION_PENDING"

    repository.statuses[FIRST_ID] = ProjectionRecord(
        **{**repository.statuses[FIRST_ID].__dict__, "status": ProjectionState.CURRENT}
    )
    page = service.find_facts(FIRST_ID, "requester", "predicate", None, None, 1)
    fields = service.list_fields(FIRST_ID, "request", 1)
    expansion = service.expand_nodes(
        FIRST_ID,
        ("urn:ecn-impact:ecn:3f2504e0-4f89-41d3-9a0c-0305e82c3301",),
        "outgoing",
        (),
        1,
        1,
    )
    replacement = service.replacement_source_objects(FIRST_ID, 1)

    assert (
        page.total_matches == 0
        and fields.page_size == 50
        and expansion.page_size == 100
        and replacement.page_size == 100
    )
    assert repository.graph_calls == [
        ("find", (FIRST_ID, "requester", "predicate", None, None, 1), {}),
        ("fields", (FIRST_ID, "request", 1), {}),
        (
            "expand",
            (
                FIRST_ID,
                ("urn:ecn-impact:ecn:3f2504e0-4f89-41d3-9a0c-0305e82c3301",),
                "outgoing",
                (),
                1,
                1,
            ),
            {},
        ),
        ("replacement", (FIRST_ID, 1), {}),
    ]


def test_current_ids_splits_ready_ecns_and_multi_ecn_reads_gate_every_one() -> None:
    """Cross-ECN searches skip pending graphs and never read a non-current one."""
    from app.ecn.rdf.service import RdfProjectionService

    repository = _Repository()
    repository.add(FIRST_ID, {"requester": "Ada"})
    repository.add(SECOND_ID, {"requester": "Lin"})
    repository.statuses[FIRST_ID] = ProjectionRecord(
        **{**repository.statuses[FIRST_ID].__dict__, "status": ProjectionState.CURRENT}
    )
    service = RdfProjectionService(repository, clock=lambda: NOW)

    assert service.current_ids([FIRST_ID, SECOND_ID]) == ([FIRST_ID], [SECOND_ID])
    with pytest.raises(PersistenceStateError):
        service.find_facts([FIRST_ID, SECOND_ID], "requester", "value", None, None, 1)
    assert repository.graph_calls == []

    service.find_facts([FIRST_ID], "requester", "value", None, None, 1)
    service.list_fields([FIRST_ID], "request", 1)
    assert [call[1][0] for call in repository.graph_calls] == [[FIRST_ID], [FIRST_ID]]
