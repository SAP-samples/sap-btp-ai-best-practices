"""Unit tests for the guarded, idle-only analysis history reset service.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_history_service.py
"""

from threading import Event, Thread

import pytest
from app.ecn.services.history import HistoryService
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.records import HistoryDeletionCounts


class _HistoryRepository:
    """Return a configured atomic reset result and record call timing."""

    def __init__(self, result: HistoryDeletionCounts | None) -> None:
        """Initialize the repository result and empty call observations."""
        self.result = result
        self.called = Event()
        self.call_count = 0

    def clear_history_if_idle(self) -> HistoryDeletionCounts | None:
        """Represent one transaction containing active check and ordered deletes."""
        self.call_count += 1
        self.called.set()
        return self.result


def _counts() -> HistoryDeletionCounts:
    """Return exact deleted-row counts for response mapping."""
    return HistoryDeletionCounts(
        attempts=4,
        analyses=3,
        jobs=3,
        ecns=3,
        uploads=1,
        bom_rows=10,
        source_artifacts=2,
    )


@pytest.mark.parametrize(
    "confirmation", ["", "reset history", "RESET HISTORY ", "RESET  HISTORY"]
)
def test_reset_requires_exact_case_sensitive_confirmation(confirmation: str) -> None:
    """Reject every value other than the exact destructive phrase."""
    repository = _HistoryRepository(_counts())
    service = HistoryService(repository, HistoryMutationGuard())
    with pytest.raises(Exception) as caught:
        service.reset_history(confirmation)
    assert getattr(caught.value, "code", None) == (
        "HISTORY_RESET_CONFIRMATION_INVALID"
    )
    assert repository.call_count == 0


def test_active_jobs_return_state_conflict_without_partial_delete() -> None:
    """Map the repository's atomic idle-check failure to a retryable conflict."""
    repository = _HistoryRepository(None)
    service = HistoryService(repository, HistoryMutationGuard())
    with pytest.raises(Exception) as caught:
        service.reset_history("RESET HISTORY")
    assert getattr(caught.value, "code", None) == "HISTORY_RESET_ACTIVE_JOBS"
    assert repository.call_count == 1


def test_idle_reset_returns_exact_repository_counts() -> None:
    """Return deleted counts while preservation remains the repository contract."""
    repository = _HistoryRepository(_counts())
    service = HistoryService(repository, HistoryMutationGuard())
    assert service.reset_history("RESET HISTORY") == _counts()
    assert repository.call_count == 1


def test_idle_reset_clears_derived_graph_only_after_relational_commit() -> None:
    """Successful authoritative reset triggers one separate derived-graph cleanup."""
    repository = _HistoryRepository(_counts())
    calls: list[str] = []
    service = HistoryService(
        repository,
        HistoryMutationGuard(),
        clear_projection_graph=lambda: calls.append("cleared"),
    )

    assert service.reset_history("RESET HISTORY") == _counts()
    assert repository.call_count == 1
    assert calls == ["cleared"]


def test_graph_cleanup_failure_keeps_committed_relational_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A derived-graph cleanup failure is logged and does not undo source deletion."""
    repository = _HistoryRepository(_counts())

    def fail_cleanup() -> None:
        """Raise a simulated derived-graph cleanup error after relational commit."""
        raise RuntimeError("graph unavailable")

    service = HistoryService(
        repository,
        HistoryMutationGuard(),
        clear_projection_graph=fail_cleanup,
    )

    assert service.reset_history("RESET HISTORY") == _counts()
    assert repository.call_count == 1
    assert "RDF projection graph cleanup failed" in caplog.text


def test_reset_waits_for_final_publication_guard_without_polling_or_sleep() -> None:
    """Serialize reset after an in-flight final publication critical section."""
    guard = HistoryMutationGuard()
    repository = _HistoryRepository(_counts())
    service = HistoryService(repository, guard)
    publication_holds_guard = Event()
    release_publication = Event()

    def publication() -> None:
        """Hold the shared guard until the test releases publication."""
        with guard.hold():
            publication_holds_guard.set()
            assert release_publication.wait(timeout=1)

    result: list[HistoryDeletionCounts] = []

    def reset() -> None:
        """Call reset and retain its exact result after guard admission."""
        result.append(service.reset_history("RESET HISTORY"))

    publishing = Thread(target=publication)
    resetting = Thread(target=reset)
    publishing.start()
    assert publication_holds_guard.wait(timeout=1)
    resetting.start()
    try:
        assert repository.called.wait(timeout=0.05) is False
    finally:
        release_publication.set()
    publishing.join(timeout=1)
    resetting.join(timeout=1)
    assert not publishing.is_alive()
    assert not resetting.is_alive()
    assert repository.called.is_set()
    assert result == [_counts()]
