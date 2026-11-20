"""Process-local concurrency boundaries for ECN lifecycle mutations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock, Semaphore


class IngestionPreparationGate:
    """Allow one memory-heavy upload preparation at a time in this process."""

    def __init__(self) -> None:
        """Create the process-local capacity-one preparation semaphore."""
        self._semaphore = Semaphore(1)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the single preparation permit for parsing and evidence derivation."""
        with self._semaphore:
            yield


class HistoryMutationGuard:
    """Serialize only short final state mutations inside one API process."""

    def __init__(self) -> None:
        """Create the reentrant lock shared by lifecycle mutation services."""
        self._lock = RLock()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the shared mutation lock for one short repository transaction."""
        with self._lock:
            yield
