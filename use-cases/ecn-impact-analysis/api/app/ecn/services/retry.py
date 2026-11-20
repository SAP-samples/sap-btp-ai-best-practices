"""Cancellation-safe application service for failed-job retries."""

from __future__ import annotations

from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.records import RetryQueueResult


class RetryService:
    """Own the guarded failed-to-queued persistence transition."""

    def __init__(
        self,
        analysis_repository: AnalysisRepository,
        mutation_guard: HistoryMutationGuard,
    ) -> None:
        """Bind the analysis repository and shared lifecycle mutation guard.

        Args:
            analysis_repository: Repository with the atomic retry transition.
            mutation_guard: Guard shared by publication, claims, retries, and reset.
        """
        self._repository = analysis_repository
        self._guard = mutation_guard

    async def queue_failed(self, ecn_row_id: str) -> RetryQueueResult:
        """Commit one retry transition through cancellation-safe blocking I/O.

        Args:
            ecn_row_id: Canonical UUID of the requested ECN occurrence.

        Returns:
            Discriminated queued, missing, or non-retryable result.
        """
        return await run_blocking(self._queue_under_guard, ecn_row_id)

    def _queue_under_guard(self, ecn_row_id: str) -> RetryQueueResult:
        """Serialize the short repository transaction with lifecycle mutations."""
        with self._guard.hold():
            return self._repository.queue_failed_job(ecn_row_id)
