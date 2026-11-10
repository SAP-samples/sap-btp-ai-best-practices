"""Bound local active work; durable snapshots remain in HANA between requests."""
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore


class WorkspaceJobs:
    """Admit at most two active jobs, without an unbounded executor queue."""

    def __init__(self, capacity=2):
        """Create a small executor and equally sized admission semaphore."""
        self.pool = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix='receivables')
        self.slots = BoundedSemaphore(capacity)

    def submit(self, claim, work):
        """Acquire capacity before a durable claim; always release it on failure/completion."""
        if not self.slots.acquire(blocking=False):
            raise RuntimeError('The workspace is processing other runs; retry shortly')
        try:
            snapshot = claim()
            future = self.pool.submit(work, snapshot)
            future.add_done_callback(self._release)
            return snapshot
        except Exception:
            self.slots.release()
            raise

    def _release(self, future):
        """Release the worker slot and log unexpected orchestration failures."""
        self.slots.release()
        if future.exception():
            import logging
            logging.getLogger(__name__).error('Workspace background job failed', exc_info=future.exception())
