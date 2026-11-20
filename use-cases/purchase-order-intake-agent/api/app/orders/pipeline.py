"""Background order processing: a pool of async workers fed by the HANA order table.

Design (restart- and multi-instance-safe without a message broker):

* **Queue = table.** New orders are rows with ``STATUS = 'QUEUED'``. A worker claims one with
  ``UPDATE ... SET STATUS = 'PROCESSING' WHERE ORDER_ID = ? AND STATUS = 'QUEUED'``; only the
  worker whose update hits one row owns the order.
* **Concurrency.** ``ORDER_WORKERS`` workers per app instance run agent calls in parallel (the
  agent runtime is shared; every order has its own context id).
* **Wake-up.** The upload endpoint sets ``app.state.order_queue_event``; idle workers also poll
  every few seconds.
* **Recovery.** On start-up, orders this instance left in PROCESSING (crash / restart) are
  requeued; a sweeper requeues PROCESSING orders older than the agent timeout (another instance
  died) and marks them FAILED after ``ORDER_MAX_ATTEMPTS``.
* **Outcome.** After the agent run the deterministic gate (``service.finalize_after_agent``)
  posts the order (auto-post) or sets NEEDS_REVIEW / NOT_AN_ORDER; agent errors and timeouts
  are retried up to ``ORDER_MAX_ATTEMPTS`` and then marked FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from ..db.engine import get_engine
from ..db.sql import fetch_all
from ..order_agent.runner import close_runtime, run_order_agent
from . import repository
from .posting import PostingError, post_order
from .service import finalize_after_agent
from .settings import get_settings

logger = logging.getLogger(__name__)
IDLE_POLL_SECONDS = 5
SWEEP_EVERY_SECONDS = 60


def instance_name() -> str:
    """Stable name of this app instance (CF instance index, else host name)."""
    index = os.getenv("CF_INSTANCE_INDEX")
    return f"cf-{index}" if index is not None else socket.gethostname()[:60]


def _now() -> datetime:
    """Current naive UTC timestamp."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def claim_next(owner: str) -> tuple[str, str | None] | None:
    """Atomically move the oldest QUEUED order to PROCESSING for ``owner``.

    Args:
        owner: Instance name stored in LOCKED_BY.

    Returns:
        (order_id, reviewer_note) of the claimed order, or None when the queue is empty.
    """
    for _ in range(5):  # another worker may win the race; try the next candidate
        candidates = fetch_all('SELECT "ORDER_ID", "REVIEWER_NOTE" FROM "POAGENT_ORDER_HEADER" '
                               'WHERE "STATUS" = \'QUEUED\' ORDER BY "CREATED_AT" LIMIT 1')
        if not candidates:
            return None
        order_id = candidates[0]["ORDER_ID"]
        with get_engine().begin() as conn:
            claimed = conn.execute(text(
                'UPDATE "POAGENT_ORDER_HEADER" SET "STATUS" = \'PROCESSING\', "LOCKED_BY" = :owner, '
                '"LOCKED_AT" = :now, "ATTEMPTS" = "ATTEMPTS" + 1, "STATUS_MESSAGE" = \'Agent is processing the order\', '
                '"UPDATED_AT" = CURRENT_UTCTIMESTAMP WHERE "ORDER_ID" = :o AND "STATUS" = \'QUEUED\''),
                {"owner": owner, "now": _now(), "o": order_id}).rowcount
        if claimed == 1:
            return order_id, candidates[0].get("REVIEWER_NOTE")
    return None


def requeue_orphans(owner: str) -> int:
    """Requeue orders this instance left in PROCESSING before a restart.

    Args:
        owner: This instance's name.

    Returns:
        Number of requeued orders.
    """
    with get_engine().begin() as conn:
        return conn.execute(text(
            'UPDATE "POAGENT_ORDER_HEADER" SET "STATUS" = \'QUEUED\', "LOCKED_BY" = NULL, "LOCKED_AT" = NULL, '
            '"STATUS_MESSAGE" = \'Requeued after restart\', "UPDATED_AT" = CURRENT_UTCTIMESTAMP '
            'WHERE "STATUS" = \'PROCESSING\' AND "LOCKED_BY" = :owner'), {"owner": owner}).rowcount


def sweep_stale(max_age_seconds: int, max_attempts: int) -> tuple[int, int]:
    """Requeue (or fail) PROCESSING orders whose lock is older than ``max_age_seconds``.

    Args:
        max_age_seconds: Lock age after which the owner is considered dead.
        max_attempts: Attempts after which the order is marked FAILED instead.

    Returns:
        (requeued, failed) counts.
    """
    cutoff = _now() - timedelta(seconds=max_age_seconds)
    with get_engine().begin() as conn:
        requeued = conn.execute(text(
            'UPDATE "POAGENT_ORDER_HEADER" SET "STATUS" = \'QUEUED\', "LOCKED_BY" = NULL, "LOCKED_AT" = NULL, '
            '"STATUS_MESSAGE" = \'Requeued: previous attempt did not finish\', "UPDATED_AT" = CURRENT_UTCTIMESTAMP '
            'WHERE "STATUS" = \'PROCESSING\' AND "LOCKED_AT" < :cutoff AND "ATTEMPTS" < :max'),
            {"cutoff": cutoff, "max": max_attempts}).rowcount
        failed = conn.execute(text(
            'UPDATE "POAGENT_ORDER_HEADER" SET "STATUS" = \'FAILED\', "LOCKED_BY" = NULL, "LOCKED_AT" = NULL, '
            '"STATUS_MESSAGE" = \'Processing did not finish after the maximum number of attempts\', '
            '"UPDATED_AT" = CURRENT_UTCTIMESTAMP '
            'WHERE "STATUS" = \'PROCESSING\' AND "LOCKED_AT" < :cutoff AND "ATTEMPTS" >= :max'),
            {"cutoff": cutoff, "max": max_attempts}).rowcount
    return requeued, failed


def post_for_gate(order_id: str) -> str:
    """Posting callback for the final gate: posts, or records an ECC rejection as FAILED.

    Args:
        order_id: Order id.

    Returns:
        Sales document number, or "" when ECC rejected the order.
    """
    try:
        return post_order(order_id)
    except PostingError as exc:
        repository.set_status(order_id, "FAILED", exc.message)
        return ""


class OrderPipeline:
    """Owns the worker and sweeper tasks of one app instance."""

    def __init__(self) -> None:
        """Create an idle pipeline; call ``start`` from the FastAPI lifespan."""
        self.settings = get_settings()
        self.owner = instance_name()
        self.event = asyncio.Event()
        self.tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Requeue own orphans and start the workers and the sweeper."""
        requeued = await asyncio.to_thread(requeue_orphans, self.owner)
        if requeued:
            logger.warning("Requeued %d orders left in PROCESSING by %s", requeued, self.owner)
        self.tasks = [asyncio.create_task(self._worker(n), name=f"order-worker-{n}")
                      for n in range(self.settings.workers)]
        self.tasks.append(asyncio.create_task(self._sweeper(), name="order-sweeper"))
        self.event.set()  # process whatever is already queued
        logger.info("Order pipeline started: %d workers on %s", self.settings.workers, self.owner)

    async def stop(self) -> None:
        """Cancel workers (their orders are requeued at the next start) and close the agent runtime."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        await close_runtime()

    async def _worker(self, number: int) -> None:
        """Claim and process orders until cancelled."""
        while True:
            try:
                claimed = await asyncio.to_thread(claim_next, self.owner)
                if claimed is None:
                    self.event.clear()
                    try:
                        await asyncio.wait_for(self.event.wait(), timeout=IDLE_POLL_SECONDS)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self.process(*claimed)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the worker alive; the order is retried by the sweeper
                logger.exception("Order worker %d failed", number)
                await asyncio.sleep(IDLE_POLL_SECONDS)

    async def process(self, order_id: str, reviewer_note: str | None = None) -> str:
        """Run the agent and the final gate for one claimed order.

        Args:
            order_id: Order in status PROCESSING owned by this instance.
            reviewer_note: Note from a reprocessing request.

        Returns:
            The resulting status.
        """
        outcome = await run_order_agent(order_id, reviewer_note)
        if outcome.outcome in ("ERROR", "TIMEOUT"):
            header = await asyncio.to_thread(repository.get_header, order_id)
            if header and header["ATTEMPTS"] < self.settings.max_attempts:
                await asyncio.to_thread(repository.set_status, order_id, "QUEUED",
                                        f"Retrying after {outcome.outcome.lower()}: {outcome.error}",
                                        LOCKED_BY=None, LOCKED_AT=None)
                self.event.set()
                return "QUEUED"
            await asyncio.to_thread(repository.set_status, order_id, "FAILED", outcome.error, LOCKED_BY=None, LOCKED_AT=None)
            return "FAILED"
        status = await asyncio.to_thread(finalize_after_agent, order_id, outcome.summary, post_for_gate)
        logger.info("Order %s finished: %s (%.0fs, %d LLM calls)", order_id, status, outcome.seconds, outcome.llm_calls)
        return status

    async def _sweeper(self) -> None:
        """Periodically requeue or fail orders whose processing instance disappeared."""
        max_age = self.settings.agent_timeout_seconds + 120
        while True:
            await asyncio.sleep(SWEEP_EVERY_SECONDS)
            try:
                requeued, failed = await asyncio.to_thread(sweep_stale, max_age, self.settings.max_attempts)
                if requeued or failed:
                    logger.warning("Sweeper: %d orders requeued, %d failed", requeued, failed)
                    self.event.set()
            except Exception:  # noqa: BLE001 - try again at the next sweep
                logger.exception("Order sweeper failed")
