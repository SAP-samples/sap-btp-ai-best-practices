"""API-owned local worker lifecycle; production may use a separate worker process."""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

from .dependencies import get_service
from .worker import worker_loop

logger = logging.getLogger(__name__)


def embedded_worker_enabled():
    """Return the explicit worker setting, defaulting to enabled outside production."""
    setting = os.getenv("WORKSPACE_EMBEDDED_WORKER")
    if setting is None:
        return os.getenv("APP_ENV", "development").lower() != "production"
    if setting.lower() not in {"true", "false", "1", "0"}:
        raise ValueError("WORKSPACE_EMBEDDED_WORKER must be true or false")
    return setting.lower() in {"true", "1"}


@asynccontextmanager
async def workspace_lifespan(app):
    """Start one local queue supervisor and stop its owned child during API shutdown."""
    if not embedded_worker_enabled():
        logger.info("Embedded optimizer worker disabled; run python -m app.workspace worker separately")
        yield
        return
    stop = threading.Event()
    thread = threading.Thread(
        target=worker_loop, args=(get_service(), stop), name="optimizer-supervisor", daemon=True
    )
    thread.start()
    logger.info("Local optimizer worker started; queued runs will execute automatically")
    try:
        yield
    finally:
        stop.set()
        await asyncio.to_thread(thread.join, 20)
        if thread.is_alive():
            logger.error("Optimizer supervisor did not stop within 20 seconds")
