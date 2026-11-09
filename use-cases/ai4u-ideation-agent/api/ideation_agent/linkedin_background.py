"""Background LinkedIn position analysis task manager.

Manages async tasks per conversation context. Results are stored in module-level
dicts (not LangGraph state) because asyncio.Task is not serializable.
Results are also persisted to HANA for durability across restarts.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TASKS: Dict[str, asyncio.Task] = {}
_RESULTS: Dict[str, Dict[str, Any]] = {}


async def _persist_to_hana(context_id: str, result: Dict[str, Any]) -> None:
    """Persist research result to HANA (fire-and-forget, never raises)."""
    try:
        from .hana_store import get_store
        store = await get_store()
        await store.save_research(context_id, "linkedin", result.get("status", "failed"), result)
    except Exception as exc:
        logger.warning("[LinkedIn-BG] HANA persist failed context=%s: %s", context_id, exc)


async def start_enrichment(context_id: str, company_name: str) -> None:
    """Launch background LinkedIn analysis for a conversation.

    Fire-and-forget: the task runs in the background and stores its result
    in _RESULTS when done. Safe to call multiple times — subsequent calls
    are no-ops if a task already exists for this context.
    """
    if context_id in _TASKS:
        logger.debug("[LinkedIn-BG] Task already exists for context=%s", context_id)
        return

    from .tools.linkedin_research import run_linkedin_research

    async def _run() -> None:
        try:
            result = await run_linkedin_research(company_name, context_id=context_id)
            _RESULTS[context_id] = result
            await _persist_to_hana(context_id, result)
            logger.info(
                "[LinkedIn-BG] Analysis complete context=%s status=%s rows=%s",
                context_id,
                result.get("status"),
                result.get("rows_analyzed"),
            )
        except Exception as exc:
            logger.error("[LinkedIn-BG] Analysis failed context=%s error=%s", context_id, exc)
            error_result = {"status": "failed", "error": str(exc)}
            _RESULTS[context_id] = error_result
            await _persist_to_hana(context_id, error_result)

    task = asyncio.create_task(_run())
    _TASKS[context_id] = task
    logger.info("[LinkedIn-BG] Started analysis for context=%s company='%s'", context_id, company_name)


def get_result(context_id: str) -> Optional[Dict[str, Any]]:
    """Non-blocking poll: returns analysis result if ready, None if pending."""
    result = _RESULTS.get(context_id)
    if result is not None:
        return result

    task = _TASKS.get(context_id)
    if task is not None and task.done():
        exc = task.exception() if not task.cancelled() else None
        if exc:
            _RESULTS[context_id] = {"status": "failed", "error": str(exc)}
        return _RESULTS.get(context_id)

    return None


def status(context_id: str) -> str:
    """Returns 'idle', 'pending', 'complete', or 'failed'."""
    if context_id in _RESULTS:
        result = _RESULTS[context_id]
        return "failed" if result.get("status") in ("failed", "error") else "complete"
    task = _TASKS.get(context_id)
    if task is not None:
        if not task.done():
            return "pending"
        exc = task.exception() if not task.cancelled() else None
        if exc:
            _RESULTS[context_id] = {"status": "failed", "error": str(exc)}
            return "failed"
        return "complete"
    return "idle"


async def load_from_hana(context_id: str) -> bool:
    """Load cached result from HANA into memory. Returns True if found."""
    if context_id in _RESULTS:
        return True
    try:
        from .hana_store import get_store
        store = await get_store()
        row = await store.get_research(context_id, "linkedin")
        if row and row.get("payload"):
            _RESULTS[context_id] = row["payload"]
            return True
    except Exception as exc:
        logger.warning("[LinkedIn-BG] HANA load failed context=%s: %s", context_id, exc)
    return False


def cleanup(context_id: str) -> None:
    """Remove references for a finished conversation."""
    _TASKS.pop(context_id, None)
    _RESULTS.pop(context_id, None)
