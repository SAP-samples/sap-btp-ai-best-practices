"""Background Perplexity enrichment task manager.

Manages async tasks per conversation context. Results are stored in module-level
dicts (not LangGraph state) because asyncio.Task is not serializable.
Results are also persisted to HANA for durability across restarts.

Two independent task tracks:
  - Company research: fires on turn 1 (broad company intelligence)
  - Problem research: fires when agent deems problem clear (domain-specific)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TASKS: Dict[str, asyncio.Task] = {}
_RESULTS: Dict[str, Dict[str, Any]] = {}

_PROBLEM_TASKS: Dict[str, asyncio.Task] = {}
_PROBLEM_RESULTS: Dict[str, Dict[str, Any]] = {}


async def _persist_to_hana(context_id: str, research_type: str, result: Dict[str, Any]) -> None:
    """Persist research result to HANA (fire-and-forget, never raises)."""
    try:
        from .hana_store import get_store
        store = await get_store()
        await store.save_research(context_id, research_type, result.get("status", "failed"), result)
    except Exception as exc:
        logger.warning("[Perplexity-BG] HANA persist failed context=%s type=%s: %s", context_id, research_type, exc)


async def start_enrichment(context_id: str, company_name: str) -> None:
    """Launch background company research for a conversation."""
    if context_id in _TASKS:
        logger.debug("[Perplexity-BG] Company task already exists for context=%s", context_id)
        return

    from .tools.perplexity_research import run_perplexity_research

    async def _run() -> None:
        try:
            result = await run_perplexity_research(company_name, context_id=context_id)
            _RESULTS[context_id] = result
            await _persist_to_hana(context_id, "perplexity_company", result)
            logger.info(
                "[Perplexity-BG] Company research complete context=%s status=%s sources=%s",
                context_id,
                result.get("status"),
                len(result.get("sources", [])),
            )
        except Exception as exc:
            logger.error("[Perplexity-BG] Company research failed context=%s error=%s", context_id, exc)
            error_result = {"status": "failed", "error": str(exc)}
            _RESULTS[context_id] = error_result
            await _persist_to_hana(context_id, "perplexity_company", error_result)

    task = asyncio.create_task(_run())
    _TASKS[context_id] = task
    logger.info("[Perplexity-BG] Started company research for context=%s company='%s'", context_id, company_name)


async def start_problem_enrichment(context_id: str, company_name: str, user_message: str) -> None:
    """Launch background problem/domain research for a conversation."""
    if context_id in _PROBLEM_TASKS:
        logger.debug("[Perplexity-BG] Problem task already exists for context=%s", context_id)
        return

    from .tools.perplexity_research import run_perplexity_problem_research

    async def _run() -> None:
        try:
            result = await run_perplexity_problem_research(company_name, user_message, context_id=context_id)
            _PROBLEM_RESULTS[context_id] = result
            await _persist_to_hana(context_id, "perplexity_problem", result)
            logger.info(
                "[Perplexity-BG] Problem research complete context=%s status=%s sources=%s",
                context_id,
                result.get("status"),
                len(result.get("sources", [])),
            )
        except Exception as exc:
            logger.error("[Perplexity-BG] Problem research failed context=%s error=%s", context_id, exc)
            error_result = {"status": "failed", "error": str(exc)}
            _PROBLEM_RESULTS[context_id] = error_result
            await _persist_to_hana(context_id, "perplexity_problem", error_result)

    task = asyncio.create_task(_run())
    _PROBLEM_TASKS[context_id] = task
    logger.info("[Perplexity-BG] Started problem research for context=%s company='%s'", context_id, company_name)


def get_result(context_id: str) -> Optional[Dict[str, Any]]:
    """Non-blocking poll: returns company research result if ready, None if pending."""
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


def get_problem_result(context_id: str) -> Optional[Dict[str, Any]]:
    """Non-blocking poll: returns problem research result if ready, None if pending."""
    result = _PROBLEM_RESULTS.get(context_id)
    if result is not None:
        return result

    task = _PROBLEM_TASKS.get(context_id)
    if task is not None and task.done():
        exc = task.exception() if not task.cancelled() else None
        if exc:
            _PROBLEM_RESULTS[context_id] = {"status": "failed", "error": str(exc)}
        return _PROBLEM_RESULTS.get(context_id)

    return None


def status(context_id: str) -> str:
    """Returns company research status: 'idle', 'pending', 'complete', or 'failed'."""
    if context_id in _RESULTS:
        result = _RESULTS[context_id]
        return "failed" if result.get("status") == "failed" else "complete"
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


def problem_status(context_id: str) -> str:
    """Returns problem research status: 'idle', 'pending', 'complete', or 'failed'."""
    if context_id in _PROBLEM_RESULTS:
        result = _PROBLEM_RESULTS[context_id]
        return "failed" if result.get("status") == "failed" else "complete"
    task = _PROBLEM_TASKS.get(context_id)
    if task is not None:
        if not task.done():
            return "pending"
        exc = task.exception() if not task.cancelled() else None
        if exc:
            _PROBLEM_RESULTS[context_id] = {"status": "failed", "error": str(exc)}
            return "failed"
        return "complete"
    return "idle"


async def load_from_hana(context_id: str) -> bool:
    """Load cached results from HANA into memory. Returns True if any found."""
    found = False
    try:
        from .hana_store import get_store
        store = await get_store()
        if context_id not in _RESULTS:
            row = await store.get_research(context_id, "perplexity_company")
            if row and row.get("payload"):
                _RESULTS[context_id] = row["payload"]
                found = True
        else:
            found = True
        if context_id not in _PROBLEM_RESULTS:
            row = await store.get_research(context_id, "perplexity_problem")
            if row and row.get("payload"):
                _PROBLEM_RESULTS[context_id] = row["payload"]
                found = True
        else:
            found = True
    except Exception as exc:
        logger.warning("[Perplexity-BG] HANA load failed context=%s: %s", context_id, exc)
    return found


def cleanup(context_id: str) -> None:
    """Remove references for a finished conversation."""
    _TASKS.pop(context_id, None)
    _RESULTS.pop(context_id, None)
    _PROBLEM_TASKS.pop(context_id, None)
    _PROBLEM_RESULTS.pop(context_id, None)
