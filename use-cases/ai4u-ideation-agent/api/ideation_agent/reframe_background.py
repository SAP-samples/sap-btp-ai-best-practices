"""Background strategic reframe task manager.

After the agent delivers a kind='final' recommendation, a background LLM call
produces a 'strategic reframe' — challenging assumptions, suggesting alternative
angles, and flagging risks. Results are persisted to HANA for durability.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_TASKS: Dict[str, asyncio.Task] = {}
_RESULTS: Dict[str, Dict[str, Any]] = {}

REFRAME_LLM_MODEL = os.getenv("REFRAME_LLM_MODEL", "gpt-5.4")
REFRAME_TEMPERATURE = float(os.getenv("REFRAME_TEMPERATURE", "0.7"))

REFRAME_SYSTEM_PROMPT = """\
You are a strategic advisor reviewing an AI solution proposal. Your job is to question whether the user is solving the RIGHT problem — not whether their solution is technically complete (that's handled separately).

Focus on:
- Assumptions about what the real problem is (vs. symptoms)
- Whether AI is the right lever, or if a process/organizational change would be more effective
- Business model, incentive, or organizational blind spots
- Whether the problem framing is too narrow or misdirected

Do NOT:
- Suggest technical improvements to the solution (e.g. "add error handling", "add a confidence threshold")
- Repeat feedback about solution completeness, edge cases, or missing features
- Be generic — reference the specific industry, domain, and choices made

Given the solution details below, produce a JSON response with exactly this structure:
{
  "challenge": "One provocative question challenging whether this is the right problem to solve (max 2 sentences)",
  "alternative_angle": "A fundamentally different framing of the problem — not a tweak to the solution, but a different way to think about what's really going wrong (max 2 sentences)",
  "risk": "The biggest strategic or business risk if the user's framing of the problem is wrong (max 2 sentences)",
  "reframe_summary": "One sentence connecting the above into an actionable 'what if' for the user to consider"
}

Be concise and punchy. Each field should be 1-2 sentences maximum.
Return ONLY the JSON object, no markdown fences or additional text."""


async def _persist_to_hana(context_id: str, result: Dict[str, Any]) -> None:
    """Persist reframe result to HANA (fire-and-forget, never raises)."""
    try:
        from .hana_store import get_store
        store = await get_store()
        await store.save_research(context_id, "strategic_reframe", result.get("status", "failed"), result)
    except Exception as exc:
        logger.warning("[Reframe-BG] HANA persist failed context=%s: %s", context_id, exc)


async def _run_reframe(context_id: str, final_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the reframe LLM call."""
    from .common import make_llm

    llm = make_llm(provider="openai", model_name=REFRAME_LLM_MODEL, temperature=REFRAME_TEMPERATURE, json_mode=True)

    user_prompt = (
        f"Industry: {final_payload.get('industry', 'Unknown')}\n"
        f"Title: {final_payload.get('title', 'Untitled')}\n"
        f"Problem: {final_payload.get('problem', '')}\n"
        f"Solution: {final_payload.get('solution', '')}\n"
        f"App Type: {final_payload.get('app_type', 'Agent')}\n"
    )
    company_context = final_payload.get("company_context", "")
    if company_context:
        user_prompt += f"Company Context: {company_context}\n"

    messages = [
        SystemMessage(content=REFRAME_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    started = time.perf_counter()
    response = await llm.ainvoke(messages)
    latency_ms = int((time.perf_counter() - started) * 1000)

    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")

    try:
        parsed = json.loads(raw_text)
        return {
            "status": "ok",
            "challenge": parsed.get("challenge", ""),
            "alternative_angle": parsed.get("alternative_angle", ""),
            "risk": parsed.get("risk", ""),
            "reframe_summary": parsed.get("reframe_summary", ""),
            "latency_ms": latency_ms,
        }
    except (json.JSONDecodeError, TypeError):
        return {"status": "failed", "error": "Invalid JSON from reframe LLM", "raw": raw_text[:500]}


async def start_reframe(context_id: str, final_payload: Dict[str, Any]) -> None:
    """Launch background reframe task for a conversation."""
    if context_id in _TASKS:
        logger.debug("[Reframe-BG] Task already exists for context=%s", context_id)
        return

    async def _run() -> None:
        try:
            result = await _run_reframe(context_id, final_payload)
            _RESULTS[context_id] = result
            await _persist_to_hana(context_id, result)
            logger.info("[Reframe-BG] Complete context=%s status=%s latency=%dms",
                        context_id, result.get("status"), result.get("latency_ms", 0))
        except Exception as exc:
            logger.error("[Reframe-BG] Failed context=%s error=%s", context_id, exc)
            error_result = {"status": "failed", "error": str(exc)}
            _RESULTS[context_id] = error_result
            await _persist_to_hana(context_id, error_result)

    task = asyncio.create_task(_run())
    _TASKS[context_id] = task
    logger.info("[Reframe-BG] Started for context=%s", context_id)


def get_result(context_id: str) -> Optional[Dict[str, Any]]:
    """Non-blocking poll: returns reframe result if ready, None if pending."""
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
    """Returns reframe status: 'idle', 'pending', 'complete', or 'failed'."""
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


async def load_from_hana(context_id: str) -> bool:
    """Load cached reframe result from HANA into memory. Returns True if found."""
    try:
        from .hana_store import get_store
        store = await get_store()
        if context_id not in _RESULTS:
            row = await store.get_research(context_id, "strategic_reframe")
            if row and row.get("payload"):
                _RESULTS[context_id] = row["payload"]
                return True
    except Exception as exc:
        logger.warning("[Reframe-BG] HANA load failed context=%s: %s", context_id, exc)
    return context_id in _RESULTS


def cleanup(context_id: str) -> None:
    """Remove references for a finished conversation."""
    _TASKS.pop(context_id, None)
    _RESULTS.pop(context_id, None)
