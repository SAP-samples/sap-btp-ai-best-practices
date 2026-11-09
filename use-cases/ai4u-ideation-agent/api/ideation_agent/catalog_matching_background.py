"""Background SAP AI catalog matching task manager.

After the agent delivers a kind='final' recommendation, a background LLM call
matches the customer's problem/solution against the SAP Discovery Center AI
catalog (~415 entries) and returns the top 3-5 relevant standard AI features
or agents. Results are persisted to HANA for durability.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_TASKS: Dict[str, asyncio.Task] = {}
_RESULTS: Dict[str, Dict[str, Any]] = {}

# ── Catalog cache ────────────────────────────────────────────────────────────
_CATALOG: Optional[List[Dict[str, str]]] = None
_CATALOG_PROMPT_BLOCK: Optional[str] = None

CATALOG_CSV_PATH = Path(__file__).parent / "data" / "AI Features Data.csv"

CATALOG_MATCH_MODEL = os.getenv("CATALOG_MATCH_MODEL", "gemini-2.5-flash")
CATALOG_MATCH_TEMPERATURE = float(os.getenv("CATALOG_MATCH_TEMPERATURE", "0.2"))


def _load_catalog() -> List[Dict[str, str]]:
    """Load and cache the SAP AI catalog from CSV."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    entries: List[Dict[str, str]] = []
    with open(CATALOG_CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "name": row.get("Name", "").strip(),
                "ai_type": row.get("AI Type", "").strip(),
                "commercial_type": row.get("Commercial Type", "").strip(),
                "product": row.get("Product", "").strip(),
                "description": row.get("Description", "").strip(),
                "product_category": row.get("Product Category", "").strip(),
                "availability": row.get("Availability", "").strip(),
                "detail_page_url": row.get("Detail Page", "").strip(),
            })
    _CATALOG = entries
    logger.info("[CatalogMatch] Loaded %d entries from %s", len(entries), CATALOG_CSV_PATH)
    return _CATALOG


def _format_catalog_for_prompt() -> str:
    """Format the full catalog as a compact numbered list for the LLM prompt."""
    global _CATALOG_PROMPT_BLOCK
    if _CATALOG_PROMPT_BLOCK is not None:
        return _CATALOG_PROMPT_BLOCK

    catalog = _load_catalog()
    lines = []
    for i, entry in enumerate(catalog, 1):
        commercial = f" [{entry['commercial_type']}]" if entry["commercial_type"] else ""
        lines.append(
            f"{i}. [{entry['ai_type']}] \"{entry['name']}\" — {entry['product']} — "
            f"\"{entry['description']}\" ({entry['availability']}){commercial}"
        )
    _CATALOG_PROMPT_BLOCK = "\n".join(lines)
    return _CATALOG_PROMPT_BLOCK


# ── LLM Prompt ───────────────────────────────────────────────────────────────

CATALOG_MATCH_SYSTEM_PROMPT = """\
You are an SAP solutions architect. Given a customer's business problem and proposed AI solution, identify which SAP standard AI features/agents from the catalog below are most relevant.

Select the TOP 1-3 entries that:
1. Solve the same or overlapping business problem
2. Could replace parts of the custom solution (the customer might not need to build it)
3. Could complement the custom solution (use alongside it)

Prioritize:
- "Generally Available" over "Beta" or "Early Adopter Care"
- Features that directly address the customer's stated problem over tangential matches
- "AI Agent" matches when the customer is building an Agent-type solution

AVOID:
- Generic "Joule with [Product]" entries that are just general-purpose copilots unless the customer's problem specifically calls for conversational navigation or Q&A within that product
- Recommending multiple entries that solve the exact same sub-problem (e.g., do not suggest two different document extraction tools — pick the best one)
- Entries whose relevance requires a stretch of interpretation (if you have to imagine a scenario to make it fit, it does not fit)

Return a JSON array (0-3 items, ordered by relevance):
[
  {
    "index": <number from the catalog list>
  }
]

If no entries are genuinely relevant, return an empty array []. Never pad with weak matches.
Return ONLY the JSON array, no markdown fences or additional text."""


# ── Background task ──────────────────────────────────────────────────────────

async def _persist_to_hana(context_id: str, result: Dict[str, Any]) -> None:
    """Persist catalog matching result to HANA (fire-and-forget, never raises)."""
    try:
        from .hana_store import get_store
        store = await get_store()
        await store.save_research(context_id, "catalog_matching", result.get("status", "failed"), result)
    except Exception as exc:
        logger.warning("[CatalogMatch] HANA persist failed context=%s: %s", context_id, exc)


async def _run_catalog_matching(context_id: str, final_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the catalog matching LLM call."""
    from .common import make_llm

    llm = make_llm(
        provider="google",
        model_name=CATALOG_MATCH_MODEL,
        temperature=CATALOG_MATCH_TEMPERATURE,
        json_mode=True,
    )

    catalog_block = _format_catalog_for_prompt()

    user_prompt = (
        f"## Customer's Problem\n"
        f"Industry: {final_payload.get('industry', 'Unknown')}\n"
        f"Problem: {final_payload.get('problem', '')}\n\n"
        f"## Proposed AI Solution\n"
        f"Title: {final_payload.get('title', 'Untitled')}\n"
        f"Type: {final_payload.get('app_type', 'Agent')}\n"
        f"Solution: {final_payload.get('solution', '')}\n\n"
        f"## SAP AI Catalog\n{catalog_block}"
    )

    messages = [
        SystemMessage(content=CATALOG_MATCH_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    started = time.perf_counter()
    response = await llm.ainvoke(messages)
    latency_ms = int((time.perf_counter() - started) * 1000)

    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")

    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            parsed = parsed.get("matches", []) if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, TypeError):
        return {"status": "failed", "error": "Invalid JSON from catalog matching LLM", "raw": raw_text[:500]}

    # Resolve indices to full catalog entries
    catalog = _load_catalog()
    matches = []
    for item in parsed[:3]:  # Cap at 3
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 1 or idx > len(catalog):
            continue
        entry = catalog[idx - 1]
        matches.append({
            "name": entry["name"],
            "ai_type": entry["ai_type"],
            "product": entry["product"],
            "product_category": entry["product_category"],
            "description": entry["description"],
            "availability": entry["availability"],
            "commercial_type": entry["commercial_type"],
            "detail_page_url": entry["detail_page_url"],
        })

    return {
        "status": "ok",
        "matches": matches,
        "candidates_evaluated": len(catalog),
        "latency_ms": latency_ms,
    }


async def start_catalog_matching(context_id: str, final_payload: Dict[str, Any], *, retried: bool = False) -> None:
    """Launch background catalog matching task for a conversation."""
    if context_id in _TASKS:
        logger.debug("[CatalogMatch] Task already exists for context=%s", context_id)
        return

    async def _run() -> None:
        try:
            result = await _run_catalog_matching(context_id, final_payload)
            _RESULTS[context_id] = result
            await _persist_to_hana(context_id, result)
            logger.info("[CatalogMatch] Complete context=%s status=%s matches=%d latency=%dms",
                        context_id, result.get("status"), len(result.get("matches", [])), result.get("latency_ms", 0))
        except Exception as exc:
            logger.error("[CatalogMatch] Failed context=%s error=%s", context_id, exc)
            error_result = {"status": "failed", "error": str(exc), "_retried": retried}
            _RESULTS[context_id] = error_result
            await _persist_to_hana(context_id, error_result)

    task = asyncio.create_task(_run())
    _TASKS[context_id] = task
    logger.info("[CatalogMatch] Started for context=%s", context_id)


def get_result(context_id: str) -> Optional[Dict[str, Any]]:
    """Non-blocking poll: returns matching result if ready, None if pending."""
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
    """Returns catalog matching status: 'idle', 'pending', 'complete', or 'failed'."""
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
    """Load cached catalog matching result from HANA into memory. Returns True if found."""
    try:
        from .hana_store import get_store
        store = await get_store()
        if context_id not in _RESULTS:
            row = await store.get_research(context_id, "catalog_matching")
            if row and row.get("payload"):
                _RESULTS[context_id] = row["payload"]
                return True
    except Exception as exc:
        logger.warning("[CatalogMatch] HANA load failed context=%s: %s", context_id, exc)
    return context_id in _RESULTS


def cleanup(context_id: str) -> None:
    """Remove references for a finished conversation."""
    _TASKS.pop(context_id, None)
    _RESULTS.pop(context_id, None)
