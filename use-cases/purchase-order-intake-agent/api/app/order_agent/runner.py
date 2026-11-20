"""Run the purchase-order agent for one order and record the run.

``get_runtime`` creates the shared ``AgentRuntime`` once per process (compiled graph, model client
and tools are reused; runs with different order ids execute fully in parallel).
``run_order_agent`` builds the input, runs the agent with a timeout, records token usage
(Cloud Logging event per model call + totals in ``POAGENT_AGENT_RUN``) and the bytes-free trace.

The status of the order is not changed here; the pipeline decides it with the deterministic gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict

from ..agent_core import AgentRuntime
from ..agent_core.runtime import _without_attachment_bytes
from ..db.engine import get_engine
from ..db.sql import insert_row
from ..observability import llm_usage_logging
from ..orders.service import new_id
from ..orders.settings import get_settings
from .prompt import build_agent_input
from .tools import build_order_tools

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "agent.yaml"
MAX_TRACE_CHARS = 2_000_000

_runtime: AgentRuntime | None = None
_runtime_lock = asyncio.Lock()


async def get_runtime() -> AgentRuntime:
    """Create (once) and return the process-wide agent runtime with the order tools.

    Returns:
        The shared AgentRuntime.
    """
    global _runtime
    async with _runtime_lock:
        if _runtime is None:
            _runtime = await AgentRuntime.create(CONFIG_PATH, extra_tools=build_order_tools())
            logger.info("Order agent runtime ready (model %s)", _runtime.config.model.name)
    return _runtime


async def close_runtime() -> None:
    """Close the shared runtime (FastAPI shutdown)."""
    global _runtime
    if _runtime is not None:
        await _runtime.aclose()
        _runtime = None


@dataclass
class RunOutcome:
    """Result of one agent run.

    Attributes:
        outcome: COMPLETED, NO_SUBMISSION, ERROR or TIMEOUT.
        summary: Final agent text for the reviewer.
        error: Error text for ERROR / TIMEOUT.
        llm_calls, input_tokens, cached_input_tokens, output_tokens: Usage totals.
        seconds: Wall-clock duration.
    """

    outcome: str
    summary: str | None = None
    error: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def _submitted(messages) -> bool:
    """True if a submit_order call in the trace was accepted."""
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "submit_order" and '"saved":true' in str(message.content):
            return True
    return False


def _record_run(run_id: str, order_id: str, started: datetime, outcome: RunOutcome, model: str,
                reviewer_note: str | None, trace: str | None) -> None:
    """Insert the POAGENT_AGENT_RUN row (blocking)."""
    with get_engine().begin() as conn:
        insert_row(conn, "POAGENT_AGENT_RUN", {
            "RUN_ID": run_id, "ORDER_ID": order_id, "STARTED_AT": started,
            "FINISHED_AT": datetime.now(timezone.utc).replace(tzinfo=None), "OUTCOME": outcome.outcome,
            "MODEL": model, "LLM_CALLS": outcome.llm_calls, "INPUT_TOKENS": outcome.input_tokens,
            "CACHED_INPUT_TOKENS": outcome.cached_input_tokens, "OUTPUT_TOKENS": outcome.output_tokens,
            "ERROR_TEXT": (outcome.error or None) and outcome.error[:2000],
            "REVIEWER_NOTE": (reviewer_note or None) and reviewer_note[:2000], "TRACE": trace,
        })


async def run_order_agent(order_id: str, reviewer_note: str | None = None) -> RunOutcome:
    """Run the agent for one order (status must already be PROCESSING).

    Args:
        order_id: Order id (also the agent context id).
        reviewer_note: Note of a reprocessing request, stored with the run.

    Returns:
        RunOutcome; exceptions of the agent run are captured as ERROR / TIMEOUT outcomes.
    """
    runtime = await get_runtime()
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    clock = time.monotonic()
    run_id = new_id()
    agent_input = await asyncio.to_thread(build_agent_input, order_id)
    messages = []
    try:
        result = await asyncio.wait_for(
            runtime.ainvoke(agent_input.text, context_id=order_id, attachments=agent_input.attachments),
            timeout=get_settings().agent_timeout_seconds,
        )
        messages = result.messages
        outcome = RunOutcome("COMPLETED" if _submitted(messages) else "NO_SUBMISSION", summary=result.output_text)
    except asyncio.TimeoutError:
        outcome = RunOutcome("TIMEOUT", error=f"agent run exceeded {get_settings().agent_timeout_seconds} s")
    except Exception as exc:  # noqa: BLE001 - recorded on the run and surfaced as FAILED by the pipeline
        logger.exception("Agent run failed for order %s", order_id)
        outcome = RunOutcome("ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        agent_input.cleanup()

    model = runtime.config.model.name
    for message in messages:
        if isinstance(message, AIMessage):
            usage = llm_usage_logging.extract_token_usage(message)
            outcome.llm_calls += 1
            outcome.input_tokens += usage.input_tokens
            outcome.cached_input_tokens += usage.cached_input_tokens
            outcome.output_tokens += usage.output_tokens
            llm_usage_logging.emit_llm_usage_event(
                route="/internal/order-agent", method="INTERNAL", actor_type="system", model=model,
                llm_endpoint="responses", input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens, output_tokens=usage.output_tokens,
                correlation_id=order_id,
            )
    outcome.seconds = round(time.monotonic() - clock, 1)
    trace = None
    if messages:
        trace = json.dumps(messages_to_dict([_without_attachment_bytes(m) for m in messages]), ensure_ascii=False,
                           default=str)[:MAX_TRACE_CHARS]
    await asyncio.to_thread(_record_run, run_id, order_id, started_at, outcome, model, reviewer_note, trace)
    return outcome
