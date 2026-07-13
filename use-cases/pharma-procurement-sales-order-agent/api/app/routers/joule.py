"""Joule adapter for the Pharma Procurement Sales Order Agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.pharma_order.graph import run_pharma_order_agent

logger = logging.getLogger(__name__)
router = APIRouter()

STREAM_IDEMPOTENCY_TTL_SECONDS = 30.0
STREAM_IDEMPOTENCY_MAX_KEYS = 256
STREAM_IDEMPOTENCY_CACHE: dict[str, float] = {}


class SampleSummaryResponse(BaseModel):
    total_items: int
    active_count: int
    status: str
    last_updated: str


@router.get("/summary", response_model=SampleSummaryResponse)
async def get_sample_summary() -> SampleSummaryResponse:
    """Small health-style response used by Joule destination checks."""
    return SampleSummaryResponse(
        total_items=42,
        active_count=37,
        status="healthy",
        last_updated="2026-01-15T10:30:00Z",
    )


def _flatten_payload(value: Any) -> list[Any]:
    flattened: list[Any] = []

    def visit(item: Any) -> None:
        flattened.append(item)
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return flattened


def _pick_question(payload: Any, fallback: str = "") -> str:
    if isinstance(payload, str) and payload.strip():
        return payload.strip()

    candidates: list[str] = []
    for item in _flatten_payload(payload):
        if isinstance(item, dict):
            for key in ("question", "query", "text", "message", "raw", "input"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
        elif isinstance(item, str) and item.strip():
            candidates.append(item.strip())

    return candidates[0] if candidates else fallback.strip()


def _joule_response(question: str, result: dict[str, Any]) -> dict[str, Any]:
    markdown = result.get("markdown") or result.get("answer") or "No answer was generated."
    return {
        "markdown": markdown,
        "answer": result.get("answer", markdown),
        "query": question,
        "agent": "pharma-order",
        "model": result.get("model"),
        "provider": result.get("provider"),
        "tool_call_count": result.get("tool_call_count", 0),
        "quickReplies": [
            {"title": "Check availability", "value": "Is this product available for shipment this week?"},
            {"title": "Recent orders", "value": "Show recent orders for this customer."},
            {"title": "Blocked orders", "value": "Which orders are currently blocked?"},
        ],
    }


async def _run_for_joule(question: str, *, route: str, method: str, correlation_id: str | None = None, client_host: str | None = None) -> dict[str, Any]:
    return await run_pharma_order_agent(
        question=question,
        include_trace=False,
        prompt_variant="joule",
        route=route,
        method=method,
        correlation_id=correlation_id,
        client_host=client_host,
    )


@router.post("/pharma-order")
async def pharma_order_joule_post(payload: Any = Body(default=None)) -> dict[str, Any]:
    question = _pick_question(payload, fallback="What can the Pharma Procurement Sales Order Agent help with?")
    result = await _run_for_joule(question, route="/api/joule/pharma-order", method="POST")
    return _joule_response(question, result)


@router.get("/pharma-order")
async def pharma_order_joule_get(
    question: str = Query(default="What is the price for Northstar for Glycemor 10 mg?")
) -> dict[str, Any]:
    result = await _run_for_joule(question, route="/api/joule/pharma-order", method="GET")
    return _joule_response(question, result)


def _sse_event(text: str) -> str:
    payload = {"message": {"parts": [{"text": text}]}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_idempotency_key(correlation_id: str, question: str) -> str:
    normalized_correlation = (correlation_id or "").strip()
    normalized_question = " ".join((question or "").casefold().split())
    if not normalized_correlation or not normalized_question:
        return ""
    return f"/api/joule/pharma-order/stream|{normalized_correlation}|{normalized_question}"


def _stream_idempotency_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else ""


def _register_stream_request(key: str, now: float) -> bool:
    """Return False when Joule repeats the same streamed request within the TTL."""
    if not key:
        return True

    expired_keys = [
        cached_key
        for cached_key, cached_at in STREAM_IDEMPOTENCY_CACHE.items()
        if now - cached_at > STREAM_IDEMPOTENCY_TTL_SECONDS
    ]
    for cached_key in expired_keys:
        STREAM_IDEMPOTENCY_CACHE.pop(cached_key, None)

    if key in STREAM_IDEMPOTENCY_CACHE:
        return False

    if len(STREAM_IDEMPOTENCY_CACHE) >= STREAM_IDEMPOTENCY_MAX_KEYS:
        oldest_key = min(STREAM_IDEMPOTENCY_CACHE, key=STREAM_IDEMPOTENCY_CACHE.get)
        STREAM_IDEMPOTENCY_CACHE.pop(oldest_key, None)

    STREAM_IDEMPOTENCY_CACHE[key] = now
    return True


@router.post("/pharma-order/stream")
async def pharma_order_joule_stream(
    http_request: Request,
    payload: Any = Body(default=None),
) -> StreamingResponse:
    started_at = time.perf_counter()
    request_trace_id = uuid.uuid4().hex[:12]
    headers = http_request.headers
    correlation_id = headers.get("x-correlationid") or headers.get("x-correlation-id") or ""
    traceparent = headers.get("traceparent", "")
    user_agent = headers.get("user-agent", "")
    client_host = http_request.client.host if http_request.client else ""
    payload_type = type(payload).__name__
    payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else []

    question = _pick_question(payload, fallback="What can the Pharma Procurement Sales Order Agent help with?")
    idempotency_key = _stream_idempotency_key(correlation_id, question)
    idempotency_hash = _stream_idempotency_hash(idempotency_key)
    is_duplicate_stream_request = not _register_stream_request(idempotency_key, started_at)

    logger.info(
        "Joule pharma-order stream start trace_id=%s correlation_id=%s traceparent=%s client=%s user_agent=%s payload_type=%s payload_keys=%s idempotency_hash=%s duplicate=%s question=%r",
        request_trace_id,
        correlation_id,
        traceparent,
        client_host,
        user_agent,
        payload_type,
        payload_keys,
        idempotency_hash,
        is_duplicate_stream_request,
        question[:500],
    )

    async def event_generator():
        if is_duplicate_stream_request:
            logger.warning(
                "Joule pharma-order stream duplicate_suppressed trace_id=%s correlation_id=%s idempotency_hash=%s",
                request_trace_id,
                correlation_id,
                idempotency_hash,
            )
            yield _sse_event("")
            return

        yield _sse_event("Pharma Procurement Sales Order Agent is reviewing the request and selecting tools...\n")
        await asyncio.sleep(0)

        try:
            result = await _run_for_joule(
                question,
                route="/api/joule/pharma-order/stream",
                method="POST",
                correlation_id=correlation_id or request_trace_id,
                client_host=user_agent or client_host,
            )
            answer = result.get("markdown") or result.get("answer") or "No readable answer was generated."
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Joule pharma-order stream emit_answer trace_id=%s correlation_id=%s elapsed_ms=%s tool_call_count=%s model=%s provider=%s answer_chars=%s",
                request_trace_id,
                correlation_id,
                elapsed_ms,
                result.get("tool_call_count", 0),
                result.get("model"),
                result.get("provider"),
                len(answer),
            )
            yield _sse_event(answer)
            logger.info(
                "Joule pharma-order stream complete trace_id=%s correlation_id=%s elapsed_ms=%s",
                request_trace_id,
                correlation_id,
                int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as error:
            logger.exception(
                "Joule pharma-order stream failed trace_id=%s correlation_id=%s elapsed_ms=%s",
                request_trace_id,
                correlation_id,
                int((time.perf_counter() - started_at) * 1000),
            )
            yield _sse_event(f"Pharma Procurement Sales Order Agent failed while processing the request: {error}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Pharma-Order-Trace-Id": request_trace_id,
        },
    )
