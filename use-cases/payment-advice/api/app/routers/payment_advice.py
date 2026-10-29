"""
Payment Advice API router (UC-01 + UC-02).

UC-01 routes – customer registry management:
    GET  /customers                          list all customers
    POST /customers/{client_key}/promote     set is_critical = true
    POST /customers/{client_key}/demote      set is_critical = false

UC-02 routes – deduction interpretation:
    POST /interpret                          run deduction-interpretation agent,
                                             stream NDJSON events

All routes live under /api/payment-advice and are protected by the shared API key.

NDJSON envelope (one JSON object per line, media type application/x-ndjson):
    {"type": "analysis", "analyzed_refs": [...], "passthrough_count": n}
    {"type": "tools",    "tools_called": [...]}
    {"type": "result",   "enriched": {...}, "unresolved_count": n}
    {"type": "error",    "message": "<error description>"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..payment_advice.customers import (
    Customer,
    list_customers,
    normalize_client_key,
    set_critical,
)
from ..deduction_agent.chat_session import run_chat_turn
from ..payment_advice.db import bootstrap, get_engine
from ..payment_advice.extraction_service import extract_to_canonical
from ..security import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# Lazily-built, cached HANA engine (tables ensured + seeded once).
_engine = None


def _engine_ready():
    """Return the cached HANA engine, ensuring config tables exist on first use."""
    global _engine
    if _engine is None:
        engine = get_engine()
        bootstrap(engine)  # idempotent: create tables + seed critical customers
        _engine = engine
    return _engine


class CustomerModel(BaseModel):
    """API representation of one customer registry row."""

    client_key: str
    display_name: str
    is_critical: bool
    status: str

    @classmethod
    def from_customer(cls, customer: Customer) -> "CustomerModel":
        """Build the API model from the internal Customer record."""
        return cls(
            client_key=customer.client_key,
            display_name=customer.display_name,
            is_critical=customer.is_critical,
            status=customer.status,
        )


@router.get("/customers", response_model=list[CustomerModel])
async def get_customers() -> list[CustomerModel]:
    """List all registered customers and their criticality."""
    return [CustomerModel.from_customer(c) for c in list_customers(_engine_ready())]


@router.post("/customers/{client_key}/promote", response_model=CustomerModel)
async def promote_customer(client_key: str) -> CustomerModel:
    """Mark a client as critical (dedicated-schema strategy)."""
    return _set_critical(client_key, True)


@router.post("/customers/{client_key}/demote", response_model=CustomerModel)
async def demote_customer(client_key: str) -> CustomerModel:
    """Mark a client as non-critical (canonical-first strategy)."""
    return _set_critical(client_key, False)


def _set_critical(client_key: str, flag: bool) -> CustomerModel:
    """Set criticality for a client, 404 if the client is not registered."""
    key = normalize_client_key(client_key)
    updated = set_critical(_engine_ready(), key, flag)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"unknown client: {key}")
    return CustomerModel.from_customer(updated)


# ---------------------------------------------------------------------------
# UC-01: document extraction (PDF/tabular -> canonical payload)
# ---------------------------------------------------------------------------


@router.post("/extract")
async def extract(
    request: Request,
    client: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Extract an uploaded advice document to a UC-01 canonical payload.

    Accepts a multipart upload (``client`` form field + ``file``), resolves or
    generates the SAP Document AI schema for that client, and runs the extraction
    pipeline. Returns the canonical ``{header, line_items}`` payload plus a small
    verification summary. The returned payload is exactly what ``POST /interpret``
    expects as its ``payload`` field, so the UI can chain extract -> interpret.

    The SAP Document AI client and settings are built once during server lifespan
    and read from ``app.state``; a 503 is returned if they are unavailable (e.g.
    missing Document AI credentials).

    Args:
        request: FastAPI Request (injected) for app.state access.
        client:  Client key or display name (multipart form field).
        file:    The uploaded advice document (PDF/XLSX/CSV/TXT ...).

    Returns:
        ``{"client_key": str, "canonical": {...}, "verify": {...}}``.
    """
    dox = getattr(request.app.state, "dox", None)
    settings = getattr(request.app.state, "settings", None)
    engine = getattr(request.app.state, "engine", None)
    if dox is None or settings is None or engine is None:
        raise HTTPException(
            status_code=503,
            detail="extraction backend unavailable (SAP Document AI / HANA not configured)",
        )

    # Persist the upload to a temp dir under its original name: the splitter detects
    # format by file suffix, and pipeline output keeps a readable ``<client>_<stem>`` name.
    tmp_dir = Path(tempfile.mkdtemp(prefix="pa_extract_"))
    tmp_path = tmp_dir / Path(file.filename or "upload").name
    try:
        tmp_path.write_bytes(await file.read())
        # pipeline.run performs blocking SAP Document AI network I/O; run it in a
        # worker thread so concurrent async requests (e.g. /interpret streams) keep flowing.
        return await asyncio.to_thread(
            extract_to_canonical,
            tmp_path,
            client,
            engine=engine,
            dox=dox,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("extraction error for client %s: %s", client, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# UC-02: deduction interpretation (NDJSON streaming)
# ---------------------------------------------------------------------------

# Single authoritative schema for structured output — imported from the shared
# module so the interpretation service and this route cannot drift apart.
from ..deduction_agent.interpretation_schema import INTERPRETATION_SCHEMA as _INTERPRETATION_SCHEMA


class InterpretRequest(BaseModel):
    """Request body for POST /interpret.

    Attributes:
        client:  Client key or display name (normalized internally).
        payload: UC-01 canonical payload dict (header + line_items).
    """

    client: str
    payload: dict[str, Any]


async def _ndjson_stream(client: str, payload: dict[str, Any], request: Request) -> AsyncIterator[str]:
    """Stream the shared interpretation service using application dependencies."""
    from ..payment_advice.interpretation_service import interpret_events
    async for event in interpret_events(client, payload, runtime=getattr(request.app.state, "runtime", None), engine=getattr(request.app.state, "engine", None)):
        yield event


@router.post("/interpret")
async def interpret(body: InterpretRequest, request: Request) -> StreamingResponse:
    """Interpret deductions in a UC-01 canonical payload using the LLM agent.

    Accepts the canonical payload and client name in the request body, runs
    the UC-02 deduction-interpretation agent, and streams the result as NDJSON
    (one JSON object per line).

    The runtime used here is created once in main.py's lifespan and stored on
    app.state, avoiding per-request initialization overhead.

    Args:
        body:    Request body containing ``client`` and ``payload``.
        request: FastAPI Request (injected by the framework) for app.state access.

    Returns:
        A StreamingResponse with media type application/x-ndjson.
    """
    return StreamingResponse(
        _ndjson_stream(body.client, body.payload, request),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# UC-02: rules-authoring chat (multi-turn, attachments, HANA-persisted playbooks)
# ---------------------------------------------------------------------------


@router.post("/rules-chat")
async def rules_chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    files: list[UploadFile] | None = File(None),
) -> dict[str, Any]:
    """Run one turn of the UC-02 rules-authoring chat.

    Multipart form fields: ``message`` + ``session_id`` (+ optional ``files``).
    The chat runtime is multi-turn (history keyed by ``session_id`` in HANA), so a
    single conversation can look up a customer's existing deduction rules, ingest
    attached PDF/Word/Excel rule documents, propose a playbook, and save/update it
    to HANA on explicit confirmation. Returns the agent's reply and the tools it
    called this turn.

    Args:
        request:    FastAPI Request (injected) for app.state access.
        message:    The user's chat message.
        session_id: Client-generated conversation id (also the runtime context id).
        files:      Optional attached rule documents for this turn.

    Returns:
        ``{"reply": str, "tools_called": [...], "session_id": str}``.
    """
    chat_runtime = getattr(request.app.state, "chat_runtime", None)
    if chat_runtime is None:
        raise HTTPException(
            status_code=503,
            detail="rules-chat backend unavailable (agent runtime not configured)",
        )

    uploads = [(Path(f.filename or "upload").name, await f.read()) for f in (files or [])]
    try:
        return await run_chat_turn(chat_runtime, session_id, message, uploads)
    except ValueError as exc:
        # Bad session id, or an unsupported/unreadable attachment (RuleSourceError
        # subclasses ValueError): treat as a client error.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("rules-chat error for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
