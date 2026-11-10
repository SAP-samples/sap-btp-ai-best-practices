"""Handover workspace router.

CRUD endpoints for handover workspaces, file uploads, context sources,
and AI-assisted pre-fill / agent chat.
Each handover is a standalone requirements-capture workspace that can
optionally link to an ideation conversation for pre-fill context.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ..auth import UserIdentity, get_current_user
from ..models.handover import (
    HandoverCreateRequest,
    HandoverFileMetadata,
    HandoverListResponse,
    HandoverPayload,
    HandoverResponse,
    HandoverUpdateRequest,
)
from ..observability.llm_usage_logging import meta_from_request
from ..services.hana_handover import get_handover_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Max file upload size: 25 MiB
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Valid file upload zones
_VALID_ZONES = {"asis", "data", "tobe", "logic", "transcript", "docs", "context"}

# Required fields that must be non-empty for validation
_REQUIRED_FIELDS = [
    "customer", "industry", "usecase_title", "problem_statement",
    "process_name", "process_freq", "data_input", "tobe_narrative",
]


# ── Handover CRUD ───────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_handover(
    body: HandoverCreateRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new handover workspace.

    Optionally link to an ideation conversation for pre-fill context.
    When a conversation is linked, it is automatically added as an
    ideation context source with extracted fields for pre-fill.
    """
    store = await get_handover_store()
    payload = body.payload.to_storage_dict() if body.payload else {}

    record = await store.create(
        user_id=user.user_id,
        payload=payload,
        linked_conversation_id=body.linked_conversation_id,
    )

    handover_id = record["id"]

    # Auto-add the linked conversation as a pre-fill source (no LLM extraction
    # here — that runs when the user clicks "Auto-fill form" via the prefill
    # endpoint, keeping creation fast).
    # Also seed the customer field from the conversation's company_name.
    if body.linked_conversation_id:
        try:
            from ideation_agent.hana_store import get_store as get_ideation_store

            ideation_store = await get_ideation_store()
            conv = await ideation_store.get_conversation_by_id(body.linked_conversation_id)
            if conv and conv.get("company_name") and not payload.get("customer"):
                await store.update_payload(handover_id, {"customer": conv["company_name"]})
        except Exception as exc:
            logger.warning("Failed to fetch company name for handover %s: %s", handover_id, exc)

        try:
            await store.add_context(
                handover_id=handover_id,
                source_type="ideation",
                source_ref=body.linked_conversation_id,
            )
        except Exception as exc:
            logger.warning(
                "Auto-add ideation context failed for handover %s: %s",
                handover_id, exc,
            )

    return {
        "id": handover_id,
        "status": record["status"],
        "linked_conversation_id": record.get("linked_conversation_id"),
        "created_at": record.get("created_at"),
    }


@router.get("")
async def list_handovers(
    user: UserIdentity = Depends(get_current_user),
) -> HandoverListResponse:
    """List all handover workspaces for the current user."""
    store = await get_handover_store()
    records = await store.list_by_user(user.user_id)
    items = [
        HandoverResponse(
            id=r["id"],
            status=r["status"],
            customer=r.get("customer"),
            usecase_title=r.get("usecase_title"),
            priority=r.get("priority"),
            linked_conversation_id=r.get("linked_conversation_id"),
            completion_pct=(r.get("payload") or {}).get("completion_pct", 0),
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at"),
            submitted_at=r.get("submitted_at"),
        )
        for r in records
    ]
    return HandoverListResponse(items=items, total=len(items))


@router.get("/{handover_id}")
async def get_handover(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Get a handover workspace with its full payload.

    Returns the payload flattened at the top level (same pattern as the
    original GTM intake form) plus id, status, and metadata fields.
    """
    store = await get_handover_store()
    record = await store.get(handover_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Flatten payload at top level for client compatibility
    payload = record.get("payload") or {}
    body: Dict[str, Any] = dict(payload)
    body["id"] = record["id"]
    body["status"] = record["status"]
    body["linked_conversation_id"] = record.get("linked_conversation_id")
    body["created_at"] = record.get("created_at")
    body["updated_at"] = record.get("updated_at")
    body["submitted_at"] = record.get("submitted_at")
    return JSONResponse(content=body)


@router.patch("/{handover_id}")
async def update_handover(
    handover_id: str,
    body: HandoverUpdateRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update handover form fields (partial JSON merge).

    The payload is merged with existing data — only the fields you send
    are overwritten.
    """
    store = await get_handover_store()

    # Verify ownership
    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    payload = body.payload.to_storage_dict()
    updated = await store.update_payload(handover_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Handover not found")

    return {"id": handover_id, "status": "updated"}


@router.post("/{handover_id}/complete")
async def complete_handover(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Mark a handover workspace as validated/complete."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Already validated — preserve original submitted_at
    if record["status"] == "validated":
        return {"id": handover_id, "status": "validated"}

    # Validate required fields before marking complete
    payload = record.get("payload") or {}
    missing = [f for f in _REQUIRED_FIELDS if not payload.get(f)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Required fields missing: {', '.join(missing)}",
        )

    await store.mark_complete(handover_id)
    return {"id": handover_id, "status": "validated"}


@router.delete("/{handover_id}")
async def delete_handover(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a handover workspace and all associated data."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    await store.delete(handover_id)
    return {"deleted": True}


@router.get("/{handover_id}/export")
async def export_handover(
    handover_id: str,
    format: str = Query("json", pattern="^(json|pdf|markdown|zip)$"),
    user: UserIdentity = Depends(get_current_user),
):
    """Export a handover in JSON, PDF, or Markdown format.

    Includes the full payload, file metadata, context sources, and
    ideation context if linked.
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    files = await store.list_files(handover_id)
    context_sources = await store.list_context(handover_id)

    export = {
        "version": "1.0",
        "id": record["id"],
        "status": record["status"],
        "linked_conversation_id": record.get("linked_conversation_id"),
        "payload": record.get("payload") or {},
        "files": [
            {"file_id": f["file_id"], "filename": f["filename"], "zone": f["zone"], "size": f["size"]}
            for f in files
        ],
        "context_sources": context_sources,
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "submitted_at": record.get("submitted_at"),
    }

    # Build a safe filename from the title
    import re
    title = (export["payload"].get("usecase_title") or handover_id or "export")
    safe_title = re.sub(r'[/\\:*?"<>|]', "_", title)[:100].strip()

    if format == "pdf":
        from ..services.handover_export import render_pdf
        pdf_bytes = render_pdf(export)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="handover-{safe_title}.pdf"'},
        )
    elif format == "markdown":
        from ..services.handover_export import render_markdown
        md_text = render_markdown(export)
        return Response(
            content=md_text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="handover-{safe_title}.md"'},
        )
    elif format == "zip":
        # Fetch all file BLOBs for the ZIP bundle
        file_contents = []
        for f in files:
            content = await store.get_file_content(f["file_id"], handover_id)
            if content and content.get("data"):
                file_contents.append({
                    "filename": content["filename"],
                    "zone": f.get("zone", "other"),
                    "data": content["data"],
                })

        # Fetch linked ideation conversation (if any)
        ideation_messages = None
        ideation_conversation = None
        linked_conv_id = record.get("linked_conversation_id")
        if not linked_conv_id:
            # Check context sources for an ideation-type link
            for src in context_sources:
                if src.get("source_type") == "ideation" and src.get("source_ref"):
                    linked_conv_id = src["source_ref"]
                    break

        if linked_conv_id:
            try:
                from ideation_agent.hana_store import get_store as get_ideation_store

                ideation_store = await get_ideation_store()
                ideation_messages = await ideation_store.get_messages(linked_conv_id)
                ideation_conversation = await ideation_store.get_conversation_by_id(linked_conv_id)
            except Exception as exc:
                logger.warning("Failed to fetch ideation conversation %s: %s", linked_conv_id, exc)

        from ..services.handover_export import render_zip
        zip_bytes = render_zip(export, file_contents, ideation_messages, ideation_conversation)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="handover-{safe_title}.zip"'},
        )
    else:
        return JSONResponse(content=export)


# ── Context Sources ─────────────────────────────────────────────────────


@router.post("/{handover_id}/context")
async def add_context_source(
    handover_id: str,
    request: Request,
    source_type: str = Form(...),
    source_ref: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Link a context source to the handover.

    source_type: ideation | document | manual
    source_ref: conversation ID (for ideation), filename (for document), or note text (for manual)
    file: optional file upload (for document type)

    Auto-extracts structured fields from the content (ideation output,
    file text, or note text) and stores them for pre-fill and agent search.
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    valid_types = {"ideation", "transcript", "document", "manual"}
    if source_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of: {', '.join(sorted(valid_types))}",
        )

    # Prevent duplicate ideation sources for the same conversation
    if source_type == "ideation" and source_ref:
        existing = await store.list_context(handover_id)
        for ctx in existing:
            if ctx.get("source_type") == "ideation" and ctx.get("source_ref") == source_ref:
                raise HTTPException(
                    status_code=409,
                    detail="This conversation is already linked as a context source",
                )

    llm_meta = meta_from_request(request, route="/api/handover/context", user_id=user.user_id)
    extracted_fields = None
    content_text = None

    # ── Ideation: extract from linked conversation ──
    if source_type == "ideation" and source_ref:
        try:
            from ideation_agent.handover_prefill import extract_from_ideation

            extracted_fields = await extract_from_ideation(source_ref, llm_usage_meta=llm_meta)
        except Exception as exc:
            logger.warning("Auto-extraction failed for ideation %s: %s", source_ref, exc)

    # ── Document: store file, extract text, run LLM extraction ──
    elif source_type == "document" and file is not None:
        try:
            file_data = await file.read()
            file_size = len(file_data)
            if file_size > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )

            filename = file.filename or "upload"
            source_ref = filename

            # Store file BLOB
            await store.add_file(
                handover_id=handover_id,
                zone="context",
                filename=filename,
                content_type=file.content_type or "application/octet-stream",
                size=file_size,
                data=file_data,
            )

            # Extract text from file
            from ideation_agent.text_extraction import extract_text_from_file

            content_text = extract_text_from_file(filename, file_data)

            # Run LLM extraction on extracted text
            if content_text:
                try:
                    from ideation_agent.handover_prefill import extract_from_text

                    extracted_fields = await extract_from_text(
                        content_text, source_type="document", llm_usage_meta=llm_meta
                    )
                except Exception as exc:
                    logger.warning("LLM extraction failed for document %s: %s", filename, exc)

        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Document processing failed: %s", exc)

    # ── Manual: extract from pasted text ──
    elif source_type == "manual" and source_ref:
        content_text = source_ref  # Store the text for agent search
        try:
            from ideation_agent.handover_prefill import extract_from_text

            extracted_fields = await extract_from_text(source_ref, source_type="manual", llm_usage_meta=llm_meta)
        except Exception as exc:
            logger.warning("Auto-extraction failed for manual note: %s", exc)

    result = await store.add_context(
        handover_id=handover_id,
        source_type=source_type,
        source_ref=source_ref,
        extracted_fields=extracted_fields,
        content_text=content_text,
    )
    return result


@router.get("/{handover_id}/context")
async def list_context_sources(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List context sources linked to the handover."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return await store.list_context(handover_id)


@router.delete("/{handover_id}/context/{context_id}")
async def remove_context_source(
    handover_id: str,
    context_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Remove a context source link."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    await store.delete_context(context_id, handover_id)
    return {"deleted": True}


# ── Pre-fill ──────────────────────────────────────────────────────────────


@router.post("/{handover_id}/prefill")
async def prefill_from_context(
    handover_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Run pre-fill extraction across all linked context sources.

    Merges extracted fields from all sources (ideation conversations,
    manual notes, etc.) and returns the combined field suggestions.
    Does NOT auto-write to the form — frontend shows inline suggestions.
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    context_sources = await store.list_context(handover_id)
    if not context_sources:
        return {"fields": {}, "field_count": 0, "source_count": 0}

    from ideation_agent.handover_prefill import (
        extract_from_ideation,
        extract_from_text,
        merge_extracted_fields,
    )

    llm_meta = meta_from_request(request, route="/api/handover/prefill", user_id=user.user_id)

    # Collect extracted fields from all sources
    all_extractions: List[Dict[str, Any]] = []
    sources_processed = 0

    for source in context_sources:
        is_ideation = source["source_type"] == "ideation"

        # Use cache for static sources (documents, manual notes).
        # Always re-extract ideation sources — conversation may have been
        # refined since the last extraction.
        if source.get("extracted_fields") and not is_ideation:
            all_extractions.append(source["extracted_fields"])
            sources_processed += 1
            continue

        # Extract now — isolate per-source errors so one failure
        # doesn't lose all other successfully-extracted fields.
        try:
            extracted = {}
            if is_ideation and source.get("source_ref"):
                extracted = await extract_from_ideation(source["source_ref"], llm_usage_meta=llm_meta)
            elif source["source_type"] == "manual" and source.get("source_ref"):
                extracted = await extract_from_text(
                    source["source_ref"], source_type="manual", llm_usage_meta=llm_meta
                )

            if extracted is not None:
                # Cache for future calls
                await store.update_context_extracted(source["id"], extracted, handover_id)
                all_extractions.append(extracted)
                sources_processed += 1
        except Exception as exc:
            logger.warning("Extraction failed for source %s: %s", source.get("id"), exc)

    # Merge all extractions (later sources override earlier for conflicts)
    merged = merge_extracted_fields(*all_extractions)

    return {
        "fields": merged,
        "field_count": len(merged),
        "source_count": sources_processed,
    }


# ── Agent Chat ─────────────────────────────────────────────────────────


@router.post("/{handover_id}/agent")
async def agent_chat(
    handover_id: str,
    request: Request,
    body: Dict[str, Any] = Body(...),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Send a message to the handover agent assistant.

    Request body:
        message: str — the user's message text
        target_section: str | null — optional section ID to focus on

    Returns a structured response with kind, message, and optional fields.
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    user_text = body.get("message", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message is required")

    target_section = body.get("target_section")

    # Load current state
    form_state = record.get("payload") or {}
    context_sources = await store.list_context(handover_id)
    prior_messages = await store.get_agent_messages(handover_id)

    # Build LLM usage meta
    llm_meta = meta_from_request(request, route="/api/handover/agent", user_id=user.user_id)

    # Run agent
    from ideation_agent.handover_agent import run_handover_agent

    result = await run_handover_agent(
        user_text=user_text,
        handover_id=handover_id,
        form_state=form_state,
        context_sources=context_sources,
        prior_messages=prior_messages,
        target_section=target_section,
        llm_usage_meta=llm_meta,
    )

    # Persist messages (append user + assistant)
    prior_messages.append({"role": "user", "content": user_text})
    prior_messages.append({
        "role": "assistant",
        "content": result.get("message", ""),
        "kind": result.get("kind"),
        "chips": result.get("suggested_replies"),
    })
    await store.save_agent_messages(handover_id, prior_messages)

    return result


@router.get("/{handover_id}/agent/history")
async def agent_history(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the agent conversation history for UI replay."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    messages = await store.get_agent_messages(handover_id)
    return {"messages": messages}


# ── File Uploads ────────────────────────────────────────────────────────


@router.post("/{handover_id}/files", response_model=HandoverFileMetadata)
async def upload_file(
    handover_id: str,
    file: UploadFile = File(...),
    zone: str = Form(...),
    user: UserIdentity = Depends(get_current_user),
) -> HandoverFileMetadata:
    """Upload a file to a specific zone of the handover workspace.

    Valid zones: asis, data, tobe, logic, transcript, docs
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if zone not in _VALID_ZONES:
        raise HTTPException(
            status_code=400,
            detail=f"zone must be one of: {', '.join(sorted(_VALID_ZONES))}",
        )

    data = await file.read()
    size = len(data)
    if size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    meta = await store.add_file(
        handover_id=handover_id,
        zone=zone,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        size=size,
        data=data,
    )
    return HandoverFileMetadata(**meta)


@router.get("/{handover_id}/files")
async def list_files(
    handover_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> List[HandoverFileMetadata]:
    """List file metadata for a handover, grouped by zone."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    files = await store.list_files(handover_id)
    return [HandoverFileMetadata(**f) for f in files]


@router.delete("/{handover_id}/files/{file_id}")
async def delete_file(
    handover_id: str,
    file_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a single uploaded file."""
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    await store.delete_file(file_id, handover_id)
    return {"deleted": True}


@router.get("/{handover_id}/files/{file_id}/content")
async def get_file_content(
    handover_id: str,
    file_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Response:
    """Serve raw file content for preview or download.

    Images and PDFs are served inline (for lightbox preview).
    Other types are served as attachment (triggers browser download).
    """
    store = await get_handover_store()

    record = await store.get(handover_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Handover not found")
    if record["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    file_data = await store.get_file_content(file_id, handover_id)
    if file_data is None:
        raise HTTPException(status_code=404, detail="File not found")

    content_type = file_data["content_type"] or "application/octet-stream"
    filename = file_data["filename"] or "download"

    # Images and PDFs served inline for lightbox; everything else as download
    if content_type.startswith("image/") or content_type == "application/pdf":
        disposition = f'inline; filename="{filename}"'
    else:
        disposition = f'attachment; filename="{filename}"'

    return Response(
        content=file_data["data"],
        media_type=content_type,
        headers={
            "Content-Disposition": disposition,
            "Cache-Control": "private, max-age=3600",
        },
    )
