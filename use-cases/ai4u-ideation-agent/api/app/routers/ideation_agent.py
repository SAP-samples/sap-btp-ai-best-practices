from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import JSONResponse

from app.observability.llm_usage_logging import (
    emit_llm_usage_event,
    emit_llm_usage_for_response,
    meta_from_mapping,
    meta_from_request,
    meta_to_mapping,
)
from ..auth import UserIdentity, get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC Error Codes
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603

# A2A-specific Error Codes
ERROR_TASK_NOT_FOUND = -32001
ERROR_TASK_NOT_CANCELABLE = -32002
ERROR_PUSH_NOT_SUPPORTED = -32003
ERROR_CONTENT_TYPE_NOT_SUPPORTED = -32005

TERMINAL_STATES = {"completed", "canceled", "failed", "rejected"}

# Model for conversation-title generation — a fast/cheap NON-reasoning model that must
# be deployed in your AI Core. Defaults to gpt-4o-mini (verified to return a clean title
# at low max_tokens). Avoid reasoning models here (gpt-5-mini, gemini-2.5-flash): with a
# small token budget they spend it on reasoning and return empty output. Override with
# IDEATION_TITLE_MODEL. gpt-* models route via the openai provider, otherwise google.
TITLE_MODEL = os.getenv("IDEATION_TITLE_MODEL", "gpt-4o-mini")

INTRO_TEXT = (
    "**Turn your challenge into a practical AI solution path.**\n\n"
    "Share the challenge, idea, or use case you want to make real with AI.\n\n"
    "Here is how I will help:\n\n"
    "- ask up to **3 focused clarification questions**\n"
    "- identify the **best-fit AI capabilities**\n"
    "- propose a **practical SAP BTP AI solution**"
)

# In-memory task store (short-lived, only for tasks/get within same session)
_TASKS: Dict[str, Dict[str, Any]] = {}

# In-flight turn registry for cooperative cancellation (the Stop button).
# Keyed context_id -> message_id -> {"cancelled": bool, "agent_task": asyncio.Task | None}.
# The client mints the message_id but never sees the server task_id mid-turn, so
# cancellation keys on (context_id, message_id).
_INFLIGHT: Dict[str, Dict[str, Dict[str, Any]]] = {}
# Per-conversation lock serializing turns on the same context_id, so a cancelled
# turn's rollback always settles before the next send reads the message count
# (prevents sequence-number collisions between overlapping turns).
_CTX_LOCKS: Dict[str, asyncio.Lock] = {}


def _ctx_lock(context_id: str) -> asyncio.Lock:
    """Return the per-conversation lock, creating it on first use.

    Safe without extra guarding: the event loop is single-threaded and there is
    no await between the get and the set.
    """
    lock = _CTX_LOCKS.get(context_id)
    if lock is None:
        lock = asyncio.Lock()
        _CTX_LOCKS[context_id] = lock
    return lock


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonrpc_error(
    req_id: Optional[Any],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": error}


def _jsonrpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def _text_part(text: str) -> Dict[str, Any]:
    return {"kind": "text", "text": text}


def _message(
    role: str,
    text: str,
    message_id: Optional[str] = None,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    msg_id = message_id or str(uuid.uuid4())
    msg: Dict[str, Any] = {
        "kind": "message",
        "role": role,
        "parts": [_text_part(text)],
        "messageId": msg_id,
    }
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    if metadata is not None:
        msg["metadata"] = metadata
    return msg


def _artifact(text: str) -> Dict[str, Any]:
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "answer.txt",
        "parts": [_text_part(text)],
    }


def _task(
    task_id: str,
    context_id: str,
    state: str,
    status_message: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": _now_iso()},
    }
    if status_message:
        task["status"]["message"] = status_message
    if history is not None:
        task["history"] = history
    if artifacts is not None:
        task["artifacts"] = artifacts
    return task


def _apply_history_length(
    task: Dict[str, Any], history_length: Optional[int]
) -> Dict[str, Any]:
    if history_length is None or history_length < 0:
        return task
    result = dict(task)
    history = task.get("history")
    if isinstance(history, list):
        result["history"] = history[-history_length:] if history_length else []
    return result


def _validate_jsonrpc_request(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return _jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request")
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        return _jsonrpc_error(
            payload.get("id"), ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request"
        )
    if "id" not in payload or payload.get("id") is None:
        return _jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request")
    if not isinstance(payload.get("method"), str):
        return _jsonrpc_error(
            payload.get("id"), ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request"
        )
    return None


def _extract_text_parts(parts: List[Dict[str, Any]]) -> Optional[str]:
    texts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        if part.get("kind") != "text":
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        texts.append(text)
    return "\n".join(texts)


def _extract_message_content(
    parts: List[Dict[str, Any]],
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Extract text AND file references from message parts.

    Accepts "text" and "file" kind parts. File parts carry
    file_id + context_id (Object Store flow — processed server-side).

    Returns:
        (text, file_parts) on success, None on invalid input.
        file_parts entries are dicts with {file_id, context_id, filename, num_outputs, original_type}.
    """
    texts: List[str] = []
    file_parts: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        kind = part.get("kind")
        if kind == "text":
            text = part.get("text")
            if not isinstance(text, str):
                return None
            texts.append(text)
        elif kind == "file":
            file_id = part.get("file_id")
            if isinstance(file_id, str) and file_id:
                file_parts.append({
                    "file_id": file_id,
                    "context_id": part.get("context_id", ""),
                    "filename": part.get("filename", ""),
                    "num_outputs": part.get("num_outputs", 1),
                    "original_type": part.get("original_type", ""),
                })
            else:
                return None  # Invalid file part — reject
        else:
            return None  # Unknown kind — reject
    return ("\n".join(texts), file_parts)


def _validate_message_params(
    params: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Validate A2A message parameters and extract content.

    Returns:
        (error, text, message_id, context_id, file_content_blocks)
        On error: (error_dict, None, None, None, [])
        On success: (None, text, message_id, context_id, file_blocks)
    """
    empty_blocks: List[Dict[str, Any]] = []

    message = params.get("message")
    if not isinstance(message, dict):
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
            empty_blocks,
        )

    role = message.get("role")
    if role not in {"user", "agent"}:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
            empty_blocks,
        )

    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
            empty_blocks,
        )

    # Try multimodal extraction first (handles both text-only and mixed)
    content_result = _extract_message_content(parts)
    if content_result is not None:
        text, file_content_blocks = content_result
    else:
        # Fall back to legacy text-only extraction for backward compat
        text = _extract_text_parts(parts)
        file_content_blocks = empty_blocks
        if text is None:
            return (
                {
                    "code": ERROR_CONTENT_TYPE_NOT_SUPPORTED,
                    "message": "Incompatible content types",
                },
                None,
                None,
                None,
                empty_blocks,
            )

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
            empty_blocks,
        )

    context_id = message.get("contextId")
    if context_id is not None and (not isinstance(context_id, str) or not context_id):
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
            empty_blocks,
        )

    return (None, text, message_id, context_id, file_content_blocks)


async def _get_store():
    from ideation_agent.hana_store import get_store
    return await get_store()


def _public_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip server-internal metadata (the agent-state snapshot) from messages
    before they are returned to the client. The snapshot can contain internal
    reasoning (e.g. reimagine_context) that the UI deliberately never shows."""
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        meta = m.get("metadata")
        if isinstance(meta, dict) and "agent_state_snapshot" in meta:
            m = {**m, "metadata": {k: v for k, v in meta.items() if k != "agent_state_snapshot"}}
        cleaned.append(m)
    return cleaned


def _restore_state_for(
    messages: List[Dict[str, Any]], sequence_num: int
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Compute the agent-state to restore when rewinding to before `sequence_num`.

    Returns (agent_state_json_or_None, remaining_messages). The state is taken from
    the highest-sequence remaining AGENT row: its `agent_state_snapshot` (exact
    resume input) when present, else reconstructed from its `ui` metadata (graceful
    fallback for pre-feature rows), else None (fresh state — no agent row remains).
    """
    remaining = [m for m in messages if int(m["sequence_num"]) < sequence_num]
    for m in reversed(remaining):
        if m.get("role") != "agent":
            continue
        meta = m.get("metadata") or {}
        snap = meta.get("agent_state_snapshot")
        if isinstance(snap, dict):
            return json.dumps(snap), remaining
        ui = meta.get("ui")
        if isinstance(ui, dict):
            return (
                json.dumps({
                    "clarification_count": ui.get("clarificationCount", 0),
                    "last_turn_kind": ui.get("kind", ""),
                    "refinement_mode": ui.get("kind") == "final",
                }),
                remaining,
            )
        break
    return None, remaining


async def _has_write_access(conv_id: str, user: UserIdentity) -> bool:
    """True if the user owns the conversation or has 'write' share access."""
    store = await _get_store()
    if await store.get_conversation(conv_id, user.user_id):
        return True
    from ..services.hana_shares import get_share_store
    share_store = await get_share_store()
    permission = await share_store.get_member_permission(conv_id, user.user_id)
    return permission == "write"


async def _rewind_conversation(conv_id: str, sequence_num: int) -> Dict[str, Any]:
    """Truncate to before `sequence_num` and restore agent state, serialized per
    context so it never races an in-flight turn. Returns the truncated transcript."""
    store = await _get_store()
    async with _ctx_lock(conv_id):
        messages = await store.get_messages(conv_id)
        restore_state, remaining = _restore_state_for(messages, sequence_num)
        await store.rewind_to(conv_id, sequence_num, restore_state, message_count=len(remaining))
        public = _public_messages(await store.get_messages(conv_id))
    return {"messages": public, "message_count": len(public)}


async def _rollback_cancelled_turn(
    store: Any, context_id: str, is_continue: bool, msg_count: int
) -> None:
    """Discard a stopped turn's own writes: delete the user row appended before the
    agent ran. (The agent row and state write are simply skipped by the caller.)
    Safe under the per-context lock — no concurrent turn can have added rows above
    msg_count, so this deletes exactly the just-added user row."""
    if is_continue:
        return  # __continue__ turns never append a user row
    try:
        await store.delete_messages_from(context_id, msg_count + 1)
    except Exception as exc:
        _logger.warning("Rollback of cancelled turn failed for %s: %s", context_id, exc)


async def _run_agent(
    text: str,
    context_id: str,
    company_name: str = "",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
    prior_state: Optional[Dict[str, Any]] = None,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
    variant: str = "standard",
    file_content_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from ideation_agent.agent import run_agent

    return await run_agent(
        text,
        context_id,
        company_name=company_name,
        prior_messages=prior_messages,
        prior_state=prior_state,
        llm_usage_meta=llm_usage_meta,
        variant=variant,
        file_content_blocks=file_content_blocks,
    )


_logger = logging.getLogger(__name__)


async def _analyze_document_via_responses_api(
    file_data: bytes, filename: str, mime_type: str
) -> Optional[str]:
    """Analyze a document using the Responses API for native understanding.

    Sends the raw file directly to GPT-5.4 via the Responses API `input_file`
    content block. The model natively processes:
    - PDF: text, images, diagrams, tables, scanned content (full vision)
    - Office (DOCX/PPTX/XLSX): text extraction (images/charts not extracted)
    - Text/Code: full content understanding
    - Email: message structure and content

    Args:
        file_data: Raw file bytes.
        filename: Original filename for context.
        mime_type: MIME type of the file (used in data URL).

    Returns:
        Extracted document content as formatted text, or None on failure.
    """
    import asyncio
    import base64

    try:
        from gen_ai_hub.proxy.native.openai.clients import OpenAI, set_deployment  # type: ignore
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client  # type: ignore

        file_b64 = base64.b64encode(file_data).decode("utf-8")

        # responses.create() is synchronous — run in thread pool to avoid blocking
        def _call_responses_api():
            proxy_client = get_proxy_client("gen-ai-hub")
            deployment = proxy_client.select_deployment(model_name="gpt-5.4")
            client = OpenAI()
            with set_deployment(deployment):
                return client.responses.create(
                    model="gpt-5.4",
                    input=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": filename,
                                "file_data": f"data:{mime_type};base64,{file_b64}",
                            },
                            {
                                "type": "input_text",
                                "text": (
                                    "Extract and describe the complete content of this document. "
                                    "Include:\n"
                                    "- All text content (preserve structure: headings, paragraphs, lists)\n"
                                    "- Descriptions of all figures, diagrams, charts, and images\n"
                                    "- All table data (preserve tabular structure)\n"
                                    "- Any metadata (title, author, dates) if visible\n\n"
                                    "Be comprehensive — capture everything in the document. "
                                    "Format your output as clean markdown."
                                ),
                            },
                        ],
                    }],
                    reasoning={"effort": "low"},
                )

        response = await asyncio.to_thread(_call_responses_api)

        extracted_text = response.output_text
        if extracted_text:
            # Choose emoji based on type
            emoji = "📄"
            if mime_type == "application/pdf":
                emoji = "📄"
            elif "spreadsheet" in mime_type or "excel" in mime_type or mime_type in ("text/csv", "text/tab-separated-values"):
                emoji = "📊"
            elif "presentation" in mime_type or "powerpoint" in mime_type:
                emoji = "📑"
            elif "word" in mime_type or "document" in mime_type or mime_type == "application/msword":
                emoji = "📝"
            elif mime_type == "message/rfc822":
                emoji = "📧"
            elif mime_type.startswith("text/") or mime_type == "application/json" or mime_type == "application/sql":
                emoji = "📋"

            return f"[{emoji} Content extracted from uploaded file: {filename}]\n\n{extracted_text}"
        return None

    except Exception as exc:
        _logger.error("Document analysis via Responses API failed for '%s': %s", filename, exc)
        return f"[📄 File uploaded: {filename} — content extraction failed, please describe what you'd like to know about this document]"


# Keep backward-compatible alias
async def _analyze_pdf_via_responses_api(pdf_data: bytes, filename: str) -> Optional[str]:
    """Backward-compatible wrapper — delegates to generalized function."""
    return await _analyze_document_via_responses_api(pdf_data, filename, "application/pdf")


async def _generate_title(
    conv_id: str,
    user_message: str,
    agent_response: str,
    company_name: str = "",
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Generate a short conversation title. Returns the title or None on failure."""
    model_name = TITLE_MODEL
    meta = meta_from_mapping(llm_usage_meta, default_route="/api/ideation-agent/a2a")
    try:
        from ideation_agent.common import make_llm

        provider = "openai" if model_name.startswith("gpt") else "google"
        llm = make_llm(provider=provider, model_name=model_name, temperature=0.3, max_tokens=20)
        prompt = (
            "Generate a 2-4 word topic for this conversation. "
            "Return ONLY the topic, no quotes or punctuation.\n\n"
            f"User: {user_message[:300]}\nAssistant: {agent_response[:300]}"
        )
        started = time.perf_counter()
        try:
            result = await llm.ainvoke(prompt)
        except Exception:
            emit_llm_usage_event(
                route=meta.route,
                method=meta.method,
                user_id=meta.user_id,
                client_host=meta.client_host,
                provider="sap-ai-core",
                model=model_name,
                llm_endpoint="chat.completions",
                input_tokens=0,
                output_tokens=0,
                outcome="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                correlation_id=meta.correlation_id,
                step_name="ideation_agent.title_generation",
                feature="ideation_agent",
            )
            raise
        emit_llm_usage_for_response(
            result,
            meta=meta,
            provider="sap-ai-core",
            model=model_name,
            llm_endpoint="chat.completions",
            latency_ms=int((time.perf_counter() - started) * 1000),
            step_name="ideation_agent.title_generation",
            feature="ideation_agent",
        )
        topic = (result.content or "").strip().strip('"\'')
        if topic:
            title = f"{company_name}: {topic}" if company_name else topic
            store = await _get_store()
            await store.update_conversation(conv_id, title=title)
            return title
    except Exception as exc:
        _logger.warning("Title generation failed for %s: %s", conv_id, exc)
    return None


async def _handle_message_send(
    req_id: Any, params: Dict[str, Any], user: UserIdentity, request: Request
) -> Dict[str, Any]:
    """Serialize turns per conversation and register them for cooperative
    cancellation (the Stop button), then run the turn. The per-context lock
    guarantees a cancelled turn's rollback settles before the next send reads the
    message count, so overlapping turns can never collide on sequence numbers."""
    error, text, message_id, provided_context_id, file_content_blocks = _validate_message_params(params)
    if error:
        return _jsonrpc_error(req_id, error["code"], error["message"])

    context_id = provided_context_id or str(uuid.uuid4())
    turn_key = message_id or str(uuid.uuid4())

    async with _ctx_lock(context_id):
        inflight = _INFLIGHT.setdefault(context_id, {})
        inflight[turn_key] = {"cancelled": False, "agent_task": None}
        try:
            return await _run_message_turn(
                req_id,
                params,
                user,
                request,
                text=text or "",
                message_id=message_id,
                provided_context_id=provided_context_id,
                file_content_blocks=file_content_blocks,
                context_id=context_id,
                turn_key=turn_key,
            )
        finally:
            ctx_map = _INFLIGHT.get(context_id)
            if ctx_map is not None:
                ctx_map.pop(turn_key, None)
                if not ctx_map:
                    _INFLIGHT.pop(context_id, None)


async def _run_message_turn(
    req_id: Any,
    params: Dict[str, Any],
    user: UserIdentity,
    request: Request,
    *,
    text: str,
    message_id: Optional[str],
    provided_context_id: Optional[str],
    file_content_blocks: List[Dict[str, Any]],
    context_id: str,
    turn_key: str,
) -> Dict[str, Any]:
    metadata = params.get("metadata") if isinstance(params, dict) else None
    company_name = str(metadata.get("company_name", "") or "").strip() if isinstance(metadata, dict) else ""
    event_id = str(metadata.get("event_id", "") or "").strip() or None if isinstance(metadata, dict) else None
    variant = str(metadata.get("variant", "") or "").strip() if isinstance(metadata, dict) else ""
    if variant != "provocateur":
        variant = "provocateur"  # Critical Mode is now the default for all users

    # Extract displayText from message-level metadata (for clean UI on reload)
    raw_message = params.get("message") if isinstance(params, dict) else {}
    msg_display_text = ""
    if isinstance(raw_message, dict):
        msg_meta = raw_message.get("metadata")
        if isinstance(msg_meta, dict):
            msg_display_text = str(msg_meta.get("displayText", "") or "").strip()

    # CSE enrichment is SAP-internal only — for external users, pre-disable CSE
    # so the agent skips it but still runs Perplexity and LinkedIn research.
    is_external = user.auth_provider == "cdc"

    store = await _get_store()
    task_id = str(uuid.uuid4())
    llm_usage_meta = meta_to_mapping(
        meta_from_request(
            request,
            route="/api/ideation-agent/a2a",
            method="POST",
            user_id=user.user_id,
            correlation_id=task_id,
        )
    )

    # Create conversation in HANA if it's the first message (no existing context_id)
    # or if a client-provided context_id doesn't have a conversation yet (file upload flow)
    if not provided_context_id:
        title = company_name if company_name else "New conversation"
        await store.create_conversation(
            conv_id=context_id,
            user_id=user.user_id,
            company_name=company_name,
            title=title,
            event_id=event_id,
        )
    else:
        conv = await store.get_conversation(context_id, user.user_id)
        if not conv:
            # Check if the user has shared access
            from ..services.hana_shares import get_share_store
            share_store = await get_share_store()
            permission = await share_store.get_member_permission(context_id, user.user_id)
            if permission:
                if permission != "write":
                    return _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "You have read-only access to this conversation")
            else:
                # No conversation exists and no shared access — create it
                # (happens when client generates context_id for file upload)
                title = company_name if company_name else "New conversation"
                await store.create_conversation(
                    conv_id=context_id,
                    user_id=user.user_id,
                    company_name=company_name,
                    title=title,
                    event_id=event_id,
                )
        elif company_name and not conv.get("company_name"):
            await store.update_conversation(
                context_id,
                company_name=company_name,
                title=company_name,
            )

    # Get current message count for sequencing
    msg_count = await store.get_message_count(context_id)

    # Handle "__continue__" — auto-follow-up after a thinking card (no user message to store)
    is_continue = (text == "__continue__")

    # Resolve file parts into LLM content blocks (from Object Store or inline)
    resolved_content_blocks: List[Dict[str, Any]] = []
    file_metadata_for_storage: List[Dict[str, Any]] = []
    if file_content_blocks:
        for fp in file_content_blocks:
            if "file_id" in fp:
                # New Object Store flow: load processed outputs
                fid = fp["file_id"]
                fctx = fp.get("context_id") or context_id
                num_outputs = fp.get("num_outputs", 1)
                original_type = fp.get("original_type", "")
                try:
                    import base64
                    from ..services.file_cache import get_file_outputs

                    # Try in-memory cache first (avoids S3 round-trip on first message)
                    cached_outputs = get_file_outputs(fctx, fid)

                    if original_type == "application/pdf":
                        # PDF: use Responses API for native understanding
                        pdf_data = None
                        if cached_outputs:
                            pdf_data = cached_outputs[0]
                        else:
                            from ..services.object_store import get_object_store
                            obj_store = await get_object_store()
                            output_key = f"{fctx}/{fid}/0.pdf"
                            pdf_data = await obj_store.download(output_key)

                        if pdf_data:
                            pdf_text = await _analyze_document_via_responses_api(
                                pdf_data, fp.get("filename", "document.pdf"), "application/pdf"
                            )
                            if pdf_text:
                                resolved_content_blocks.append({
                                    "type": "text",
                                    "text": pdf_text,
                                })

                    elif original_type and not original_type.startswith("image/"):
                        # Documents (Office, text, email, etc.): use Responses API
                        from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
                        if original_type in DOCUMENT_MIME_TYPES:
                            doc_data = None
                            if cached_outputs:
                                doc_data = cached_outputs[0]
                            else:
                                from ..services.object_store import get_object_store
                                obj_store = await get_object_store()
                                # Determine extension from MIME type
                                _doc_ext_map = {
                                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                                    "application/msword": "doc",
                                    "application/vnd.ms-powerpoint": "ppt",
                                    "application/vnd.ms-excel": "xls",
                                    "application/vnd.oasis.opendocument.text": "odt",
                                    "application/vnd.oasis.opendocument.presentation": "odp",
                                    "application/vnd.oasis.opendocument.spreadsheet": "ods",
                                    "application/rtf": "rtf",
                                    "text/csv": "csv",
                                    "text/tab-separated-values": "tsv",
                                    "message/rfc822": "eml",
                                    "text/plain": "txt",
                                    "text/markdown": "md",
                                    "application/json": "json",
                                    "text/xml": "xml",
                                    "text/yaml": "yaml",
                                    "text/x-python": "py",
                                    "text/javascript": "js",
                                    "text/html": "html",
                                    "text/css": "css",
                                    "application/sql": "sql",
                                }
                                doc_ext = _doc_ext_map.get(original_type, "bin")
                                output_key = f"{fctx}/{fid}/0.{doc_ext}"
                                doc_data = await obj_store.download(output_key)

                            if doc_data:
                                doc_text = await _analyze_document_via_responses_api(
                                    doc_data, fp.get("filename", "document"), original_type
                                )
                                if doc_text:
                                    resolved_content_blocks.append({
                                        "type": "text",
                                        "text": doc_text,
                                    })

                    else:
                        # Images: existing image_url flow
                        if cached_outputs:
                            for img_data in cached_outputs[:num_outputs]:
                                b64 = base64.b64encode(img_data).decode("ascii")
                                resolved_content_blocks.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}",
                                        "detail": "high",
                                    },
                                })
                        else:
                            # Cache miss: fall back to S3 download
                            from ..services.object_store import get_object_store
                            obj_store = await get_object_store()
                            for idx in range(num_outputs):
                                output_key = f"{fctx}/{fid}/{idx}.png"
                                img_data = await obj_store.download(output_key)
                                if img_data:
                                    b64 = base64.b64encode(img_data).decode("ascii")
                                    resolved_content_blocks.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{b64}",
                                            "detail": "high",
                                        },
                                    })
                except Exception as exc:
                    _logger.warning("Failed to load file %s from Object Store: %s", fid, exc)

                file_metadata_for_storage.append({
                    "file_id": fid,
                    "filename": fp.get("filename", ""),
                    "context_id": fctx,
                    "num_outputs": num_outputs,
                    "original_type": original_type,
                })

    if not is_continue:
        # Persist user message (include sender attribution for shared conversations)
        user_msg_metadata = {}
        if msg_display_text:
            user_msg_metadata["displayText"] = msg_display_text
        # Always include sender info for message attribution
        user_msg_metadata["sender_uid"] = user.user_id
        sender_name = ""
        if user.given_name:
            sender_name = user.given_name
            if user.family_name:
                sender_name += f" {user.family_name}"
        if sender_name:
            user_msg_metadata["sender_name"] = sender_name

        # Store file metadata for conversation replay (lightweight — no base64)
        if file_metadata_for_storage:
            user_msg_metadata["files"] = file_metadata_for_storage

        await store.append_message(
            conv_id=context_id,
            msg_id=message_id or str(uuid.uuid4()),
            sequence_num=msg_count + 1,
            role="user",
            content=text or "",
            metadata=user_msg_metadata if user_msg_metadata else None,
        )

    # Load prior messages for existing conversations (HANA is the source of truth)
    prior_messages = None
    prior_state = None
    if provided_context_id and msg_count > 0:
        stored_msgs = await store.get_messages(context_id)
        prior_messages = stored_msgs[:-1] if stored_msgs else None
        # Try owner access first, then shared access for agent state
        conv = await store.get_conversation(context_id, user.user_id)
        if not conv:
            conv = await store.get_conversation_by_id(context_id)
        if conv and conv.get("agent_state"):
            prior_state = conv["agent_state"]

    # Run the agent
    # For external users, pre-set cse_status to "disabled" so the agent
    # skips CSE but still triggers Perplexity and LinkedIn with company_name.
    if is_external and prior_state is None:
        prior_state = {"cse_status": "disabled"}
    elif is_external and prior_state is not None and not prior_state.get("cse_status"):
        prior_state["cse_status"] = "disabled"

    inflight_entry = _INFLIGHT.get(context_id, {}).get(turn_key)

    async def _canceled_response() -> Dict[str, Any]:
        """Discard this turn's writes and return a terminal 'canceled' task."""
        await _rollback_cancelled_turn(store, context_id, is_continue, msg_count)
        remaining = _public_messages(await store.get_messages(context_id))
        history = [
            _message(
                role=m["role"],
                text=m["content"] or "",
                message_id=m["id"],
                task_id=task_id,
                context_id=context_id,
                metadata=m.get("metadata"),
            )
            for m in remaining
        ]
        return _jsonrpc_result(
            req_id,
            _task(task_id=task_id, context_id=context_id, state="canceled", history=history),
        )

    # Stop may have arrived during file resolution — skip the agent entirely.
    if inflight_entry is not None and inflight_entry.get("cancelled"):
        return await _canceled_response()

    try:
        agent_task = asyncio.ensure_future(_run_agent(
            text or "",
            context_id,
            company_name=company_name,
            prior_messages=prior_messages,
            prior_state=prior_state,
            llm_usage_meta=llm_usage_meta,
            variant=variant,
            file_content_blocks=resolved_content_blocks or None,
        ))
        if inflight_entry is not None:
            inflight_entry["agent_task"] = agent_task
        agent_result = await agent_task
    except asyncio.CancelledError:
        # Distinguish our Stop (inner task cancelled) from an outer request cancel.
        if inflight_entry is not None and inflight_entry.get("cancelled"):
            return await _canceled_response()
        raise
    except Exception as exc:
        return _jsonrpc_error(
            req_id,
            ERROR_INTERNAL,
            "Internal server error",
            {"detail": str(exc)},
        )

    # Stop may have arrived just as the agent finished (or was swallowed by the
    # agent) — discard before we persist anything.
    if inflight_entry is not None and inflight_entry.get("cancelled"):
        return await _canceled_response()

    agent_text = str(agent_result.get("text", "") or "")
    ui_metadata = agent_result.get("ui", {}) if isinstance(agent_result, dict) else {}

    # Build the agent-state snapshot BEFORE persisting the agent message, so it can
    # be embedded in that message's metadata. This dict is exactly the resume input
    # run_agent reads back, so restoring it on a rewind is a behaviorally-identical
    # resume at that turn (question count + sticky reimagine/provocateur fields).
    from ideation_agent import cse_background, linkedin_background, perplexity_background
    agent_state = {
        "clarification_count": ui_metadata.get("clarificationCount", 0),
        "last_turn_kind": ui_metadata.get("kind", ""),
        "refinement_mode": ui_metadata.get("kind") == "final",
        "cse_status": cse_background.status(context_id),
        "perplexity_status": perplexity_background.status(context_id),
        "perplexity_problem_status": perplexity_background.problem_status(context_id),
        "linkedin_status": linkedin_background.status(context_id),
    }
    # Persist provocateur/reimagine state for multi-turn continuity
    prov_state = agent_result.get("provocateur_state")
    if prov_state and isinstance(prov_state, dict):
        agent_state.update(prov_state)

    # Client-facing metadata (UI only) vs. persisted metadata (+ internal snapshot).
    # The snapshot is stripped from all client reads via _public_messages().
    client_metadata = {"ui": ui_metadata} if ui_metadata else None
    persist_metadata = dict(client_metadata) if client_metadata else {}
    persist_metadata["agent_state_snapshot"] = agent_state

    # Persist agent message (carrying the per-turn state snapshot)
    agent_msg_id = str(uuid.uuid4())
    agent_seq = msg_count + 1 if is_continue else msg_count + 2
    await store.append_message(
        conv_id=context_id,
        msg_id=agent_msg_id,
        sequence_num=agent_seq,
        role="agent",
        content=agent_text,
        metadata=persist_metadata,
    )

    # Persist conversation-level snapshot (unchanged resume contract)
    await store.update_conversation(
        context_id,
        message_count=agent_seq,
        agent_state_json=json.dumps(agent_state),
    )

    # Trigger strategic reframe in background when agent produces a final recommendation
    if ui_metadata.get("kind") == "final":
        from ideation_agent import reframe_background
        reframe_payload = {
            "industry": ui_metadata.get("industry", ""),
            "title": ui_metadata.get("title", ""),
            "problem": ui_metadata.get("problem", ""),
            "solution": ui_metadata.get("solution", ""),
            "app_type": ui_metadata.get("app_type", "Agent"),
            "company_context": ui_metadata.get("companyContext", ""),
        }
        await reframe_background.start_reframe(context_id, reframe_payload)

        # Trigger SAP AI catalog matching
        from ideation_agent import catalog_matching_background
        catalog_payload = {
            "industry": ui_metadata.get("industry", ""),
            "title": ui_metadata.get("title", ""),
            "problem": ui_metadata.get("problem", ""),
            "solution": ui_metadata.get("solution", ""),
            "app_type": ui_metadata.get("app_type", "Agent"),
        }
        await catalog_matching_background.start_catalog_matching(context_id, catalog_payload)

    # Generate title for first exchange (started in parallel with agent, awaited now)
    generated_title = None
    if msg_count == 0 and (text or agent_text):
        generated_title = await _generate_title(
            context_id,
            text or "",
            agent_text,
            company_name,
            llm_usage_meta=llm_usage_meta,
        )

    # Build A2A response (load full history from HANA for the response)
    all_messages = _public_messages(await store.get_messages(context_id))
    history = []
    for msg in all_messages:
        history.append(_message(
            role=msg["role"],
            text=msg["content"] or "",
            message_id=msg["id"],
            task_id=task_id,
            context_id=context_id,
            metadata=msg.get("metadata"),
        ))

    agent_msg = _message(
        role="agent",
        text=agent_text,
        message_id=agent_msg_id,
        task_id=task_id,
        context_id=context_id,
        metadata=client_metadata,
    )

    task = _task(
        task_id=task_id,
        context_id=context_id,
        state="completed",
        status_message=agent_msg,
        history=history,
        artifacts=[_artifact(agent_text)],
    )
    if ui_metadata:
        task["metadata"] = {"ui": ui_metadata}
    if generated_title:
        task.setdefault("metadata", {})
        task["metadata"]["generatedTitle"] = generated_title

    metadata = params.get("metadata") if isinstance(params, dict) else None
    include_tool_calls = bool(
        isinstance(metadata, dict) and metadata.get("includeToolCalls")
    )
    if include_tool_calls:
        task.setdefault("metadata", {})
        task["metadata"]["toolCalls"] = agent_result.get("tool_calls", [])

    if os.getenv("A2A_LOG_TOOL_CALLS", "false").strip().lower() in {"1", "true", "yes"}:
        print(f"[ideation-a2a] tool_calls={agent_result.get('tool_calls', [])}")

    _TASKS[task_id] = {"task": task, "context_id": context_id}

    config = params.get("configuration") if isinstance(params, dict) else None
    history_length = config.get("historyLength") if isinstance(config, dict) else None
    task_response = _apply_history_length(
        task,
        history_length if isinstance(history_length, int) else None,
    )

    return _jsonrpc_result(req_id, task_response)


@router.post("/bootstrap")
async def bootstrap(
    user: UserIdentity = Depends(get_current_user),
    event_id: Optional[str] = Body(None, embed=True),
) -> Dict[str, Any]:
    context_id = str(uuid.uuid4())
    ui_metadata = {
        "ui": {
            "kind": "intro",
            "allowFreeform": True,
            "clarificationCount": 0,
        }
    }
    intro_message = _message(
        role="agent",
        text=INTRO_TEXT,
        context_id=context_id,
        metadata=ui_metadata,
    )

    # Create conversation in HANA
    store = await _get_store()
    await store.create_conversation(
        conv_id=context_id,
        user_id=user.user_id,
        title="New conversation",
        event_id=event_id,
    )
    # Persist the intro message
    await store.append_message(
        conv_id=context_id,
        msg_id=intro_message["messageId"],
        sequence_num=1,
        role="agent",
        content=INTRO_TEXT,
        metadata=ui_metadata,
    )
    await store.update_conversation(context_id, message_count=1)

    return {
        "contextId": context_id,
        "message": {
            "role": "agent",
            "text": INTRO_TEXT,
            "metadata": ui_metadata,
        },
    }


# ---------- Conversation CRUD endpoints ----------


@router.get("/conversations")
async def list_conversations(
    user: UserIdentity = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    event_id: Optional[str] = Query(None),
    include_shared: bool = Query(False),
) -> Dict[str, Any]:
    """List the authenticated user's conversations.

    Args:
        event_id: If provided, list conversations for that event.
                  If omitted, list personal (non-event) conversations only.
        include_shared: If true, also include conversations shared with this user.
    """
    store = await _get_store()
    conversations = await store.list_conversations(
        user.user_id, limit=limit, offset=offset, event_id=event_id
    )

    # Attach shared_member_count to owner's conversations that have members
    from ..services.hana_shares import get_share_store
    share_store = await get_share_store()
    shared_counts = await share_store.get_shared_conversation_ids(user.user_id)
    for conv in conversations:
        count = shared_counts.get(conv.get("id", ""), 0)
        if count > 0:
            conv["shared_member_count"] = count

    shared_conversations = []
    if include_shared or event_id:
        shared_rows = await share_store.get_shared_conversations(
            user.user_id, event_id=event_id
        )

        # Batch-resolve owner emails for rows missing owner_name
        missing_owner_uids = [
            row.get("owner_uid", "")
            for row in shared_rows
            if not row.get("owner_name") and row.get("owner_uid")
        ]
        owner_email_map: Dict[str, str] = {}
        if missing_owner_uids:
            from ..services.hana_submissions import get_submission_store
            sub_store = await get_submission_store()
            for uid in set(missing_owner_uids):
                email = await sub_store.get_email_by_uid(uid)
                if email:
                    owner_email_map[uid] = email

        for row in shared_rows:
            owner_name = row.get("owner_name", "")
            if not owner_name:
                owner_uid = row.get("owner_uid", "")
                # Use email as fallback display name
                email = owner_email_map.get(owner_uid, "")
                if email:
                    owner_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
                elif "@" in owner_uid:
                    # CDC users have email as UID
                    owner_name = owner_uid.split("@")[0].replace(".", " ").replace("_", " ").title()

            shared_conversations.append({
                "id": row["id"],
                "title": row.get("title", "Untitled"),
                "status": row.get("status", "active"),
                "message_count": row.get("message_count", 0),
                "created_at": row.get("created_at", ""),
                "updated_at": row.get("updated_at", ""),
                "shared": True,
                "permission": row.get("permission", "read"),
                "owner_uid": row.get("owner_uid", ""),
                "owner_name": owner_name,
            })

    return {"conversations": conversations, "shared_conversations": shared_conversations}


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Get a single conversation's metadata. Supports owner and shared access."""
    store = await _get_store()
    conv = await store.get_conversation(conv_id, user.user_id)
    if conv:
        conv["access_level"] = "owner"
        return JSONResponse(conv)

    # Check shared access
    from ..services.hana_shares import get_share_store
    share_store = await get_share_store()
    permission = await share_store.get_member_permission(conv_id, user.user_id)
    if permission:
        conv = await store.get_conversation_by_id(conv_id)
        if conv:
            conv["access_level"] = permission
            return JSONResponse(conv)

    return JSONResponse({"error": "Conversation not found"}, status_code=404)


@router.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Get all messages for a conversation. Supports owner and shared access."""
    store = await _get_store()
    conv = await store.get_conversation(conv_id, user.user_id)
    access_level = "owner" if conv else None

    if not conv:
        # Check shared access
        from ..services.hana_shares import get_share_store
        share_store = await get_share_store()
        permission = await share_store.get_member_permission(conv_id, user.user_id)
        if permission:
            conv = await store.get_conversation_by_id(conv_id)
            access_level = permission

    if not conv:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)

    messages = _public_messages(await store.get_messages(conv_id))
    response_data = {
        "messages": messages,
        "contextId": conv_id,
        "access_level": access_level,
    }
    # Include event_id so the frontend can auto-activate event mode
    event_id = conv.get("event_id") if conv else None
    if event_id:
        response_data["event_id"] = event_id
    return JSONResponse(response_data)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Delete a conversation and all its messages. Owner only. Cascades to shares."""
    store = await _get_store()
    deleted = await store.delete_conversation(conv_id, user.user_id)
    if not deleted:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)

    # Cascade: remove all shares and members for this conversation
    try:
        from ..services.hana_shares import get_share_store
        share_store = await get_share_store()
        await share_store.delete_shares_for_conversation(conv_id)
    except Exception as exc:
        logger.warning("Failed to cascade-delete shares for %s: %s", conv_id, exc)

    # Cascade: remove leaderboard submissions for this conversation
    try:
        from ..services.hana_submissions import get_submission_store
        sub_store = await get_submission_store()
        await sub_store.delete_by_context(conv_id)
    except Exception as exc:
        logger.warning("Failed to cascade-delete submissions for %s: %s", conv_id, exc)

    return JSONResponse({"deleted": True})


@router.patch("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Rename a conversation."""
    store = await _get_store()
    conv = await store.get_conversation(conv_id, user.user_id)
    if not conv:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    body = await request.json()
    title = (body.get("title") or "").strip()
    if not title:
        return JSONResponse({"error": "Title is required"}, status_code=400)
    await store.update_conversation(conv_id, title=title)
    return JSONResponse({"ok": True})


@router.post("/conversations/{conv_id}/rewind")
async def rewind_conversation_endpoint(
    conv_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Truncate a conversation to before a sequence number and restore agent state.

    Used by Edit-last-message: the client rewinds to the edited user message's
    sequence, then re-sends the edited text through the normal /a2a path (reusing
    all file/provocateur/title/research logic). Write access required.
    """
    if not await _has_write_access(conv_id, user):
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    seq = body.get("sequenceNum")
    message_id = body.get("messageId")
    if isinstance(message_id, str) and message_id:
        # Resolve the message's server sequence (robust — the client tracks messageId,
        # not the fragile local-array position).
        store = await _get_store()
        messages = await store.get_messages(conv_id)
        target = next((m for m in messages if m.get("id") == message_id), None)
        if target is None:
            return JSONResponse({"error": "messageId not found"}, status_code=404)
        seq = int(target["sequence_num"])
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        return JSONResponse(
            {"error": "sequenceNum or messageId is required"}, status_code=400
        )
    result = await _rewind_conversation(conv_id, seq)
    result["contextId"] = conv_id
    return JSONResponse(result)


@router.post("/conversations/{conv_id}/cancel")
async def cancel_turn_endpoint(
    conv_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> JSONResponse:
    """Stop an in-flight turn (the Stop button), or discard it if it just finished.

    If the turn is still running: flag it + cancel the agent task and return
    immediately (does NOT take the per-context lock, so it never blocks behind the
    running turn). If the turn already completed in the race window: rewind it away
    by its messageId (timing-gap fallback). Write access required.
    """
    if not await _has_write_access(conv_id, user):
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    message_id = str(body.get("messageId") or "").strip()
    if not message_id:
        return JSONResponse({"error": "messageId is required"}, status_code=400)

    entry = _INFLIGHT.get(conv_id, {}).get(message_id)
    if entry is not None:
        entry["cancelled"] = True
        agent_task = entry.get("agent_task")
        if agent_task is not None and not agent_task.done():
            agent_task.cancel()
        return JSONResponse({"cancelled": True, "mode": "in_flight"})

    # Timing-gap fallback: the turn already committed — rewind it away by messageId.
    store = await _get_store()
    messages = await store.get_messages(conv_id)
    target = next(
        (m for m in messages if m.get("id") == message_id and m.get("role") == "user"),
        None,
    )
    if target is None:
        return JSONResponse({"cancelled": False, "reason": "not_found"})
    await _rewind_conversation(conv_id, int(target["sequence_num"]))
    return JSONResponse({"cancelled": True, "mode": "rewound"})


# ---------- Existing endpoints ----------


def _handle_tasks_get(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    task_id = params.get("id") if isinstance(params, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "Invalid method parameters")
    if task_id not in _TASKS:
        return _jsonrpc_error(req_id, ERROR_TASK_NOT_FOUND, "Task not found")

    task = _TASKS[task_id]["task"]
    history_length = params.get("historyLength") if isinstance(params, dict) else None
    task_response = _apply_history_length(
        task,
        history_length if isinstance(history_length, int) else None,
    )
    return _jsonrpc_result(req_id, task_response)


def _handle_tasks_cancel(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    task_id = params.get("id") if isinstance(params, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "Invalid method parameters")
    if task_id not in _TASKS:
        return _jsonrpc_error(req_id, ERROR_TASK_NOT_FOUND, "Task not found")

    task = _TASKS[task_id]["task"]
    state = task.get("status", {}).get("state")
    if state in TERMINAL_STATES:
        return _jsonrpc_error(
            req_id, ERROR_TASK_NOT_CANCELABLE, "Task cannot be canceled"
        )

    task["status"] = {"state": "canceled", "timestamp": _now_iso()}
    return _jsonrpc_result(req_id, task)


@router.post("/a2a")
async def a2a_endpoint(
    request: Request, user: UserIdentity = Depends(get_current_user)
) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            _jsonrpc_error(None, ERROR_PARSE, "Invalid JSON payload"),
            status_code=400,
        )

    validation_error = _validate_jsonrpc_request(payload)
    if validation_error:
        return JSONResponse(validation_error, status_code=400)

    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return JSONResponse(
            _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "Invalid method parameters"),
            status_code=200,
        )

    if method == "message/send":
        response = await _handle_message_send(req_id, params, user, request)
    elif method == "tasks/get":
        response = _handle_tasks_get(req_id, params)
    elif method == "tasks/cancel":
        response = _handle_tasks_cancel(req_id, params)
    elif isinstance(method, str) and method.startswith("tasks/pushNotificationConfig/"):
        response = _jsonrpc_error(
            req_id,
            ERROR_PUSH_NOT_SUPPORTED,
            "Push Notification is not supported",
        )
    else:
        response = _jsonrpc_error(req_id, ERROR_METHOD_NOT_FOUND, "Method not found")

    return JSONResponse(response, status_code=200)


@router.get("/enrichment/{context_id}")
async def get_enrichment(context_id: str, user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Poll CSE + Perplexity + LinkedIn enrichment status for the research drawer UI."""
    from ideation_agent import cse_background, linkedin_background, perplexity_background
    from ideation_agent.tools.company_search import cse_enabled

    # CSE is SAP-internal only — hide from external users and when not configured
    is_internal = user.auth_provider != "cdc" and cse_enabled()

    # Load from HANA if not in memory (e.g. after server restart)
    if is_internal and cse_background.status(context_id) == "idle":
        await cse_background.load_from_hana(context_id)
    if perplexity_background.status(context_id) == "idle":
        await perplexity_background.load_from_hana(context_id)
    if linkedin_background.status(context_id) == "idle":
        await linkedin_background.load_from_hana(context_id)

    cse_status = cse_background.status(context_id) if is_internal else "disabled"
    cse_result = (
        cse_background.get_result(context_id)
        if is_internal and cse_status in ("complete", "failed")
        else None
    )

    px_status = perplexity_background.status(context_id)
    px_result = (
        perplexity_background.get_result(context_id)
        if px_status in ("complete", "failed")
        else None
    )

    px_problem_status = perplexity_background.problem_status(context_id)
    px_problem_result = (
        perplexity_background.get_problem_result(context_id)
        if px_problem_status in ("complete", "failed")
        else None
    )

    li_status = linkedin_background.status(context_id)
    li_result = (
        linkedin_background.get_result(context_id)
        if li_status in ("complete", "failed")
        else None
    )

    return {
        "status": cse_status,
        "result": cse_result,
        "perplexity": {
            "status": px_status,
            "result": px_result,
            "problem_status": px_problem_status,
            "problem_result": px_problem_result,
        },
        "linkedin": {
            "status": li_status,
            "result": li_result,
        },
    }


@router.get("/reframe/{context_id}")
async def get_reframe(context_id: str, user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Poll strategic reframe status for a conversation."""
    from ideation_agent import reframe_background

    # Load from HANA if not in memory (e.g. after server restart)
    if reframe_background.status(context_id) == "idle":
        await reframe_background.load_from_hana(context_id)

    reframe_status = reframe_background.status(context_id)
    reframe_result = (
        reframe_background.get_result(context_id)
        if reframe_status in ("complete", "failed")
        else None
    )

    # Auto-trigger (Idea 6): if conversation has a final but no reframe yet, generate one
    if reframe_status == "idle":
        store = await _get_store()
        messages = await store.get_messages(context_id)
        # Find the latest final message's metadata
        final_payload = None
        for msg in reversed(messages or []):
            meta = msg.get("metadata") or {}
            ui = meta.get("ui") or {}
            if ui.get("kind") == "final":
                final_payload = {
                    "industry": ui.get("industry", ""),
                    "title": ui.get("title", ""),
                    "problem": ui.get("problem", ""),
                    "solution": ui.get("solution", ""),
                    "app_type": ui.get("app_type", "Agent"),
                    "company_context": ui.get("companyContext", ""),
                }
                break
        if final_payload:
            await reframe_background.start_reframe(context_id, final_payload)
            reframe_status = "pending"

    return {
        "status": reframe_status,
        "result": reframe_result,
    }


@router.post("/reframe/{context_id}/regenerate")
async def regenerate_reframe(context_id: str, user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Force re-generate the strategic reframe for a conversation."""
    from ideation_agent import reframe_background

    # Clear existing result so a fresh one is generated
    reframe_background.cleanup(context_id)

    # Find the latest final message to rebuild the payload
    store = await _get_store()
    messages = await store.get_messages(context_id)
    final_payload = None
    for msg in reversed(messages or []):
        meta = msg.get("metadata") or {}
        ui = meta.get("ui") or {}
        if ui.get("kind") == "final":
            final_payload = {
                "industry": ui.get("industry", ""),
                "title": ui.get("title", ""),
                "problem": ui.get("problem", ""),
                "solution": ui.get("solution", ""),
                "app_type": ui.get("app_type", "Agent"),
                "company_context": ui.get("companyContext", ""),
            }
            break

    if not final_payload:
        return {"status": "idle", "result": None}

    await reframe_background.start_reframe(context_id, final_payload)
    return {"status": "pending", "result": None}


@router.get("/catalog-matches/{context_id}")
async def get_catalog_matches(context_id: str, user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Poll SAP AI catalog matching status for a conversation."""
    from ideation_agent import catalog_matching_background

    # Load from HANA if not in memory (e.g. after server restart)
    if catalog_matching_background.status(context_id) == "idle":
        await catalog_matching_background.load_from_hana(context_id)

    match_status = catalog_matching_background.status(context_id)
    match_result = (
        catalog_matching_background.get_result(context_id)
        if match_status in ("complete", "failed")
        else None
    )

    # Auto-trigger on "idle" or retry once on "failed" (transient errors)
    if match_status in ("idle", "failed"):
        if match_status == "failed":
            # Only retry once — prevents infinite loops on permanent failures
            fail_result = catalog_matching_background.get_result(context_id)
            if fail_result and fail_result.get("_retried"):
                return {"status": match_status, "result": match_result}
            catalog_matching_background.cleanup(context_id)
        store = await _get_store()
        messages = await store.get_messages(context_id)
        final_payload = None
        for msg in reversed(messages or []):
            meta = msg.get("metadata") or {}
            ui = meta.get("ui") or {}
            if ui.get("kind") == "final":
                final_payload = {
                    "industry": ui.get("industry", ""),
                    "title": ui.get("title", ""),
                    "problem": ui.get("problem", ""),
                    "solution": ui.get("solution", ""),
                    "app_type": ui.get("app_type", "Agent"),
                }
                break
        if final_payload:
            await catalog_matching_background.start_catalog_matching(
                context_id, final_payload, retried=(match_status == "failed")
            )
            match_status = "pending"
            match_result = None

    return {
        "status": match_status,
        "result": match_result,
    }


@router.post("/catalog-matches/{context_id}/regenerate")
async def regenerate_catalog_matches(context_id: str, user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Force re-generate catalog matches for a conversation."""
    from ideation_agent import catalog_matching_background

    # Clear existing result so a fresh one is generated
    catalog_matching_background.cleanup(context_id)

    # Find the latest final message to rebuild the payload
    store = await _get_store()
    messages = await store.get_messages(context_id)
    final_payload = None
    for msg in reversed(messages or []):
        meta = msg.get("metadata") or {}
        ui = meta.get("ui") or {}
        if ui.get("kind") == "final":
            final_payload = {
                "industry": ui.get("industry", ""),
                "title": ui.get("title", ""),
                "problem": ui.get("problem", ""),
                "solution": ui.get("solution", ""),
                "app_type": ui.get("app_type", "Agent"),
            }
            break

    if not final_payload:
        return {"status": "idle", "result": None}

    await catalog_matching_background.start_catalog_matching(context_id, final_payload)
    return {"status": "pending", "result": None}


@router.get("/company-suggestions")
async def company_suggestions(q: str = "", user: UserIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    """Autocomplete suggestions for company name input (SAP-internal only)."""
    if user.auth_provider == "cdc":
        return {"suggestions": []}

    from ideation_agent.tools.company_search import cse_enabled, cse_suggest_companies

    if not cse_enabled():
        return {"suggestions": []}

    candidates = await cse_suggest_companies(q)
    return {"suggestions": candidates}


@router.get("/.well-known/agent-card.json")
async def agent_card(request: Request) -> Dict[str, Any]:
    base_url = f"{str(request.base_url).rstrip('/')}/api/ideation-agent"
    return {
        "protocolVersion": "0.3.0",
        "name": os.getenv("A2A_V2_AGENT_NAME", "Ideation Agent"),
        "description": os.getenv(
            "A2A_V2_AGENT_DESCRIPTION",
            "Conversational SAP BTP AI ideation agent over A2A JSON-RPC.",
        ),
        "url": f"{base_url}/a2a",
        "preferredTransport": "JSONRPC",
        "additionalInterfaces": [
            {"url": f"{base_url}/a2a", "transport": "JSONRPC"},
        ],
        "version": os.getenv("A2A_AGENT_VERSION", "0.1.0"),
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "ideation-interview",
                "name": "Ideation Interview",
                "description": "Designs SAP BTP AI scenarios through a guided interview.",
                "tags": ["ideation", "sap-btp", "ai"],
                "examples": [
                    "Help me design an AI use case for procurement adoption."
                ],
            }
        ],
    }
