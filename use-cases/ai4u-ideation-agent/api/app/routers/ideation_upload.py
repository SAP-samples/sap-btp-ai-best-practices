"""Upload endpoint for multimodal file processing.

Accepts image uploads, validates and processes them via file_registry,
stores original + processed outputs in Object Store (with meta.json),
and returns file_id for the client to include in A2A messages.

Available to all authenticated users.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from fastapi.responses import Response

from ..auth import UserIdentity, get_current_user
from ..services.file_cache import cache_file_outputs
from ..services.object_store import get_object_store
from ideation_agent.file_processor import MAX_FILE_SIZE_BYTES, MAX_FILES_PER_MESSAGE
from ideation_agent.file_registry import (
    ProcessedFile,
    ProcessingError,
    process_file,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    context_id: str = Query(..., description="Conversation context ID for path scoping"),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Upload and process a file for multimodal AI analysis.

    Stores original + processed outputs in Object Store with meta.json.
    Returns file_id for the client to reference in A2A messages.

    Responses:
        200: Successful upload with file_id
        400: Invalid file type or corrupt image
        413: File too large
    """
    # Read file data with size limit enforcement
    file_data = await file.read()

    if len(file_data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(file_data) / 1024 / 1024:.1f} MB "
                   f"(max {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB)",
        )

    if len(file_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    # Generate file ID
    file_id = str(uuid.uuid4())
    filename = file.filename or "unknown"

    # Process the file through the registry
    result = process_file(
        file_data=file_data,
        filename=filename,
        content_type=file.content_type or "",
        file_id=file_id,
    )

    if isinstance(result, ProcessingError):
        if result.code == "too_large":
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=result.error)

    # Cache processed outputs for immediate consumption by A2A handler
    # (eliminates S3 round-trip on first message)
    cache_file_outputs(context_id, file_id, result.outputs)

    # Store in Object Store (persistence for conversation reload/history)
    store = await get_object_store()

    # Determine file extension from original type
    ext_map = {
        # Images
        "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
        "image/webp": "webp",
        # PDF
        "application/pdf": "pdf",
        # Office XML
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        # Legacy Office
        "application/msword": "doc",
        "application/vnd.ms-powerpoint": "ppt",
        "application/vnd.ms-excel": "xls",
        # OpenDocument
        "application/vnd.oasis.opendocument.text": "odt",
        "application/vnd.oasis.opendocument.presentation": "odp",
        "application/vnd.oasis.opendocument.spreadsheet": "ods",
        # RTF
        "application/rtf": "rtf",
        # Data
        "text/csv": "csv",
        "text/tab-separated-values": "tsv",
        # Email / Web Archive
        "message/rfc822": "eml",
        # Text / Code
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
    ext = ext_map.get(result.original_type, "bin")

    # Store original
    original_key = f"{context_id}/{file_id}/original.{ext}"
    await store.upload(original_key, file_data, result.original_type)

    # Store processed outputs
    is_image = result.original_type.startswith("image/")
    for i, output_bytes in enumerate(result.outputs):
        if is_image:
            output_key = f"{context_id}/{file_id}/{i}.png"
            await store.upload(output_key, output_bytes, "image/png")
        else:
            # Documents (PDF, Office, text, etc.) — store with original extension
            output_key = f"{context_id}/{file_id}/{i}.{ext}"
            await store.upload(output_key, output_bytes, result.original_type)

    # Store meta.json (self-describing file metadata)
    meta = {
        "filename": filename,
        "type": result.original_type,
        "num_outputs": result.num_outputs,
        "dimensions": result.dimensions,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    if result.metadata:
        meta["file_metadata"] = result.metadata
    meta_key = f"{context_id}/{file_id}/meta.json"
    await store.upload(meta_key, json.dumps(meta).encode(), "application/json")

    logger.info(
        "File uploaded: file=%s id=%s context=%s original=%d bytes "
        "outputs=%d dimensions=%s user=%s",
        filename,
        file_id,
        context_id,
        len(file_data),
        result.num_outputs,
        result.dimensions,
        user.user_id,
    )

    return {
        "file_id": file_id,
        "filename": filename,
        "original_type": result.original_type,
        "dimensions": result.dimensions,
        "num_outputs": result.num_outputs,
    }


# --- File retrieval endpoints -------------------------------------------------


async def _verify_file_access(context_id: str, user: UserIdentity) -> None:
    """Verify user has access to the conversation (owner or shared member).

    Raises HTTPException 404 if access denied.
    """
    from ideation_agent.hana_store import get_store
    store = await get_store()

    # Check owner access
    conv = await store.get_conversation(context_id, user.user_id)
    if conv:
        return

    # Check shared access
    from ..services.hana_shares import get_share_store
    share_store = await get_share_store()
    permission = await share_store.get_member_permission(context_id, user.user_id)
    if permission:
        return

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/files/{file_id}/full")
async def get_full_file(
    file_id: str,
    context_id: str = Query(...),
    user: UserIdentity = Depends(get_current_user),
) -> Response:
    """Serve full-resolution file (output 0) for lightbox display.

    Supports images (PNG), PDFs, and other document types.
    Tries meta.json first to determine type, then falls back to common extensions.
    Security: verifies user has access to the conversation.
    """
    await _verify_file_access(context_id, user)

    store = await get_object_store()

    # Try to read meta.json to determine the file type
    meta_key = f"{context_id}/{file_id}/meta.json"
    meta_data = await store.download(meta_key)
    if meta_data is not None:
        try:
            meta = json.loads(meta_data)
            file_type = meta.get("type", "")
            # Determine extension from the stored type
            from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
            if file_type.startswith("image/"):
                # Images are always stored as PNG
                key = f"{context_id}/{file_id}/0.png"
                data = await store.download(key)
                if data is not None:
                    return Response(
                        content=data,
                        media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"},
                    )
            elif file_type in DOCUMENT_MIME_TYPES:
                # Determine extension from meta
                ext_map_local = {
                    "application/pdf": "pdf",
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
                doc_ext = ext_map_local.get(file_type, "bin")
                key = f"{context_id}/{file_id}/0.{doc_ext}"
                data = await store.download(key)
                if data is not None:
                    headers = {"Cache-Control": "private, max-age=3600"}
                    # Add Content-Disposition for non-PDF documents (download trigger)
                    # PDFs are served inline for lightbox iframe rendering
                    if file_type != "application/pdf":
                        filename = meta.get("filename", f"file.{doc_ext}")
                        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
                    return Response(
                        content=data,
                        media_type=file_type,
                        headers=headers,
                    )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: try common extensions (backward compatibility)
    pdf_key = f"{context_id}/{file_id}/0.pdf"
    data = await store.download(pdf_key)
    if data is not None:
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    png_key = f"{context_id}/{file_id}/0.png"
    data = await store.download(png_key)
    if data is not None:
        return Response(
            content=data,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/files/{file_id}/output/{index}")
async def get_output(
    file_id: str,
    index: int,
    context_id: str = Query(...),
    user: UserIdentity = Depends(get_current_user),
) -> Response:
    """Serve a specific processed output (for multi-page documents).

    Security: verifies user has access to the conversation.
    """
    await _verify_file_access(context_id, user)

    store = await get_object_store()
    key = f"{context_id}/{file_id}/{index}.png"
    data = await store.download(key)

    if data is None:
        raise HTTPException(status_code=404, detail="Output not found")

    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
