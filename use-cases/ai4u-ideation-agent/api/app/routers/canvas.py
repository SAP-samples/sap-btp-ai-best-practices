"""Canvas router — persist tile positions per user per workspace.

Supports shared canvases (company/event) via resolve_canvas_with_access_check.
All canvases use the same endpoints: GET (load), PUT (full-board save), DELETE (single tile).
Shared canvases use SSE (GET /stream) for near-instant push sync.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth import UserIdentity, get_current_user
from ..services.hana_canvas import (
    get_canvas_store,
    resolve_canvas_with_access_check,
    wait_for_canvas_change,
    join_canvas,
    leave_canvas,
    get_canvas_viewers,
    MAX_TILES_PER_CANVAS,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Models ────────────────────────────────────────────────────────

class TileData(BaseModel):
    tile_id: str = Field(..., max_length=64)
    tile_type: str = Field(..., max_length=20)  # 'entry' | 'card' | 'note'
    context_id: Optional[str] = Field(None, max_length=64)
    title: str = Field("", max_length=512)
    body: str = ""
    pos_x: int
    pos_y: int
    color: str = Field("", max_length=20)
    z_index: int = 0


class TileResponse(TileData):
    still_valid: bool = True


class TilesResponse(BaseModel):
    tiles: List[TileResponse]
    server_time: str


class BulkSaveRequest(BaseModel):
    tiles: List[TileData] = Field(..., max_length=MAX_TILES_PER_CANVAS)


class BulkSaveResponse(BaseModel):
    saved: int


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/{workspace_id}/tiles", response_model=TilesResponse)
async def get_tiles(
    workspace_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> TilesResponse:
    """Load tiles for a workspace (personal or shared)."""
    try:
        canvas, is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s user=%s: %s", workspace_id, user.user_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    store = await get_canvas_store()

    try:
        if is_shared:
            rows = await store.get_shared_tiles(canvas["canvas_id"])
        else:
            rows = await store.get_workspace_tiles(user.user_id, workspace_id)
    except Exception as exc:
        logger.error("Failed to load canvas tiles ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to load canvas tiles") from exc

    tiles = [
        TileResponse(
            tile_id=row["tile_id"],
            tile_type=row["tile_type"],
            context_id=row.get("context_id"),
            title=row.get("title", ""),
            body=row.get("body", ""),
            pos_x=int(row.get("pos_x", 0)),
            pos_y=int(row.get("pos_y", 0)),
            color=row.get("color", ""),
            z_index=int(row.get("z_index", 0)),
            still_valid=bool(row.get("still_valid", 1)),
        )
        for row in rows
    ]

    return TilesResponse(
        tiles=tiles,
        server_time=datetime.now(timezone.utc).isoformat(),
    )


@router.put("/{workspace_id}/tiles", response_model=BulkSaveResponse)
async def save_tiles(
    workspace_id: str,
    body: BulkSaveRequest,
    user: UserIdentity = Depends(get_current_user),
) -> BulkSaveResponse:
    """Bulk-save the full board state for a workspace.

    Personal canvases: upsert + prune (full-replace semantics).
    Shared canvases: upsert-only (removals use DELETE /tiles/{tile_id}).
    """
    try:
        canvas, is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    store = await get_canvas_store()
    tiles_dicts = [t.model_dump() for t in body.tiles]

    try:
        await store.bulk_save(canvas["canvas_id"], tiles_dicts, skip_prune=is_shared)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to save canvas tiles ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to save canvas tiles") from exc

    return BulkSaveResponse(saved=len(tiles_dicts))


class TilePositionUpdate(BaseModel):
    tile_id: str = Field(..., max_length=64)
    pos_x: int
    pos_y: int
    z_index: int = 0


class BatchPositionRequest(BaseModel):
    moves: List[TilePositionUpdate] = Field(..., max_length=MAX_TILES_PER_CANVAS)


@router.patch("/{workspace_id}/tiles")
async def patch_tile_positions(
    workspace_id: str,
    body: BatchPositionRequest,
    user: UserIdentity = Depends(get_current_user),
):
    """Lightweight position-only update for one or more tiles.

    Much faster than full-board PUT — only updates pos_x, pos_y, z_index.
    Used for drag-and-drop moves on shared canvases.
    """
    try:
        canvas, _is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    from ..services.hana_canvas import notify_canvas_changed
    from ideation_agent.hana_store import _execute

    canvas_id = canvas["canvas_id"]
    now = datetime.now(timezone.utc)

    for move in body.moves:
        await _execute(
            "UPDATE CANVAS_TILES SET POS_X = ?, POS_Y = ?, Z_INDEX = ?, UPDATED_AT = ? WHERE TILE_ID = ? AND CANVAS_ID = ?",
            (move.pos_x, move.pos_y, move.z_index, now, move.tile_id, canvas_id),
        )

    await _execute(
        "UPDATE CANVASES SET UPDATED_AT = ? WHERE CANVAS_ID = ?",
        (now, canvas_id),
    )
    notify_canvas_changed(canvas_id)

    return {"status": "ok", "updated": len(body.moves)}


@router.delete("/{workspace_id}/tiles/{tile_id}")
async def delete_tile(
    workspace_id: str,
    tile_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Remove a single tile from a canvas."""
    try:
        canvas, _is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    store = await get_canvas_store()

    try:
        deleted = await store.delete_tile(canvas["canvas_id"], tile_id)
    except Exception as exc:
        logger.error("Failed to delete tile %s ws=%s: %s", tile_id, workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to delete tile") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Tile not found")

    return {"status": "ok"}


# ── Tile locking ─────────────────────────────────────────────────────

@router.post("/{workspace_id}/tiles/{tile_id}/lock")
async def lock_tile(
    workspace_id: str,
    tile_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Acquire an edit lock on a tile. Returns 409 if already locked."""
    try:
        canvas, _is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    store = await get_canvas_store()
    acquired = await store.acquire_lock(canvas["canvas_id"], tile_id, user.user_id)
    if not acquired:
        raise HTTPException(status_code=409, detail="Tile is locked by another user")
    return {"status": "locked"}


@router.post("/{workspace_id}/tiles/{tile_id}/unlock")
async def unlock_tile(
    workspace_id: str,
    tile_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Release an edit lock on a tile."""
    try:
        canvas, _is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    store = await get_canvas_store()
    await store.release_lock(canvas["canvas_id"], tile_id, user.user_id)
    return {"status": "unlocked"}


# ── SSE stream ──────────────────────────────────────────────────────

def _row_to_tile_dict(row: dict) -> dict:
    """Convert a DB row to a JSON-serializable tile dict (for SSE payloads)."""
    return {
        "tile_id": row["tile_id"],
        "tile_type": row["tile_type"],
        "context_id": row.get("context_id"),
        "title": row.get("title", ""),
        "body": row.get("body", ""),
        "pos_x": int(row.get("pos_x", 0)),
        "pos_y": int(row.get("pos_y", 0)),
        "color": row.get("color", ""),
        "z_index": int(row.get("z_index", 0)),
        "still_valid": bool(row.get("still_valid", 1)),
        "locked_by": row.get("locked_by") or None,
    }


@router.get("/{workspace_id}/stream")
async def stream_tiles(
    workspace_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """SSE: push full tile snapshot whenever the canvas changes.

    The client connects via fetch() with Authorization header (not native
    EventSource which can't set headers). Pushes immediately on connect,
    then again whenever CANVASES.UPDATED_AT changes.
    """
    try:
        canvas, is_shared = await resolve_canvas_with_access_check(
            workspace_id, user.user_id, user.email
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("SSE: failed to resolve canvas ws=%s: %s", workspace_id, exc)
        raise HTTPException(status_code=502, detail="Failed to resolve canvas") from exc

    canvas_id = canvas["canvas_id"]
    display_name = f"{user.given_name} {user.family_name}".strip() or user.email or user.user_id

    async def event_generator():
        store = await get_canvas_store()
        last_updated_at = None

        join_canvas(canvas_id, user.user_id, display_name)
        try:
            while True:
                try:
                    current = await store.get_canvas_updated_at(canvas_id)
                    if current != last_updated_at:
                        last_updated_at = current
                        if is_shared:
                            rows = await store.get_shared_tiles(canvas_id)
                        else:
                            rows = await store.get_workspace_tiles(user.user_id, workspace_id)
                        tiles = [_row_to_tile_dict(r) for r in rows]
                        payload = json.dumps({
                            "tiles": tiles,
                            "viewers": get_canvas_viewers(canvas_id),
                            "server_time": current.isoformat() if current else "",
                        })
                        yield f"data: {payload}\n\n"
                except Exception as exc:
                    logger.warning("SSE loop error ws=%s: %s", workspace_id, exc)
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                    break

                # Wait for a write signal or timeout (2s fallback ensures liveness)
                await wait_for_canvas_change(canvas_id, timeout=2.0)
        finally:
            leave_canvas(canvas_id, user.user_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
