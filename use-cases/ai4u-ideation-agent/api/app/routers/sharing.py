"""Sharing router — conversation sharing via link/code.

Allows conversation owners to create share links (read-only or write access)
and recipients to join shared conversations.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import UserIdentity, get_current_user
from ..services.hana_shares import get_share_store

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_current_user)])


# ── Request/Response models ─────────────────────────────────────────────────


class CreateShareRequest(BaseModel):
    permission: str = "read"  # 'read' | 'write'


class ShareInfo(BaseModel):
    share_id: str
    share_code: str
    permission: str
    created_at: str


class ShareMemberInfo(BaseModel):
    uid: str
    email: str
    display_name: str
    permission: str
    joined_at: str


class ShareDetailsResponse(BaseModel):
    shares: List[ShareInfo]
    members: List[ShareMemberInfo]


class SharePreview(BaseModel):
    """Limited info for the join page (before the user joins)."""
    conversation_id: str
    title: str
    permission: str
    owner_name: str
    share_code: str


class JoinResponse(BaseModel):
    conversation_id: str
    permission: str
    title: str


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _verify_conversation_owner(conv_id: str, user_id: str) -> Dict[str, Any]:
    """Verify the user owns the conversation. Raises 404 or 403 if not."""
    from ideation_agent.hana_store import get_store

    store = await get_store()
    conv = await store.get_conversation(conv_id, user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/conversations/{conv_id}/share", response_model=ShareInfo)
async def create_share(
    conv_id: str,
    request: CreateShareRequest,
    user: UserIdentity = Depends(get_current_user),
) -> ShareInfo:
    """Create a new share link for a conversation. Owner only."""
    if request.permission not in ("read", "write"):
        raise HTTPException(status_code=400, detail="Permission must be 'read' or 'write'")

    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()
    # Build creator display name from user identity
    creator_name = ""
    if user.given_name:
        creator_name = user.given_name
        if user.family_name:
            creator_name += f" {user.family_name}"
    try:
        result = await share_store.create_share(
            conversation_id=conv_id,
            uid=user.user_id,
            permission=request.permission,
            creator_name=creator_name,
        )
    except Exception as exc:
        logger.error("Failed to create share: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create share link") from exc

    return ShareInfo(
        share_id=result["share_id"],
        share_code=result["share_code"],
        permission=result["permission"],
        created_at=result["created_at"],
    )


@router.get("/conversations/{conv_id}/share", response_model=ShareDetailsResponse)
async def get_shares(
    conv_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> ShareDetailsResponse:
    """List all active shares and members for a conversation. Owner only."""
    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()
    shares = await share_store.get_shares_for_conversation(conv_id)
    members = await share_store.get_members(conv_id)

    return ShareDetailsResponse(
        shares=[
            ShareInfo(
                share_id=s["share_id"],
                share_code=s["share_code"],
                permission=s["permission"],
                created_at=s["created_at"],
            )
            for s in shares
        ],
        members=[
            ShareMemberInfo(
                uid=m["uid"],
                email=m.get("email", ""),
                display_name=m.get("display_name", ""),
                permission=m["permission"],
                joined_at=m.get("joined_at", ""),
            )
            for m in members
        ],
    )


class UpdatePermissionRequest(BaseModel):
    permission: str


@router.put("/conversations/{conv_id}/share/permission")
async def update_share_permission(
    conv_id: str,
    request: UpdatePermissionRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update the default permission for the share link (new joiners only). Owner only."""
    if request.permission not in ("read", "write"):
        raise HTTPException(status_code=400, detail="Permission must be 'read' or 'write'")

    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()

    # Update only the share link default — existing members keep their individual permissions
    await share_store.update_share_permission(conv_id, request.permission)

    return {"status": "updated", "permission": request.permission}


@router.put("/conversations/{conv_id}/share/members/{target_uid}/permission")
async def update_member_permission(
    conv_id: str,
    target_uid: str,
    request: UpdatePermissionRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update a single member's permission. Owner only."""
    if request.permission not in ("read", "write"):
        raise HTTPException(status_code=400, detail="Permission must be 'read' or 'write'")

    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()
    updated = await share_store.update_member_permission(conv_id, target_uid, request.permission)
    if not updated:
        raise HTTPException(status_code=404, detail="Member not found")

    return {"status": "updated", "uid": target_uid, "permission": request.permission}


@router.delete("/conversations/{conv_id}/share/{share_id}")
async def revoke_share(
    conv_id: str,
    share_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Revoke a share link. Owner only."""
    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()
    success = await share_store.revoke_share(share_id, user.user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Share link not found")

    return {"status": "revoked", "message": "Share link has been revoked"}


@router.delete("/conversations/{conv_id}/share/members/{target_uid}")
async def remove_share_member(
    conv_id: str,
    target_uid: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Remove a member from a shared conversation. Owner only."""
    await _verify_conversation_owner(conv_id, user.user_id)

    share_store = await get_share_store()
    removed = await share_store.remove_member(conv_id, target_uid)
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")

    return {"status": "removed", "message": "Member has been removed"}


@router.get("/shared/preview/{code}", response_model=SharePreview)
async def get_share_preview(
    code: str,
    user: UserIdentity = Depends(get_current_user),
) -> SharePreview:
    """Preview a shared conversation before joining (via share code)."""
    share_store = await get_share_store()

    share = await share_store.get_share_by_code(code.upper())
    if not share:
        raise HTTPException(status_code=404, detail="Share link not found or has been revoked")

    # Fetch conversation info (without ownership check — we verified via share code)
    from ideation_agent.hana_store import get_store
    store = await get_store()
    conv = await store.get_conversation_by_id(share["conversation_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation no longer exists")

    # Get owner display name
    owner_name = conv.get("user_id", "Unknown")
    # Try to get a better display name from the conversation's company_name or user_id
    # For now, use a short form of user_id
    owner_id = conv.get("user_id", "")
    if "@" in owner_id:
        owner_name = owner_id.split("@")[0].replace(".", " ").replace("_", " ").title()

    return SharePreview(
        conversation_id=share["conversation_id"],
        title=conv.get("title", "Untitled conversation"),
        permission=share["permission"],
        owner_name=owner_name,
        share_code=share["share_code"],
    )


@router.post("/shared/join/{code}", response_model=JoinResponse)
async def join_shared_conversation(
    code: str,
    user: UserIdentity = Depends(get_current_user),
) -> JoinResponse:
    """Join a shared conversation via share code."""
    share_store = await get_share_store()

    share = await share_store.get_share_by_code(code.upper())
    if not share:
        raise HTTPException(status_code=404, detail="Share link not found or has been revoked")

    conversation_id = share["conversation_id"]

    # Don't let owner join their own conversation
    from ideation_agent.hana_store import get_store
    store = await get_store()
    conv = await store.get_conversation_by_id(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation no longer exists")

    if conv.get("user_id") == user.user_id:
        return JoinResponse(
            conversation_id=conversation_id,
            permission="owner",
            title=conv.get("title", "Untitled conversation"),
        )

    # Build display name
    display_name = ""
    if user.given_name:
        display_name = user.given_name
        if user.family_name:
            display_name += f" {user.family_name}"
    if not display_name and user.email:
        display_name = user.email.split("@")[0].replace(".", " ").replace("_", " ").title()

    await share_store.join_share(
        conversation_id=conversation_id,
        uid=user.user_id,
        email=user.email,
        display_name=display_name,
        permission=share["permission"],
        share_id=share["share_id"],
    )

    return JoinResponse(
        conversation_id=conversation_id,
        permission=share["permission"],
        title=conv.get("title", "Untitled conversation"),
    )
