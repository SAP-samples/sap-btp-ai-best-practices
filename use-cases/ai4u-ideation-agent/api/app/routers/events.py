"""Events router — scoped leaderboards for workshops, hackathons, and competitions."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth import UserIdentity, get_current_user
from ..services.hana_events import get_event_store

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_current_user)])


# ── Request/Response models ─────────────────────────────────────────────────


class CreateEventRequest(BaseModel):
    name: str


class CreateEventResponse(BaseModel):
    event_id: str
    join_code: str
    name: str
    status: str
    created_at: str


class EventInfo(BaseModel):
    event_id: str
    join_code: str
    name: str
    status: str
    created_at: str
    created_by_uid: str
    created_by_email: str
    member_count: int
    role: Optional[str] = None
    joined_at: Optional[str] = None


class EventPreview(BaseModel):
    """Limited event info for non-members (join page rendering)."""
    event_id: str
    join_code: str
    name: str
    status: str
    member_count: int
    is_member: bool
    role: Optional[str] = None


class StatusRequest(BaseModel):
    status: str  # 'active' | 'ended'


class RoleRequest(BaseModel):
    role: str  # 'admin' | 'member'


class MemberInfo(BaseModel):
    uid: str
    email: str
    display_name: str
    role: str
    joined_at: str


class LeaderboardEntry(BaseModel):
    rank: int
    display_name: str
    uid: str
    industry: str
    title: str
    problem: str
    solution: str
    app_type: str
    evaluator_feedback: str
    evaluator_detailed_analysis: str
    evaluator_suggestions: List[str]
    appropriateness_level: str
    score: int
    rating: int
    timestamp: str
    context_id: Optional[str] = None
    is_current_user: bool


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_display_name(email: str, given_name: str = "", family_name: str = "") -> str:
    """Format display name as full name."""
    if given_name:
        if family_name:
            return f"{given_name} {family_name}"
        return given_name
    # Fallback: use email prefix
    local_part = email.split("@")[0] if "@" in email else email
    return local_part.replace(".", " ").replace("_", " ").title()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("", response_model=CreateEventResponse)
async def create_event(
    request: CreateEventRequest,
    user: UserIdentity = Depends(get_current_user),
) -> CreateEventResponse:
    """Create a new event. The creator is auto-joined as admin."""
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Event name is required")

    store = await get_event_store()

    try:
        result = await store.create_event(
            name=request.name.strip(),
            uid=user.user_id,
            email=user.email,
            given_name=user.given_name,
            family_name=user.family_name,
        )
    except Exception as exc:
        logger.error("Failed to create event: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create event") from exc

    return CreateEventResponse(**result)


@router.get("/my", response_model=List[EventInfo])
async def get_my_events(
    user: UserIdentity = Depends(get_current_user),
) -> List[EventInfo]:
    """List all events the user is a member of (active + ended)."""
    store = await get_event_store()

    try:
        rows = await store.get_user_events(user.user_id)
    except Exception as exc:
        logger.error("Failed to list user events: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list events") from exc

    events = []
    for row in rows:
        member_count = await store.count_event_members(row["event_id"])
        events.append(EventInfo(
            event_id=row["event_id"],
            join_code=row["join_code"],
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            created_by_uid=row["created_by_uid"],
            created_by_email=row["created_by_email"],
            member_count=member_count,
            role=row.get("role"),
            joined_at=row.get("joined_at"),
        ))

    return events


@router.get("/by-code/{code}", response_model=EventPreview)
async def get_event_by_code(
    code: str,
    user: UserIdentity = Depends(get_current_user),
) -> EventPreview:
    """Resolve a join code to event info. Works for both members and non-members."""
    store = await get_event_store()

    event = await store.get_event_by_code(code.upper())
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    member_count = await store.count_event_members(event["event_id"])
    role = await store.get_member_role(event["event_id"], user.user_id)

    return EventPreview(
        event_id=event["event_id"],
        join_code=event["join_code"],
        name=event["name"],
        status=event["status"],
        member_count=member_count,
        is_member=role is not None,
        role=role,
    )


@router.post("/{event_id}/join")
async def join_event(
    event_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Join an event as a member."""
    store = await get_event_store()

    event = await store.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event["status"] == "ended":
        raise HTTPException(status_code=400, detail="Cannot join an ended event")

    joined = await store.join_event(
        event_id=event_id,
        uid=user.user_id,
        email=user.email,
        given_name=user.given_name,
        family_name=user.family_name,
    )

    if not joined:
        return {"status": "already_member", "message": "Already a member of this event"}

    return {"status": "joined", "message": "Successfully joined event"}


@router.post("/{event_id}/leave")
async def leave_event(
    event_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Leave an event."""
    store = await get_event_store()

    left = await store.leave_event(event_id, user.user_id)
    if not left:
        raise HTTPException(status_code=404, detail="Not a member of this event")

    return {"status": "left", "message": "Successfully left event"}


@router.patch("/{event_id}/status")
async def update_event_status(
    event_id: str,
    request: StatusRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Toggle event status between 'active' and 'ended' (admin only)."""
    if request.status not in ("active", "ended"):
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'ended'")

    store = await get_event_store()

    success = await store.set_event_status(event_id, user.user_id, request.status)
    if not success:
        raise HTTPException(status_code=403, detail="Only event admins can change status")

    return {"status": request.status, "message": f"Event is now {request.status}"}


@router.get("/{event_id}/leaderboard", response_model=List[LeaderboardEntry])
async def get_event_leaderboard(
    event_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user: UserIdentity = Depends(get_current_user),
) -> List[LeaderboardEntry]:
    """Get the event leaderboard (members only)."""
    store = await get_event_store()

    # Verify membership
    role = await store.get_member_role(event_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this event")

    try:
        rows = await store.get_event_leaderboard(event_id, limit=limit)
    except Exception as exc:
        logger.error("Failed to query event leaderboard: %s", exc)
        raise HTTPException(status_code=500, detail="Leaderboard query failed") from exc

    entries: List[LeaderboardEntry] = []
    for index, row in enumerate(rows, start=1):
        row_uid = str(row.get("uid", ""))
        is_current = row_uid == user.user_id

        display_name = _format_display_name(
            row.get("email", ""),
            given_name=row.get("given_name", ""),
            family_name=row.get("family_name", ""),
        )

        entries.append(LeaderboardEntry(
            rank=index,
            display_name=display_name,
            uid=row_uid,
            industry=row.get("industry", ""),
            title=row.get("title", ""),
            problem=row.get("problem", ""),
            solution=row.get("solution", ""),
            app_type=row.get("app_type", ""),
            evaluator_feedback=row.get("evaluator_feedback", ""),
            evaluator_detailed_analysis=row.get("evaluator_detailed_analysis", ""),
            evaluator_suggestions=row.get("evaluator_suggestions", []),
            appropriateness_level=row.get("appropriateness_level", ""),
            score=int(row.get("score", 0)),
            rating=int(row.get("rating", 0)),
            timestamp=row.get("submitted_at", ""),
            context_id=row.get("context_id"),
            is_current_user=is_current,
        ))

    return entries


@router.get("/{event_id}/members", response_model=List[MemberInfo])
async def get_event_members(
    event_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> List[MemberInfo]:
    """List all members of an event (members only)."""
    store = await get_event_store()

    # Verify membership
    role = await store.get_member_role(event_id, user.user_id)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a member of this event")

    try:
        rows = await store.get_event_members(event_id)
    except Exception as exc:
        logger.error("Failed to list event members: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list members") from exc

    members = []
    for row in rows:
        display_name = _format_display_name(
            row.get("email", ""),
            given_name=row.get("given_name", ""),
            family_name=row.get("family_name", ""),
        )
        members.append(MemberInfo(
            uid=row["uid"],
            email=row["email"],
            display_name=display_name,
            role=row["role"],
            joined_at=row.get("joined_at", ""),
        ))

    return members


@router.patch("/{event_id}/members/{target_uid}/role")
async def update_member_role(
    event_id: str,
    target_uid: str,
    request: RoleRequest,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Promote/demote a member (admin only)."""
    if request.role not in ("admin", "member"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'member'")

    store = await get_event_store()

    success = await store.set_member_role(event_id, user.user_id, target_uid, request.role)
    if not success:
        raise HTTPException(
            status_code=403,
            detail="Only admins can change roles, or target is not a member",
        )

    return {"status": "updated", "role": request.role}
