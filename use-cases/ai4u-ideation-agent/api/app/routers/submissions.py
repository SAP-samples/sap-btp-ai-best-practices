"""Submissions router — save and retrieve evaluated ideation solutions."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..auth import UserIdentity, get_current_user
from ..services.cdc import _extract_email_domain, _is_public_email_domain
from ..services.hana_submissions import get_submission_store

logger = logging.getLogger(__name__)
router = APIRouter()


class SubmissionRequest(BaseModel):
    industry: str
    title: str
    problem: str
    solution: str
    app_type: str = "Agent"
    company_context: Optional[str] = None
    sources: Optional[List[str]] = None
    evaluator_feedback: str = ""
    evaluator_detailed_analysis: str = ""
    evaluator_suggestions: Optional[List[str]] = None
    appropriateness_level: str = ""
    score: int = 0
    rating: int = 0
    context_id: Optional[str] = None
    message_index: Optional[int] = None
    event_id: Optional[str] = None


class SubmissionResponse(BaseModel):
    success: bool
    submission_id: str
    message: str


class HistoryEntry(BaseModel):
    submission_id: str
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
    submitted_at: str
    context_id: Optional[str] = None
    handover_status: Optional[str] = None


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    total: int
    limit: int
    cursor: int


async def _verify_context_access(uid: str, context_id: str) -> bool:
    """Check if user has access to a conversation (owner or shared member).

    Returns True if the user owns the conversation or is a shared member.
    """
    try:
        from ideation_agent.hana_store import get_store
        from ..services.hana_shares import get_share_store

        hana_store = await get_store()
        conv = await hana_store.get_conversation(context_id, uid)
        if conv:
            return True

        share_store = await get_share_store()
        permission = await share_store.get_member_permission(context_id, uid)
        return permission is not None
    except Exception as exc:
        logger.debug("Access check failed for context %s, uid %s: %s", context_id, uid, exc)
        return False


def _dedup_by_message_index(rows: list) -> list:
    """Deduplicate rows by message_index, keeping the highest score for each.

    When multiple collaborators have evaluations for the same message_index,
    only the best-scoring one is returned.
    """
    best_by_index: dict = {}
    for row in rows:
        idx = row.get("message_index")
        existing = best_by_index.get(idx)
        if existing is None or int(row.get("score", 0)) > int(existing.get("score", 0)):
            best_by_index[idx] = row
    return list(best_by_index.values())


@router.post("/submissions", response_model=SubmissionResponse)
async def submit_solution(
    request: SubmissionRequest,
    user: UserIdentity = Depends(get_current_user),
) -> SubmissionResponse:
    """Save an evaluated ideation solution to the leaderboard."""
    store = await get_submission_store()

    # Determine company domain: for shared conversations, use the owner's domain
    # so the entry appears on the correct company leaderboard.
    company_domain = ""
    context_id = request.context_id or ""
    conv_data: Optional[Dict[str, Any]] = None  # loaded once, reused for event_id

    if context_id:
        try:
            from ideation_agent.hana_store import get_store

            hana_store = await get_store()
            conv = await hana_store.get_conversation(context_id, user.user_id)
            if conv:
                # User is the owner — use their own domain
                domain = _extract_email_domain(user.email)
                company_domain = domain if domain and not _is_public_email_domain(domain) else ""
                conv_data = conv  # owner's conv record includes EVENT_ID
            else:
                # User is a shared member — use the owner's domain
                conv_data = await hana_store.get_conversation_by_id(context_id)
                if conv_data:
                    owner_uid = conv_data.get("user_id", "")
                    # Look up owner's email from their submissions
                    owner_email = await store.get_email_by_uid(owner_uid)
                    if owner_email:
                        domain = _extract_email_domain(owner_email)
                        company_domain = domain if domain and not _is_public_email_domain(domain) else ""
        except Exception as exc:
            logger.debug("Could not resolve owner domain for context %s: %s", context_id, exc)
            # Fallback to submitting user's domain
            domain = _extract_email_domain(user.email)
            company_domain = domain if domain and not _is_public_email_domain(domain) else ""
    else:
        domain = _extract_email_domain(user.email)
        company_domain = domain if domain and not _is_public_email_domain(domain) else ""

    # Resolve event_id: use request value, or fall back to conversation's event
    resolved_event_id = request.event_id or ""
    if not resolved_event_id and context_id and conv_data:
        resolved_event_id = conv_data.get("event_id") or ""

    submission_data: Dict[str, Any] = {
        "uid": user.user_id,
        "email": user.email,
        "company_domain": company_domain,
        "industry": request.industry,
        "title": request.title,
        "problem": request.problem,
        "solution": request.solution,
        "app_type": request.app_type,
        "company_context": request.company_context or "",
        "sources": request.sources or [],
        "evaluator_feedback": request.evaluator_feedback,
        "evaluator_detailed_analysis": request.evaluator_detailed_analysis,
        "evaluator_suggestions": request.evaluator_suggestions or [],
        "appropriateness_level": request.appropriateness_level,
        "score": request.score,
        "rating": request.rating,
        "context_id": context_id,
        "message_index": request.message_index,
        "event_id": resolved_event_id,
        "source": "ideation-agent",
    }

    try:
        submission_id = await store.upsert_submission(submission_data)
        return SubmissionResponse(
            success=True,
            submission_id=submission_id,
            message="Solution submitted successfully",
        )
    except Exception as exc:
        logger.error("Failed to save submission for user %s: %s", user.user_id, exc)
        return SubmissionResponse(
            success=False,
            submission_id="",
            message=f"Failed to save submission: {exc}",
        )


@router.get("/submissions/by-context/{context_id}/all")
async def get_all_submissions_by_context(
    context_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Fetch ALL stored evaluations for a conversation.

    Returns all evaluations keyed by message_index so the frontend
    can restore eval cards for every final message on reload.
    For shared conversations, returns evaluations from any collaborator
    (deduplicated by message_index — highest score wins).
    """
    store = await get_submission_store()

    # Verify user has access to this conversation
    has_access = await _verify_context_access(user.user_id, context_id)
    if not has_access:
        return {"entries": []}

    try:
        rows = await store.get_all_by_context_any_user(context_id)
    except Exception as exc:
        logger.error("Failed to fetch submissions for context %s: %s", context_id, exc)
        return {"entries": []}

    # Deduplicate by message_index (highest score wins)
    rows = _dedup_by_message_index(rows)

    entries = []
    for row in rows:
        # Skip error evals
        if int(row.get("score", 0)) == 500 and row.get("evaluator_feedback", "") == "Evaluation Error":
            continue
        entries.append({
            "messageIndex": row.get("message_index"),
            "score": int(row.get("score", 0)),
            "feedback": row.get("evaluator_feedback", ""),
            "detailedAnalysis": row.get("evaluator_detailed_analysis", ""),
            "suggestions": row.get("evaluator_suggestions", []),
            "appropriatenessLevel": row.get("appropriateness_level", ""),
            "isOnLeaderboard": bool(row.get("is_on_leaderboard", True)),
        })

    # Normalize: only ONE entry should be marked as on leaderboard.
    # Multiple can appear when different collaborators each have IS_ON_LEADERBOARD=TRUE.
    # Keep only the last (most recent) one marked.
    leaderboard_entries = [e for e in entries if e["isOnLeaderboard"]]
    if len(leaderboard_entries) > 1:
        # Mark all as false, then re-mark the last one
        for e in entries:
            e["isOnLeaderboard"] = False
        leaderboard_entries[-1]["isOnLeaderboard"] = True

    return {"entries": entries}


@router.get("/submissions/by-context/{context_id}")
async def get_submission_by_context(
    context_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """Fetch a stored evaluation by context_id.

    Returns the best evaluation for this conversation (any collaborator),
    or {found: false} if none exists.
    """
    store = await get_submission_store()

    # Verify user has access to this conversation
    has_access = await _verify_context_access(user.user_id, context_id)
    if not has_access:
        return {"found": False}

    try:
        rows = await store.get_all_by_context_any_user(context_id)
    except Exception as exc:
        logger.error("Failed to fetch submission for context %s: %s", context_id, exc)
        return {"found": False}

    if not rows:
        return {"found": False}

    # Pick the best-scoring evaluation
    row = max(rows, key=lambda r: int(r.get("score", 0)))

    # Don't return failed evaluation attempts (error fallback data)
    if int(row.get("score", 0)) == 500 and row.get("evaluator_feedback", "") == "Evaluation Error":
        return {"found": False}

    return {
        "found": True,
        "score": int(row.get("score", 0)),
        "feedback": row.get("evaluator_feedback", ""),
        "detailedAnalysis": row.get("evaluator_detailed_analysis", ""),
        "suggestions": row.get("evaluator_suggestions", []),
        "appropriatenessLevel": row.get("appropriateness_level", ""),
        "success": True,
    }


@router.get("/submissions/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int = Query(default=0, ge=0),
    user: UserIdentity = Depends(get_current_user),
) -> HistoryResponse:
    """Get the current user's submission history."""
    store = await get_submission_store()

    rows = await store.get_user_history(user.user_id, limit=limit, offset=cursor)
    total = await store.count_user_history(user.user_id)

    # Batch-lookup handover status for entries that have a context_id
    handover_statuses: dict = {}
    context_ids = [r.get("context_id") for r in rows if r.get("context_id")]
    if context_ids:
        try:
            from ..services.hana_handover import get_handover_store
            h_store = await get_handover_store()
            for cid in context_ids:
                status = await h_store.get_status_for_conversation(cid)
                if status:
                    handover_statuses[cid] = status
        except Exception:
            pass  # Graceful degradation — handover status is optional

    entries = [
        HistoryEntry(
            submission_id=row.get("submission_id", ""),
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
            submitted_at=row.get("submitted_at", ""),
            context_id=row.get("context_id"),
            handover_status=handover_statuses.get(row.get("context_id")),
        )
        for row in rows
    ]

    return HistoryResponse(
        entries=entries,
        total=total,
        limit=limit,
        cursor=cursor,
    )
