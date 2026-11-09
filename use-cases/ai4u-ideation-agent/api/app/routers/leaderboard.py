"""Leaderboard router — company-level rankings of ideation submissions."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth import UserIdentity, get_current_user
from ..services.cdc import _extract_email_domain, _is_public_email_domain
from ..services.hana_submissions import get_submission_store

logger = logging.getLogger(__name__)
router = APIRouter()


class LeaderboardEntry(BaseModel):
    rank: int
    displayName: str
    email: str
    industry: str
    title: str
    problem: str
    solution: str
    app_type: str
    evaluatorFeedback: str
    evaluatorDetailedAnalysis: str
    evaluatorSuggestions: List[str]
    appropriatenessLevel: str
    score: int
    rating: int
    timestamp: str
    contextId: Optional[str] = None
    isCurrentUser: bool


class LeaderboardResponse(BaseModel):
    companyDomain: str
    totalEntries: int
    entries: List[LeaderboardEntry]


def _format_display_name(email: str, given_name: str = "", family_name: str = "") -> str:
    """Format display name as full name."""
    if given_name:
        if family_name:
            return f"{given_name} {family_name}"
        return given_name
    # Fallback: use email prefix
    local_part = email.split("@")[0] if "@" in email else email
    return local_part.replace(".", " ").replace("_", " ").title()


@router.get("/leaderboard/company", response_model=LeaderboardResponse)
async def get_company_leaderboard(
    limit: int = Query(default=50, ge=1, le=200),
    user: UserIdentity = Depends(get_current_user),
) -> LeaderboardResponse:
    """Get the company leaderboard for the current user's corporate domain."""
    if not user.email:
        raise HTTPException(
            status_code=401,
            detail="User email not available.",
        )

    company_domain = _extract_email_domain(user.email)
    if not company_domain or _is_public_email_domain(company_domain):
        raise HTTPException(
            status_code=403,
            detail=(
                "Company leaderboard is only available for users "
                "with a corporate email domain."
            ),
        )

    store = await get_submission_store()

    try:
        total = await store.count_company_entries(company_domain)
        rows = await store.get_company_leaderboard(company_domain, limit=limit)
    except Exception as exc:
        logger.error("Failed to query leaderboard for domain '%s': %s", company_domain, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Leaderboard query failed: {exc}",
        ) from exc

    entries: List[LeaderboardEntry] = []
    for index, row in enumerate(rows, start=1):
        row_email = str(row.get("email", "")).strip().lower()
        is_current = row_email == user.email.lower()

        display_name = _format_display_name(
            row_email,
            given_name=user.given_name if is_current else "",
            family_name=user.family_name if is_current else "",
        )

        entries.append(LeaderboardEntry(
            rank=index,
            displayName=display_name,
            email=row_email,
            industry=row.get("industry", ""),
            title=row.get("title", ""),
            problem=row.get("problem", ""),
            solution=row.get("solution", ""),
            app_type=row.get("app_type", ""),
            evaluatorFeedback=row.get("evaluator_feedback", ""),
            evaluatorDetailedAnalysis=row.get("evaluator_detailed_analysis", ""),
            evaluatorSuggestions=row.get("evaluator_suggestions", []),
            appropriatenessLevel=row.get("appropriateness_level", ""),
            score=int(row.get("score", 0)),
            rating=int(row.get("rating", 0)),
            timestamp=row.get("submitted_at", ""),
            contextId=row.get("context_id"),
            isCurrentUser=is_current,
        ))

    logger.info(
        "Company leaderboard for '%s': total=%d, returned=%d",
        company_domain, total, len(entries),
    )

    return LeaderboardResponse(
        companyDomain=company_domain,
        totalEntries=total,
        entries=entries,
    )
