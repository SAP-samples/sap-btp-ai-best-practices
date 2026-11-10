"""Version check router — exposes update availability to all authenticated users."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import UserIdentity, get_current_user
from ..services.version_check import check_for_update

router = APIRouter()


@router.get("/version-check")
async def version_check(
    _user: UserIdentity = Depends(get_current_user),
) -> dict:
    """Check if a newer version of the application is available.

    Available to ALL authenticated users (not admin-only).
    The upstream check is cached in-memory for 1 hour.

    Returns:
        updateAvailable: true (newer exists) / false (up-to-date) / null (check failed or disabled)
        currentVersion: currently deployed version string
        latestVersion: upstream version string (null if check failed)
        checkEnabled: whether the update check feature is active
    """
    return await check_for_update()
