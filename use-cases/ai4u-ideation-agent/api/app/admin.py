"""Admin access control for the AI4U Ideation Agent.

Provides FastAPI dependencies that restrict endpoints to admin users.
Admin status is determined by:
1. ADMIN_EMAILS env-var allowlist (always checked first)
2. CDC data.role field (when CDC is configured)
3. HANA users table role field (when CDC is not configured)
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException, status

from .auth import UserIdentity, cdc_enabled, get_current_user

logger = logging.getLogger(__name__)


def _user_is_admin(user: UserIdentity) -> bool:
    """Check if a user has admin access.

    Order:
    1. ADMIN_EMAILS env-var allowlist (comma-separated emails)
    2. CDC data.role == "admin" (when CDC_API_KEY is set)
    3. HANA users table role == "admin" (fallback when CDC off)

    Returns False on any lookup failure (non-blocking).
    """
    if not user.email:
        return False

    # 1. Env-var allowlist
    allowlist = os.getenv("ADMIN_EMAILS", "")
    if allowlist:
        admin_emails = [e.strip().lower() for e in allowlist.split(",") if e.strip()]
        if user.email.lower() in admin_emails:
            return True

    # 2. CDC lookup (when configured)
    if cdc_enabled():
        try:
            from .services.cdc import _search_account_by_email, _get_account_data

            uid = _search_account_by_email(user.email)
            if not uid:
                return False

            data = _get_account_data(uid)
            return data.get("role") == "admin"
        except Exception as exc:
            logger.warning("Admin check (CDC) failed for %s: %s", user.email, exc)
            return False

    # 3. HANA users table fallback (when CDC not configured)
    try:
        import asyncio
        from .services.hana_users import get_user_role

        # _user_is_admin is sync but hana_users is async — bridge via event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Can't await in a sync context within a running loop —
            # ADMIN_EMAILS is the primary mechanism; HANA role is best-effort
            return False
        else:
            role = asyncio.run(get_user_role(user.email))
            return role == "admin"
    except Exception as exc:
        logger.warning("Admin check (HANA) failed for %s: %s", user.email, exc)
        return False


async def require_admin(
    user: UserIdentity = Depends(get_current_user),
) -> UserIdentity:
    """FastAPI dependency that enforces admin access. Raises 403 if not admin."""
    if not _user_is_admin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def is_admin(
    user: UserIdentity = Depends(get_current_user),
) -> bool:
    """Non-blocking admin check — returns True/False instead of raising."""
    return _user_is_admin(user)
