"""Signed-in user information for the UI header.

``GET /api/me`` returns who is signed in, so the UI can show the user's name
next to the "Sign out" link. Authentication itself is enforced by
:func:`app.security.require_user`; this router only reports its result.

``sign_in_required`` tells the UI whether this deployment has sign-in at all:
it is false in the open showcase deployment (``AUTH_MODE=public``) and in
local development, where there is no session to end.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.security import AuthenticatedUser, require_user, sign_in_required

router = APIRouter(prefix="/api", tags=["user-session"])


class CurrentUserResponse(BaseModel):
    """Public view of the signed-in user.

    Attributes:
        user_id: Logon name of the user.
        display_name: Full name, else email, else logon name.
        email: Email address, when the identity provider supplies it.
        sign_in_required: True when users sign in (``AUTH_MODE=xsuaa``); false
            in the public showcase and in local development.
    """

    user_id: str
    display_name: str
    email: Optional[str] = None
    sign_in_required: bool = True


@router.get("/me", response_model=CurrentUserResponse)
def current_user(user: AuthenticatedUser = Depends(require_user)) -> CurrentUserResponse:
    """Return the identity of the caller.

    Args:
        user: User resolved by the authentication dependency.

    Returns:
        CurrentUserResponse with the id, a display name, the email and whether
        this deployment requires sign-in.
    """
    return CurrentUserResponse(
        user_id=user.user_id,
        display_name=user.display_name,
        email=user.email,
        sign_in_required=sign_in_required(),
    )
