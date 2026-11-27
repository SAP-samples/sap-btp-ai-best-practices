"""User authentication for the workspace API.

Every user-facing route depends on :func:`require_user`. In Cloud Foundry the
application router (``approuter/``) signs the user in through SAP BTP (XSUAA)
and forwards the user's JWT in ``Authorization: Bearer <token>``. This module
validates that token against the bound XSUAA service instance and checks that
the user holds the application's single access scope (``$XSAPPNAME.User``,
granted by the ``ProductionWheel_User`` role collection, see
``xs-security.json``).

Authentication modes, selected with the ``AUTH_MODE`` environment variable:

* ``xsuaa`` (default): validate the forwarded JWT. Fails closed: without a
  bound XSUAA service every protected call returns 500.
* ``local``: developer mode for running the API on a laptop without BTP. Every
  request is treated as the fixed user ``local-developer``. Refused when
  ``APP_ENV=production``.
* ``public``: open showcase deployment (``manifest-public.yaml``, chosen with
  ``./deploy.sh public``). Nobody signs in; every request is treated as the
  fixed user ``public``. Allowed in production because it is an explicit
  deployment choice, never a fallback: a missing ``AUTH_MODE`` still means
  ``xsuaa``.

Status codes returned by :func:`require_user`:

* 401 - no token, or the token is invalid or expired (the caller must sign in).
* 403 - valid token, but the user lacks the access role.
* 500 - the server is misconfigured (no XSUAA binding, unknown mode, or local
  mode in production).

The ``local`` and ``public`` modes never return 401 or 403.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Local scope name (without the "$XSAPPNAME." prefix) that grants access.
REQUIRED_SCOPE = "User"
# Name of the XSUAA service label/tag searched for in VCAP_SERVICES.
XSUAA_LABEL = "xsuaa"
AUTH_MODE_XSUAA = "xsuaa"
AUTH_MODE_LOCAL = "local"
AUTH_MODE_PUBLIC = "public"
AUTH_MODES = {AUTH_MODE_XSUAA, AUTH_MODE_LOCAL, AUTH_MODE_PUBLIC}

# auto_error=False lets require_user answer 401 itself, with a clear message,
# instead of FastAPI's generic 403 for a missing header.
BEARER_SCHEME = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity of the signed-in caller, taken from a validated token.

    Attributes:
        user_id: Logon name of the user (or the client id for technical tokens).
        email: User email address, when the identity provider supplies it.
        given_name: First name, when supplied.
        family_name: Last name, when supplied.
    """

    user_id: str
    email: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Return a readable name for the UI header: full name, else email, else id."""
        full_name = " ".join(part for part in (self.given_name, self.family_name) if part)
        return full_name or self.email or self.user_id


# Fixed identity used only in AUTH_MODE=local (never in production).
LOCAL_USER = AuthenticatedUser(
    user_id="local-developer",
    email=None,
    given_name="Local",
    family_name="Developer",
)

# Fixed identity used only in AUTH_MODE=public (open showcase deployment).
PUBLIC_USER = AuthenticatedUser(
    user_id="public",
    email=None,
    given_name="Public",
    family_name="access",
)


class MissingScopeError(Exception):
    """Raised when a valid token does not carry the application's access scope."""


def auth_mode() -> str:
    """Return the configured authentication mode.

    Returns:
        ``"xsuaa"``, ``"local"`` or ``"public"``.

    Raises:
        HTTPException: 500 when ``AUTH_MODE`` holds an unknown value, so a typo
            never silently weakens authentication.
    """
    mode = (os.getenv("AUTH_MODE") or AUTH_MODE_XSUAA).strip().lower()
    if mode not in AUTH_MODES:
        logger.error("Unknown AUTH_MODE %r; refusing all requests", mode)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Authentication is misconfigured on the server")
    return mode


def sign_in_required() -> bool:
    """Return True when callers must sign in (``AUTH_MODE=xsuaa``).

    The UI uses this (through ``GET /api/me``) to decide whether to show the
    "Sign out" link.

    Raises:
        HTTPException: 500 for an unknown ``AUTH_MODE`` (see :func:`auth_mode`).
    """
    return auth_mode() == AUTH_MODE_XSUAA


def is_production() -> bool:
    """Return True when the API runs with ``APP_ENV=production``."""
    return os.getenv("APP_ENV", "").strip().lower() == "production"


def load_xsuaa_credentials() -> Optional[Dict[str, Any]]:
    """Read the credentials of the bound XSUAA service from ``VCAP_SERVICES``.

    Cloud Foundry injects ``VCAP_SERVICES`` as JSON keyed by service label. The
    XSUAA instance is found by its ``xsuaa`` label, or by an ``xsuaa`` tag when
    it is bound as a user-provided service.

    Returns:
        The credentials dictionary (``clientid``, ``xsappname``, ``url``,
        ``uaadomain``, ``verificationkey``...), or None when no XSUAA service is
        bound or ``VCAP_SERVICES`` is not valid JSON.
    """
    raw = os.getenv("VCAP_SERVICES")
    if not raw:
        return None
    try:
        services = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("VCAP_SERVICES is not valid JSON")
        return None
    for label, instances in services.items():
        for instance in instances or []:
            tags = instance.get("tags") or []
            if label == XSUAA_LABEL or XSUAA_LABEL in tags:
                credentials = instance.get("credentials")
                if credentials:
                    return credentials
    return None


def validate_token(token: str, credentials: Dict[str, Any]) -> AuthenticatedUser:
    """Validate an XSUAA access token and build the caller identity.

    The validation (signature against the XSUAA token keys, expiry, issuer zone
    and audience) is done by SAP's ``sap-xssec`` library. This call is blocking
    (it may download and cache the signing keys), so async callers should run
    it in a thread pool.

    Args:
        token: Raw JWT taken from the ``Authorization`` header.
        credentials: XSUAA credentials from :func:`load_xsuaa_credentials`.

    Returns:
        The authenticated user.

    Raises:
        MissingScopeError: The token is valid but lacks ``$XSAPPNAME.User``.
        Exception: Any validation failure raised by ``sap-xssec`` (invalid
            signature, expired token, wrong audience...).
    """
    # Imported lazily so environments that only use AUTH_MODE=local (unit tests,
    # laptops) do not need the library loaded at import time.
    from sap.xssec import create_security_context_xsuaa

    context = create_security_context_xsuaa(token, credentials)
    if not context.check_local_scope(REQUIRED_SCOPE):
        raise MissingScopeError(REQUIRED_SCOPE)
    return AuthenticatedUser(
        user_id=context.get_logon_name() or context.get_clientid(),
        email=context.get_email(),
        given_name=context.get_given_name(),
        family_name=context.get_family_name(),
    )


async def require_user(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Security(BEARER_SCHEME),
) -> AuthenticatedUser:
    """FastAPI dependency that admits only signed-in users with the access role.

    Args:
        request: Incoming request; the user is also stored in ``request.state.user``.
        bearer: Parsed ``Authorization: Bearer`` header, or None when absent.

    Returns:
        The authenticated user.

    Raises:
        HTTPException: 401 (not signed in / invalid token), 403 (missing role)
            or 500 (server misconfiguration). See the module docstring.
    """
    mode = auth_mode()
    if mode == AUTH_MODE_LOCAL:
        if is_production():
            logger.error("AUTH_MODE=local is not allowed with APP_ENV=production")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Authentication is misconfigured on the server")
        request.state.user = LOCAL_USER
        return LOCAL_USER
    if mode == AUTH_MODE_PUBLIC:
        request.state.user = PUBLIC_USER
        return PUBLIC_USER

    credentials = load_xsuaa_credentials()
    if credentials is None:
        logger.error("No XSUAA service is bound; refusing all requests")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "User authentication is not configured on the server")

    if bearer is None or bearer.scheme.lower() != "bearer" or not bearer.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Sign-in required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = await run_in_threadpool(validate_token, bearer.credentials, credentials)
    except MissingScopeError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Your user does not have access to this application")
    except Exception as error:  # sap-xssec raises ValueError/RuntimeError/DecodeError
        # Never log the token itself; the error type is enough to diagnose.
        logger.warning("Rejected access token: %s", type(error).__name__)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Your session is invalid or has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    request.state.user = user
    return user
