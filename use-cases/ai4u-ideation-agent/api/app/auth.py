from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .services.cdc import validate_cdc_jwt

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class UserIdentity:
    user_id: str
    email: str
    given_name: str
    family_name: str
    user_uuid: str
    auth_provider: str = "xsuaa"  # "xsuaa" | "cdc"


@dataclass
class XSUAAConfig:
    client_id: str
    xsuaa_url: str
    jwks_uri: str

    @classmethod
    def from_environment(cls) -> "XSUAAConfig":
        vcap = os.getenv("VCAP_SERVICES")
        if vcap:
            try:
                services = json.loads(vcap)
                xsuaa_binding = services.get("xsuaa", [{}])[0].get("credentials", {})
                url = xsuaa_binding["url"]
                return cls(
                    client_id=xsuaa_binding["clientid"],
                    xsuaa_url=url,
                    jwks_uri=f"{url}/token_keys",
                )
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                logger.warning("Failed to parse VCAP_SERVICES for XSUAA: %s", exc)

        xsuaa_url = os.getenv("XSUAA_URL", "").rstrip("/")
        client_id = os.getenv("XSUAA_CLIENT_ID", "")
        if not xsuaa_url or not client_id:
            raise RuntimeError(
                "XSUAA not configured. Set VCAP_SERVICES (BTP) or XSUAA_URL + XSUAA_CLIENT_ID env vars."
            )
        return cls(
            client_id=client_id,
            xsuaa_url=xsuaa_url,
            jwks_uri=f"{xsuaa_url}/token_keys",
        )


class JWKSCache:
    _TTL_SECONDS = 86400  # 24 hours

    def __init__(self) -> None:
        self._keys: Dict[str, Any] = {}
        self._fetched_at: float = 0.0
        self._jwks_uri: Optional[str] = None

    def _is_stale(self) -> bool:
        return time.time() - self._fetched_at > self._TTL_SECONDS

    async def _refresh(self, jwks_uri: str) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(jwks_uri)
            resp.raise_for_status()
            data = resp.json()

        self._keys = {}
        for key_data in data.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                self._keys[kid] = key_data
        self._jwks_uri = jwks_uri
        self._fetched_at = time.time()
        logger.info("Refreshed JWKS cache with %d keys", len(self._keys))

    async def get_key(self, kid: str, jwks_uri: str) -> Dict[str, Any]:
        if self._is_stale() or self._jwks_uri != jwks_uri or kid not in self._keys:
            await self._refresh(jwks_uri)
        key_data = self._keys.get(kid)
        if not key_data:
            raise ValueError(f"Key ID '{kid}' not found in JWKS")
        return key_data


_jwks_cache = JWKSCache()
_xsuaa_config: Optional[XSUAAConfig] = None


def _get_xsuaa_config() -> XSUAAConfig:
    global _xsuaa_config
    if _xsuaa_config is None:
        _xsuaa_config = XSUAAConfig.from_environment()
    return _xsuaa_config


def _is_cdc_token(token: str) -> bool:
    """Peek at unverified payload to detect CDC tokens (contain 'apiKey' claim)."""
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        return "apiKey" in unverified
    except Exception:
        return False


def cdc_enabled() -> bool:
    """Returns True when CDC authentication is configured (CDC_API_KEY is set)."""
    return bool(os.getenv("CDC_API_KEY", "").strip())


async def _verify_xsuaa_token(token: str) -> UserIdentity:
    """Validate an XSUAA JWT and return UserIdentity."""
    config = _get_xsuaa_config()

    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise ValueError("Token header missing 'kid'")

    key_data = await _jwks_cache.get_key(kid, config.jwks_uri)

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)

    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience=config.client_id,
        options={"verify_aud": True, "verify_exp": True, "verify_iss": False},
    )

    return UserIdentity(
        user_id=payload.get("user_name", payload.get("sub", "")),
        email=payload.get("email", ""),
        given_name=payload.get("given_name", ""),
        family_name=payload.get("family_name", ""),
        user_uuid=payload.get("user_id", payload.get("sub", "")),
        auth_provider="xsuaa",
    )


def _verify_cdc_token(token: str) -> UserIdentity:
    """Validate a CDC/Gigya JWT and return UserIdentity."""
    validated = validate_cdc_jwt(token)

    return UserIdentity(
        user_id=validated.get("sub", ""),
        email=validated.get("email", ""),
        given_name=validated.get("firstName", validated.get("first_name", "")),
        family_name=validated.get("lastName", validated.get("last_name", "")),
        user_uuid=validated.get("sub", ""),
        auth_provider="cdc",
    )


async def verify_token(token: str) -> UserIdentity:
    """Detect token type and validate accordingly.

    CDC tokens contain an 'apiKey' claim; XSUAA tokens do not.
    When CDC is not configured, CDC tokens are rejected gracefully.
    """
    if _is_cdc_token(token):
        if not cdc_enabled():
            raise ValueError("CDC authentication is not configured on this server")
        return _verify_cdc_token(token)
    return await _verify_xsuaa_token(token)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> UserIdentity:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = await verify_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
