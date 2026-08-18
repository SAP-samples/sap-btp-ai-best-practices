"""
SAP Dynamic Credentials

Reads SAP connection credentials from HTTP request headers (set by frontend
from sessionStorage) with fallback to .env settings.

Headers (sent by frontend):
    X-SAP-Base-URL   — S/4HANA base URL
    X-SAP-Client     — SAP client number
    X-SAP-Username   — SAP username
    X-SAP-Password   — SAP password
    X-SAP-Verify     — SSL verify ("true"/"false")

Security:
    - Credentials are NEVER logged
    - Credentials are NEVER stored server-side
    - Each request is independently authenticated
    - Falls back to .env if headers are absent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Union

import requests
import urllib3
from fastapi import Request

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SapConfig:
    """SAP connection configuration for a single request."""

    base_url: str
    client: str
    username: str
    password: str = field(repr=False)  # never printed
    verify: Union[bool, str] = False

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.username and self.password)

    def build_session(self) -> requests.Session:
        """Build an authenticated requests.Session from this config."""
        if not self.is_configured:
            raise RuntimeError(
                "SAP credentials not configured. "
                "Set them in Settings → S4 Access or in backend/.env"
            )

        verify: Union[bool, str] = self.verify
        if not verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        session = requests.Session()
        session.auth = (self.username, self.password)
        session.verify = verify
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "sap-client": self.client,
            }
        )
        return session


def get_sap_config(request: Request | None = None) -> SapConfig:
    """
    Build SAP config from request headers, falling back to .env settings.

    Priority:
    1. X-SAP-* request headers (set by frontend sessionStorage)
    2. .env / settings (static fallback)
    """
    if request is not None:
        headers = request.headers
        base_url = headers.get("X-SAP-Base-URL", "").strip()
        client = headers.get("X-SAP-Client", "").strip()
        username = headers.get("X-SAP-Username", "").strip()
        password = headers.get("X-SAP-Password", "").strip()
        verify_str = headers.get("X-SAP-Verify", "").strip().lower()

        # If all required headers are present, use them
        if base_url and username and password:
            verify: Union[bool, str] = verify_str not in ("false", "0", "no", "")
            logger.debug(
                "SAP config from request headers | base_url=%s | client=%s | user=%s",
                base_url,
                client or "(default)",
                username,
            )
            return SapConfig(
                base_url=base_url.rstrip("/"),
                client=client or settings.S4_CLIENT,
                username=username,
                password=password,
                verify=verify,
            )

    # Fallback to .env settings
    logger.debug("SAP config from .env settings | base_url=%s", settings.S4_BASE_URL)
    return SapConfig(
        base_url=settings.S4_BASE_URL.rstrip("/") if settings.S4_BASE_URL else "",
        client=settings.S4_CLIENT,
        username=settings.S4_USERNAME,
        password=settings.S4_PASSWORD,
        verify=settings.S4_VERIFY,
    )