"""
S/4HANA HTTP Session Client.

Supports both static (.env) and dynamic (request headers) credentials.

Usage:
    # Static (legacy, from .env):
    session = get_s4_session()
    base_url = get_s4_base_url()

    # Dynamic (from request headers):
    from S4.sap_credentials import get_sap_config
    config = get_sap_config(request)
    session = config.build_session()
    base_url = config.base_url
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import requests
import urllib3
from fastapi import Request

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static URL (used as fallback when no request context is available)
# ---------------------------------------------------------------------------
_STATIC_BASE_URL: str = settings.S4_BASE_URL.rstrip("/") if settings.S4_BASE_URL else ""

# Keep legacy module-level constants for backward compatibility
BASE_URL: str = _STATIC_BASE_URL
API_SO: str = f"{BASE_URL}/sap/opu/odata/sap/API_SALES_ORDER_SRV"


def sess(request: Optional[Request] = None) -> requests.Session:
    """
    Build and return an authenticated requests.Session for S/4HANA.

    If a FastAPI Request is provided, credentials are read from
    X-SAP-* headers (set by frontend sessionStorage).
    Otherwise falls back to .env settings.
    """
    from S4.sap_credentials import get_sap_config
    config = get_sap_config(request)
    return config.build_session()


def get_s4_session(request: Optional[Request] = None) -> requests.Session:
    """Alias for sess() — returns an authenticated S/4HANA session."""
    return sess(request)


def get_s4_base_url(request: Optional[Request] = None) -> str:
    """Return the S/4HANA base URL (no trailing slash)."""
    if request is not None:
        from S4.sap_credentials import get_sap_config
        config = get_sap_config(request)
        if config.base_url:
            return config.base_url
    return _STATIC_BASE_URL