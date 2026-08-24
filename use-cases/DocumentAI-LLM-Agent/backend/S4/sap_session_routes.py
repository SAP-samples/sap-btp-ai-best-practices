"""
SAP Session Routes

POST /api/sap/session/login  — validate SAP credentials and test connection
GET  /api/sap/session/status — check if current request has valid SAP config

Improvements (PROD debug):
- Split timeout: (connect=10s, read=30s)
- Detailed error_type classification
- Full exception class + cause logged
- SSL verify mode logged explicitly
- Separate handling for DNS, TCP refused, SSL handshake, timeout
"""

from __future__ import annotations

import logging
import socket as _socket

import requests
import urllib3
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from S4.sap_credentials import SapConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sap/session", tags=["SAP Session"])

# Timeouts: (connect_timeout, read_timeout)
_CONNECT_TIMEOUT = 10   # seconds to establish TCP connection
_READ_TIMEOUT    = 30   # seconds to wait for SAP response


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SapLoginRequest(BaseModel):
    base_url: str = Field(..., description="S/4HANA base URL")
    client: str = Field(default="100", description="SAP client number")
    username: str = Field(..., description="SAP username")
    password: str = Field(..., description="SAP password")
    verify: bool = Field(default=False, description="Verify SSL certificate")


class SapLoginResponse(BaseModel):
    success: bool
    message: str
    error_type: str = ""          # DNS_FAILURE | TCP_REFUSED | TIMEOUT | SSL_ERROR | AUTH | HTTP_ERROR | UNKNOWN
    http_status: int | None = None
    details: str = ""
    base_url: str = ""
    client: str = ""
    username: str = ""
    ssl_verify: bool = False
    # password is NEVER returned


class SapStatusResponse(BaseModel):
    configured: bool
    source: str  # "headers" | "env" | "none"
    base_url: str = ""
    client: str = ""
    username: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_connection_error(exc: requests.exceptions.ConnectionError) -> tuple[str, str]:
    """
    Classify a ConnectionError into (error_type, human_message).

    Inspects the exception chain to distinguish:
    - DNS resolution failure
    - TCP connection refused
    - Network unreachable / no route
    - Generic connection error
    """
    cause = str(exc).lower()
    original = repr(exc)

    if "name or service not known" in cause or "nodename nor servname" in cause or "getaddrinfo" in cause:
        return "DNS_FAILURE", "DNS resolution failed — hostname cannot be resolved"
    if "connection refused" in cause or "errno 111" in cause or "errno 61" in cause:
        return "TCP_REFUSED", "TCP connection refused — SAP port is closed or firewall is blocking"
    if "network is unreachable" in cause or "no route to host" in cause or "errno 101" in cause or "errno 113" in cause:
        return "NETWORK_UNREACHABLE", "Network unreachable — backend cannot reach SAP subnet (VPN/firewall?)"
    if "timed out" in cause or "timeout" in cause:
        return "CONNECT_TIMEOUT", "TCP connect timed out — SAP host is unreachable or filtered"

    return "CONNECTION_ERROR", f"Connection error: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=SapLoginResponse,
    summary="Validate SAP credentials and test connection",
    description=(
        "Validates SAP credentials by calling "
        "API_BUSINESS_PARTNER/A_BusinessPartner?$top=1&$format=json. "
        "Credentials are NOT stored server-side. "
        "Frontend must store them in sessionStorage and send as X-SAP-* headers."
    ),
)
async def sap_login(body: SapLoginRequest) -> SapLoginResponse:
    """
    Test SAP credentials by calling A_BusinessPartner?$top=1&$format=json.

    Detailed diagnostics:
    - Logs resolved URL, SSL verify mode, timeout config
    - Classifies errors: DNS, TCP refused, network unreachable, SSL, auth, timeout
    - Returns error_type for frontend to display actionable message
    - Password is NEVER returned or logged
    """
    if not body.base_url or not body.username or not body.password:
        return SapLoginResponse(
            success=False,
            error_type="MISSING_FIELDS",
            message="Missing required fields: base_url, username, password",
            details="All three fields are required to test the connection.",
        )

    base_url = body.base_url.rstrip("/")

    logger.info(
        "SAP login test | base_url=%s | client=%s | user=%s | ssl_verify=%s | "
        "connect_timeout=%ds | read_timeout=%ds",
        base_url,
        body.client,
        body.username,
        body.verify,
        _CONNECT_TIMEOUT,
        _READ_TIMEOUT,
    )

    # Build session
    if not body.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    session.auth = (body.username, body.password)
    session.verify = body.verify
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "sap-client": body.client,
        }
    )

    test_url = f"{base_url}/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"
    params = {
        "sap-client": body.client,
        "$top": "1",
        "$format": "json",
    }

    logger.info(
        "SAP HTTP GET | url=%s | params=%s | verify=%s",
        test_url, params, body.verify,
    )

    try:
        resp = session.get(
            test_url,
            params=params,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        http_status = resp.status_code
        response_preview = resp.text[:500] if resp.text else ""

        logger.info(
            "SAP response | status=%d | content_type=%s | preview=%s",
            http_status,
            resp.headers.get("Content-Type", ""),
            response_preview,
        )

        if http_status == 200:
            logger.info("SAP connection test SUCCESS | user=%s | base_url=%s", body.username, base_url)
            return SapLoginResponse(
                success=True,
                message="✅ SAP Connection Successful",
                http_status=http_status,
                details=f"Connected to {base_url} as {body.username}",
                base_url=base_url,
                client=body.client,
                username=body.username,
                ssl_verify=body.verify,
            )

        elif http_status == 401:
            logger.warning("SAP auth failed | user=%s | status=401 | base_url=%s", body.username, base_url)
            return SapLoginResponse(
                success=False,
                error_type="AUTH",
                message="❌ Invalid credentials — check username and password",
                http_status=http_status,
                details=response_preview,
                base_url=base_url,
                ssl_verify=body.verify,
            )

        elif http_status == 403:
            logger.warning("SAP forbidden | user=%s | status=403", body.username)
            return SapLoginResponse(
                success=False,
                error_type="AUTH",
                message="❌ SAP authorization issue — user lacks API_BUSINESS_PARTNER access",
                http_status=http_status,
                details=response_preview,
                base_url=base_url,
                ssl_verify=body.verify,
            )

        elif http_status == 406:
            return SapLoginResponse(
                success=False,
                error_type="HTTP_ERROR",
                message="❌ SAP returned 406 Not Acceptable — OData format issue",
                http_status=http_status,
                details=f"SAP rejected the request format. Response: {response_preview}",
                base_url=base_url,
                ssl_verify=body.verify,
            )

        elif http_status == 404:
            return SapLoginResponse(
                success=False,
                error_type="HTTP_ERROR",
                message="❌ SAP endpoint not found (404) — check base URL and SAP client",
                http_status=http_status,
                details=f"URL tried: {test_url} | Response: {response_preview}",
                base_url=base_url,
                ssl_verify=body.verify,
            )

        else:
            return SapLoginResponse(
                success=False,
                error_type="HTTP_ERROR",
                message=f"❌ SAP returned HTTP {http_status}",
                http_status=http_status,
                details=response_preview,
                base_url=base_url,
                ssl_verify=body.verify,
            )

    except requests.exceptions.SSLError as exc:
        logger.error(
            "SAP SSL error | base_url=%s | exc_type=%s | detail=%s",
            base_url, type(exc).__name__, exc,
        )
        return SapLoginResponse(
            success=False,
            error_type="SSL_ERROR",
            message="❌ SSL/TLS error — SAP uses a self-signed cert; disable SSL verification",
            details=f"{type(exc).__name__}: {exc}",
            base_url=base_url,
            ssl_verify=body.verify,
        )

    except requests.exceptions.ConnectTimeout as exc:
        logger.error(
            "SAP connect timeout | base_url=%s | connect_timeout=%ds | exc=%s",
            base_url, _CONNECT_TIMEOUT, exc,
        )
        return SapLoginResponse(
            success=False,
            error_type="CONNECT_TIMEOUT",
            message=f"❌ TCP connect timed out after {_CONNECT_TIMEOUT}s — SAP host unreachable or filtered",
            details=f"Host: {base_url} | Timeout: {_CONNECT_TIMEOUT}s connect / {_READ_TIMEOUT}s read",
            base_url=base_url,
            ssl_verify=body.verify,
        )

    except requests.exceptions.ReadTimeout as exc:
        logger.error(
            "SAP read timeout | base_url=%s | read_timeout=%ds | exc=%s",
            base_url, _READ_TIMEOUT, exc,
        )
        return SapLoginResponse(
            success=False,
            error_type="READ_TIMEOUT",
            message=f"❌ SAP did not respond within {_READ_TIMEOUT}s — system may be overloaded",
            details=f"Connected but no response within {_READ_TIMEOUT}s",
            base_url=base_url,
            ssl_verify=body.verify,
        )

    except requests.exceptions.ConnectionError as exc:
        error_type, human_msg = _classify_connection_error(exc)
        logger.error(
            "SAP connection error | base_url=%s | error_type=%s | exc_type=%s | detail=%s",
            base_url, error_type, type(exc).__name__, exc,
        )
        return SapLoginResponse(
            success=False,
            error_type=error_type,
            message=f"❌ {human_msg}",
            details=(
                f"Base URL: {base_url}\n"
                f"Exception: {type(exc).__name__}\n"
                f"Detail: {exc}\n\n"
                f"If SAP IP is private (10.x / 192.168.x / 172.x), the PROD backend "
                f"may not have network access to the corporate SAP subnet."
            ),
            base_url=base_url,
            ssl_verify=body.verify,
        )

    except requests.exceptions.Timeout as exc:
        logger.error("SAP timeout | base_url=%s | exc=%s", base_url, exc)
        return SapLoginResponse(
            success=False,
            error_type="TIMEOUT",
            message="❌ Connection timed out — SAP system may be unreachable",
            details=f"Timeout ({_CONNECT_TIMEOUT}s connect / {_READ_TIMEOUT}s read) connecting to {test_url}",
            base_url=base_url,
            ssl_verify=body.verify,
        )

    except Exception as exc:
        logger.exception(
            "Unexpected SAP login error | base_url=%s | exc_type=%s",
            base_url, type(exc).__name__,
        )
        return SapLoginResponse(
            success=False,
            error_type="UNKNOWN",
            message=f"❌ Unexpected error: {type(exc).__name__}",
            details=str(exc),
            base_url=base_url,
            ssl_verify=body.verify,
        )


@router.get(
    "/status",
    response_model=SapStatusResponse,
    summary="Check SAP configuration status for current request",
)
async def sap_status(request: Request) -> SapStatusResponse:
    """
    Returns whether the current request has SAP credentials configured.
    Indicates source: 'headers' (from frontend) or 'env' (from .env).
    """
    from S4.sap_credentials import get_sap_config

    headers = request.headers
    has_headers = bool(
        headers.get("X-SAP-Base-URL")
        and headers.get("X-SAP-Username")
        and headers.get("X-SAP-Password")
    )

    config = get_sap_config(request)

    if not config.is_configured:
        return SapStatusResponse(configured=False, source="none")

    return SapStatusResponse(
        configured=True,
        source="headers" if has_headers else "env",
        base_url=config.base_url,
        client=config.client,
        username=config.username,
    )