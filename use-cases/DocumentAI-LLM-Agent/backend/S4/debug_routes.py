"""
SAP Diagnostic Routes

GET /api/debug/ping-sap  — low-level connectivity test to SAP host

Performs three independent tests:
  1. TCP socket connect (raw connectivity, no HTTP)
  2. HTTPS request with verify=False (ignores cert)
  3. HTTPS request with verify=True  (checks cert validity)

Returns a structured payload with per-test results and a human-readable
diagnosis — useful to determine whether the PROD backend can reach the
corporate SAP subnet.

NOTE: This endpoint is for temporary diagnostics only.
      Credentials are read from X-SAP-* headers or query param.
      No credentials are stored or logged.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import ssl
import time
import urllib.parse
from typing import Optional

import requests
import urllib3
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["Debug / Diagnostics"])

_SOCKET_TIMEOUT = 8    # seconds for raw TCP test
_HTTPS_TIMEOUT  = 12   # seconds for HTTPS test


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class PingSapResponse(BaseModel):
    # Target
    base_url: str
    host: str
    port: int

    # Test 1 – raw TCP socket
    socket_reachable: bool
    socket_latency_ms: float | None = None
    socket_error: str | None = None

    # Test 2 – HTTPS (verify=False)
    https_reachable: bool
    https_status: int | None = None
    https_error: str | None = None

    # Test 3 – SSL certificate check (verify=True)
    ssl_valid: bool | None = None
    ssl_error: str | None = None

    # Summary
    error_type: str | None = None   # NETWORK_UNREACHABLE | TCP_REFUSED | TIMEOUT | SSL_ERROR | OK
    diagnosis: str
    recommendation: str


# ---------------------------------------------------------------------------
# Synchronous test helpers (run in thread pool)
# ---------------------------------------------------------------------------

def _test_socket(host: str, port: int) -> tuple[bool, float | None, str | None]:
    """
    Attempt a raw TCP connection to host:port.
    Returns (reachable, latency_ms, error_message).
    """
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=_SOCKET_TIMEOUT):
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return True, latency_ms, None
    except socket.timeout:
        return False, None, f"TCP connect timed out after {_SOCKET_TIMEOUT}s"
    except ConnectionRefusedError:
        return False, None, f"TCP connection refused on port {port}"
    except OSError as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _test_https(base_url: str, verify: bool) -> tuple[bool, int | None, str | None]:
    """
    Attempt an HTTPS GET to base_url with the given verify setting.
    Returns (reachable, http_status, error_message).
    """
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    probe_url = f"{base_url.rstrip('/')}/sap/opu/odata/sap/API_BUSINESS_PARTNER"
    try:
        resp = requests.get(
            probe_url,
            verify=verify,
            timeout=(_HTTPS_TIMEOUT, _HTTPS_TIMEOUT),
            allow_redirects=True,
        )
        return True, resp.status_code, None
    except requests.exceptions.SSLError as exc:
        return False, None, f"SSLError: {exc}"
    except requests.exceptions.ConnectTimeout:
        return False, None, f"Connect timed out after {_HTTPS_TIMEOUT}s"
    except requests.exceptions.ReadTimeout:
        return False, None, f"Read timed out after {_HTTPS_TIMEOUT}s"
    except requests.exceptions.ConnectionError as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _run_all_tests(base_url: str) -> PingSapResponse:
    """Run all three tests synchronously and build the response."""
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or base_url
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    logger.info(
        "ping-sap | base_url=%s | host=%s | port=%d",
        base_url, host, port,
    )

    # ── Test 1: TCP socket ────────────────────────────────────────────────
    sock_ok, sock_latency, sock_err = _test_socket(host, port)
    logger.info(
        "ping-sap socket | reachable=%s | latency=%s ms | error=%s",
        sock_ok, sock_latency, sock_err,
    )

    # ── Test 2: HTTPS verify=False ────────────────────────────────────────
    https_ok, https_status, https_err = _test_https(base_url, verify=False)
    logger.info(
        "ping-sap https(verify=False) | reachable=%s | status=%s | error=%s",
        https_ok, https_status, https_err,
    )

    # ── Test 3: SSL verify=True ───────────────────────────────────────────
    ssl_valid: bool | None = None
    ssl_error: str | None = None
    if sock_ok:  # only test SSL if TCP works
        _, _, ssl_err_raw = _test_https(base_url, verify=True)
        ssl_valid = ssl_err_raw is None
        ssl_error = ssl_err_raw
        logger.info(
            "ping-sap ssl(verify=True) | valid=%s | error=%s",
            ssl_valid, ssl_error,
        )

    # ── Diagnosis ─────────────────────────────────────────────────────────
    error_type, diagnosis, recommendation = _build_diagnosis(
        host=host,
        port=port,
        base_url=base_url,
        sock_ok=sock_ok,
        sock_err=sock_err,
        https_ok=https_ok,
        https_err=https_err,
        ssl_valid=ssl_valid,
        ssl_error=ssl_error,
    )

    return PingSapResponse(
        base_url=base_url,
        host=host,
        port=port,
        socket_reachable=sock_ok,
        socket_latency_ms=sock_latency,
        socket_error=sock_err,
        https_reachable=https_ok,
        https_status=https_status,
        https_error=https_err,
        ssl_valid=ssl_valid,
        ssl_error=ssl_error,
        error_type=error_type,
        diagnosis=diagnosis,
        recommendation=recommendation,
    )


def _is_private_ip(host: str) -> bool:
    """Return True if host looks like a private/corporate IP."""
    return (
        host.startswith("10.")
        or host.startswith("192.168.")
        or host.startswith("172.")
        or host == "localhost"
        or host == "127.0.0.1"
    )


def _build_diagnosis(
    *,
    host: str,
    port: int,
    base_url: str,
    sock_ok: bool,
    sock_err: str | None,
    https_ok: bool,
    https_err: str | None,
    ssl_valid: bool | None,
    ssl_error: str | None,
) -> tuple[str, str, str]:
    """
    Return (error_type, diagnosis, recommendation) based on test results.
    """
    is_private = _is_private_ip(host)

    if sock_ok and https_ok:
        ssl_note = ""
        if ssl_valid is False:
            ssl_note = f" SSL cert is self-signed/invalid ({ssl_error})."
        return (
            "OK",
            f"Backend can reach SAP at {base_url}.{ssl_note}",
            "Connection is working. If login still fails, check credentials or SAP client number.",
        )

    if sock_ok and not https_ok:
        if https_err and "ssl" in https_err.lower():
            return (
                "SSL_ERROR",
                f"TCP port {port} is open but HTTPS/SSL handshake failed: {https_err}",
                "Disable SSL verification (verify=False) in the Settings page.",
            )
        return (
            "HTTP_ERROR",
            f"TCP port {port} is open but HTTPS request failed: {https_err}",
            "Check SAP service status and OData endpoint availability.",
        )

    # Socket failed
    if sock_err and ("refused" in sock_err.lower()):
        return (
            "TCP_REFUSED",
            f"TCP connection refused on {host}:{port}. SAP port is closed or firewall is blocking.",
            "Verify the SAP port number and firewall rules.",
        )

    if sock_err and ("timed out" in sock_err.lower() or "timeout" in sock_err.lower()):
        if is_private:
            return (
                "NETWORK_UNREACHABLE",
                (
                    f"Cannot reach {host}:{port} — TCP connect timed out after {_SOCKET_TIMEOUT}s. "
                    f"The SAP IP {host} is a private/corporate address. "
                    f"The PROD backend (Cloud Foundry) is outside the corporate network and "
                    f"cannot access this IP without a VPN tunnel or network peering."
                ),
                (
                    "Options:\n"
                    "1. Set up a VPN/tunnel from CF to the corporate network\n"
                    "2. Expose SAP via a public reverse proxy with IP allowlisting\n"
                    "3. Use SAP Cloud Connector (SCC) to bridge CF ↔ on-premise SAP"
                ),
            )
        return (
            "TIMEOUT",
            f"TCP connect to {host}:{port} timed out after {_SOCKET_TIMEOUT}s.",
            "Check network routing and firewall rules between CF and SAP.",
        )

    if is_private:
        return (
            "NETWORK_UNREACHABLE",
            (
                f"Cannot reach {host}:{port}. "
                f"The SAP IP {host} is a private/corporate address. "
                f"The PROD backend (Cloud Foundry) is outside the corporate network."
            ),
            (
                "Options:\n"
                "1. Set up a VPN/tunnel from CF to the corporate network\n"
                "2. Expose SAP via a public reverse proxy with IP allowlisting\n"
                "3. Use SAP Cloud Connector (SCC) to bridge CF ↔ on-premise SAP"
            ),
        )

    return (
        "CONNECTION_ERROR",
        f"Cannot reach {host}:{port}. Error: {sock_err}",
        "Check network connectivity and firewall rules.",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/ping-sap",
    response_model=PingSapResponse,
    summary="Diagnostic: test TCP + HTTPS + SSL connectivity to SAP",
    description=(
        "Performs three independent connectivity tests to the SAP host:\n"
        "1. Raw TCP socket connect\n"
        "2. HTTPS request (verify=False)\n"
        "3. SSL certificate validation (verify=True)\n\n"
        "Useful to diagnose whether the PROD backend can reach the corporate SAP subnet.\n"
        "No credentials are required or stored."
    ),
)
async def ping_sap(
    request: Request,
    base_url: Optional[str] = Query(
        None,
        description="SAP base URL to test (e.g. https://<sap-host>:<port>). "
                    "Falls back to X-SAP-Base-URL header, then to .env S4_BASE_URL.",
    ),
) -> PingSapResponse:
    """
    Diagnostic endpoint: test TCP + HTTPS + SSL connectivity to SAP.

    Priority for base_url:
    1. ?base_url= query parameter
    2. X-SAP-Base-URL request header
    3. S4_BASE_URL from .env / settings
    """
    # Resolve base_url
    resolved_url = (
        base_url
        or request.headers.get("X-SAP-Base-URL", "").strip()
        or settings.S4_BASE_URL
        or ""
    )

    if not resolved_url:
        return PingSapResponse(
            base_url="",
            host="",
            port=0,
            socket_reachable=False,
            socket_error="No SAP base URL provided",
            https_reachable=False,
            https_error="No SAP base URL provided",
            error_type="MISSING_URL",
            diagnosis="No SAP base URL configured. Provide ?base_url= or set S4_BASE_URL in .env",
            recommendation="Provide the SAP base URL via query parameter or .env",
        )

    resolved_url = resolved_url.rstrip("/")
    logger.info("ping-sap request | resolved_url=%s", resolved_url)

    # Run blocking tests in thread pool
    result = await asyncio.to_thread(_run_all_tests, resolved_url)
    return result


# ---------------------------------------------------------------------------
# Material debug endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/material/{material_code}",
    summary="Debug: raw SAP payload for a material code",
    description=(
        "Calls A_Product with $filter=Product eq '{material_code}' and "
        "$expand=to_Description. Returns the unmodified SAP JSON so you can "
        "verify exactly where the description is stored.\n\n"
        "Example: GET /api/debug/material/TG11"
    ),
)
def debug_material(material_code: str, request: Request) -> dict:
    """
    Raw SAP inspection for a material code.

    Returns:
      - status_code, ok, url
      - results_count
      - summary: list of {Product, ProductType, BaseUnit, to_Description[]}
      - raw_sap_response: unmodified SAP JSON
    """
    from S4.s4_client import get_s4_session, get_s4_base_url

    session = get_s4_session(request)
    base_url = get_s4_base_url(request)
    url = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product"

    safe_code = material_code.replace("'", "''")
    params = {
        "sap-client": settings.S4_CLIENT,
        "$filter": f"Product eq '{safe_code}'",
        "$expand": "to_Description",
        "$format": "json",
    }

    logger.info(
        "debug_material | code=%r | url=%s | params=%s",
        material_code, url, params,
    )

    try:
        resp = session.get(url, params=params, timeout=15)
        raw_json: Any = None
        try:
            raw_json = resp.json()
        except Exception:
            raw_json = resp.text

        results = []
        if isinstance(raw_json, dict):
            results = raw_json.get("d", {}).get("results", [])

        summary = []
        for product in results:
            prod_code = product.get("Product", "")
            prod_type = product.get("ProductType", "")
            base_unit = product.get("BaseUnit", "")
            to_desc = (product.get("to_Description") or {}).get("results", [])
            desc_entries = [
                {
                    "Language": d.get("Language", ""),
                    "ProductDescription": d.get("ProductDescription", ""),
                }
                for d in to_desc
            ]
            summary.append(
                {
                    "Product": prod_code,
                    "ProductType": prod_type,
                    "BaseUnit": base_unit,
                    "to_Description": desc_entries,
                }
            )
            logger.info(
                "debug_material result | Product=%r | ProductType=%r | BaseUnit=%r | descriptions=%s",
                prod_code, prod_type, base_unit, desc_entries,
            )

        return {
            "status_code": resp.status_code,
            "ok": resp.ok,
            "url": str(resp.url),
            "results_count": len(results),
            "summary": summary,
            "raw_sap_response": raw_json,
        }

    except Exception as exc:
        logger.error("debug_material error | code=%r | error=%s", material_code, exc)
        return {"error": str(exc)}
