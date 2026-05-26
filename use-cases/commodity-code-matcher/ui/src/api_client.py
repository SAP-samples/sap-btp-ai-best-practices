"""
HTTP client wrappers for talking to the FastAPI backend.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional

import requests

from .utils import get_api_base_url

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}


def _build_url(path: str) -> str:
    """Build an absolute backend URL from a relative API path.

    Args:
        path: API path such as ``/api/health``.

    Returns:
        Absolute URL using ``API_BASE_URL``.
    """

    base = get_api_base_url().rstrip("/")
    return f"{base}{path}"


def _auth_headers() -> Dict[str, str]:
    """Return API-key headers when the UI is configured with ``API_KEY``.

    Returns:
        Header dictionary for authenticated backend requests, or an empty
        dictionary for local unauthenticated development.
    """

    api_key = os.getenv("API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


def _error_payload(exc: Exception, response: Optional[requests.Response] = None) -> Dict:
    """Build a consistent UI error payload from a failed HTTP request.

    Args:
        exc: Exception raised by ``requests``.
        response: Optional response object that may contain JSON details.

    Returns:
        Dictionary with ``error``, optional ``status``, and optional raw body.
    """

    error_body = None
    message: object = str(exc)
    if response is not None:
        try:
            error_body = response.json()
            message = error_body.get("detail") if isinstance(error_body, dict) else error_body
        except Exception:
            message = str(exc)
    status_code = getattr(response, "status_code", None)
    return {"error": message, "status": status_code, "raw_error": error_body}


def healthcheck() -> Dict:
    """Call the backend health endpoint.

    Returns:
        Health payload from the API, or an error payload for display.
    """

    try:
        resp = requests.get(_build_url("/api/health"), headers=_auth_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Healthcheck failed")
        return {"error": str(exc)}


def get_job_status(job_id: str) -> Dict:
    """Fetch current status for an extraction job.

    Args:
        job_id: Identifier returned by ``POST /api/extraction/run``.

    Returns:
        Job status payload, or an error payload for display.
    """

    try:
        resp = requests.get(_build_url(f"/api/extraction/jobs/{job_id}"), headers=_auth_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        logger.exception("Job status request failed")
        return _error_payload(exc, getattr(exc, "response", None))
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Unexpected error while polling extraction job")
        return {"error": str(exc)}


def run_extraction(
    files: List,
    *,
    on_status: Optional[Callable[[Dict], None]] = None,
    poll_interval_seconds: float = 2,
    slow_poll_interval_seconds: float = 5,
    slow_after_seconds: float = 60,
) -> Dict:
    """Submit PDFs and poll until the extraction job reaches a terminal state.

    Args:
        files: Streamlit uploaded PDF objects.
        on_status: Optional callback invoked after every polling response.
        poll_interval_seconds: Poll interval used during the first minute.
        slow_poll_interval_seconds: Poll interval used after the first minute.
        slow_after_seconds: Elapsed seconds before switching to slow polling.

    Returns:
        Final job status payload, or an error payload for display.
    """

    if not files:
        raise ValueError("Please upload at least one PDF.")

    file_payload = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]
    data = {
        "llm_verify": "true",
        "llm_model": "gpt-4.1",
    }
    try:
        resp = requests.post(
            _build_url("/api/extraction/run"),
            files=file_payload,
            data=data,
            headers=_auth_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        submission = resp.json()
    except requests.HTTPError as exc:
        logger.exception("Extraction request failed")
        return _error_payload(exc, getattr(exc, "response", None))
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Unexpected error during extraction request")
        return {"error": str(exc)}

    job_id = submission.get("job_id")
    if not job_id:
        return {"error": "Extraction API did not return a job ID.", "raw_error": submission}

    started_at = time.monotonic()
    while True:
        status_payload = get_job_status(job_id)
        if on_status is not None:
            on_status(status_payload)
        if status_payload.get("error"):
            return status_payload

        job_status = str(status_payload.get("status", "")).upper()
        if job_status in TERMINAL_STATUSES:
            return status_payload

        elapsed = time.monotonic() - started_at
        interval = slow_poll_interval_seconds if elapsed >= slow_after_seconds else poll_interval_seconds
        time.sleep(interval)


def download_output(job_id: str) -> Optional[bytes]:
    """Retrieve the generated Excel file for a completed job.

    Args:
        job_id: Identifier returned by the extraction submission endpoint.

    Returns:
        Raw Excel bytes, or ``None`` if the download fails.
    """

    try:
        resp = requests.get(
            _build_url(f"/api/extraction/jobs/{job_id}/download"),
            headers=_auth_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Failed to download output")
        return None
