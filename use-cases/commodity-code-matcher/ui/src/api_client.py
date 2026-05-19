"""
HTTP client wrappers for talking to the FastAPI backend.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

from .utils import get_api_base_url

logger = logging.getLogger(__name__)


def _build_url(path: str) -> str:
    base = get_api_base_url().rstrip("/")
    return f"{base}{path}"


def healthcheck() -> Dict:
    try:
        resp = requests.get(_build_url("/api/health"), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Healthcheck failed")
        return {"error": str(exc)}


def run_extraction(files: List) -> Dict:
    """Send PDFs to the extraction endpoint using default flags."""

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
            timeout=None,  # potentially long-running
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        logger.exception("Extraction request failed")
        error_body = None
        try:
            error_body = resp.json()  # type: ignore[name-defined]
            message = error_body.get("detail") if isinstance(error_body, dict) else error_body
        except Exception:
            message = str(exc)
        status_code = getattr(exc.response, "status_code", None)
        return {"error": message, "status": status_code, "raw_error": error_body}
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Unexpected error during extraction request")
        return {"error": str(exc)}


def download_output(download_path: str) -> Optional[bytes]:
    """Retrieve the generated Excel file."""

    try:
        resp = requests.get(_build_url("/api/extraction/download"), params={"path": download_path}, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # pragma: no cover - UI helper
        logger.exception("Failed to download output")
        return None
