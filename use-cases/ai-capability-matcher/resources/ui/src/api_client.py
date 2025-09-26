import os
import json
import math
from typing import Dict, Any, Any as AnyType

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

"""
HTTP client for the Streamlit UI to talk to the backend API.

This module centralizes request settings like base URL and timeouts. For large
matching jobs that generate embeddings and call LLMs, default HTTP timeouts are
often insufficient. We therefore make long read timeouts configurable via env.

Environment variables (optional):
- API_BASE_URL: Base URL for the backend (default: http://localhost:8000)
- API_KEY:      API key header value, if the API requires it
- API_TIMEOUT_SECONDS: Total timeout used as read timeout if specific values
  are not supplied (default: 900 seconds)
- API_CONNECT_TIMEOUT_SECONDS: Connection timeout (default: 10 seconds)
- API_READ_TIMEOUT_SECONDS: Read timeout (default: equals API_TIMEOUT_SECONDS)
"""

# Load environment variables from .env file (when present)
load_dotenv()

# API base configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

# Timeout configuration
# Use a short connect timeout to fail fast on network issues, but a very long
# read timeout to allow the backend to process large datasets.
_default_total_timeout = float(os.getenv("API_TIMEOUT_SECONDS", "900"))
_connect_timeout = float(os.getenv("API_CONNECT_TIMEOUT_SECONDS", "10"))
_read_timeout = float(os.getenv("API_READ_TIMEOUT_SECONDS", str(_default_total_timeout)))

# Requests accepts a (connect, read) timeout tuple
REQUEST_TIMEOUT = (_connect_timeout, _read_timeout)


def _sanitize_json(value: AnyType) -> AnyType:
    """Recursively convert payload into JSON-safe values.

    - Replaces NaN/NaT/None-like with None
    - Replaces +/-Inf with None
    - Converts numpy scalar types to native Python types
    """

    # Handle dict
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}

    # Handle list/tuple
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]

    # Handle numpy scalars
    if isinstance(value, (np.generic,)):
        value = value.item()

    # Handle floats (including numpy floats after item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # Handle pandas NA-like via numpy/pandas checks without importing pandas
    try:
        # np.isnan for non-numeric raises, so protect it
        if value is not None and hasattr(np, "isnan"):
            if isinstance(value, (np.floating,)) and np.isnan(value):
                return None
    except Exception:
        pass

    # Leave other primitives as-is
    return value


def make_api_request(
    endpoint: str,
    method: str = "POST",
    payload: Dict[str, Any] | None = None,
    timeout: tuple[float, float] | float | None = None,
) -> Dict[str, Any]:
    """
    Make a request to the backend API.

    Args:
        endpoint: The API endpoint to call (e.g., "/chat/openai").
        method: The HTTP method to use ("GET", "POST", etc.).
        payload: The JSON payload to send with the request.
        timeout: Optional timeout override passed to requests. Defaults to
                 REQUEST_TIMEOUT (connect, read) if not provided.

    Returns:
        A dictionary containing the JSON response from the API.
    """
    api_url = f"{API_BASE_URL}{endpoint}"
    effective_timeout = timeout if timeout is not None else REQUEST_TIMEOUT

    try:
        if method.upper() == "POST":
            safe_payload = _sanitize_json(payload or {})
            data = json.dumps(safe_payload, allow_nan=False)
            response = requests.post(
                api_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": API_KEY,
                },
                timeout=effective_timeout,
            )
        else:
            response = requests.get(api_url, timeout=effective_timeout)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {response.text}",
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot connect to API server at {api_url}. Make sure the API service is running.",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out.",
        }
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}
