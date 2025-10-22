from __future__ import annotations

import os
import logging
import urllib3
import requests
from typing import Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def _get_api_base_url() -> str:
    """Resolve API base URL from environment with a reasonable default.

    Returns:
        Base URL string like "https://host" or "http://localhost:8000".
    """
    return os.getenv("API_BASE_URL", "http://localhost:8000")


def _get_api_key() -> Optional[str]:
    """Fetch API key from environment at call time.

    Returns:
        The API key string if present; otherwise None.
    """
    return os.getenv("API_KEY")


def _get_ssl_verify() -> bool:
    """Resolve SSL verification behavior from env.

    Returns:
        True to verify SSL certs (default), False to disable verification.
    """
    val = os.getenv("SSL_VERIFY")
    if val is None:
        return True
    return str(val).strip().lower() not in {"false", "0", "no"}


def make_api_request(
    endpoint: str,
    method: str = "POST",
    payload: Dict[str, Any] | None = None,
    timeout_s: int | None = None,
) -> Dict[str, Any]:
    """
    Make a request to the backend API.

    Args:
        endpoint: The API endpoint to call (e.g., "/chat/openai").
        method: The HTTP method to use ("GET", "POST", etc.).
        payload: The JSON payload to send with the request.
        timeout_s: The timeout in seconds for the request.

    Returns:
        A dictionary containing the JSON response from the API.
    """
    api_base = _get_api_base_url()
    api_key = _get_api_key()
    ssl_verify = _get_ssl_verify()
    api_url = f"{api_base}{endpoint}"

    if not api_key:
        # Surface a clear error if API key is missing in the UI runtime
        logger.warning("API client missing API_KEY env var; refusing request to %s", endpoint)
        return {
            "success": False,
            "error": "API_KEY is not set in UI environment; cannot authenticate to backend API.",
        }

    try:
        if method.upper() == "POST":
            response = requests.post(
                api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                timeout=timeout_s or 1200,
                verify=ssl_verify,
            )
        else:
            response = requests.get(
                api_url,
                headers={
                    "X-API-Key": api_key,
                },
                timeout=timeout_s or 10,
                verify=ssl_verify,
            )

        if response.status_code == 200:
            return response.json()
        else:
            # Try to extract error detail from API body if available
            detail: Optional[str] = None
            try:
                body = response.json()
                detail = body.get("detail") if isinstance(body, dict) else None
            except Exception:
                detail = None
            return {
                "success": False,
                "error": f"API request failed ({response.status_code}). {detail or response.text}",
            }
    except requests.exceptions.SSLError as e:
        return {
            "success": False,
            "error": f"SSL Certificate verification failed. Set SSL_VERIFY=false to bypass. Error: {str(e)}",
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


def list_suppliers() -> Dict[str, Any]:
    """Call GET /api/v1/kg/static/list to fetch supplier IDs and names.

    Uses an extended timeout to better handle cold starts on Cloud Foundry.
    """
    return make_api_request("/api/v1/kg/static/list", method="GET", timeout_s=60)


def get_supplier_kg(supplier_id: str) -> Dict[str, Any]:
    """Call GET /api/v1/kg/static/get/{id} to fetch the KG JSON (when needed client-side)."""
    return make_api_request(f"/api/v1/kg/static/get/{supplier_id}", method="GET", timeout_s=60)


def ask_chat(question: str, supplier1_id: str, supplier2_id: str, model: str | None = None, supplier1_name: str | None = None, supplier2_name: str | None = None) -> Dict[str, Any]:
    """Call POST /api/v1/chat/ask with supplier IDs to get consolidated markdown answer."""

    anonymization_prompt = "Please anonymize supplier names consistently as 'SupplierA' and 'SupplierB' in the response. Do not include real supplier names in the response."
    payload: Dict[str, Any] = {
        "question": f"{anonymization_prompt} {question}",
        "supplier1": {"id": supplier1_id},
        "supplier2": {"id": supplier2_id},
    }
    if supplier1_name:
        payload["supplier1"]["name"] = supplier1_name
    if supplier2_name:
        payload["supplier2"]["name"] = supplier2_name
    if model:
        payload["model"] = model
    # Chat can take longer due to multiple LLM calls
    return make_api_request("/api/v1/chat/ask", method="POST", payload=payload, timeout_s=180)


# Analyzer helpers

def analyze_cost(supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    return make_api_request("/api/v1/analyze/cost", payload={"supplier_id": supplier_id, "kg_path": kg_path, "model": model})


def analyze_risk(supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    return make_api_request("/api/v1/analyze/risk", payload={"supplier_id": supplier_id, "kg_path": kg_path, "model": model})


def analyze_parts(supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None, core_part_categories: Optional[list[str]] = None) -> Dict[str, Any]:
    return make_api_request(
        "/api/v1/analyze/parts",
        payload={"supplier_id": supplier_id, "kg_path": kg_path, "model": model, "core_part_categories": core_part_categories},
    )


def analyze_homepage(supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    return make_api_request("/api/v1/analyze/homepage", payload={"supplier_id": supplier_id, "kg_path": kg_path, "model": model})


def analyze_tqdcs(supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None, prior_analyses: Optional[Dict[str, Any]] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    return make_api_request(
        "/api/v1/analyze/tqdcs",
        payload={"supplier_id": supplier_id, "kg_path": kg_path, "model": model, "prior_analyses": prior_analyses, "weights": weights},
    )


def analyze_compare(
    supplier1_name: str,
    supplier2_name: str,
    supplier1_analyses: Optional[Dict[str, Any]] = None,
    supplier2_analyses: Optional[Dict[str, Any]] = None,
    tqdcs_weights: Optional[Dict[str, float]] = None,
    generate_metrics: bool = True,
    generate_strengths_weaknesses: bool = True,
    generate_recommendation_and_split: bool = True,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    # Call compare endpoint
    response: Dict[str, Any] = make_api_request(
        "/api/v1/analyze/compare",
        payload={
            "supplier1_name": supplier1_name,
            "supplier2_name": supplier2_name,
            "supplier1_analyses": supplier1_analyses,
            "supplier2_analyses": supplier2_analyses,
            "tqdcs_weights": tqdcs_weights,
            "generate_metrics": generate_metrics,
            "generate_strengths_weaknesses": generate_strengths_weaknesses,
            "generate_recommendation_and_split": generate_recommendation_and_split,
            "model": model,
        },
    )

    # Normalize shape: backend may return the comparison object directly
    # while callers expect { "comparison": {...} }. Only wrap successful
    # dict responses that do not already include a 'comparison' key.
    try:
        if isinstance(response, dict) and response.get("success", True) is not False:
            if "comparison" not in response:
                return {"comparison": response}
    except Exception:
        # In case of unexpected shapes, fall through and return as-is
        pass
    return response


def analyze_complete(
    supplier1_id: str,
    supplier2_id: str,
    model: Optional[str] = None,
    comparator_model: Optional[str] = None,
    core_part_categories: Optional[list[str]] = None,
    tqdcs_weights: Optional[Dict[str, float]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    # Complete orchestration may be long-running; extend timeout generously
    return make_api_request(
        "/api/v1/analyze/complete",
        payload={
            "supplier1_id": supplier1_id,
            "supplier2_id": supplier2_id,
            "model": model,
            "comparator_model": comparator_model,
            "core_part_categories": core_part_categories,
            "tqdcs_weights": tqdcs_weights,
            "force_refresh": force_refresh,
        },
        timeout_s=900,
    )


def analyze_ensure(
    supplier1_id: str,
    supplier2_id: str,
    model: Optional[str] = None,
    comparator_model: Optional[str] = None,
    core_part_categories: Optional[list[str]] = None,
    tqdcs_weights: Optional[Dict[str, float]] = None,
    force_refresh: bool = False,
    generate_metrics: bool = True,
    generate_strengths_weaknesses: bool = True,
    generate_recommendation_and_split: bool = True,
) -> Dict[str, Any]:
    return make_api_request(
        "/api/v1/analyze/ensure",
        payload={
            "supplier1_id": supplier1_id,
            "supplier2_id": supplier2_id,
            "model": model,
            "comparator_model": comparator_model,
            "core_part_categories": core_part_categories,
            "tqdcs_weights": tqdcs_weights,
            "force_refresh": force_refresh,
            "generate_metrics": generate_metrics,
            "generate_strengths_weaknesses": generate_strengths_weaknesses,
            "generate_recommendation_and_split": generate_recommendation_and_split,
        },
        timeout_s=900,
    )


def analyze_cache_status(
    supplier1_id: str,
    supplier2_id: str,
    core_part_categories: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return make_api_request(
        "/api/v1/analyze/cache_status",
        payload={
            "supplier1_id": supplier1_id,
            "supplier2_id": supplier2_id,
            "core_part_categories": core_part_categories,
        },
        timeout_s=60,
    )


def analyze_supplier_full(
    supplier_id: str,
    model: Optional[str] = None,
    core_part_categories: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    return make_api_request(
        "/api/v1/analyze/supplier_full",
        payload={
            "supplier_id": supplier_id,
            "model": model,
            "core_part_categories": core_part_categories,
            "force_refresh": force_refresh,
        },
        timeout_s=600,
    )
