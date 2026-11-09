"""External API tools.

This module contains tools that call external HTTP services. These tools act as
HTTP clients - the actual API servers run separately (potentially on different
infrastructure).

Configuration:
    EXTERNAL_API_BASE_URL: Base URL for the external API (default: http://localhost:8001)
    EXTERNAL_API_TIMEOUT: HTTP timeout in seconds (default: 10)

Example:
    To add a new external API tool:

    @tool("myservice.action")
    async def my_service_tool(param: str) -> Dict[str, Any]:
        '''Description for the LLM.'''
        result = await _http_get_async("/api/my-endpoint", {"param": param})
        return {"result": result}
"""
from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from langchain_core.tools import tool


_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "http://localhost:8001").rstrip("/")
_HTTP_TIMEOUT = float(os.getenv("EXTERNAL_API_TIMEOUT", "10"))


async def _http_get_async(path: str, params: Dict[str, Any]) -> Any:
    """Make an async HTTP GET request to the external API.

    Args:
        path: API endpoint path (e.g., "/api/echo").
        params: Query parameters to send.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: If the response status is 4xx or 5xx.
    """
    url = f"{_API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _http_post_async(path: str, data: Dict[str, Any]) -> Any:
    """Make an async HTTP POST request to the external API.

    Args:
        path: API endpoint path (e.g., "/api/submit").
        data: JSON body to send.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: If the response status is 4xx or 5xx.
    """
    url = f"{_API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()


@tool("external_echo")
async def external_echo_tool(text: str) -> Dict[str, Any]:
    """Call the external API echo endpoint and return its response.

    This is an example tool demonstrating how to call external HTTP services.
    The actual echo server runs separately (see external_api_example/).
    """
    payload = await _http_get_async("/api/echo", {"text": text})
    return {"echo": payload}


# List of all external API tools to be registered with the agent
EXTERNAL_TOOLS = [external_echo_tool]
