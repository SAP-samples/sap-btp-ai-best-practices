"""Tool registry for the A2A LangGraph agent.

This module aggregates tools from three sources:
1. Custom tools (tools/custom.py) - Local LangGraph tools
2. External API tools (tools/external.py) - HTTP clients calling external services
3. MCP tools (tools/mcp_server.py) - Tools via Model Context Protocol

The main entry point is `get_all_tools()` which returns all available tools.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from .custom import CUSTOM_TOOLS
from .external import EXTERNAL_TOOLS


def _default_mcp_config() -> Dict[str, Any]:
    """Default MCP configuration using stdio transport.

    Spawns the local MCP server (tools/mcp_server.py) as a subprocess.
    This is suitable for development and single-instance deployments.

    Returns:
        MCP client configuration dictionary.
    """
    mcp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))
    return {
        "local_tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [mcp_path],
        }
    }


def _load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from environment or use default.

    If MCP_SERVERS_JSON is set, parses it as JSON configuration.
    Otherwise, uses the default local stdio configuration.

    Returns:
        MCP client configuration dictionary.
    """
    config_json = os.getenv("MCP_SERVERS_JSON")
    if not config_json:
        return _default_mcp_config()
    return json.loads(config_json)


def _mcp_enabled() -> bool:
    """Check if MCP tools are enabled."""
    return os.getenv("MCP_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


_MCP_CLIENT: Optional[MultiServerMCPClient] = None
_MCP_TOOLS: Optional[List[Any]] = None
_MCP_LOCK = asyncio.Lock()


async def load_mcp_tools() -> List[Any]:
    """Load tools from configured MCP servers.

    Tools are loaded once and cached. The MCP client connection is maintained
    for the lifetime of the process.

    Returns:
        List of LangChain-compatible tool objects from MCP servers.
    """
    global _MCP_CLIENT, _MCP_TOOLS
    if not _mcp_enabled():
        return []
    async with _MCP_LOCK:
        if _MCP_TOOLS is not None:
            return _MCP_TOOLS
        config = _load_mcp_config()
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        _MCP_CLIENT = client
        _MCP_TOOLS = tools
        return tools


async def get_all_tools() -> List[Any]:
    """Get all available tools from all sources.

    Combines:
    - Custom tools (calculator, etc.)
    - External API tools (HTTP clients)
    - MCP tools (if enabled)

    Returns:
        List of all LangChain-compatible tool objects.
    """
    tools = list(CUSTOM_TOOLS)
    tools.extend(EXTERNAL_TOOLS)
    tools.extend(await load_mcp_tools())
    return tools
