"""MCP (Model Context Protocol) server providing basic tools.

This server exposes tools via the MCP protocol. By default, it runs as a subprocess
spawned by the agent via stdio transport. For production deployments, you can run
it as a standalone HTTP server and configure MCP_SERVERS_JSON accordingly.

Running as standalone:
    python -m a2a_langgraph_template.tools.mcp_server

Tools provided:
    - add(a, b): Addition
    - multiply(a, b): Multiplication
    - echo(text): Returns input text
    - get_utc_time(): Current UTC timestamp
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("basic_tools")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


@mcp.tool()
def get_utc_time() -> Dict[str, str]:
    """Get the current UTC time as an ISO 8601 timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat()}


if __name__ == "__main__":
    mcp.run()
