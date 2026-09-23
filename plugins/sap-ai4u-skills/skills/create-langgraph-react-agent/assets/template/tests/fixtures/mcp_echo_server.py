"""Run a tiny read-only stdio MCP server for integration tests.

Example:
    .venv/bin/python tests/fixtures/mcp_echo_server.py
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("template-agent-test")


@server.tool()
def echo(text: str) -> str:
    """Return the supplied text unchanged."""

    return text


if __name__ == "__main__":
    server.run(transport="stdio")
