"""Load namespaced LangChain tools from configured MCP servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MCPSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPManager:
    """Own MCP clients, loaded tools, and optional-server diagnostics."""

    tools: list[BaseTool] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)
    _clients: list[MultiServerMCPClient] = field(default_factory=list)

    @classmethod
    async def create(cls, settings: MCPSettings) -> "MCPManager":
        """Connect to enabled servers and return their prefixed tools.

        Required server failures abort startup. Optional failures are retained in
        diagnostics so callers can report degraded capability explicitly.
        """

        manager = cls()
        for name, server in settings.servers.items():
            if not server.enabled:
                manager.diagnostics[name] = "disabled"
                continue
            try:
                client = MultiServerMCPClient(
                    {name: server.connection()},
                    tool_name_prefix=True,
                    handle_tool_errors=True,
                )
                tools = await client.get_tools(server_name=name)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if server.required:
                    raise RuntimeError(f"Required MCP server {name!r} failed: {message}") from exc
                LOGGER.warning("Optional MCP server %s failed: %s", name, message)
                manager.diagnostics[name] = message
                continue
            manager._clients.append(client)
            manager.tools.extend(tools)
            manager.diagnostics[name] = f"connected ({len(tools)} tools)"
        return manager

    async def aclose(self) -> None:
        """Close clients that expose an async close hook and release references."""

        for client in self._clients:
            close = getattr(client, "aclose", None)
            if close:
                await close()
        self._clients.clear()
