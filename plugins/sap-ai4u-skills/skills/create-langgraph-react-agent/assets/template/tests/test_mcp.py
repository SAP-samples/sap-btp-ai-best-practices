"""Verify the official adapter with a real local stdio MCP process."""

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from template_agent.config import AgentConfig, MCPServerSettings, MCPSettings
from template_agent.mcp import MCPManager
from template_agent.memory import Conversation
from template_agent.runtime import AgentRuntime
from template_agent.skills import SkillLoader


class MCPReactModel(BaseChatModel):
    """Call the local MCP echo tool once and then provide a final response."""

    @property
    def _llm_type(self) -> str:
        """Return the fake model identifier required by BaseChatModel."""

        return "mcp-react-test-model"

    def bind_tools(self, tools: object, **kwargs: object) -> "MCPReactModel":
        """Retain the normal bind-tools interface for graph compilation."""

        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        """Emit one namespaced MCP tool call followed by a grounded answer."""

        if isinstance(messages[-1], ToolMessage):
            assert "hello-react" in str(messages[-1].content)
            response = AIMessage(content="echoed hello-react")
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "local_echo",
                        "args": {"text": "hello-react"},
                        "id": "mcp-echo",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


class LocalConversationStore:
    """Provide local conversation storage for the MCP graph test."""

    max_messages = 40

    def __init__(self) -> None:
        """Initialize empty in-process storage."""

        self.values: dict[str, Conversation] = {}

    def load(self, context_id: str) -> Conversation:
        """Load one context's turns."""

        return list(self.values.get(context_id, []))

    def save(self, context_id: str, messages: Conversation) -> None:
        """Save one context's turns."""

        self.values[context_id] = list(messages)

    def clear(self, context_id: str) -> None:
        """Clear one context's turns."""

        self.values.pop(context_id, None)

    def close(self) -> None:
        """Close no external resources."""


def _stdio_settings() -> MCPSettings:
    """Return settings for the repository's real local echo MCP server."""

    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    return MCPSettings(
        servers={
            "local": MCPServerSettings(
                transport="stdio",
                command=sys.executable,
                args=[str(server)],
            )
        }
    )


@pytest.mark.asyncio
async def test_loads_and_invokes_namespaced_stdio_tool() -> None:
    """Discover and call a real MCP tool through MultiServerMCPClient."""

    manager = await MCPManager.create(_stdio_settings())
    try:
        assert manager.diagnostics["local"] == "connected (1 tools)"
        assert [tool.name for tool in manager.tools] == ["local_echo"]
        result = await manager.tools[0].ainvoke({"text": "hello"})
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "hello"
    finally:
        await manager.aclose()


@pytest.mark.asyncio
async def test_stdio_tool_runs_through_complete_react_loop(tmp_path: Path) -> None:
    """Exercise agent -> real stdio MCP tool -> agent through ToolNode."""

    skills = tmp_path / "skills"
    skills.mkdir()
    config = AgentConfig.model_validate(
        {
            "base_prompt": "Use the echo tool.",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skills)},
            "memory": {"enabled": False},
        }
    )
    manager = await MCPManager.create(_stdio_settings())
    runtime = AgentRuntime(
        config,
        MCPReactModel(),
        SkillLoader(skills),
        manager,
        LocalConversationStore(),
    )
    try:
        result = await runtime.ainvoke("Echo the test value.", "mcp-local")
        assert result.output_text == "echoed hello-react"
        assert any(isinstance(message, ToolMessage) for message in result.messages)
    finally:
        await runtime.aclose()
