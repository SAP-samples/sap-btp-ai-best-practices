"""Verify the graph topology, ReAct loop, structured pass, and concise memory.

Converted from the scaffold's pytest-based test to stdlib unittest so it runs
under ``unittest discover`` without requiring pytest.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from app.deduction_agent.template_agent.config import AgentConfig
from app.deduction_agent.template_agent.mcp import MCPManager
from app.deduction_agent.template_agent.memory import Conversation
from app.deduction_agent.template_agent.runtime import AgentRuntime
from app.deduction_agent.template_agent.skills import SkillLoader


class StructuredAnswer(BaseModel):
    """Represent the schema requested in the test structured pass."""

    answer: str


class RecordingModel(BaseChatModel):
    """Request two skills, observe their ToolMessage, then answer."""

    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        """Return the fake model identifier required by BaseChatModel."""
        return "recording-test-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RecordingModel":
        """Return this fake while retaining the normal bind-tools interface."""
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        """Return a deterministic provider-native formatting stand-in."""
        if isinstance(schema, dict):
            return RunnableLambda(lambda messages: {"answer": "structured-json"})
        return RunnableLambda(lambda messages: StructuredAnswer(answer="structured"))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Emit a tool call first and a final answer after tool execution."""
        self.calls.append(messages)
        if isinstance(messages[-1], ToolMessage):
            assert "SKILL: alpha" in messages[-1].content
            assert "SKILL: beta" in messages[-1].content
            answer = AIMessage(content="skills loaded")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_names": ["alpha", "beta"]},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


class RecordingStore:
    """Provide an in-memory ConversationStore test double."""

    max_messages = 40

    def __init__(self) -> None:
        """Initialize empty storage."""
        self.values: dict[str, Conversation] = {}

    def ensure(self) -> None:
        """Perform no setup."""

    def load(self, context_id: str) -> Conversation:
        """Return a defensive copy of stored turns."""
        return list(self.values.get(context_id, []))

    def save(self, context_id: str, messages: Conversation) -> None:
        """Record a defensive copy of stored turns."""
        self.values[context_id] = list(messages)

    def clear(self, context_id: str) -> None:
        """Delete one context."""
        self.values.pop(context_id, None)

    def close(self) -> None:
        """Close nothing."""


def _skill(root: Path, name: str) -> None:
    """Create a minimal valid skill fixture."""
    directory = root / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n\n# {name}\n",
        encoding="utf-8",
    )


class TestRuntime(unittest.IsolatedAsyncioTestCase):
    """Test the AgentRuntime graph topology, ReAct loop, and structured output."""

    async def test_runtime_executes_batch_skill_react_loop_and_structured_pass(
        self,
    ) -> None:
        """Exercise the complete graph while persisting only user/final turns."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skill_root = tmp_path / "skills"
            skill_root.mkdir()
            _skill(skill_root, "alpha")
            _skill(skill_root, "beta")
            config = AgentConfig.model_validate(
                {
                    "base_prompt": "Base prompt",
                    "model": {"provider": "openai", "name": "fake"},
                    "skills": {"directory": str(skill_root)},
                    "memory": {"enabled": False},
                }
            )
            model = RecordingModel()
            store = RecordingStore()
            runtime = AgentRuntime(
                config,
                model,
                SkillLoader(skill_root),
                MCPManager(),
                store,
            )

            result = await runtime.ainvoke(
                "load what you need",
                "ctx",
                response_model=StructuredAnswer,
            )

            self.assertEqual(result.output_text, "skills loaded")
            self.assertEqual(result.output_parsed, StructuredAnswer(answer="structured"))
            self.assertTrue(
                any(isinstance(message, ToolMessage) for message in result.messages)
            )
            self.assertEqual(
                store.values["ctx"],
                [
                    {"role": "user", "content": "load what you need"},
                    {"role": "assistant", "content": "skills loaded"},
                ],
            )
            first_system = model.calls[0][0].content
            self.assertIn("alpha: alpha description", first_system)
            self.assertIn("beta: beta description", first_system)
            graph = runtime.graph.get_graph()
            self.assertTrue({"load_skill_node", "agent", "tools"}.issubset(graph.nodes))

            json_result = await runtime.ainvoke(
                "load for JSON output",
                "ctx-json",
                response_model={
                    "title": "Answer",
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            )
            self.assertEqual(json_result.output_parsed, {"answer": "structured-json"})


if __name__ == "__main__":
    unittest.main()
