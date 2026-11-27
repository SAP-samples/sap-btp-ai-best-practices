"""Regressions for provider rejection of replayed encrypted reasoning items.

Run:
    cd api && ../.venv/bin/python -m pytest tests/test_reasoning_replay.py -q
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.agent.config import load_config
from app.agent.memory import NullConversationStore
from app.agent.runtime import AgentRuntime, _is_reasoning_replay_error, _without_reasoning
from app.agent.skills import SkillLoader

REASONING = {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}


class ReplayRejected(Exception):
    """Mimic openai.BadRequestError attributes for invalid_encrypted_content."""

    status_code = 400
    code = "invalid_encrypted_content"


def test_only_invalid_encrypted_content_is_recognized():
    """Other 400s and other codes keep their normal failure path."""
    assert _is_reasoning_replay_error(ReplayRejected())
    assert not _is_reasoning_replay_error(SimpleNamespace(status_code=400, code="other"))
    assert not _is_reasoning_replay_error(ValueError("boom"))


def test_without_reasoning_keeps_text_tool_calls_and_ids():
    """Reasoning blocks and kwargs are removed; untouched messages stay identical."""
    human = HumanMessage(content="hi")
    plain = AIMessage(content="answer", id="a0")
    reasoned = AIMessage(
        content=[REASONING, {"type": "text", "text": "ok"}],
        additional_kwargs={"reasoning": {"id": "rs_2"}, "other": 1},
        tool_calls=[{"id": "c1", "name": "ping", "args": {}}],
        id="a1",
    )
    out = _without_reasoning([human, plain, reasoned])
    assert out[0] is human and out[1] is plain
    assert out[2].content == [{"type": "text", "text": "ok"}]
    assert out[2].additional_kwargs == {"other": 1}
    assert (out[2].id, out[2].tool_calls[0]["id"]) == ("a1", "c1")


def test_graph_retries_once_without_replayed_reasoning():
    """A rejected replay is retried without reasoning and is not resent later in the turn."""

    @tool
    def ping() -> str:
        """Return a fixed value so the graph takes a second model step."""
        return "pong"

    class Model:
        """Scripted model: tool call, one replay rejection, then two final steps."""

        def __init__(self):
            self.calls = []
            self.script = [
                AIMessage(content=[REASONING], tool_calls=[{"id": "c1", "name": "ping", "args": {}}], id="a1"),
                ReplayRejected(),
                AIMessage(content=[REASONING], tool_calls=[{"id": "c2", "name": "ping", "args": {}}], id="a2"),
                AIMessage(content="Done", id="a3"),
            ]

        def bind_tools(self, tools, **kwargs):
            """Bind no provider; keep the scripted sequence."""
            return self

        async def ainvoke(self, messages, *args, **kwargs):
            """Record the request and return or raise the next scripted item."""
            self.calls.append(list(messages))
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    def reasoning_count(messages):
        """Count AI messages that still carry a reasoning block."""
        return sum(
            isinstance(m, AIMessage) and isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "reasoning" for b in m.content)
            for m in messages
        )

    async def exercise():
        """Run the real graph with the scripted model."""
        config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
        model = Model()
        runtime = AgentRuntime(
            config, model, SkillLoader(config.skills.directory),
            SimpleNamespace(tools=[]), NullConversationStore(config.memory.max_messages), [ping],
        )
        result = await runtime.ainvoke("go", "replay-test", session_history=[])
        assert result.output_text == "Done"
        # Calls: first step, rejected step, its retry, final step after the second tool.
        assert len(model.calls) == 4
        assert reasoning_count(model.calls[1]) == 1
        assert reasoning_count(model.calls[2]) == 0
        # a1 was rewritten in state; only the new a2 reasoning is replayed afterwards.
        assert reasoning_count(model.calls[3]) == 1

    asyncio.run(exercise())
