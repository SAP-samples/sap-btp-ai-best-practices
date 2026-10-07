"""Guard the structured-output transport that OpenAI accepts for plant-rule interpretation.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_structured_output_transport.py -q

OpenAI's native response-format schema rejects ``oneOf``, which Pydantic emits for the
discriminated ``BusinessConstraint`` union inside ``Interpretation.rules``. The runtime
must therefore request tool-call structured output for OpenAI models.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.agent.config import load_config
from app.agent.memory import NullConversationStore
from app.agent.runtime import AgentRuntime, _structured_output_kwargs
from app.agent.skills import SkillLoader
from app.workspace.profile_interpretation import Interpretation


def test_interpretation_schema_still_needs_tool_transport():
    """The rule union is emitted as oneOf, which is why the default OpenAI path fails."""
    items = Interpretation.model_json_schema()["properties"]["rules"]["items"]
    assert "oneOf" in items


def test_openai_payload_uses_forced_tool_not_response_format():
    """The real OpenAI Responses payload carries the schema as tool parameters."""
    model = ChatOpenAI(model="gpt-5.4", api_key="offline-test", use_responses_api=True)
    bound = model.with_structured_output(Interpretation, **_structured_output_kwargs("openai")).first
    payload = model._get_request_payload([("user", "interpret")], **bound.kwargs)
    assert "text_format" not in payload and "text" not in payload
    assert [tool["name"] for tool in payload["tools"]] == ["Interpretation"]
    assert payload["tool_choice"] == {"type": "function", "name": "Interpretation"}
    assert "oneOf" in json.dumps(payload["tools"][0]["parameters"])


def test_runtime_selects_transport_per_provider():
    """OpenAI formatters use function calling; other providers keep the library default."""
    seen = []

    class Model(FakeMessagesListChatModel):
        """Capture the formatter request without contacting a provider."""

        def bind_tools(self, tools, **kwargs):
            """Answer the agent stage directly."""
            return RunnableLambda(lambda _messages: AIMessage(content="No rules."))

        def with_structured_output(self, schema, **kwargs):
            """Record the transport keywords the runtime chose."""
            seen.append(kwargs)
            return RunnableLambda(lambda _messages: {"rules": []})

    async def exercise(provider):
        """Run one structured invocation for the given provider."""
        config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
        config.model.provider = provider
        runtime = AgentRuntime(config, Model(responses=[]), SkillLoader(config.skills.directory),
            SimpleNamespace(tools=[]), NullConversationStore(config.memory.max_messages))
        await runtime.ainvoke("Translate", "transport-check", session_history=[], response_model=dict)

    asyncio.run(exercise("openai"))
    asyncio.run(exercise("claude"))
    assert seen == [{"method": "function_calling"}, {}]
