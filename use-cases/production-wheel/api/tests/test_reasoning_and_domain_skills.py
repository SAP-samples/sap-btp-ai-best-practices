"""Verify Claude request controls and required guidance through both model stages."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent.config import ModelSettings, load_config
from app.agent.providers import create_chat_model
from app.agent.runtime import AgentRuntime
from app.agent.skills import SkillLoader, SkillError
from app.agent.memory import NullConversationStore


def test_claude_effort_reaches_converse_payload(monkeypatch):
    """All explicit supported controls survive the SAP constructor and Bedrock serializer."""
    import gen_ai_hub.proxy.langchain.amazon as sdk
    constructor = Mock()
    constructor.get_corresponding_model_id.return_value = 'anthropic.claude-sonnet-4-6'
    monkeypatch.setattr(sdk, 'ChatBedrockConverse', constructor)
    for effort in (None, 'none', 'low', 'medium', 'high', 'max'):
        create_chat_model(ModelSettings(provider='claude', name='anthropic--claude-4.6-sonnet',
                                       reasoning_effort=effort))
        args = constructor.call_args.kwargs
        model = ChatBedrockConverse.model_construct(model_id=args['model'],
            additional_model_request_fields=args.get('additional_model_request_fields'))
        payload = model._converse_params()
        expected = (None if effort is None else {'thinking': {'type': 'disabled'}} if effort == 'none'
            else {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}})
        assert payload.get('additionalModelRequestFields') == expected
    with pytest.raises(ValueError, match='reasoning_effort'):
        create_chat_model(ModelSettings(provider='claude', name='anthropic--claude-4.6-sonnet', reasoning_effort='typo'))
    with pytest.raises(ValueError, match='temperature'):
        create_chat_model(ModelSettings(provider='claude', name='anthropic--claude-4.6-sonnet', reasoning_effort='high', temperature=0.6))


def test_required_skills_reach_agent_and_structured_formatter():
    """Both model requests receive the complete guides even without a load_skill call."""
    seen = []

    class Model(FakeMessagesListChatModel):
        """Capture both invocation stages without a remote provider or tool selection."""

        def bind_tools(self, tools, **kwargs):
            """Use the real graph with a deterministic captured answer."""
            return RunnableLambda(self.answer)

        def answer(self, messages):
            """Capture the initial domain instructions and return a final answer."""
            seen.append(messages)
            return AIMessage(content='The requested bounds contradict each other.')

        def with_structured_output(self, schema, **kwargs):
            """Capture the formatting request rather than contacting a provider."""
            return RunnableLambda(self.format_answer)

        def format_answer(self, messages):
            """Return a small parsed response while retaining the formatter prompt."""
            seen.append(messages)
            return {'rules': []}

    async def exercise():
        """Run the normal graph and prove a missing required guide stops the request."""
        config = load_config(Path(__file__).parents[1] / 'app/agent/config/agent.yaml')
        assert config.skills.required == ['contraint_translation', 'optimization_explanation', 'settings-vocabulary']
        runtime = AgentRuntime(config, Model(responses=[]), SkillLoader(config.skills.directory),
            SimpleNamespace(tools=[]), NullConversationStore(config.memory.max_messages))
        await runtime.ainvoke('Every group must have at least 3 and at most 2 products.',
            'skill-check', session_history=[], response_model=dict)
        assert len(seen) == 2
        for messages in seen:
            prompt = messages[0].content
            assert '===== SKILL: contraint_translation =====' in prompt
            assert '===== SKILL: optimization_explanation =====' in prompt
            assert 'Inclusive per-group OR' in prompt and 'Exact cover' in prompt
        config.skills.required = ['missing-required-guide']
        with pytest.raises(SkillError, match='Unknown skill'):
            await runtime.ainvoke('Translate', 'skill-check', session_history=[])

    asyncio.run(exercise())
