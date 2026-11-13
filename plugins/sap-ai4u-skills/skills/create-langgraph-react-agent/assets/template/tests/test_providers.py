"""Guard self-contained SAP AI Core Responses history serialization.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_providers.py -q
"""

from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from template_agent.config import ModelSettings
from template_agent.providers import ResponsesCompatibleChatOpenAI, create_chat_model


def test_responses_replay_keeps_content_and_call_links_without_storage_ids():
    """A real serializer can replay reasoning and multiple calls without stored items."""
    model = ResponsesCompatibleChatOpenAI.model_construct(
        model_name="gpt-6.1-sol", proxy_model_name="gpt-6.1-sol",
        use_responses_api=True, store=False, include=["reasoning.encrypted_content"], n=1,
    )
    assistant = AIMessage(
        id="resp_example", response_metadata={"id": "resp_example"},
        content=[
            {"type": "reasoning", "id": "rs_example", "summary": [],
             "encrypted_content": "opaque"},
            {"type": "text", "id": "msg_example", "text": "Checking both lines."},
            {"type": "function_call", "id": "fc_one", "call_id": "call_one",
             "name": "lookup", "arguments": '{"line":1}'},
            {"type": "function_call", "id": "fc_two", "call_id": "call_two",
             "name": "lookup", "arguments": '{"line":2}'},
        ],
        tool_calls=[
            {"id": "call_one", "name": "lookup", "args": {"line": 1}},
            {"id": "call_two", "name": "lookup", "args": {"line": 2}},
        ],
    )
    messages = [
        HumanMessage(content="Check the lines."), assistant,
        ToolMessage(content="ordinary", tool_call_id="call_one"),
        ToolMessage(content="high speed", tool_call_id="call_two"),
    ]
    payload = model._get_request_payload(messages)
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload and "n" not in payload
    assert all("id" not in item for item in payload["input"])
    reasoning = next(i for i in payload["input"] if i["type"] == "reasoning")
    assert reasoning == {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    text = next(i for i in payload["input"] if i.get("role") == "assistant")
    assert text["content"][0]["text"] == "Checking both lines."
    calls = [i for i in payload["input"] if i["type"] == "function_call"]
    outputs = [i for i in payload["input"] if i["type"] == "function_call_output"]
    assert [(i["call_id"], i["arguments"]) for i in calls] == [
        ("call_one", '{"line":1}'), ("call_two", '{"line":2}'),
    ]
    assert [(i["call_id"], i["output"]) for i in outputs] == [
        ("call_one", "ordinary"), ("call_two", "high speed"),
    ]
    # Strip only the outgoing payload: graph message IDs are still needed by its
    # reducer, and stored call IDs must remain available for future tool steps.
    assert assistant.id == "resp_example"
    assert assistant.content[0]["id"] == "rs_example"
    assert assistant.content[2]["id"] == "fc_one"


def test_responses_without_encrypted_reasoning_drops_bare_reasoning_reference():
    """Older provider responses without opaque content require no server-side lookup."""
    model = ResponsesCompatibleChatOpenAI.model_construct(
        model_name="gpt-6.1-sol", proxy_model_name="gpt-6.1-sol",
        use_responses_api=True, store=False,
    )
    payload = model._get_request_payload([
        HumanMessage(content="hello"),
        AIMessage(content=[{"type": "reasoning", "id": "rs_old", "summary": []},
                           {"type": "text", "id": "msg_old", "text": "Ready"}]),
    ])
    assert [i["type"] for i in payload["input"]] == ["message", "message"]
    assert payload["input"][1]["content"][0]["text"] == "Ready"


def test_factory_requests_stateless_reasoning_only_for_responses(monkeypatch):
    """Agent, formatter and summarizer share the same stateless model construction."""
    from template_agent import providers

    constructor = Mock()
    monkeypatch.setattr(providers, "ResponsesCompatibleChatOpenAI", constructor)
    create_chat_model(ModelSettings(provider="openai", name="gpt-6.1-sol",
                                    reasoning_effort="high"))
    assert constructor.call_args.kwargs["store"] is False
    assert constructor.call_args.kwargs["include"] == ["reasoning.encrypted_content"]
    assert constructor.call_args.kwargs["reasoning_effort"] == "high"
    create_chat_model(ModelSettings(provider="openai", name="gpt-6.1-sol",
                                    use_responses_api=False))
    assert "store" not in constructor.call_args.kwargs
    assert "include" not in constructor.call_args.kwargs


def test_chat_completions_serialization_keeps_normal_parameters():
    """The compatibility layer changes only Responses transport payloads."""
    model = ResponsesCompatibleChatOpenAI.model_construct(
        model_name="gpt-6.1-sol", proxy_model_name="gpt-6.1-sol", use_responses_api=False, n=1,
    )
    payload = model._get_request_payload([HumanMessage(content="hello")])
    assert payload["n"] == 1
    assert payload["messages"] == [{"content": "hello", "role": "user"}]
    assert "input" not in payload and "store" not in payload



def test_generated_reasoning_is_preserved_in_sync_and_async_results():
    """Actual Responses result parsing retains opaque content for graph and HANA history."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openai.types.responses import Response, ResponseReasoningItem

    response = Response.model_construct(
        id="resp_generated", created_at=1, model="gpt-6.1-sol", object="response",
        status="completed", error=None, usage=None,
        output=[ResponseReasoningItem(id="rs_generated", summary=[],
                                      encrypted_content="opaque", type="reasoning")],
    )
    raw = SimpleNamespace(parse=lambda: response)
    sync_create = Mock(return_value=raw)
    async_create = AsyncMock(return_value=raw)
    model = ResponsesCompatibleChatOpenAI.model_construct(
        model_name="gpt-6.1-sol", proxy_model_name="gpt-6.1-sol",
        use_responses_api=True, store=False, include=["reasoning.encrypted_content"],
        client=SimpleNamespace(),
        root_client=SimpleNamespace(responses=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=sync_create))),
        root_async_client=SimpleNamespace(responses=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=async_create))),
    )
    for result in (model._generate([HumanMessage(content="hello")]),
                   asyncio.run(model._agenerate([HumanMessage(content="hello")]))):
        message = result.generations[0].message
        assert message.id == "resp_generated"
        assert message.content[0]["id"] == "rs_generated"
        assert message.content[0]["encrypted_content"] == "opaque"
    assert sync_create.call_args.kwargs["store"] is False
    assert async_create.call_args.kwargs["include"] == ["reasoning.encrypted_content"]


def test_claude_model_name_maps_to_non_empty_bedrock_model_id():
    """Keep existing Claude deployment-to-Converse model ID mapping unchanged."""
    from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse

    assert ChatBedrockConverse.get_corresponding_model_id(
        "anthropic--claude-4.6-sonnet"
    ) == "anthropic.claude-sonnet-4-6"
