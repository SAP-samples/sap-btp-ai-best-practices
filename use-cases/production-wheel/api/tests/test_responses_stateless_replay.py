"""Guard self-contained SAP AI Core Responses history serialization.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_responses_stateless_replay.py -q
"""

from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.config import ModelSettings
from app.agent.providers import ResponsesCompatibleChatOpenAI, create_chat_model


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
    from app.agent import providers

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


def test_missing_provider_resource_is_reported_as_a_model_request_failure():
    """An upstream 404 points to AI Core without exposing raw provider response data."""
    from app.routers.workspace_chat import chat_error_message

    class MissingResource(Exception):
        """Represent a missing provider resource with a private response body."""

        status_code = 404

    message = chat_error_message(MissingResource("private provider response"))
    assert "SAP AI Core" in message and "selected model request" in message
    assert "continue independently" in message
    assert "private provider response" not in message
    assert "Check the selected workspace" not in message
