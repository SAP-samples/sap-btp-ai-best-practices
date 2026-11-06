"""Verify provider-factory compatibility details without live model calls."""

from typing import Any

from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from template_agent.providers import ResponsesCompatibleChatOpenAI, _without_reasoning


def test_responses_compatibility_removes_n_only_for_responses(monkeypatch: Any) -> None:
    """Ensure the SAP default ``n`` cannot reach the native Responses client."""

    monkeypatch.setattr(
        "gen_ai_hub.proxy.langchain.openai.ChatOpenAI._get_request_payload",
        lambda self, input_, stop=None, **kwargs: {"n": 1, "input": []},
    )
    monkeypatch.setattr(
        "gen_ai_hub.proxy.langchain.openai.ChatOpenAI._use_responses_api",
        lambda self, payload: True,
    )
    model = object.__new__(ResponsesCompatibleChatOpenAI)

    payload = model._get_request_payload("hello")

    assert payload == {"input": []}


def _result_with(content: Any) -> ChatResult:
    """Wrap one AI message in a ChatResult as the model returns it."""

    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_reasoning_blocks_are_dropped_and_other_blocks_kept() -> None:
    """SAP AI Core rejects replayed reasoning; tool calls and text must survive."""

    content = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA..."},
        {"type": "function_call", "name": "add", "call_id": "call_1", "arguments": "{}"},
        {"type": "text", "text": "Done."},
    ]

    message = _without_reasoning(_result_with(content)).generations[0].message

    assert [block["type"] for block in message.content] == ["function_call", "text"]


def test_plain_text_answer_is_unchanged() -> None:
    """String content (no content blocks) passes through untouched."""

    message = _without_reasoning(_result_with("Done.")).generations[0].message

    assert message.content == "Done."


def test_claude_model_name_maps_to_non_empty_bedrock_model_id() -> None:
    """Map the SAP deployment name to the model ID required by Converse."""

    assert (
        ChatBedrockConverse.get_corresponding_model_id(
            "anthropic--claude-4.6-sonnet"
        )
        == "anthropic.claude-sonnet-4-6"
    )
