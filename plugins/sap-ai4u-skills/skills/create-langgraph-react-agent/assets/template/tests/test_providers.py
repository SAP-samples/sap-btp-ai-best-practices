"""Verify provider-factory compatibility details without live model calls."""

from typing import Any

from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse

from template_agent.providers import ResponsesCompatibleChatOpenAI


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


def test_claude_model_name_maps_to_non_empty_bedrock_model_id() -> None:
    """Map the SAP deployment name to the model ID required by Converse."""

    assert (
        ChatBedrockConverse.get_corresponding_model_id(
            "anthropic--claude-4.6-sonnet"
        )
        == "anthropic.claude-sonnet-4-6"
    )
