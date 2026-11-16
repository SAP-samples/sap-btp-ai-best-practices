"""Construct SAP Generative AI Hub LangChain models from one small config."""

from __future__ import annotations

from typing import Any

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from .config import ModelSettings


class ResponsesCompatibleChatOpenAI(SAPChatOpenAI):
    """Remove a SAP-wrapper default unsupported by OpenAI Responses calls.

    SAP AI SDK 7.2.0 always contributes the Chat Completions ``n`` parameter.
    The native Responses client rejects that keyword, so it is removed only
    after LangChain has selected and serialized a Responses API request.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the normal LangChain payload without Responses-incompatible ``n``."""

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api(payload):
            payload.pop("n", None)
        return payload


# Responses-API settings that make every call self-contained. By default a follow-up
# call in the ReAct loop (after a tool call or skill load) references the previous
# reasoning items by their stored id. SAP AI Core does not reliably resolve those ids
# and answers "404 Resource not found", which aborted interpretation and rule
# authoring. With storage off and the encrypted reasoning returned, LangChain sends
# the prior reasoning inline instead, so no stored item is ever looked up.
RESPONSES_STATELESS_SETTINGS: dict[str, object] = {
    "store": False,
    "include": ["reasoning.encrypted_content"],
}


def create_chat_model(settings: ModelSettings) -> BaseChatModel:
    """Create the configured GPT, Gemini, or Claude chat model.

    Args:
        settings: Provider-neutral deployment settings.

    Returns:
        A LangChain-compatible SAP Generative AI Hub chat model.
    """

    common: dict[str, object] = {}
    if settings.temperature is not None:
        common["temperature"] = settings.temperature

    if settings.provider == "openai":
        if settings.max_tokens is not None:
            common["max_completion_tokens"] = settings.max_tokens
        if settings.reasoning_effort:
            common["reasoning_effort"] = settings.reasoning_effort
        if settings.use_responses_api:
            # Copy the list so a model instance never shares mutable state with the constant.
            common["store"] = RESPONSES_STATELESS_SETTINGS["store"]
            common["include"] = list(RESPONSES_STATELESS_SETTINGS["include"])
        return ResponsesCompatibleChatOpenAI(
            proxy_model_name=settings.name,
            use_responses_api=settings.use_responses_api,
            **common,
        )

    if settings.provider == "gemini":
        from gen_ai_hub.proxy.langchain.google_genai import ChatGoogleGenerativeAI

        if settings.max_tokens is not None:
            common["max_tokens"] = settings.max_tokens
        if settings.reasoning_effort in {"minimal", "low", "medium", "high"}:
            common["thinking_level"] = settings.reasoning_effort
        return ChatGoogleGenerativeAI(proxy_model_name=settings.name, **common)

    from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse

    if settings.max_tokens is not None:
        common["max_tokens"] = settings.max_tokens
    model_id = ChatBedrockConverse.get_corresponding_model_id(settings.name)
    return ChatBedrockConverse(
        model=model_id,
        model_name=settings.name,
        **common,
    )
