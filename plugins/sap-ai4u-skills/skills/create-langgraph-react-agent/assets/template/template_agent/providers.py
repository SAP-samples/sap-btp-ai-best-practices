"""Construct SAP Generative AI Hub LangChain models from one small config."""

from __future__ import annotations

from typing import Any

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from .config import ModelSettings


class ResponsesCompatibleChatOpenAI(SAPChatOpenAI):
    """SAP Gen AI Hub ChatOpenAI adapted to the Responses API on SAP AI Core.

    - SAP AI SDK 7.2.0 always contributes the Chat Completions ``n`` parameter.
      The native Responses client rejects that keyword, so it is removed only
      after LangChain has selected and serialized a Responses API request.
    - Earlier output item IDs can require provider-side storage that AI Core
      cannot reliably resolve. Remove IDs from outgoing message, reasoning and
      function-call items, preserving encrypted reasoning and call_id links.
      Returned messages remain intact for graph state and HANA conversation memory.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize self-contained history while preserving reasoning and tool links.

        Args:
            input_: LangChain messages or another supported prompt input.
            stop: Optional stop sequences passed to the normal serializer.
            kwargs: Bound model/tool parameters for the request.
        Returns:
            Responses payload without chat-only n or provider output item IDs.
            Encrypted content, function call_id and graph messages are preserved;
            other transports retain their normal payload.
        """

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api(payload):
            payload.pop("n", None)
            # store=False drops message IDs in LangChain, but encrypted reasoning
            # and function-call items still retain IDs. These IDs address provider
            # storage; call_id joins a call to its output and must stay intact.
            replay_types = {"message", "reasoning", "function_call", "custom_tool_call"}
            payload["input"] = [
                {key: value for key, value in item.items() if key != "id"}
                if item.get("type") in replay_types else item
                for item in payload["input"]
            ]
        return payload


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
            # Carry complete history locally instead of looking up stored responses.
            # Opaque reasoning can be replayed without exposing or decrypting it.
            common["store"] = False
            common["include"] = ["reasoning.encrypted_content"]
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
