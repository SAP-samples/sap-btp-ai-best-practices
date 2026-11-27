"""Construct SAP Generative AI Hub LangChain models from one small config."""

from __future__ import annotations

from typing import Any

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from ..observability import LLMUsageCallback
from .config import ModelSettings


class ResponsesCompatibleChatOpenAI(SAPChatOpenAI):
    """Serialize self-contained Responses requests for SAP AI Core.

    SAP AI SDK 7.2.0 always contributes the Chat Completions ``n`` parameter.
    The native Responses client rejects that keyword. AI Core also cannot
    reliably resolve replayed response item IDs, even with ``store=False``.
    Keep full content and tool-call links but remove storage-dependent IDs.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return serialized input without unsupported parameters or stored-item IDs.

        Args:
            input_: LangChain messages or another supported prompt input.
            stop: Optional stop sequences forwarded to the normal serializer.
            kwargs: Bound model/tool parameters for this request.
        Returns:
            Provider payload. Responses history keeps encrypted reasoning,
            assistant text and function ``call_id`` links without item ``id``
            fields; Chat Completions retain their normal serialization.
        """

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api(payload):
            payload.pop("n", None)
            # Item IDs address provider-side storage; call_id joins each local
            # function call to its output and must be preserved. LangChain's
            # store=False handles text IDs, but retains reasoning/function IDs.
            replay_types = {"message", "reasoning", "function_call", "custom_tool_call"}
            payload["input"] = [
                {key: value for key, value in item.items() if key != "id"}
                if item.get("type") in replay_types else item
                for item in payload["input"]
            ]
        return payload


def _usage_callback(settings: ModelSettings, llm_endpoint: str) -> LLMUsageCallback:
    """Build the token-usage logger labelled with this deployment and endpoint family."""

    return LLMUsageCallback(model=settings.name, llm_endpoint=llm_endpoint)


def create_chat_model(settings: ModelSettings) -> BaseChatModel:
    """Create the configured GPT, Gemini, or Claude chat model.

    Args:
        settings: Provider-neutral deployment settings.

    Returns:
        A LangChain-compatible SAP Generative AI Hub chat model. Every call made
        through it logs one token-usage event (including cache read/write tokens).
    """

    common: dict[str, object] = {}
    if settings.temperature is not None:
        common["temperature"] = settings.temperature

    if settings.provider == "openai":
        if settings.max_tokens is not None:
            common["max_completion_tokens"] = settings.max_tokens
        if settings.reasoning_effort:
            common["reasoning_effort"] = settings.reasoning_effort
        return ResponsesCompatibleChatOpenAI(
            proxy_model_name=settings.name,
            use_responses_api=settings.use_responses_api,
            # Conversation state is supplied by the application. Request opaque
            # reasoning content so it can be replayed without provider storage.
            **({"store": False, "include": ["reasoning.encrypted_content"]}
               if settings.use_responses_api else {}),
            callbacks=[_usage_callback(settings, "responses" if settings.use_responses_api else "chat.completions")],
            **common,
        )

    if settings.provider == "gemini":
        from gen_ai_hub.proxy.langchain.google_genai import ChatGoogleGenerativeAI

        if settings.max_tokens is not None:
            common["max_tokens"] = settings.max_tokens
        if settings.reasoning_effort in {"minimal", "low", "medium", "high"}:
            common["thinking_level"] = settings.reasoning_effort
        return ChatGoogleGenerativeAI(
            proxy_model_name=settings.name,
            callbacks=[_usage_callback(settings, "generateContent")],
            **common,
        )

    from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse

    if settings.reasoning_effort is not None:
        effort = settings.reasoning_effort
        if effort not in {"none", "low", "medium", "high", "max"}:
            raise ValueError("Claude reasoning_effort must be none, low, medium, high or max")
        if effort != "none" and settings.temperature is not None:
            raise ValueError("Set temperature to null when enabling Claude adaptive thinking")
        common["additional_model_request_fields"] = (
            {"thinking": {"type": "disabled"}} if effort == "none" else
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
        )
    if settings.max_tokens is not None:
        common["max_tokens"] = settings.max_tokens
    model_id = ChatBedrockConverse.get_corresponding_model_id(settings.name)
    return ChatBedrockConverse(
        model=model_id,
        model_name=settings.name,
        callbacks=[_usage_callback(settings, "bedrock.converse")],
        **common,
    )
