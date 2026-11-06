"""Construct SAP Generative AI Hub LangChain models from one small config."""

from __future__ import annotations

from typing import Any

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as SAPChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

from .config import ModelSettings


def _without_reasoning(result: ChatResult) -> ChatResult:
    """Drop Responses API reasoning blocks from the generated messages.

    SAP AI Core answers every request that carries a reasoning item from an
    earlier step with ``404 Resource not found`` (or hangs), whether the item is
    sent inline, with ``store=False`` or referenced via ``previous_response_id``.
    The blocks only hold encrypted, unreadable content, so they are removed as
    soon as the model answers: they never reach the agent state, are never sent
    back and are never stored in a trace. The model still sees all earlier tool
    calls and results and reasons again in every step.

    Args:
        result: Chat result returned by the model.

    Returns:
        The same result with ``type == "reasoning"`` content blocks removed.
    """

    for generation in result.generations:
        message = generation.message
        if isinstance(message.content, list):
            message.content = [
                block
                for block in message.content
                if not (isinstance(block, dict) and block.get("type") == "reasoning")
            ]
    return result


class ResponsesCompatibleChatOpenAI(SAPChatOpenAI):
    """SAP Gen AI Hub ChatOpenAI adapted to the Responses API on SAP AI Core.

    - SAP AI SDK 7.2.0 always contributes the Chat Completions ``n`` parameter.
      The native Responses client rejects that keyword, so it is removed only
      after LangChain has selected and serialized a Responses API request.
    - Reasoning blocks are dropped from every answer (see ``_without_reasoning``).
      This is unconditional while AI Core rejects replayed reasoning (2026-11);
      remove these overrides and ``store=False`` once it accepts them again.
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

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        """Generate as usual, then drop reasoning blocks from the answer."""

        return _without_reasoning(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        """Generate as usual (async), then drop reasoning blocks from the answer."""

        return _without_reasoning(await super()._agenerate(*args, **kwargs))


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
            # Stateless calls: nothing is kept on the server, and a function call
            # sent without its (dropped) reasoning item is accepted.
            common["store"] = False
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
