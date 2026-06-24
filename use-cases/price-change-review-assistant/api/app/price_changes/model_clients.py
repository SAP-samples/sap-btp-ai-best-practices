from __future__ import annotations

import re
from typing import Any, Literal


OPENAI_RESPONSES_MODEL_PATTERN = re.compile(r"^gpt-(?:[5-9]|\d{2,})(?:[.-].*)?$")


def infer_genai_model_provider(model_name: str) -> Literal["openai", "gemini", "claude"]:
    """Infer the SAP Gen AI Hub provider wrapper from a configured model name.

    Args:
        model_name: SAP Gen AI Hub model deployment name.

    Returns:
        Provider family used by the shared LangChain chat-model factory.
    """
    normalized = model_name.lower()
    if "gemini" in normalized:
        return "gemini"
    if "claude" in normalized or normalized.startswith("anthropic--"):
        return "claude"
    return "openai"


def should_use_openai_responses_api(model_name: str) -> bool:
    """Return whether an OpenAI model should use the native Responses API.

    Args:
        model_name: SAP Gen AI Hub model deployment name.

    Returns:
        True when the model is an OpenAI GPT-5 or newer model name.
    """
    normalized = model_name.strip().lower()
    return infer_genai_model_provider(normalized) == "openai" and bool(
        OPENAI_RESPONSES_MODEL_PATTERN.match(normalized)
    )


def make_genai_hub_chat_model(model_name: str) -> Any:
    """Create a provider-aware SAP Gen AI Hub LangChain chat model.

    Args:
        model_name: SAP Gen AI Hub model deployment name.

    Returns:
        LangChain-compatible chat model for OpenAI-compatible, Gemini, or Claude deployments.
    """
    from gen_ai_hub.proxy import get_proxy_client

    provider = infer_genai_model_provider(model_name)
    if provider == "gemini":
        from gen_ai_hub.proxy.langchain.google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            proxy_client=get_proxy_client(),
            temperature=0,
            max_tokens=4096,
        )
    if provider == "claude":
        from gen_ai_hub.proxy.langchain.amazon import ChatBedrock

        return ChatBedrock(
            model=model_name,
            temperature=0,
            model_kwargs={"max_tokens": 4096},
        )

    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

    return ChatOpenAI(proxy_model_name=model_name, proxy_client=get_proxy_client())
