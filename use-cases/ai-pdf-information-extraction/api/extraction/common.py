"""
Common utilities for PDF extraction.

Provides LLM factory functions and shared types using SAP Generative AI Hub
LangChain proxy wrappers.
"""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv

# Load credentials from the project-root .env, matching the notebook behavior
load_dotenv()

# from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore
# from gen_ai_hub.proxy.langchain.amazon import ChatBedrock  # type: ignore
# from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI  # type: ignore


def make_llm(
    provider: Literal["openai", "bedrock", "vertex"],
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
):
    """
    Create an LLM compatible with LangChain from SAP GenAI Hub proxies.

    - provider: which backend to use
    - model_name: proxy model name, e.g. "gpt-4o", "anthropic--claude-3.5-sonnet", "gemini-2.5-pro"
    - temperature, max_tokens, top_p: generation controls
    """
    if provider == "openai":
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
        if max_tokens is None:
            return ChatOpenAI(
                proxy_model_name=model_name,
                temperature=temperature,
            )
        else:
            return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "bedrock":
        from gen_ai_hub.proxy.langchain.amazon import ChatBedrock
        if max_tokens is None:
            return ChatBedrock(
                model_name=model_name,
                temperature=temperature,
            )
        else:
            return ChatBedrock(
                model_name=model_name,
                temperature=temperature,
                model_kwargs={
                    "max_tokens": max_tokens,
                    **({"top_p": top_p} if top_p is not None else {}),
                },
            )
    if provider == "vertex":
        from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI
        from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel


        # For Vertex AI via GenAI Hub, max tokens parameter name differs
        if max_tokens is None:
            llm= ChatVertexAI(
                model_name=model_name,
                temperature=temperature,
            )
        else:
            llm= ChatVertexAI(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        llm.genaihub_client = GenerativeModel(
            model_name=model_name,
            proxy_client=None
        )
        return llm

    raise ValueError(f"Unknown provider: {provider}")
