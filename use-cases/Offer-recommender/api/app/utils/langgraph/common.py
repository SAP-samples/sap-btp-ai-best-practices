"""
Common utilities for LangGraph agents.

Provides LLM factory functions and shared types using SAP Generative AI Hub
LangChain proxy wrappers shown in the notebooks.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

# Load credentials from the project-root .env, matching the notebook behavior
load_dotenv()

def make_llm(
    provider: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: Optional[float] = None,
):
    """
    Create an LLM compatible with LangChain from SAP GenAI Hub proxies.

    - provider: must be "openai"
    - model_name: proxy model name, e.g. "gpt-4.1"
    - temperature, max_tokens, top_p: generation controls
    """
    if provider == "openai":
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError("Only OpenAI provider is supported in this deployment.")


__all__ = ['make_llm']

