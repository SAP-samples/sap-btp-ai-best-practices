"""
Common utilities for LangGraph agents.

Provides LLM factory functions using SAP Generative AI Hub
LangChain proxy wrappers.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI


def make_llm(
    provider: Literal["openai"] = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """
    Create an LLM compatible with LangChain from SAP GenAI Hub proxies.

    Args:
        provider: Which backend to use (currently only "openai" supported)
        model_name: Proxy model name, e.g. "gpt-4o". If None, uses config default.
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens for response

    Returns:
        LangChain compatible chat model
    """
    if model_name is None:
        from ...config import LLM_MODEL_NAME
        model_name = LLM_MODEL_NAME

    if provider == "openai":
        return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: {provider}")


__all__ = ['make_llm']
