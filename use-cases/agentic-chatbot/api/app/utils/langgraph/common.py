"""
Common utilities for LangGraph agents.

Provides LLM factory functions and shared types using SAP Generative AI Hub
LangChain proxy wrappers shown in the notebooks.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from dotenv import load_dotenv

# Load credentials from the project-root .env, matching the notebook behavior
load_dotenv()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore
from gen_ai_hub.proxy.langchain.amazon import ChatBedrock  # type: ignore
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI  # type: ignore


def make_llm(
    provider: Literal["openai", "bedrock", "vertex"],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: Optional[float] = None,
):
    """
    Create an LLM compatible with LangChain from SAP GenAI Hub proxies.

    - provider: which backend to use
    - model_name: proxy model name, e.g. "gpt-4o", "anthropic--claude-3.5-sonnet", "gemini-2.5-pro"
    - temperature, max_tokens, top_p: generation controls
    """
    if provider == "openai":
        return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "bedrock":
        return ChatBedrock(
            model_name=model_name,
            temperature=temperature,
            model_kwargs={
                "max_tokens": max_tokens,
                **({"top_p": top_p} if top_p is not None else {}),
            },
        )
    if provider == "vertex":
        # For Vertex AI via GenAI Hub, max tokens parameter name differs
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: {provider}")


# Import visualization utilities for easy access
try:
    from visualization_utils import (
        visualize_graph,
        format_state_transition,
        format_message,
        show_execution_flow,
        create_progress_bar,
        format_execution_summary,
        format_react_step,
        ExecutionTracker
    )
    
    __all__ = [
        'make_llm',
        'visualize_graph',
        'format_state_transition', 
        'format_message',
        'show_execution_flow',
        'create_progress_bar',
        'format_execution_summary',
        'format_react_step',
        'ExecutionTracker'
    ]
    
except ImportError:
    # Graceful fallback if visualization utils not available
    __all__ = ['make_llm']


# Mermaid/PNG rendering helper for compiled LangGraph apps
try:
    # Preferred import in LangChain/LangGraph 0.2+
    from langchain_core.runnables.graph import MermaidDrawMethod  # type: ignore
except Exception:  # pragma: no cover
    MermaidDrawMethod = None  # type: ignore


def save_graph_mermaid_png(app, output_path: str) -> bool:
    """Render and save a Mermaid PNG for a compiled LangGraph application.

    - app: Compiled LangGraph application returned by builder.compile()
    - output_path: Absolute path to write the PNG to

    Returns True on success, False otherwise.
    """
    try:
        graph = app.get_graph()
        # Use Mermaid API by default; falls back to pyppeteer if unavailable
        if MermaidDrawMethod is not None:
            png_bytes = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        else:
            png_bytes = graph.draw_mermaid_png()

        # Ensure directory exists and write bytes atomically
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        return True
    except Exception as exc:
        # Best-effort logging; do not raise to avoid breaking agent runs
        print(f"[mermaid] Failed to render PNG to {output_path}: {exc}")
        return False


