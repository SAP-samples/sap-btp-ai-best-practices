"""
Common utilities for the forecasting agent.

Provides LLM factory functions using SAP Generative AI Hub
LangChain proxy wrappers.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from dotenv import load_dotenv

# Load credentials from the project-root .env
load_dotenv()



def make_llm(
    provider: Literal["openai", "bedrock", "vertexai"],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 16384,
    top_p: Optional[float] = None,
):
    """
    Create an LLM compatible with LangChain from SAP GenAI Hub proxies.

    Parameters
    ----------
    provider : str
        Which backend to use: "openai", "bedrock", or "vertexai"
    model_name : str
        Proxy model name, e.g. "gpt-4.1", "anthropic--claude-3.5-sonnet", "gemini-2.5-pro"
    temperature : float
        Sampling temperature (default: 0.2)
    max_tokens : int
        Maximum tokens to generate (default: 2048)
    top_p : float, optional
        Top-p sampling parameter

    Returns
    -------
    BaseChatModel
        LangChain-compatible chat model
    """
    if provider == "openai":
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore

        # GPT-5 does not use max_tokens or temperature parameters
        is_gpt5 = model_name.lower() == "gpt-5"

        kwargs = {"proxy_model_name": model_name}
        if not is_gpt5:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        return ChatOpenAI(**kwargs)
    elif provider == "bedrock":
        from gen_ai_hub.proxy.langchain import init_llm  # type: ignore

        return init_llm(
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p if top_p is not None else 1.0,
        )
    elif provider == "vertexai":
        from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI  # type: ignore
        from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel  # type: ignore

        chat = ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        chat.genaihub_client = GenerativeModel(
            model_name=model_name,
            proxy_client=None,
        )
        return chat
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Mermaid/PNG rendering helper for compiled LangGraph apps
try:
    from langchain_core.runnables.graph import MermaidDrawMethod  # type: ignore
except Exception:
    MermaidDrawMethod = None  # type: ignore


def save_graph_mermaid_png(app, output_path: str) -> bool:
    """
    Render and save a Mermaid PNG for a compiled LangGraph application.

    Parameters
    ----------
    app : CompiledGraph
        Compiled LangGraph application returned by builder.compile()
    output_path : str
        Absolute path to write the PNG to

    Returns
    -------
    bool
        True on success, False otherwise
    """
    try:
        graph = app.get_graph()
        draw_kwargs = {}
        requested_pyppeteer = False

        # Prefer the local Pyppeteer renderer to avoid network dependency on mermaid.ink
        if MermaidDrawMethod is not None:
            draw_kwargs["draw_method"] = MermaidDrawMethod.PYPPETEER
            requested_pyppeteer = True

        try:
            png_bytes = graph.draw_mermaid_png(**draw_kwargs)
        except Exception as pyppeteer_exc:
            if requested_pyppeteer:
                print(
                    f"[mermaid] Pyppeteer render failed, falling back to default: "
                    f"{pyppeteer_exc}"
                )
            png_bytes = graph.draw_mermaid_png()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        return True
    except Exception as exc:
        print(f"[mermaid] Failed to render PNG to {output_path}: {exc}")
        return False


def normalize_llm_response(content) -> str:
    """
    Normalize LLM response content to a plain string.

    Different LLM providers return content in different formats:
    - OpenAI: Returns a plain string
    - Gemini/Vertex AI: Returns a list of content blocks like [{'type': 'text', 'text': '...'}]

    This function handles both formats and returns a plain string.

    Args:
        content: The response content from an LLM message

    Returns:
        str: The normalized text content
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else str(content)
    return str(content) if content else ""


__all__ = ["make_llm", "save_graph_mermaid_png", "normalize_llm_response"]
