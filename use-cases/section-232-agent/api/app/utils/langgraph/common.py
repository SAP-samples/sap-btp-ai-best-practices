"""Common LangGraph helpers backed by SAP Generative AI Hub."""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv


load_dotenv()


def make_llm(
    provider: Literal["openai", "bedrock", "vertex"],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: Optional[float] = None,
):
    """Create a LangChain-compatible chat model through SAP Gen AI Hub."""
    if provider == "openai":
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "bedrock":
        from gen_ai_hub.proxy.langchain.amazon import ChatBedrock  # type: ignore

        return ChatBedrock(
            model_name=model_name,
            temperature=temperature,
            model_kwargs={
                "max_tokens": max_tokens,
                **({"top_p": top_p} if top_p is not None else {}),
            },
        )
    if provider == "vertex":
        from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI  # type: ignore

        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    raise ValueError(f"Unknown provider: {provider}")
