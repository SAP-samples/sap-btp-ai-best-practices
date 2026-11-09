from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()


def make_llm(
    provider: Literal["openai", "bedrock", "vertex", "google"],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    top_p: Optional[float] = None,
    json_mode: bool = False,
    model_kwargs: Optional[dict] = None,
):
    if provider == "openai":
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore
        kwargs: dict = {
            "proxy_model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Pass model_kwargs that ChatOpenAI accepts as top-level params directly
        if model_kwargs:
            for key, value in model_kwargs.items():
                kwargs[key] = value
        llm = ChatOpenAI(**kwargs)
        if json_mode:
            llm = llm.bind(response_format={"type": "json_object"})
        return llm
    if provider == "bedrock":
        from gen_ai_hub.proxy.langchain.amazon import ChatBedrock  # type: ignore
        return ChatBedrock(
            model_name=model_name,
            temperature=temperature,
            model_kwargs={
                "max_tokens": max_tokens
            },
        )
    if provider in ("vertex", "google"):
        from gen_ai_hub.proxy.langchain.google_genai import ChatGoogleGenerativeAI  # type: ignore
        llm = ChatGoogleGenerativeAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if json_mode:
            llm = llm.bind(response_mime_type="application/json")
        return llm
    raise ValueError(f"Unknown provider: {provider}")
