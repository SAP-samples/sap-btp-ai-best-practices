from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def make_llm(
    model_name: str = "gpt-4.1",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Create the SAP AI Core-backed OpenAI chat model.

    This project always uses SAP AI Core via gen_ai_hub with an OpenAI model.
    """
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # type: ignore

    # Reasoning-model deployments do not universally accept sampling controls.
    options = {} if model_name.startswith('gpt-5') else {'temperature': temperature}
    if top_p is not None and not model_name.startswith('gpt-5'): options['top_p'] = top_p
    if max_tokens is not None: options['max_completion_tokens'] = max_tokens
    return ChatOpenAI(proxy_model_name=model_name, **options)
