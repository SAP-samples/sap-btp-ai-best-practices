# llm_client.py
from typing import Optional

# Optional: real AI Core proxy client
try:
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
    _proxy_client = get_proxy_client('gen-ai-hub')
    _model_name = "gpt-4.1"
except Exception:
    _proxy_client = None
    ChatOpenAI = None  # type: ignore
    _model_name = None

def ask_llm_simple(prompt: str) -> str:
    """
    Try to answer using AI Core proxy (if available). Fallback: echo the prompt.
    """
    if ChatOpenAI and _proxy_client and _model_name:
        try:
            llm = ChatOpenAI(proxy_model_name=_model_name, proxy_client=_proxy_client)
            result = llm.invoke(prompt)
            return result.content
        except Exception as e:
            return f"(LLM error) {e}\n\n{prompt}"
    # fallback
    return f"(LLM fallback)\n{prompt}"

def ask_llm_confirmation(prompt: str, context: str = "", validation_results: str = "") -> str:
    """
    Lightweight placeholder: concatenates inputs (kept for backward-compat).
    """
    base = prompt.strip()
    if context:
        base += f"\nContext: {context}"
    if validation_results:
        base += f"\nSummary: {validation_results}"
    return base
