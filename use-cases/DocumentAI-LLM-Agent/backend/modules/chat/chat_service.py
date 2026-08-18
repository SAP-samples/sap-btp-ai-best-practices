"""
chat_service.py
---------------
Streaming chat service for DocAI Assistant.

Uses SAP Gen AI Hub LLM (same pattern as modules/genai/llm_client.py)
with LangChain streaming to yield NDJSON events as a synchronous generator.

The caller (api.py) runs this in a thread pool so the event loop is not blocked.
"""

import json
import logging
from typing import Generator, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the DocAI Assistant. You help users understand invoice extraction results, "
    "answer questions about the pipeline (SAP Document AI + GenAI Hub), and guide them "
    "through posting invoices to S/4HANA FI. You can explain extraction fields, confidence "
    "scores, routing decisions, and FI posting results."
)


def stream_chat(
    message: str,
    history: list[dict],
    context: Optional[dict] = None,
) -> Generator[str, None, None]:
    """
    Stream a chat response as NDJSON lines (synchronous generator).

    Each yielded value is a JSON string terminated with a newline:
      {"type": "delta", "content": "..."}   — one per LLM chunk
      {"type": "done"}                       — signals end of stream
      {"type": "error", "message": "..."}    — on failure

    Args:
        message:  The current user message.
        history:  List of prior turns as {"role": "user"|"assistant", "content": "..."}.
        context:  Optional dict with the current extraction result to include as context.
    """
    try:
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except ImportError as exc:
        yield json.dumps({"type": "error", "message": f"Missing dependency: {exc}"}) + "\n"
        return

    # Build system prompt, optionally enriched with extraction context
    system_content = SYSTEM_PROMPT
    if context:
        system_content += (
            "\n\nCurrent extraction context (use this to answer user questions):\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

    messages = [SystemMessage(content=system_content)]

    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=message))

    # Initialise the LLM (same pattern as llm_client.get_llm)
    try:
        proxy_client = get_proxy_client("gen-ai-hub")
        llm = ChatOpenAI(
            proxy_model_name="gpt-4o",
            proxy_client=proxy_client,
            streaming=True,
        )
    except Exception as exc:
        logger.error("Failed to initialise SAP Gen AI Hub client: %s", exc)
        yield json.dumps({"type": "error", "message": f"LLM initialisation failed: {exc}"}) + "\n"
        return

    # Stream response chunks via LangChain llm.stream()
    try:
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield json.dumps({"type": "delta", "content": content}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
    except Exception as exc:
        logger.error("Streaming error: %s", exc)
        yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
