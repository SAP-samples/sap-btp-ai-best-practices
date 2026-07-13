from __future__ import annotations

import sys
import types

from app.utils.langgraph.common import make_llm


def test_make_llm_openai_provider_does_not_require_optional_vertex_dependency(monkeypatch) -> None:
    """OpenAI construction should not import optional Vertex dependencies."""
    class ChatOpenAI:
        """Minimal stand-in for the GenAI Hub OpenAI LangChain wrapper."""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    gen_ai_hub = types.ModuleType("gen_ai_hub")
    proxy = types.ModuleType("gen_ai_hub.proxy")
    langchain = types.ModuleType("gen_ai_hub.proxy.langchain")
    openai = types.ModuleType("gen_ai_hub.proxy.langchain.openai")
    openai.ChatOpenAI = ChatOpenAI

    monkeypatch.setitem(sys.modules, "gen_ai_hub", gen_ai_hub)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy", proxy)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.langchain", langchain)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.langchain.openai", openai)

    llm = make_llm("openai", "gpt-4o-mini", temperature=0.0, max_tokens=32)

    assert llm is not None
    assert llm.__class__.__name__ == "ChatOpenAI"
