"""Skill-aware LangGraph ReAct runtime, vendored from the create-langgraph-react-agent template (v0.5.0).

Graph: START -> load_skill_node -> compact_context -> agent -> tools -> compact_context -> agent -> END

Local modifications compared with the template (keep this list current when updating):

1. This ``__init__`` does not import the template's A2A server, so the ``a2a-sdk`` dependency is
   not needed; ``a2a_server.py`` and the CLI ``__main__.py`` are not copied. The runtime is embedded
   in the FastAPI app instead (``app.order_agent.runner``).
2. No code changes in the copied modules. Use-case behaviour is configured in
   ``app/order_agent/config/agent.yaml`` (summarization and HANA memory disabled, A2A disabled,
   higher recursion limit) and added through ``extra_tools``.
"""

from .models import AgentResult, Attachment
from .runtime import AgentRuntime

__all__ = ["AgentResult", "AgentRuntime", "Attachment"]
