"""Purchase-order agent: configuration, skills, tools, prompt building and run wrapper.

Built on the vendored ReAct runtime in ``app.agent_core``. The agent extracts, resolves and
submits orders; it never posts them (posting is decided by the deterministic gate).
"""
