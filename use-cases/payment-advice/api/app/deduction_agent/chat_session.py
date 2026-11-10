"""
Session state for the UC-02 rules-authoring chat (the /rules-chat endpoint).

The chat is multi-turn: conversation history lives in the agent runtime's HANA
memory (keyed by ``session_id`` == ``context_id``). This module owns the OTHER
piece of per-session state that memory does not cover: the set of uploaded rule
documents (PDF/DOCX/XLSX) whose bytes must persist on disk across turns so the
agent's ``read_rule_source`` tool can re-read them in a later turn.

``run_chat_turn`` binds the accumulated session files to the rule-source
allowlist for the duration of one ``ainvoke`` and resets it afterwards, so the
agent can only read documents this session actually uploaded.

ponytail: in-memory session registry + temp files, single process, no eviction.
Fine for the PoC. Move to a shared store (HANA/object store) + TTL cleanup only
if this runs multi-instance or sessions pile up.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .rule_sources import bind_rule_sources, reset_rule_sources
from .trace import extract_tool_calls

# session_id -> {"dir": Path, "files": dict[filename -> Path]}
_sessions: dict[str, dict[str, Any]] = {}

# Untrusted session ids come from the client; constrain to a filesystem-safe set
# (crypto.randomUUID() from the UI matches) so they cannot escape the temp root.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _session(session_id: str) -> dict[str, Any]:
    """Return (creating if needed) the on-disk state for one chat session.

    Raises:
        ValueError: If ``session_id`` is not a filesystem-safe token.
    """
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError("session_id must be 1-128 chars of [A-Za-z0-9_-]")
    state = _sessions.get(session_id)
    if state is None:
        session_dir = Path(tempfile.mkdtemp(prefix=f"pa_chat_{session_id}_"))
        state = {"dir": session_dir, "files": {}}
        _sessions[session_id] = state
    return state


def store_uploads(session_id: str, uploads: Sequence[tuple[str, bytes]]) -> list[Path]:
    """Persist this turn's uploads into the session dir and return ALL session files.

    Files are keyed by name, so re-uploading the same name overwrites (keeping the
    allowlist's unique-name requirement satisfied). Returns the full accumulated
    file list so every previously uploaded document stays readable.

    Args:
        session_id: Client-supplied session identifier.
        uploads: ``(filename, bytes)`` pairs read from the multipart request.

    Returns:
        Absolute paths of every document uploaded in this session so far.
    """
    state = _session(session_id)
    files: dict[str, Path] = state["files"]
    for raw_name, data in uploads:
        safe_name = Path(raw_name or "upload").name  # strip any directory component
        dest = state["dir"] / safe_name
        dest.write_bytes(data)
        files[safe_name] = dest
    return list(files.values())


async def run_chat_turn(
    chat_runtime: Any,
    session_id: str,
    message: str,
    uploads: Sequence[tuple[str, bytes]] = (),
) -> dict[str, Any]:
    """Run one chat turn: persist uploads, bind them, invoke the agent, reset.

    Args:
        chat_runtime: The memory-enabled ``AgentRuntime`` for rule authoring.
        session_id: Conversation key (also the runtime ``context_id``).
        message: The user's chat message.
        uploads: ``(filename, bytes)`` pairs attached this turn (may be empty).

    Returns:
        ``{"reply": str, "tools_called": [...], "session_id": str}``.
    """
    files = store_uploads(session_id, uploads)

    # Nudge the agent to actually read attachments by naming them in the prompt.
    # (The runtime's system prompt is static; this per-turn hint is how it learns
    # which documents are available without us mutating the runtime.)
    prompt = message
    if files:
        names = ", ".join(sorted(path.name for path in files))
        prompt = (
            f"{message}\n\n[Rule documents available this session via "
            f"list_rule_sources / read_rule_source: {names}]"
        )

    # Only bind when files exist: bind_rule_sources rejects an empty allowlist, and
    # a pure-text turn (e.g. "show the rules for northwind") needs no rule sources.
    token = bind_rule_sources(files) if files else None
    from .tools.customer_admin import current_request
    request_token = current_request.set(message)
    try:
        result = await chat_runtime.ainvoke(prompt, context_id=session_id)
    finally:
        current_request.reset(request_token)
        if token is not None:
            reset_rule_sources(token)

    return {
        "reply": result.output_text,
        "tools_called": extract_tool_calls(getattr(result, "messages", []) or []),
        "session_id": session_id,
    }
