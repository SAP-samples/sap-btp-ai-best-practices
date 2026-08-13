"""In-memory conversation history store for GR/IR chat sessions."""

from __future__ import annotations


class GrirSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list] = {}

    def get_or_create(self, session_id: str) -> list:
        """Return the message history for session_id, creating it if absent."""
        return self._sessions.setdefault(session_id, [])

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
