"""
Conversation context manager.
Stores conversation history in memory, keyed by conversation_id.
Implements context window trimming to avoid token overflow.
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum number of message pairs (user + assistant) to keep
MAX_TURNS = 10
# Maximum total characters in context window
MAX_CONTEXT_CHARS = 8000


class ConversationTurn:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = time.time()


class ConversationContext:
    """Holds the full state of a single conversation session."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.turns: List[ConversationTurn] = []
        self.entities: Dict = {}   # Last detected entities (for follow-up resolution)
        self.last_active = time.time()

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role, content))
        self.last_active = time.time()
        self._trim_context()

    def _trim_context(self) -> None:
        """Keep only the last MAX_TURNS pairs and trim by character count."""
        max_messages = MAX_TURNS * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]

        # Trim by character count (keep most recent)
        total_chars = sum(len(t.content) for t in self.turns)
        while total_chars > MAX_CONTEXT_CHARS and len(self.turns) > 2:
            removed = self.turns.pop(0)
            total_chars -= len(removed.content)

    def get_history_text(self) -> str:
        """Return conversation history as formatted text for prompt injection."""
        lines = []
        for turn in self.turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        return "\n".join(lines)

    def update_entities(self, entities: Dict) -> None:
        """Merge newly detected entities into the session context."""
        for key, value in entities.items():
            if value is not None and value != "":
                self.entities[key] = value

    def get_entity(self, key: str, default=None):
        return self.entities.get(key, default)


class ContextManager:
    """In-memory conversation context store (singleton per process)."""

    def __init__(self):
        self._store: Dict[str, ConversationContext] = {}

    def get_or_create(self, conversation_id: str) -> ConversationContext:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationContext(conversation_id)
            logger.info("chat.context: new conversation %s", conversation_id)
        return self._store[conversation_id]

    def cleanup_old(self, max_age_seconds: int = 3600) -> None:
        """Remove conversations idle for more than max_age_seconds."""
        now = time.time()
        to_remove = [
            cid for cid, ctx in self._store.items()
            if now - ctx.last_active > max_age_seconds
        ]
        for cid in to_remove:
            del self._store[cid]
        if to_remove:
            logger.info("chat.context: cleaned up %d stale conversations", len(to_remove))

    def count(self) -> int:
        return len(self._store)