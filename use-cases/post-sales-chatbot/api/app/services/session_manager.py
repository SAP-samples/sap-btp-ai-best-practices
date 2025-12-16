"""Session management service for tracking conversation state."""

from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
from threading import Lock


@dataclass
class SessionState:
    """Holds conversation state for a single session."""

    client_id: Optional[int] = None
    selected_vin: Optional[str] = None
    last_intent: Optional[str] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            'client_id': self.client_id,
            'selected_vin': self.selected_vin,
            'last_intent': self.last_intent,
            'chat_history': self.chat_history,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
        }


class SessionManager:
    """Thread-safe in-memory session storage."""

    def __init__(self, expiry_minutes: int = 60):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._expiry = timedelta(minutes=expiry_minutes)

    def get_or_create(self, session_id: Optional[str] = None) -> tuple[str, SessionState]:
        """Get existing session or create new one.

        Args:
            session_id: Optional session ID to retrieve. If None, creates new session.

        Returns:
            Tuple of (session_id, SessionState)
        """
        with self._lock:
            # Clean up expired sessions periodically
            self._cleanup_expired_unsafe()

            if session_id and session_id in self._sessions:
                state = self._sessions[session_id]
                state.last_accessed = datetime.now()
                return session_id, state

            # Create new session
            new_id = str(uuid.uuid4())
            new_state = SessionState()
            self._sessions[new_id] = new_state
            return new_id, new_state

    def get(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            SessionState if found, None otherwise.
        """
        with self._lock:
            if session_id in self._sessions:
                state = self._sessions[session_id]
                state.last_accessed = datetime.now()
                return state
            return None

    def update(self, session_id: str, state: SessionState) -> None:
        """Update session state.

        Args:
            session_id: The session ID to update.
            state: The new session state.
        """
        with self._lock:
            state.last_accessed = datetime.now()
            self._sessions[session_id] = state

    def reset(self, session_id: str) -> SessionState:
        """Reset session to initial state while preserving the session ID.

        Args:
            session_id: The session ID to reset.

        Returns:
            New SessionState for the session.
        """
        with self._lock:
            new_state = SessionState()
            self._sessions[session_id] = new_state
            return new_state

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def _cleanup_expired_unsafe(self) -> None:
        """Remove expired sessions. Must be called with lock held."""
        now = datetime.now()
        expired = [
            sid for sid, state in self._sessions.items()
            if now - state.last_accessed > self._expiry
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions.")

    def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            now = datetime.now()
            expired = [
                sid for sid, state in self._sessions.items()
                if now - state.last_accessed > self._expiry
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Singleton instance with default expiry from config
def _get_session_manager() -> SessionManager:
    """Get session manager with configured expiry."""
    try:
        from ..config import SESSION_EXPIRY_MINUTES
        return SessionManager(expiry_minutes=SESSION_EXPIRY_MINUTES)
    except ImportError:
        return SessionManager(expiry_minutes=60)


session_manager = _get_session_manager()
