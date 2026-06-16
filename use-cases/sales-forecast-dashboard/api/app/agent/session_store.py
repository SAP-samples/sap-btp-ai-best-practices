"""
Thread-safe session storage for multi-user support.

Each user session gets its own:
- SessionManager instance (not singleton)
- AgentState with scenarios and predictions
- Conversation checkpointer for multi-turn dialogue
- Cached resources (models can be shared read-only)

The module provides:
- SessionStore: Manages all user sessions with LRU eviction
- UserSession: Wraps SessionManager with agent and checkpointer
- Context variable: Allows tools to access current session
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.session import SessionManager


# Context variable for current session (used by tools via get_session())
_current_session: contextvars.ContextVar[SessionManager] = contextvars.ContextVar(
    "current_session"
)


def get_current_session() -> SessionManager:
    """
    Get the current user session from context.

    Called by tools via get_session() to access the current user's
    SessionManager without needing to pass it explicitly.

    Returns
    -------
    SessionManager
        The current user's session manager

    Raises
    ------
    RuntimeError
        If called outside of a session context
    """
    try:
        return _current_session.get()
    except LookupError:
        raise RuntimeError(
            "No session context. Ensure the request is wrapped with "
            "set_current_session() or use SessionStore.get_or_create()."
        )


def set_current_session(session: SessionManager) -> contextvars.Token:
    """
    Set the current session in context.

    Parameters
    ----------
    session : SessionManager
        The session to make current

    Returns
    -------
    contextvars.Token
        Token for resetting the context later
    """
    return _current_session.set(session)


def reset_current_session(token: contextvars.Token) -> None:
    """
    Reset the current session context.

    Parameters
    ----------
    token : contextvars.Token
        Token from set_current_session()
    """
    _current_session.reset(token)


@dataclass
class UserSession:
    """
    Encapsulates all state for a single user session.

    Combines a SessionManager with the LangGraph agent and checkpointer
    for multi-turn conversations.
    """

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Session manager holds agent state (scenarios, predictions, etc.)
    session_manager: SessionManager = field(default=None)

    # LangGraph agent and checkpointer for conversation memory
    _agent: Any = field(default=None, repr=False)
    _checkpointer: MemorySaver = field(default=None, repr=False)

    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        """Initialize session manager if not provided."""
        if self.session_manager is None:
            self.session_manager = SessionManager(session_id=self.session_id)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def _build_agent(self):
        """Build the LangGraph agent with checkpointer."""
        from app.agent.agent import build_agent_with_checkpointer

        self._agent, self._checkpointer = build_agent_with_checkpointer()

    def get_agent(self):
        """
        Get or create the LangGraph agent.

        Lazily builds the agent on first access.
        """
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._build_agent()
        return self._agent

    def run_query(self, message: str) -> Dict[str, Any]:
        """
        Run a query through the agent.

        Parameters
        ----------
        message : str
            User message to process

        Returns
        -------
        Dict[str, Any]
            Result with final_response, tool_calls, and messages
        """
        self.touch()

        agent = self.get_agent()
        config = {"configurable": {"thread_id": self.session_id}}

        # Set this session as current for tools
        token = set_current_session(self.session_manager)
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )

            # Store messages in session manager for report generation
            self.session_manager.set_messages(result["messages"])

            # Extract tool calls
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "name": tc.get("name"),
                            "args": tc.get("args", {}),
                        })

            # Get final response
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                content = final_message.content
                # Handle different content formats
                if isinstance(content, str):
                    final_response = content
                elif isinstance(content, list):
                    # Gemini/Vertex AI format
                    final_response = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    final_response = str(content)
            else:
                final_response = str(final_message)

            return {
                "final_response": final_response,
                "tool_calls": tool_calls,
                "messages": result["messages"],
            }
        finally:
            reset_current_session(token)

    def get_info(self) -> Dict[str, Any]:
        """
        Get session information.

        Returns
        -------
        Dict[str, Any]
            Session metadata and state summary
        """
        summary = self.session_manager.get_session_summary()
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            **summary,
        }

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get conversation history.

        Parameters
        ----------
        limit : int
            Maximum number of messages to return

        Returns
        -------
        Dict[str, Any]
            Conversation history
        """
        messages = self.session_manager.get_messages()
        history = []
        for msg in messages[-limit:]:
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            history.append({
                "role": msg_type,
                "content": content,
            })
        return {
            "session_id": self.session_id,
            "messages": history,
            "total_messages": len(messages),
        }

    def reset(self) -> None:
        """Reset session state while preserving conversation."""
        self.session_manager.reset()


class SessionStore:
    """
    Thread-safe storage for user sessions.

    Features:
    - Per-user session isolation
    - Automatic session cleanup (configurable TTL)
    - Memory-bounded via LRU eviction
    - Thread-safe operations
    """

    def __init__(
        self,
        max_sessions: Optional[int] = None,
        session_ttl_hours: Optional[int] = None,
    ):
        """
        Initialize the session store.

        Parameters
        ----------
        max_sessions : int, optional
            Maximum number of concurrent sessions.
            Defaults to MAX_SESSIONS env var or 100.
        session_ttl_hours : int, optional
            Session time-to-live in hours.
            Defaults to SESSION_TTL_HOURS env var or 24.
        """
        self.max_sessions = max_sessions or int(os.getenv("MAX_SESSIONS", "100"))
        self.session_ttl_hours = session_ttl_hours or int(
            os.getenv("SESSION_TTL_HOURS", "24")
        )
        self.session_ttl_seconds = self.session_ttl_hours * 3600

        self._sessions: OrderedDict[str, UserSession] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> UserSession:
        """
        Get existing session or create new one.

        Parameters
        ----------
        session_id : str
            Unique session identifier

        Returns
        -------
        UserSession
            The user session
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                # Move to end (LRU)
                self._sessions.move_to_end(session_id)
                return session

            # Create new session
            session = UserSession(session_id=session_id)
            self._sessions[session_id] = session

            # Evict oldest if at capacity
            while len(self._sessions) > self.max_sessions:
                oldest_id = next(iter(self._sessions))
                del self._sessions[oldest_id]

            return session

    def get(self, session_id: str) -> Optional[UserSession]:
        """
        Get session by ID.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        Optional[UserSession]
            The session, or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
                self._sessions.move_to_end(session_id)
            return session

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        Remove expired sessions.

        Returns
        -------
        int
            Number of sessions removed
        """
        now = datetime.now()
        with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if (now - session.last_activity).total_seconds() > self.session_ttl_seconds
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.

        Returns
        -------
        List[Dict[str, Any]]
            List of session summaries
        """
        with self._lock:
            return [session.get_info() for session in self._sessions.values()]

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


# Global session store instance (created on import)
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """
    Get the global session store.

    Creates the store on first access.

    Returns
    -------
    SessionStore
        The global session store
    """
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


__all__ = [
    "SessionStore",
    "UserSession",
    "get_session_store",
    "get_current_session",
    "set_current_session",
    "reset_current_session",
]
