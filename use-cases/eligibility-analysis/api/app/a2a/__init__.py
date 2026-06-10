"""A2A LangGraph agent package for eligibility queries."""
try:
    from .a2a_server import router as a2a_router
except ImportError:  # pragma: no cover
    a2a_router = None

__all__ = ["a2a_router"]
