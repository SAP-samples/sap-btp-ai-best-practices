"""Process-local service factory; all durable application state is in HANA."""

from functools import lru_cache
from .repository import HanaRepository
from .service import WorkspaceService


@lru_cache(maxsize=1)
def get_service():
    """Construct the shared service without initializing model or database on import."""
    return WorkspaceService(HanaRepository())
