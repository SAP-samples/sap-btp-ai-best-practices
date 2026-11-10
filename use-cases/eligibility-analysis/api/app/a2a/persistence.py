"""HANA conversation service and authenticated shared-key domain identity."""
import hashlib
import os
from functools import lru_cache
from .conversation_store import ConversationStore
from ..services.database.backend import get_backend


def access_domain():
    """Derive an opaque domain from server authentication; never persist the credential."""
    key=os.getenv('API_KEY')
    if not key:raise ValueError('API authentication is not configured')
    return hashlib.sha256(('a2a-access-domain:'+key).encode()).hexdigest()


@lru_cache(maxsize=1)
def get_conversation_store():
    """Require HANA for application conversation history and initialize its tables."""
    backend=get_backend()
    if not backend.is_hana:raise ValueError('HANA is required for durable assistant conversations')
    return ConversationStore(backend)
