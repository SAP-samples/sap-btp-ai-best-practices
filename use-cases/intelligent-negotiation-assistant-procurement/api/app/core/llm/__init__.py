"""
LLM abstraction layer for Knowledge Graph extraction.

This module provides a unified interface for different LLM providers
through the gen_ai_hub proxy.
"""

from .factory import create_llm, get_model_info
from .config import MODEL_CONFIGS, SUPPORTED_MODELS, DEFAULT_MODEL

__all__ = [
    "create_llm",
    "get_model_info",
    "MODEL_CONFIGS",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL"
]