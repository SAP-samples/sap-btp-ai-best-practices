"""
Factory for creating LLM instances.

This module provides a factory function to create LLM instances
based on model names, handling provider selection automatically.
"""

import os
import logging
from typing import Optional, Dict, Any
from . import config as cfg


logger = logging.getLogger(__name__)


def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> Any:
    """
    Create an LLM instance based on model name.
    
    Args:
        model_name: Name of the model to use. If None, uses DEFAULT_MODEL
                   or LLM_MODEL environment variable.
        temperature: Temperature for generation (0.0 = deterministic)
        **kwargs: Additional model-specific parameters
        
    Returns:
        A langchain chat model instance (ChatOpenAI, ChatBedrock, or ChatVertexAI)
        
    Raises:
        ValueError: If model_name is not supported
        ImportError: If required dependencies are not installed
    """
    # Determine model name
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", cfg.DEFAULT_MODEL)
    
    # Validate model name
    if model_name not in cfg.MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {', '.join(cfg.MODEL_CONFIGS.keys())}"
        )
    
    # Get model configuration
    config = cfg.MODEL_CONFIGS[model_name]
    provider = config["provider"]
    
    # Handle model name aliases
    actual_model_name = config.get("actual_model_name", model_name)
    
    # Merge default configuration with provided kwargs
    provider_defaults = cfg.PROVIDER_DEFAULTS.get(provider, {})
    all_kwargs = {**provider_defaults, **kwargs}
    
    logger.info(
        f"Creating LLM instance: model={actual_model_name}, "
        f"provider={provider}, temperature={temperature}"
    )
    
    try:
        if provider == "openai":
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
            
            # Check if model supports temperature
            supports_temperature = config.get("supports_temperature", True)
            
            if supports_temperature:
                return ChatOpenAI(
                    proxy_model_name=actual_model_name,
                    temperature=temperature,
                    **all_kwargs
                )
            else:
                # O3 and similar models don't support temperature
                return ChatOpenAI(
                    proxy_model_name=actual_model_name,
                    **all_kwargs
                )
        
        elif provider == "bedrock":
            from gen_ai_hub.proxy.langchain.amazon import ChatBedrock
            return ChatBedrock(
                model_name=actual_model_name,
                temperature=temperature,
                model_kwargs={
                    "top_p": all_kwargs.get('top_p', 1.0)
                }
            )
        
        elif provider == "vertexai":
            from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI
            from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel
            
            chat = ChatVertexAI(
                model_name=actual_model_name,
                temperature=temperature,
                **all_kwargs
            )
            
            # Set up the genaihub client as shown in the notebook
            try:
                chat.genaihub_client = GenerativeModel(
                    model_name=actual_model_name,
                    proxy_client=None  # Will use default proxy client
                )
            except Exception as e:
                # Log warning but continue - some versions might not need this
                logger.warning(f"Could not set genaihub_client: {e}")
            
            return chat
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    except Exception as e:
        logger.error(f"Failed to create LLM instance: {str(e)}")
        raise


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model is not supported
    """
    if model_name not in cfg.MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    return cfg.MODEL_CONFIGS[model_name]