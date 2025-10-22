"""
Model configurations and constants.

This module contains configuration for all supported models including
their default parameters and capabilities.
"""

from typing import Dict, Any, List

# Model configuration with default parameters
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-4.1": {
        "provider": "openai",
        "temperature": 0.0,
        "description": "OpenAI GPT-4.1 - Enhanced reasoning and analysis",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 800000  # 800K tokens context window
    },
    "o3": {
        "provider": "openai",
        "temperature": 0.0,
        "description": "OpenAI O3 - Advanced reasoning model",
        "supports_json": True,
        "supports_temperature": False,  # O3 doesn't support temperature parameter
        "max_tokens": 128000  # 128K tokens context window
    },
    "gpt-5": {
        "provider": "openai",
        "temperature": 0.0,
        "description": "OpenAI GPT-5 - Enhanced reasoning and analysis",
        "supports_json": True,
        "supports_temperature": False,
        "max_tokens": 250000  # 250K tokens context window
    },
    "gpt-5-mini": {
        "provider": "openai",
        "temperature": 0.0,
        "description": "OpenAI GPT-5 Mini - Enhanced reasoning and analysis",
        "supports_json": True,
        "supports_temperature": False,
        "max_tokens": 250000  # 250K tokens context window
    },
    "gemini-2.5-pro": {
        "provider": "vertexai",
        "temperature": 0.0,
        "description": "Google Gemini 2.5 Pro - Large context window and multimodal",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 800000  # 800K tokens context window
    },
    "gemini-2.5-flash": {
        "provider": "vertexai",
        "temperature": 0.0,
        "description": "Google Gemini 2.5 Flash - Fast and efficient model",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 800000  # 800K tokens context window
    },
    "anthropic--claude-4-sonnet": {
        "provider": "bedrock",
        "temperature": 0.0,
        "description": "Anthropic Claude 4 Sonnet - Balanced performance and reasoning",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 200000  # 200K tokens context window
    },
    # Alias for easier use
    "claude-4-sonnet": {
        "provider": "bedrock",
        "temperature": 0.0,
        "description": "Anthropic Claude 4 Sonnet - Balanced performance and reasoning",
        "supports_json": True,
        "supports_temperature": True,
        "max_tokens": 200000,  # 200K tokens context window
        "actual_model_name": "anthropic--claude-4-sonnet"
    }
}

# List of supported models
SUPPORTED_MODELS: List[str] = list(MODEL_CONFIGS.keys())

# Default model if none specified
DEFAULT_MODEL = "gpt-4.1"

# Provider-specific defaults
PROVIDER_DEFAULTS = {
    "openai": {
        "request_timeout": 600,
        "max_retries": 3
    },
    "bedrock": {
        "region_name": "us-east-1",
        "request_timeout": 600
    },
    "vertexai": {
        "location": "us-central1",
        # "request_timeout": 600
    }
}